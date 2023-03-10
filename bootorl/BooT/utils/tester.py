# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
import random
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from .sample import sample_rollout, top_k_logits, filter_cdf


def to(xs, device):
    return [x.to(device) for x in xs]


class Tester:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger if hasattr(config, "logger") else None

        self.n_epochs = 0
        self.n_tokens = 0

        # For boot-r (bootstrap-repeat)
        self.generated_dataset = []

    def _sample_from_sequence_logits(
            self, logits, dim_obs, dim_act, num_samples, temperature=1.0,
            k_obs=1, k_act=None, k_rew=1, cdf_obs=None, cdf_act=0.6, cdf_rew=None,
            real_rew=True, real_rtg=True,
        ):
        # Sample tokens from a sequence of logits, used in teacher-forcing generation
        bs, vs = logits.shape[0], logits.shape[-1]
        logits = logits / temperature
        confidence = 0
        raw_probs = logits.softmax(dim=-1)

        obs_logits = logits[:, :, :dim_obs].reshape(-1, vs)
        if cdf_obs is not None:
            obs_logits = filter_cdf(obs_logits, cdf_obs)
        if k_obs is not None:
            obs_logits = top_k_logits(obs_logits, k_obs)
        obs_probs = obs_logits.softmax(dim=-1)
        obs_indices = torch.multinomial(obs_probs, num_samples=num_samples, replacement=True)
        obs_indices = obs_indices.reshape(bs, -1, dim_obs, num_samples).permute(3, 0, 1, 2)

        act_logits = logits[:, :, dim_obs: dim_obs + dim_act].reshape(-1, vs)
        if cdf_act is not None:
            act_logits = filter_cdf(act_logits, cdf_act)
        if k_act is not None:
            act_logits = top_k_logits(act_logits, max(k_act, num_samples))
        act_probs = act_logits.softmax(dim=-1)
        act_indices = torch.multinomial(act_probs, num_samples=num_samples, replacement=True)
        act_indices = act_indices.reshape(bs, -1, dim_act, num_samples).permute(3, 0, 1, 2)

        rew_logits = logits[:, :, -2: -1].reshape(-1, vs)
        if cdf_rew is not None:
            rew_logits = filter_cdf(rew_logits, cdf_rew)
        if k_rew is not None:
            rew_logits = top_k_logits(rew_logits, max(k_rew, num_samples))
        rew_probs = rew_logits.softmax(dim=-1)
        rew_indices = torch.multinomial(rew_probs, num_samples=num_samples, replacement=True)
        rew_indices = rew_indices.reshape(bs, -1, 1, num_samples).permute(3, 0, 1, 2)

        rtg_logits = logits[:, :, -1:].reshape(-1, vs)
        if cdf_rew is not None:
            rtg_logits = filter_cdf(rtg_logits, cdf_rew)
        if k_rew is not None:
            rtg_logits = top_k_logits(rtg_logits, max(k_rew, num_samples))
        rtg_probs = rtg_logits.softmax(dim=-1)
        rtg_indices = torch.multinomial(rtg_probs, num_samples=num_samples, replacement=True)
        rtg_indices = rtg_indices.reshape(bs, -1, 1, num_samples).permute(3, 0, 1, 2)

        raw_probs = raw_probs.reshape(-1, vs)
        # (num_samples, batch_size, rollout_len, dim_obs + dim_act) --> (-1, dim_obs + dim_act)
        raw_probs_indices = torch.arange(raw_probs.shape[0]).repeat(num_samples)
        chosen_indices = torch.cat([obs_indices, act_indices, rew_indices, rtg_indices], dim=-1).reshape(-1)
        raw_probs = raw_probs[raw_probs_indices, chosen_indices].reshape(num_samples, bs, -1, dim_obs + dim_act + 2)
        chosen_indices = chosen_indices.reshape(num_samples, bs, -1, dim_obs + dim_act + 2)
        confidence = torch.log10(raw_probs + 1e-15).sum(dim=(2, 3))

        return chosen_indices, confidence, raw_probs

    def generate(self, model, x, y, mask, generation_type, **kwargs):
        if generation_type == "teacherforcing" or generation_type == "tf":
            x_generate, mask, confidence = self.generate_teacher_forcing(model, x, y, mask, **kwargs)
        elif generation_type == "autoregressive" or generation_type == "ar":
            x_generate, mask, confidence = self.generate_autoregressive(model, x, y, mask, **kwargs)
        else:
            raise NotImplementedError()
        return x_generate, mask, confidence

    def generate_autoregressive(self, model, x, y, mask, **kwargs):
        raw_model = model.module if hasattr(model, "module") else model
        obs_dim, act_dim, trans_dim = raw_model.observation_dim, raw_model.action_dim, raw_model.transition_dim
        x_crop = torch.cat([x, y[:, [-1]]], dim=1).repeat_interleave(kwargs.get("generation_num", 4), dim=0)
        start_idx = x_crop.shape[1] - kwargs.get("generation_len", 1) * trans_dim
        x_generate = x_crop[:, :start_idx]
        bs = x_generate.shape[0]

        # Indices used for confidence calculation
        # row_indices: like (0, ...(dim-2 repeats), 0, 1, .., 1, ..., bs-1, ..., bs-1)
        # dim_indices: like (0, 1, 2, ..., dim-1, ...(bs-2 repeats), 0, 1, 2, ..., dim-1)
        obs_row_indices = torch.arange(bs).repeat_interleave(obs_dim, dim=0)
        obs_dim_indices = torch.arange(obs_dim).unsqueeze(0).repeat(bs, 1).flatten()
        act_row_indices = torch.arange(bs).repeat_interleave(act_dim, dim=0)
        act_dim_indices = torch.arange(act_dim).unsqueeze(0).repeat(bs, 1).flatten()
        rew_row_indices = torch.arange(bs)
        rew_dim_indices = torch.zeros(bs, dtype=torch.long)

        confidence_sum = torch.zeros(bs, device=x.device)
        confidence_nums = torch.zeros(1, device=x.device)

        generation_len = kwargs.get("generation_len", 1)
        for i in range(generation_len):
            r_idx = start_idx + (i + 1) * trans_dim - 2

            x_generate, obs_probs = sample_rollout(
                model, x_generate, obs_dim, topk=kwargs.get("k_obs", 1), cdf=kwargs.get("cdf_obs", None)
            )
            obs_indices = x_generate[:, -obs_dim:].flatten()
            obs_probs = obs_probs[obs_row_indices, obs_dim_indices, obs_indices].reshape(bs, obs_dim)
            confidence_sum += torch.log10(obs_probs + 1e-15).sum(dim=-1)
            confidence_nums += obs_dim

            x_generate, act_probs = sample_rollout(
                model, x_generate, act_dim, topk=kwargs.get("k_act", None), cdf=kwargs.get("cdf_act", 0.6)
            )
            act_indices = x_generate[:, -act_dim:].flatten()
            act_probs = act_probs[act_row_indices, act_dim_indices, act_indices].reshape(bs, act_dim)
            confidence_sum += torch.log10(act_probs + 1e-15).sum(dim=-1)
            confidence_nums += act_dim

            if kwargs.get("generation_real_r", True):
                x_generate = torch.cat([x_generate, x_crop[:, r_idx: r_idx + 1]], dim=1)
            else:
                x_generate, rew_probs = sample_rollout(
                    model, x_generate, 1, topk=kwargs.get("k_rew", 1), cdf=kwargs.get("cdf_rew", None)
                )
                rew_indices = x_generate[:, -1:].flatten()
                rew_probs = rew_probs[rew_row_indices, rew_dim_indices, rew_indices].reshape(bs, 1)
                confidence_sum += torch.log10(rew_probs + 1e-15).sum(dim=-1)
                confidence_nums += 1

            if kwargs.get("generation_real_R", True):
                x_generate = torch.cat([x_generate, x_crop[:, r_idx + 1: r_idx + 2]], dim=1)
            else:
                x_generate, val_probs = sample_rollout(
                    model, x_generate, 1, topk=kwargs.get("k_rew", 1), cdf=kwargs.get("cdf_rew", None)
                )
                val_indices = x_generate[:, -1:].flatten()
                val_probs = val_probs[rew_row_indices, rew_dim_indices, val_indices].reshape(bs, 1)
                confidence_sum += torch.log10(val_probs + 1e-15).sum(dim=-1)
                confidence_nums += 1
        
        confidence = confidence_sum / confidence_nums
        mask = mask.repeat_interleave(kwargs.get("generation_num", 4), dim=0)

        if kwargs.get("generation_confidence_type") == "thresh":
            # Filter by a hard confidence threshold
            confidence_thresh = kwargs.get("generation_confidence_factor", -0.4)
            if confidence_thresh > 0:
                confidence_thresh = -confidence_thresh
            x_generate, mask, confidence = self.filter_by_confidence_thresh(
                x_generate, mask, confidence, confidence_thresh
            )
        elif kwargs.get("generation_confidence_type") == "ratio":
            # Filter by a percentage of trajectories with top confidence
            confidence_ratio = kwargs.get("generation_confidence_factor", 0.1)
            x_generate, mask, confidence = self.filter_by_confidence_ratio(
                x_generate, mask, confidence, confidence_ratio
            )
        elif kwargs.get("generation_confidence_type") is None or kwargs.get("generation_confidence_type") == "none":
            pass
        else:
            raise NotImplementedError()

        return x_generate, mask, confidence

    def generate_teacher_forcing(self, model, x, y, mask, **kwargs):
        # Rollout arguments
        generation_len = kwargs.get("generation_len", 1)
        generation_num = kwargs.get("generation_num", 4)
        generation_real_r = kwargs.get("generation_real_r", True)
        generation_real_R = kwargs.get("generation_real_R", True)

        raw_model = model.module if hasattr(model, "module") else model
        obs_dim, act_dim, trans_dim = raw_model.observation_dim, raw_model.action_dim, raw_model.transition_dim
        x_next_logits, _, _ = model(x, return_info=False)
        batch_size, vocab_size = x_next_logits.shape[0], x_next_logits.shape[-1]
        x_origin = torch.cat([x, y[:, [-1]]], dim=1)
        x_generate = x_origin.reshape(batch_size, -1, trans_dim)
        x_next_logits = x_next_logits[:, -generation_len * trans_dim:].reshape(
            batch_size, generation_len, trans_dim, vocab_size
        )
        chosen_indices, confidence, raw_probs = self._sample_from_sequence_logits(
            x_next_logits, dim_obs=obs_dim, dim_act=act_dim, num_samples=generation_num, temperature=1
        )
        # (generation_num, batch_size, timestep_num, transition_dim)
        x_generate = x_generate.unsqueeze(0).repeat(generation_num, 1, 1, 1)
        x_generate[:, :, -generation_len:, :-2] = chosen_indices[:, :, :, :-2]
        if not generation_real_r:
            x_generate[:, :, -generation_len:, [-2]] = chosen_indices[:, :, :, [-2]]
        if not generation_real_R:
            x_generate[:, :, -generation_len:, [-1]] = chosen_indices[:, :, :, [-1]]
        x_generate = x_generate.reshape(batch_size * generation_num, -1)
        confidence = confidence.reshape(batch_size * generation_num)
        mask = mask.repeat(generation_num, 1)

        if kwargs.get("generation_confidence_type") == "thresh":
            # Filter by a hard confidence threshold
            confidence_thresh = kwargs.get("generation_confidence_factor", -0.4)
            if confidence_thresh > 0:
                confidence_thresh = -confidence_thresh
            x_generate, mask, confidence = self.filter_by_confidence_thresh(
                x_generate, mask, confidence, confidence_thresh
            )
        elif kwargs.get("generation_confidence_type") == "ratio":
            # Filter by a percentage of trajectories with top confidence
            confidence_ratio = kwargs.get("generation_confidence_factor", 0.1)
            x_generate, mask, confidence = self.filter_by_confidence_ratio(
                x_generate, mask, confidence, confidence_ratio
            )
        elif kwargs.get("generation_confidence_type") is None or kwargs.get("generation_confidence_type") == "none":
            pass
        else:
            raise NotImplementedError()

        return x_generate, mask, confidence

    def filter_by_confidence_thresh(self, x, mask, confidence, confidence_thresh):
        chosen = confidence > confidence_thresh
        return x[chosen], mask[chosen], confidence[chosen]

    def filter_by_confidence_ratio(self, x, mask, confidence, confidence_ratio):
        rollout_num = x.shape[0]
        chosen_num = int(rollout_num * confidence_ratio)
        _, chosen = torch.topk(confidence, chosen_num)
        return x[chosen], mask[chosen], confidence[chosen]

    def generate_data(
        self, model, dataset, n_epochs=1, with_tqdm=True, log_freq=10, bootstrap_kwargs=None,
    ):
        config = self.config
        device = next(model.parameters()).device
        model.train(False)

        loader = DataLoader(
            dataset, shuffle=True, pin_memory=True, batch_size=config.batch_size, num_workers=config.num_workers
        )

        info_dict = {
            "loss": [], "acc": [], "nei_acc": [], "err": [], "err_2": [],
            "sequence_num": 0, "bootstrap_num": 0, "bootstrap_loss": [], "bootstrap_acc": [],
            "bootstrap_nei_acc": [], "bootstrap_err": [], "bootstrap_err_2": [],
        }

        for _ in range(n_epochs):
            pbar = tqdm(enumerate(loader), total=len(loader), leave=False) if with_tqdm else enumerate(loader)
            for it, (batch_X, batch_Y, mask, batch_origin, indices) in pbar:
                if it >= 50:  break

                indices = torch.stack(indices, dim=1)
                batch_X, batch_Y, mask = to([batch_X, batch_Y, mask], device)

                with torch.set_grad_enabled(False):
                    batch_generated_tf_realr, _, _ = self.generate(
                        model, batch_X, batch_Y, mask, 
                        generation_type="tf", generation_len=1, generation_num=1,
                        generation_confidence_type=None, generation_confidence_factor=0,
                        generation_real_r=True, generation_real_R=True
                    )
                    batch_generated_ar_realr, _, _ = self.generate(
                        model, batch_X, batch_Y, mask, 
                        generation_type="ar", generation_len=1, generation_num=1,
                        generation_confidence_type=None, generation_confidence_factor=0,
                        generation_real_r=True, generation_real_R=True
                    )
                    batch_generated_tf_genr, _, _ = self.generate(
                        model, batch_X, batch_Y, mask, 
                        generation_type="tf", generation_len=1, generation_num=1,
                        generation_confidence_type=None, generation_confidence_factor=0,
                        generation_real_r=False, generation_real_R=False
                    )
                    batch_generated_ar_genr, _, _ = self.generate(
                        model, batch_X, batch_Y, mask, 
                        generation_type="ar", generation_len=1, generation_num=1,
                        generation_confidence_type=None, generation_confidence_factor=0,
                        generation_real_r=False, generation_real_R=False
                    )
                
                # print(batch_Y[-1, -10:])
                # print(batch_generated_tf_realr[-1, -10:])
                # print(batch_generated_ar_realr[-1, -10:])
                # print(batch_generated_tf_genr[-1, -10:])
                # print(batch_generated_ar_genr[-1, -10:])

                augmented_data = [
                    {
                        "indices": indices[i].clone().detach().cpu(),
                        "origin": batch_origin[i].clone().detach().cpu(), 
                        "discretized": (batch_X[i].clone().detach().cpu(), batch_Y[i].clone().detach().cpu()),
                        "generated_tf_realr": batch_generated_tf_realr[i].clone().detach().cpu(),
                        "generated_ar_realr": batch_generated_ar_realr[i].clone().detach().cpu(),
                        "generated_tf_genr": batch_generated_tf_genr[i].clone().detach().cpu(),
                        "generated_ar_genr": batch_generated_ar_genr[i].clone().detach().cpu(),
                        "mask": mask[i].clone().detach().cpu(),
                    }
                    for i in range(batch_origin.shape[0])
                ]

                self.generated_dataset.extend(augmented_data)

                # report progress
                if with_tqdm:
                    desc = f"Epoch {self.n_epochs:>3}, Iter {it:>4} | bootstrap num {info_dict['bootstrap_num']}"
                    pbar.set_description(desc)
                else:
                    if it % log_freq == 0 or it == len(loader) - 1:
                        if self.logger is not None:
                            self.logger.debug(f"Epoch {self.n_epochs:>3} [ {it:4d} / {len(loader):4d} ]  "
                                              f"bootstrap num {info_dict['bootstrap_num']}")
                        else:
                            print(f"[ utils/trainer ] Epoch {self.n_epochs:>3} [ {it:4d} / {len(loader):4d} ]  "
                                  f"bootstrap num {info_dict['bootstrap_num']}")

            self.n_epochs += 1

        return self.generated_dataset
