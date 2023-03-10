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


class Trainer:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger if hasattr(config, "logger") else None

        self.n_epochs = 0
        self.n_tokens = 0
        self.optimizer = None

        # For boot-r (bootstrap-repeat)
        self.generated_dataset = []

    def get_optimizer(self, model):
        if self.optimizer is None:
            if self.logger is not None:
                self.logger.debug(f"Making optimizer at epoch {self.n_epochs}")
            else:
                print(f'[ utils/trainer ] Making optimizer at epoch {self.n_epochs}')
            raw_model = model.module if hasattr(model, "module") else model
            self.optimizer = raw_model.configure_optimizers(self.config)
        return self.optimizer

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
        obs_row_indices = torch.arange(bs).repeat_interleave(obs_dim, dim=0)
        obs_dim_indices = torch.arange(obs_dim).unsqueeze(0).repeat(bs, 1).flatten()
        act_row_indices = torch.arange(bs).repeat_interleave(act_dim, dim=0)
        act_dim_indices = torch.arange(act_dim).unsqueeze(0).repeat(bs, 1).flatten()
        rew_row_indices = torch.arange(bs)
        rew_dim_indices = torch.zeros(bs).long()

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
        # (rollout_num, batch_size, timestep_num, transition_dim)
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

    def train(
        self, model, dataset, n_epochs=1, with_tqdm=True, log_freq=100, bootstrap_kwargs=None,
    ):
        # Rollout arguments
        bootstrap = bootstrap_kwargs.get("bootstrap", True)
        bootstrap_type = bootstrap_kwargs.get("bootstrap_type", "once")
        generation_type = bootstrap_kwargs.get("generation_type", "autoregressive")
        generation_epoch_thresh = bootstrap_kwargs.get("generation_epoch_thresh", 20)
        generation_len = bootstrap_kwargs.get("generation_len", 1)
        generation_num = bootstrap_kwargs.get("generation_num", 4)
        generation_confidence_type = bootstrap_kwargs.get("generation_confidence_type", "ratio")
        generation_confidence_factor = bootstrap_kwargs.get("generation_confidence_factor", 0.10)
        generation_real_r = bootstrap_kwargs.get("generation_real_r", True)
        generation_real_R = bootstrap_kwargs.get("generation_real_R", True)

        config = self.config
        optimizer = self.get_optimizer(model)
        device = next(model.parameters()).device
        model.train(True)
        vocab_size = dataset.n_bins if hasattr(dataset, "n_bins") else -1
        perform_bootstrap = (bootstrap and self.n_epochs >= generation_epoch_thresh)

        loader = DataLoader(
            dataset, shuffle=True, pin_memory=True, batch_size=config.batch_size, num_workers=config.num_workers
        )

        info_dict = {
            "loss": [], "acc": [], "nei_acc": [], "err": [], "err_2": [],
            "sequence_num": 0, "bootstrap_num": 0, "bootstrap_loss": [], "bootstrap_acc": [],
            "bootstrap_nei_acc": [], "bootstrap_err": [], "bootstrap_err_2": [],
        }

        for _ in range(n_epochs):
            if perform_bootstrap and bootstrap_type == "repeat" and len(self.generated_dataset) > 0:
                generated_dataset_loader = DataLoader(
                    self.generated_dataset, shuffle=True, pin_memory=True,
                    batch_size=config.batch_size, num_workers=config.num_workers
                )
                pbar = tqdm(enumerate(generated_dataset_loader), leave=False) if with_tqdm \
                       else enumerate(generated_dataset_loader)
                for it, (batch_X, batch_Y, mask) in pbar:
                    batch_X, batch_Y, mask = to([batch_X, batch_Y, mask], device)
                    with torch.set_grad_enabled(True):
                        logits, loss, info = model(batch_X, batch_Y, mask)
                        loss = loss.mean()
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    info_dict["bootstrap_num"] += batch_X.shape[0]
                    for k in ["loss", "acc", "nei_acc", "err", "err_2"]:
                        info_dict["bootstrap_" + k].append(info[k])
                    
                    # This is only to display current LR. LR decay is not performed when training with augmented data
                    if config.lr_decay:
                        if self.n_tokens < config.warmup_tokens:
                            lr_mult = float(self.n_tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.n_tokens - config.warmup_tokens) / \
                                       float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = 0.1 if progress > 0.5 else max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                    else:
                        lr = config.learning_rate

                    if with_tqdm:
                        desc = f"Epoch {self.n_epochs:>3} Boot-r, Iter {it:>4} | loss {loss.item():.5f} | " \
                               f"lr {lr:.4e} | bootstrap num {info_dict['bootstrap_num']}"
                        pbar.set_description(desc)
                    else:
                        if it % log_freq == 0 or it == len(generated_dataset_loader) - 1:
                            if self.logger is not None:
                                self.logger.debug(f"Epoch {self.n_epochs:>3} Boot-r [ {it:4d} / {len(generated_dataset_loader):4d} ]  "
                                                  f"train loss {loss.item():.5f} | lr {lr:.3e} | "
                                                  f"bootstrap num {info_dict['bootstrap_num']}")
                            else:
                                print(f"[ utils/trainer ] Epoch {self.n_epochs:>3} Boot-r "
                                      f"[ {it:4d} / {len(generated_dataset_loader):4d} ]  train loss {loss.item():.5f} | "
                                      f"lr {lr:.3e} | bootstrap num {info_dict['bootstrap_num']}")


            pbar = tqdm(enumerate(loader), total=len(loader), leave=False) if with_tqdm else enumerate(loader)
            for it, (batch_X, batch_Y, mask) in pbar:

                # Train with original data
                batch_X, batch_Y, mask = to([batch_X, batch_Y, mask], device)
                with torch.set_grad_enabled(True):
                    logits, loss, info = model(batch_X, batch_Y, mask)
                    loss = loss.mean()
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()
                info_dict["sequence_num"] += batch_X.shape[0]
                for k in ["loss", "acc", "nei_acc", "err", "err_2"]:
                    info_dict[k].append(info[k])

                # Train with generated data as scheduled
                if perform_bootstrap:
                    with torch.set_grad_enabled(False):
                        batch_generated, mask_generated, confidence_generated = self.generate(
                            model, batch_X, batch_Y, mask, 
                            generation_type=generation_type,
                            generation_len=generation_len,
                            generation_num=generation_num,
                            generation_confidence_type=generation_confidence_type,
                            generation_confidence_factor=generation_confidence_factor,
                            generation_real_r=generation_real_r,
                            generation_real_R=generation_real_R
                        )
                    batch_X_generated = batch_generated[:, :-1].clone().detach()
                    batch_Y_generated = batch_generated[:, 1:].clone().detach()

                    if bootstrap_type == "once":
                        with torch.set_grad_enabled(True):
                            logits_rollout, loss_rollout, info_rollout = model(
                                batch_X_generated, batch_Y_generated, mask_generated
                            )
                            loss_rollout = loss_rollout.mean()
                        model.zero_grad()
                        loss_rollout.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()
                        info_dict["bootstrap_num"] += batch_X_generated.shape[0]
                        for k in ["loss", "acc", "nei_acc", "err", "err_2"]:
                            info_dict["bootstrap_" + k].append(info_rollout[k])
                    elif bootstrap_type == "repeat":
                        batch_X_generated = batch_X_generated.cpu()
                        batch_Y_generated = batch_Y_generated.cpu()
                        mask_generated = mask_generated.cpu()
                        augmented_data = [
                            (batch_X_generated[i], batch_Y_generated[i], mask_generated[i]) 
                            for i in range(batch_X_generated.shape[0])
                        ]
                        self.generated_dataset.extend(augmented_data)
                    else:
                        raise NotImplementedError()

                # decay the learning rate based on our progress
                if config.lr_decay:
                    self.n_tokens += (batch_Y != vocab_size).sum() # number of tokens processed this step
                    if self.n_tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.n_tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.n_tokens - config.warmup_tokens) / \
                                   float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = 0.1 if progress > 0.5 else max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                # report progress
                if with_tqdm:
                    desc = f"Epoch {self.n_epochs:>3}, Iter {it:>4} | loss {loss.item():.5f} | lr {lr:.4e} | " \
                           f"bootstrap num {info_dict['bootstrap_num']}"
                    pbar.set_description(desc)
                else:
                    if it % log_freq == 0 or it == len(loader) - 1:
                        if self.logger is not None:
                            self.logger.debug(f"Epoch {self.n_epochs:>3} [ {it:4d} / {len(loader):4d} ]  "
                                              f"train loss {loss.item():.5f} | lr {lr:.3e} | "
                                              f"bootstrap num {info_dict['bootstrap_num']}")
                        else:
                            print(f"[ utils/trainer ] Epoch {self.n_epochs:>3} [ {it:4d} / {len(loader):4d} ]  "
                                  f"train loss {loss.item():.5f} | lr {lr:.3e} | "
                                  f"bootstrap num {info_dict['bootstrap_num']}")

            info_ = {}
            for k, v in info_dict.items():
                if isinstance(v, list) and len(v) > 0:
                    info_[k] = torch.cat(to(v, v[0].device), dim=0).nanmean().item()
                elif isinstance(v, int) or isinstance(v, float):
                    info_[k] = v

            self.n_epochs += 1

        return info_
