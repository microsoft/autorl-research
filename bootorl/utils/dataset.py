# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
import gym
import os
import contextlib
from itertools import accumulate


@contextlib.contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stderr(fnull) as err, contextlib.redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    import d4rl


def load_environment(name):
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env


class Discretizer:
    def __init__(self, data, n_bins):
        # Percentile discretization to each dimension of `data`.
        # Expect data shape (N, transition_dim)
        bins = np.percentile(data, np.linspace(0, 100, n_bins + 1), axis=0).T
        discrete_data = [np.digitize(data[..., dim], bins[dim]) - 1 for dim in range(data.shape[-1])]
        discrete_data = np.stack(np.clip(discrete_data, 0, n_bins - 1), axis=-1)

        self.n_tokens = data.shape[-1]
        self.n_bins = n_bins
        self.bins = bins
        self.data = data
        self.discrete_data = discrete_data

        self._test()

    def __call__(self, x):
        indices = self.discretize(x)
        recon = self.reconstruct(indices)
        error = np.abs(recon - x).max(0)
        return indices, recon, error

    def _test(self):
        inds = np.random.randint(0, len(self.data), size=10)
        X = self.data[inds]
        indices = self.discretize(X)
        recon = self.reconstruct(indices)
        # make sure reconstruction error is less than the max allowed per dimension
        error = np.abs(X - recon).max(0)
        diffs = self.bins[:, 1:] - self.bins[:, :-1]
        assert (error <= diffs.max(axis=1)).all()
        # re-discretize reconstruction and make sure it is the same as original indices
        indices_2 = self.discretize(recon)
        assert (indices == indices_2).all()
        return True

    def discretize(self, x, subslice=(None, None)):
        # Discretize `x`` with shape (N, dim) according to the [subslice[0]: subslice[1]] bins
        # assert (subslice[1] - subslice[0] == x.shape[-1])
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        if x.ndim == 1:
            x = x[None]
        bins = self.bins[subslice[0]: subslice[1]]
        discrete_data = [np.digitize(x[..., dim], bins[dim]) - 1 for dim in range(x.shape[-1])]
        discrete_data = np.stack(np.clip(discrete_data, 0, self.n_bins - 1), axis=-1)
        return discrete_data

    def reconstruct(self, indices, subslice=(None, None)):
        # Reconstruct (discrete) `x` with shape (N, dim) according to the [subslice[0]: subslice[1]] bins
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        if indices.ndim == 1:
            indices = indices[None]
        indices = np.clip(indices, 0, self.n_bins - 1)
        bin_data = (self.bins[subslice[0]: subslice[1], :-1] + self.bins[subslice[0]: subslice[1], 1:]) / 2
        recon = [bin_data[dim, indices[..., dim]] for dim in range(indices.shape[-1])]
        recon = np.stack(recon, axis=-1)
        return recon

    def expectation(self, probs, subslice):
        # Calculate the expectation of `prob` with shape (N, n_bins) according to the `subslice-th` bins
        if torch.is_tensor(probs):
            probs = probs.detach().cpu().numpy()
        if probs.ndim == 1:
            probs = probs[None]
        bin_data = (self.bins[subslice, :-1] + self.bins[subslice, 1:]) / 2
        exp = probs @ bin_data
        return exp

    def percentile(self, probs, percentile, subslice):
        # Reconstruct with the percentile of `prob` with shape (N, n_bins) according to the `subslice-th` bins
        bin_data = (self.bins[subslice, :-1] + self.bins[subslice, 1:]) / 2
        probs_cumsum = np.cumsum(probs, axis=-1)
        indices = (probs_cumsum < percentile).sum(dim=-1).type(torch.long)
        return bin_data[indices]

    def value_expectation(self, probs):
        # Calculate expected reward and reward-to-go (value)
        probs = probs[..., :-1]
        torch_device = probs.device
        torch_dtype = probs.dtype
        if torch.is_tensor(probs):
            probs = probs.detach().cpu().numpy()
            return_torch = True
        else:
            return_torch = False
        rewards = self.expectation(probs[:, 0], subslice=-2)
        next_values = self.expectation(probs[:, 1], subslice=-1)
        if return_torch:
            rewards = torch.tensor(rewards, dtype=torch_dtype, device=torch_device)
            next_values = torch.tensor(next_values, dtype=torch_dtype, device=torch_device)
        return rewards, next_values

    def value_fn(self, probs, percentile):
        torch_device = probs.device
        torch_dtype = probs.dtype
        if percentile == 'mean':
            return self.value_expectation(probs)
        else:
            percentile = float(percentile)
        probs = probs[..., :-1]
        if torch.is_tensor(probs):
            probs = probs.detach().cpu().numpy()
            return_torch = True
        else:
            return_torch = False
        rewards = self.percentile(probs[:, 0], percentile, subslice=-2)
        next_values = self.percentile(probs[:, 1], percentile, subslice=-1)
        if return_torch:
            rewards = torch.tensor(rewards, dtype=torch_dtype, device=torch_device)
            next_values = torch.tensor(next_values, dtype=torch_dtype, device=torch_device)
        return rewards, next_values


class DiscretizedDataset(torch.utils.data.Dataset):
    def __init__(
        self, env, n_bins=100, sequence_length=10, discount=0.99, max_path_length=1000, penalty=None,
        device='cuda:0', discretizer=None, logger=None,
    ):
        self.logger = logger
        if self.logger is not None:
            logger.debug(f"Sequence length: {sequence_length} | Max path length: {max_path_length}")
        else:
            print(f'[ utils/datasets ] Sequence length: {sequence_length} | Max path length: {max_path_length}')
        self.env = load_environment(env) if type(env) is str else env
        self.sequence_length = sequence_length
        self.discount = discount
        self.penalty = penalty
        self.max_path_length = max_path_length
        self.device = device

        # Load raw dataset
        raw_dataset = self.env.unwrapped.get_dataset()
        s = raw_dataset['observations'].astype(np.float64)
        a = raw_dataset['actions'].astype(np.float64)
        r = raw_dataset['rewards'].astype(np.float64)
        raw_r = raw_dataset['rewards'].astype(np.float64)
        done_idxs = np.argwhere(raw_dataset["terminals"] | raw_dataset["timeouts"]).flatten()
        penalty_idxs = np.argwhere(raw_dataset["terminals"]).flatten()
        idx_ = [0] + (done_idxs + 1).tolist()

        # Penalize reward (by default subtract 100) if stopped before timestep limit
        if penalty is not None:
            r[penalty_idxs] = penalty
        raw_returns = [raw_r[idx_[i]: idx_[i+1]].sum() for i in range(len(idx_) - 1)]
        returns = [r[idx_[i]: idx_[i+1]].sum() for i in range(len(idx_) - 1)]

        # Calculate reward-to-go
        rtgs = np.zeros_like(r)
        for i in range(done_idxs.shape[0]):
            start_idx = 0 if i == 0 else done_idxs[i - 1] + 1
            reversed_rewards = r[start_idx: done_idxs[i] + 1][::-1]
            cumulative_rewards = list(accumulate(reversed_rewards, lambda x, y: x * discount + y))
            rtgs[start_idx: done_idxs[i]] = cumulative_rewards[-2::-1]

        # Concatenate (s, a, r, R), and pad to full length (1000)
        joined = np.concatenate([s, a, r[..., None], rtgs[..., None]], axis=-1)
        joined_seg = [joined[idx_[i]: idx_[i+1]] for i in range(len(idx_) - 1)]
        joined_seg_len = [len(l) for l in joined_seg]
        joined_segmented = np.zeros((len(joined_seg), max_path_length + sequence_length - 1, joined.shape[-1]), dtype=np.float64)
        termination_flags = np.zeros((len(joined_seg), max_path_length + sequence_length - 1), dtype=bool)
        for i in range(len(joined_seg)):
            joined_segmented[i, :len(joined_seg[i])] = joined_seg[i]
            termination_flags[i, len(joined_seg[i]):] = True

        # Map each index to a trajectory
        indices = []
        for path_ind, length in enumerate(joined_seg_len):
            for i in range(length - 1):
                indices.append((path_ind, i, i + sequence_length))
        indices = np.array(indices)

        self.indices = indices
        self.observation_dim = s.shape[1]
        self.action_dim = a.shape[1]
        self.joined_dim = joined_segmented.shape[-1]
        self.joined_segmented = torch.tensor(joined_segmented, device='cpu', dtype=torch.float64).contiguous()
        self.termination_flags = termination_flags
        self.path_lengths = np.array(joined_seg_len)
        self.raw_returns = np.array(raw_returns)
        self.returns = np.array(returns)

        # Initialize discretizer
        self.discretizer = Discretizer(joined, n_bins) if discretizer is None else discretizer
        self.n_bins = n_bins if discretizer is None else discretizer.n_bins

        joined_discretized = self.discretizer.discretize(joined_segmented)
        joined_discretized[termination_flags] = self.n_bins
        joined_discretized = torch.tensor(joined_discretized, device='cpu', dtype=torch.long).contiguous()
        self.joined_discretized = joined_discretized

        # Used for adding Gaussian noise
        self.raw_data_mean = np.mean(joined, axis=0)
        self.raw_data_std = np.mean(joined, axis=0)

        self._log()
    
    def _log(self):
        logger = self.logger
        if logger is None:
            print(f"Load dataset: {self.env.name}")
            print(f"Observation dim: {self.observation_dim}")
            print(f"Action dim: {self.action_dim}")
            print(f"Joined dim: {self.joined_dim}")
            print(f"Dataset size: {len(self.indices)}")
            print(f"Return mean / std / max: {self.raw_returns.mean():.2f} / "
                  f"{self.raw_returns.std():.2f} / {self.raw_returns.max():.2f}")
            print(f"Length mean / std / max: {self.path_lengths.mean():.2f} / "
                  f"{self.path_lengths.std():.2f} / {self.path_lengths.max():.2f}")
        else:
            logger.debug(f"Load dataset: {self.env.name}")
            logger.debug(f"Observation dim: {self.observation_dim}")
            logger.debug(f"Action dim: {self.action_dim}")
            logger.debug(f"Joined dim: {self.joined_dim}")
            logger.debug(f"Dataset size: {len(self.indices)}")
            logger.debug(f"Return mean / std / max: {self.raw_returns.mean():.2f} / "
                         f"{self.raw_returns.std():.2f} / {self.raw_returns.max():.2f}")
            logger.debug(f"Length mean / std / max: {self.path_lengths.mean():.2f} / "
                         f"{self.path_lengths.std():.2f} / {self.path_lengths.max():.2f}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]
        path_length = self.path_lengths[path_ind]
        joined_discrete = self.joined_discretized[path_ind, start_ind: end_ind]

        # Mask out parts of the trajectories that extend beyond the max path length
        mask = torch.ones(joined_discrete.shape, dtype=torch.bool)
        traj_inds = torch.arange(start_ind, end_ind)
        mask[traj_inds > path_length] = 0

        # Flatten everything
        joined_discrete, mask = joined_discrete.view(-1), mask.view(-1)
        X, Y, mask = joined_discrete[:-1], joined_discrete[1:], mask[1:]
        return X, Y, mask


class NoisyDiscretizedDataset(DiscretizedDataset):
    def set_noisy_length(self, length=1):
        # Add noise to the last `length` timesteps
        self.noisy_length = length

    def add_noise(self, sigma=3e-4, noise_target="s"):
        if noise_target == "s":
            noise_dim = self.observation_dim
        elif noise_target == "sa":
            noise_dim = self.observation_dim + self.action_dim
        else:
            raise NotImplementedError()

        self.gaussian_noise = np.random.normal(scale=sigma, size=self.joined_segmented.shape)
        self.gaussian_noise = self.gaussian_noise * self.raw_data_std + self.raw_data_mean

        self.noisy_joined_segmented = self.joined_segmented
        self.noisy_joined_segmented[..., :noise_dim] += self.gaussian_noise[..., :noise_dim]

        # Discretize noisy data using original discretizer
        noisy_joined_discretized = self.discretizer.discretize(self.noisy_joined_segmented)
        noisy_joined_discretized[self.termination_flags] = self.n_bins
        noisy_joined_discretized = torch.tensor(noisy_joined_discretized, device='cpu', dtype=torch.long).contiguous()
        self.noisy_joined_discretized = noisy_joined_discretized

        if self.logger is not None:
            self.logger.debug(f"Adding Gaussian random noise to target \"{noise_target}\" with sigma {sigma:.2e}")
        else:
            print(f'[ utils/dataset ] Adding Gaussian random noise to target \"{noise_target}\" with sigma {sigma:.2e}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]
        path_length = self.path_lengths[path_ind]
        
        if self.noisy_length >= end_ind - start_ind:
            joined_discrete = self.noisy_joined_discretized[path_ind, start_ind: end_ind]
        else:
            original_joined_discrete = self.joined_discretized[path_ind, start_ind: end_ind - self.noisy_length]
            noisy_joined_discrete = self.noisy_joined_discretized[path_ind, end_ind - self.noisy_length: end_ind]
            joined_discrete = torch.cat([original_joined_discrete, noisy_joined_discrete], dim=0)

        # Mask out parts of the trajectories that extend beyond the max path length
        mask = torch.ones(joined_discrete.shape, dtype=torch.bool)
        traj_inds = torch.arange(start_ind, end_ind)
        mask[traj_inds > path_length] = 0

        # Flatten everything
        joined_discrete, mask = joined_discrete.view(-1), mask.view(-1)
        X, Y, mask = joined_discrete[:-1], joined_discrete[1:], mask[1:]


class AnalyzeDiscretizedDataset(DiscretizedDataset):
    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]
        path_length = self.path_lengths[path_ind]
        joined_origin = self.joined_segmented[path_ind, start_ind: end_ind]
        joined_discrete = self.joined_discretized[path_ind, start_ind: end_ind]

        # Mask out parts of the trajectories that extend beyond the max path length
        mask = torch.ones(joined_discrete.shape, dtype=torch.bool)
        traj_inds = torch.arange(start_ind, end_ind)
        mask[traj_inds > path_length] = 0

        # Flatten everything
        joined_discrete, mask = joined_discrete.view(-1), mask.view(-1)
        X, Y, mask = joined_discrete[:-1], joined_discrete[1:], mask[1:]

        return X, Y, mask, joined_origin.view(-1), (path_ind, start_ind, end_ind)
