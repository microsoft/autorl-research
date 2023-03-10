# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import torch
import os
import sys
sys.path.append(os.getcwd())
from utils import DiscretizedDataset
from itertools import accumulate

pd.set_option("display.precision", 2)


def calc_mmd(x, y):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q

    borrowed from https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy/notebook
    """
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx, ry = (xx.diag().unsqueeze(0).expand_as(xx)), (yy.diag().unsqueeze(0).expand_as(yy))
    dxx, dyy, dxy = rx.t() + rx - 2. * xx, ry.t() + ry - 2. * yy, rx.t() + ry - 2. * zz
    XX, YY, XY = (torch.zeros(xx.shape), torch.zeros(xx.shape), torch.zeros(xx.shape))
    bandwidth_range = [10, 15, 20, 50]
    for a in bandwidth_range:
        XX += torch.exp(-0.5*dxx/a)
        YY += torch.exp(-0.5*dyy/a)
        XY += torch.exp(-0.5*dxy/a)
    return torch.mean(XX + YY - 2. * XY)


def calc_r(real_r, pred_r_list):
    reversed_rewards = real_r[::-1]
    cumulative_rewards = list(accumulate(reversed_rewards, lambda x, y: x * 0.99 + y))
    real_R = np.array(cumulative_rewards)
    pred_r = [t[0] for t in pred_r_list]
    reversed_rewards = pred_r[::-1]
    cumulative_rewards = list(accumulate(reversed_rewards, lambda x, y: x * 0.99 + y))
    pred_R = np.array(cumulative_rewards)
    return real_r, real_R, pred_r, pred_R


def calc_dist(target="boot_genr_repeat_ar", env="halfcheetah-medium"):
    transitions = np.load(f"./logs/{env}/distribution/{target}/transitions.npz")
    trans_origin = transitions["trans_origin"][:10000]
    trans_discretized = transitions["trans_discretized"][:10000]
    trans_generated_discretized = transitions[f"trans_generated_{target.split('_')[3]}_{target.split('_')[1]}"][:10000]
    trans_generated_reconstruct = transitions[f"trans_recon_{target.split('_')[3]}_{target.split('_')[1]}"][:10000]
    rmse = np.sqrt(((trans_generated_discretized - trans_discretized) ** 2).mean())
    print(f"\tRMSE (state): {rmse:.4f}")
    mmd = calc_mmd(trans_generated_reconstruct, trans_origin)
    print(f"\tMMD (10^-3): {mmd * 1000:.4f}")
    return rmse, mmd


def calc_boot_dist():
    dist = {}
    for target in ["boot_realr_once_ar", "boot_realr_once_tf", "boot_genr_once_ar", "boot_genr_once_tf"]:
        for game in ["halfcheetah", "hopper", "walker2d"]:
            for level in ["medium", "medium-replay", "medium-expert"]:
                rmse, mmd = calc_dist(target=target, env=f"{game}-{level}")
                dist[(target, "rmse", game, level)] = rmse
                dist[(target, "mmd", game, level)] = mmd
    multi_idx = pd.MultiIndex.from_tuples(dist, names=["target", "metric", "game", "level"])
    dist = pd.DataFrame(dist.values(), index=multi_idx).sort_index()
    dist = dist.groupby(["target", "metric"]).mean()
    dist.to_csv("./analysis/dist.csv", index=False)
    print(dist)
    return dist


def calc_eval_dist(game="halfcheetah", level="medium"):
    dist = {}
    for target in ["boot_genr_once_ar", "boot_genr_once_tf"]:
        for game in ["halfcheetah", "hopper", "walker2d"]:
            for level in ["medium", "medium-replay", "medium-expert"]:
                df = pd.read_csv(f"./logs/{game}-{level}/plan_analysis/{target}/reward_analysis.csv", sep="\t")
                real_s = np.array([eval(r) for r in df["rollout_states"]])
                pred_s = np.array([eval(r) for r in df["predict_states"]])
                real_r = np.array([r for r in df["real_rewards"]])
                pred_r_list = np.array([eval(r) for r in df["rollout_values"]])
                real_r, real_R, pred_r, pred_R = calc_r(real_r, pred_r_list)

                dataset = DiscretizedDataset(logger=None, env=f"{game}-{level}-v2", n_bins=100, sequence_length=10, penalty=-100, discount=0.99)
                discretizer = dataset.discretizer
                mean, std = dataset.raw_data_mean, dataset.raw_data_std

                real_traj = np.zeros((len(real_s), dataset.joined_dim), dtype=real_s.dtype)
                real_traj[:, :dataset.observation_dim] = real_s
                real_traj[:, -2] = real_r
                real_traj[:, -1] = real_R
                pred_traj = np.zeros((len(pred_s), dataset.joined_dim), dtype=pred_s.dtype)
                pred_traj[:, :dataset.observation_dim] = pred_s
                pred_traj[:, -2] = pred_r
                pred_traj[:, -1] = pred_R

                mmd = calc_mmd(real_traj, pred_traj)
                print(f"\tMMD (10^-3): {mmd * 1000:.4f}")

                real_traj = discretizer.discretize(real_traj)
                pred_traj = discretizer.discretize(pred_traj)
                rmse = np.sqrt(((real_traj - pred_traj) ** 2).mean())
                print(f"\tRMSE (state): {rmse:.4f}")

                dist[(target, "rmse", game, level)] = rmse
                dist[(target, "mmd", game, level)] = mmd * 1000

    multi_idx = pd.MultiIndex.from_tuples(dist, names=["target", "metric", "game", "level"])
    dist = pd.DataFrame(dist.values(), index=multi_idx).sort_index()
    dist = dist.groupby(["target", "metric"]).mean()
    dist.to_csv("./analysis/eval_dist.csv")
    print(dist)


def calc_noise_eval_dist(game="halfcheetah", level="medium"):
    dist = {}
    for target in ["boot_s4rl_noise_last"]:
        for game in ["halfcheetah", "hopper", "walker2d"]:
            for level in ["medium", "medium-replay", "medium-expert"]:
                df = pd.read_csv(f"./logs/{game}-{level}/plan_analysis/{target}/reward_analysis.csv", sep="\t")
                real_s = np.array([eval(r) for r in df["rollout_states"]])
                pred_s = np.array([eval(r) for r in df["predict_states"]])
                real_r = np.array([r for r in df["real_rewards"]])
                pred_r_list = np.array([eval(r) for r in df["rollout_values"]])
                real_r, real_R, pred_r, pred_R = calc_r(real_r, pred_r_list)

                dataset = DiscretizedDataset(logger=None, env=f"{game}-{level}-v2", n_bins=100, sequence_length=10, penalty=-100, discount=0.99)
                discretizer = dataset.discretizer
                mean, std = dataset.raw_data_mean, dataset.raw_data_std

                real_traj = np.zeros((len(real_s), dataset.joined_dim), dtype=real_s.dtype)
                real_traj[:, :dataset.observation_dim] = real_s
                real_traj[:, -2] = real_r
                real_traj[:, -1] = real_R
                pred_traj = np.zeros((len(pred_s), dataset.joined_dim), dtype=pred_s.dtype)
                pred_traj[:, :dataset.observation_dim] = pred_s
                pred_traj[:, -2] = pred_r
                pred_traj[:, -1] = pred_R

                noisy_traj = real_traj.copy()

                noise = np.random.normal(scale=3e-4, size=noisy_traj.shape)
                noise = noise * std + mean
                noisy_traj[:, :dataset.observation_dim] += noise[:, :dataset.observation_dim]

                mmd = calc_mmd(real_traj, noisy_traj)
                print(f"\tMMD (10^-3): {mmd * 1000:.4f}")

                real_traj = discretizer.discretize(real_traj)
                noisy_traj = discretizer.discretize(noisy_traj)
                rmse = np.sqrt(((real_traj - noisy_traj) ** 2).mean())
                print(f"\tRMSE (state): {rmse:.4f}")

                dist[(target, "rmse", game, level)] = rmse
                dist[(target, "mmd", game, level)] = mmd * 1000

    multi_idx = pd.MultiIndex.from_tuples(dist, names=["target", "metric", "game", "level"])
    dist = pd.DataFrame(dist.values(), index=multi_idx).sort_index()
    dist = dist.groupby(["target", "metric"]).mean()
    dist.to_csv("./analysis/noise_eval_dist.csv")
    print(dist)


calc_boot_dist()
calc_eval_dist()
calc_noise_eval_dist()
