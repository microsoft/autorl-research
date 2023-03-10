# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from PIL import Image
from utils import DiscretizedDataset
from utils import Renderer


def load_states(target="boot_genr_repeat_ar", env="halfcheetah-medium"):
    transitions = np.load(f"./logs/{env}/distribution/{target}/transitions.npz")
    trans_origin = transitions["trans_origin"]
    trans_generated_reconstruct = transitions[f"trans_recon_{target.split('_')[3]}_{target.split('_')[1]}"]
    dataset = DiscretizedDataset(logger=None, env=f"{env}-v2", n_bins=100, sequence_length=10, penalty=-100, discount=0.99)
    discretizer = dataset.discretizer
    mean, std = dataset.raw_data_mean, dataset.raw_data_std

    noise = np.random.normal(scale=3e-4, size=trans_origin.shape)
    noise = noise * std + mean
    trans_noisy_state = trans_origin.copy()
    trans_noisy_state[:, :dataset.observation_dim] += noise[:, :dataset.observation_dim]
    trans_discretized_noisy_state = discretizer.discretize(trans_noisy_state)
    trans_recon_noisy_state = discretizer.reconstruct(trans_discretized_noisy_state)

    state_dim = dataset.observation_dim
    state_origin = trans_origin[:, :state_dim]
    state_generated_reconstruct = trans_generated_reconstruct[:, :state_dim]
    state_noisy_reconstruct = trans_recon_noisy_state[:, :state_dim]
    return state_origin, state_generated_reconstruct, state_noisy_reconstruct


def render(target="boot_genr_repeat_ar", game="halfcheetah", level="medium", num=5):
    save_dir = f"./analysis/images/{target}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    renderer = Renderer(env=f"{game}-{level}-v2")
    state_origin, state_generated_reconstruct, state_noisy_reconstruct = load_states(target=target, env=f"{game}-{level}")

    for i in range(num):
        img_file_name = os.path.join(save_dir, f"{game}-{level}_original_{i}.jpg")
        img = renderer.render(state_origin[i], dim=2048)
        img = Image.fromarray(img)
        img.save(img_file_name)
    for i in range(num):
        img_file_name = os.path.join(save_dir, f"{game}-{level}_generated_{i}.jpg")
        img = renderer.render(state_generated_reconstruct[i], dim=2048)
        img = Image.fromarray(img)
        img.save(img_file_name)
    for i in range(num):
        img_file_name = os.path.join(save_dir, f"{game}-{level}_noisy_{i}.jpg")
        img = renderer.render(state_noisy_reconstruct[i], dim=2048)
        img = Image.fromarray(img)
        img.save(img_file_name)


def render_all():
    dist = {}
    for target in ["boot_genr_once_ar", "boot_genr_once_tf"]:
        for game in ["halfcheetah", "hopper", "walker2d"]:
            for level in ["medium", "medium-replay", "medium-expert"]:
                render(target=target, game=game, level=level)


render_all()
