# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
sys.path.append(os.getcwd())
from utils import DiscretizedDataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22


def plot_tsne(game="halfcheetah", level="medium", sample=1, update=True):
    if not update and os.path.exists(f"./logs/{game}-{level}/distribution/reduced_tsne.npz"):
        data = np.load(f"./logs/{game}-{level}/distribution/reduced_tsne.npz")
        reduced_origin =   data["reduced_origin"]
        reduced_recon_ar = data["reduced_recon_ar"]
        reduced_recon_tf = data["reduced_recon_tf"]
        reduced_noise =    data["reduced_noise"]
        reduced_merged = np.concatenate([reduced_origin, reduced_recon_ar, reduced_recon_tf, reduced_noise])
    else:
        trans = []
        for i, scheme in enumerate(["boot_genr_once_ar", "boot_genr_once_tf"]):
            transitions = np.load(f"./logs/{game}-{level}/distribution/{scheme}/transitions.npz")
            trans_origin = transitions["trans_origin"][:10000]
            trans_generated_reconstruct = transitions[f"trans_recon_{scheme.split('_')[3]}_{scheme.split('_')[1]}"][:10000]
            if i == 0:
                trans.append(trans_origin)

                dataset = DiscretizedDataset(logger=None, env=f"{game}-{level}-v2", n_bins=100, sequence_length=10, penalty=-100, discount=0.99)
                discretizer = dataset.discretizer
                mean, std = dataset.raw_data_mean, dataset.raw_data_std
                trans_noise = trans_origin.copy()
                noise = np.random.normal(scale=3e-4, size=trans_noise.shape)
                noise = noise * std + mean
                trans_noise[:, :dataset.observation_dim] += noise[:, :dataset.observation_dim]
                trans_noise = discretizer.discretize(trans_noise)
                trans_noise = discretizer.reconstruct(trans_noise)

            trans.append(trans_generated_reconstruct)

        trans.append(trans_noise)
        trans_merged = np.concatenate(trans)
        print(trans_merged.shape)

        from sklearn.manifold import TSNE
        n_components = 2
        tsne = TSNE(n_components)
        reduced_merged = tsne.fit_transform(trans_merged)
        reduced_origin =   reduced_merged[    0: 10000]
        reduced_recon_ar = reduced_merged[10000: 20000]
        reduced_recon_tf = reduced_merged[20000: 30000]
        reduced_noise =    reduced_merged[30000: 40000]
        np.savez(
            f"./logs/{game}-{level}/distribution/reduced_tsne.npz",
            reduced_origin=reduced_origin,
            reduced_recon_ar=reduced_recon_ar,
            reduced_recon_tf=reduced_recon_tf,
            reduced_noise=reduced_noise,
        )

    x_min, y_min = reduced_merged.min(axis=0)
    x_max, y_max = reduced_merged.max(axis=0)
    x_left, x_right = 0.5 * (x_max + x_min) - 0.55 * (x_max - x_min),  0.5 * (x_max + x_min) + 0.55 * (x_max - x_min)
    y_bottom, y_top = 0.5 * (y_max + y_min) - 0.55 * (y_max - y_min),  0.5 * (y_max + y_min) + 0.80 * (y_max - y_min)

    fig, axes = plt.subplots(1, 4, figsize=(4 * 6, 3.8), dpi=320)
    for i in range(4):
        axes[i].set_xlim(x_left, x_right)
        axes[i].set_ylim(y_bottom, y_top)
        axes[i].grid(ls="--", alpha=0.5)

    axes[0].scatter(reduced_origin[::sample, 0], reduced_origin[::sample, 1], marker='o', s=18, c=f"#2980b9", alpha=0.15*sample)
    axes[0].set_title('Original Dataset', y=-0.33)
    h = mlines.Line2D([], [], color='#2980b9', marker='o', linestyle='None', markersize=12, label='Original Data')
    axes[0].legend(handles=[h], bbox_to_anchor=(0.01, 0.99), loc='upper left', borderpad=0.2, handlelength=1, handletextpad=0.4, borderaxespad=0.1)

    h = axes[1].scatter(reduced_recon_tf[::sample, 0], reduced_recon_tf[::sample, 1], marker='^', s=18, c=f"#c0392b", alpha=0.15*sample)
    axes[1].set_title('Teacher-forcing Generation', y=-0.33)
    h = mlines.Line2D([], [], color='#c0392b', marker='^', linestyle='None', markersize=12, label='Generated Data')
    axes[1].legend(handles=[h], bbox_to_anchor=(0.01, 0.99), loc='upper left', borderpad=0.2, handlelength=1, handletextpad=0.4, borderaxespad=0.1)

    h = axes[2].scatter(reduced_recon_ar[::sample, 0], reduced_recon_ar[::sample, 1], marker='^', s=18, c=f"#f39c12", alpha=0.15*sample)
    axes[2].set_title('Autoregressive Generation', y=-0.33)
    h = mlines.Line2D([], [], color='#f39c12', marker='^', linestyle='None', markersize=12, label='Generated Data')
    axes[2].legend(handles=[h], bbox_to_anchor=(0.01, 0.99), loc='upper left', borderpad=0.2, handlelength=1, handletextpad=0.4, borderaxespad=0.1)

    axes[3].scatter(reduced_origin[::sample, 0], reduced_origin[::sample, 1], marker='o', s=18, c=f"#2980b9", alpha=0.15*sample)
    axes[3].scatter(reduced_recon_tf[::sample, 0], reduced_recon_tf[::sample, 1], marker='^', s=18, c=f"#c0392b", alpha=0.15*sample)
    axes[3].scatter(reduced_recon_ar[::sample, 0], reduced_recon_ar[::sample, 1], marker='^', s=18, c=f"#f39c12", alpha=0.12*sample)
    axes[3].set_title(f'Augmented Dataset', y=-0.33)

    plt.tight_layout()
    print(f"Saving figure to `./analysis/images/tsne_distribution/tsne_distribution_{game}-{level}`")
    plt.savefig(f"./analysis/images/tsne_distribution/tsne_distribution_{game}-{level}.pdf")
    plt.savefig(f"./analysis/images/tsne_distribution/tsne_distribution_{game}-{level}.png")
    plt.clf()


    # ======================= Noise =======================
    fig, axes = plt.subplots(1, 3, figsize=(3 * 6, 3.8), dpi=320)
    for i in range(3):
        axes[i].set_xlim(x_left, x_right)
        axes[i].set_ylim(y_bottom, y_top)
        axes[i].grid(ls="--", alpha=0.5)

    axes[0].scatter(reduced_origin[::sample, 0], reduced_origin[::sample, 1], marker='o', s=18, c=f"#2980b9", alpha=0.15*sample)
    axes[0].set_title('Original Dataset', y=-0.33)
    h = mlines.Line2D([], [], color='#2980b9', marker='o', linestyle='None', markersize=12, label='Original Data')
    axes[0].legend(handles=[h], bbox_to_anchor=(0.01, 0.99), loc='upper left', borderpad=0.2, handlelength=1, handletextpad=0.4, borderaxespad=0.1)

    h = axes[1].scatter(reduced_noise[::sample, 0], reduced_noise[::sample, 1], marker='^', s=18, c=f"#27ae60", alpha=0.15*sample)
    axes[1].set_title('Noisy Data', y=-0.33)
    h = mlines.Line2D([], [], color='#27ae60', marker='^', linestyle='None', markersize=12, label='Generated Data')
    axes[1].legend(handles=[h], bbox_to_anchor=(0.01, 0.99), loc='upper left', borderpad=0.2, handlelength=1, handletextpad=0.4, borderaxespad=0.1)

    axes[2].scatter(reduced_origin[::sample, 0], reduced_origin[::sample, 1], marker='o', s=18, c=f"#2980b9", alpha=0.15*sample)
    axes[2].scatter(reduced_noise[::sample, 0], reduced_noise[::sample, 1], marker='^', s=18, c=f"#27ae60", alpha=0.12*sample)
    axes[2].set_title(f'Augmented Dataset', y=-0.33)

    plt.tight_layout()
    print(f"Saving figure to `./analysis/images/tsne_distribution/tsne_distribution_{game}-{level}_noise`")
    plt.savefig(f"./analysis/images/tsne_distribution/tsne_distribution_{game}-{level}_noise.pdf")
    plt.savefig(f"./analysis/images/tsne_distribution/tsne_distribution_{game}-{level}_noise.png")
    plt.clf()


for game in ["halfcheetah", "hopper", "walker2d"]:
    for level in ["medium", "medium-replay", "medium-expert"]:
        plot_tsne(game, level, sample=4, update=False)

