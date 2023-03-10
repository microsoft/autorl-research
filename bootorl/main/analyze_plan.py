# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import glob
import random

import numpy as np
import pandas as pd
import torch
from argparse import Namespace

sys.path.append(os.getcwd())

import utils
from utils import DiscretizedDataset
from utils import plan
from utils import Renderer
from model import GPT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_ckpt_epoch(loadpath, target="latest"):
    assert target in ["all", "latest", "earliest"] or isinstance(target, int)
    states = glob.glob1(loadpath, 'state_*')
    epochs = [int(state.replace('state_', '').replace('.pt', '')) for state in states]
    if target == "all":
        return epochs
    if target == "latest":
        return  [max(epochs)]
    elif target == "earliest":
        return [min(epochs)]
    elif isinstance(target, int):
        assert target in epochs
        return [target]


# Setup
parser = utils.ArgParser()
args = parser.parse_args()
logger = parser.get_logger()
set_seed(args.seed)

# Environment
env = utils.load_environment(args.dataset)

# Dataset
dataset = DiscretizedDataset(
    logger=logger,
    env=args.dataset,
    n_bins=args.n_bins,
    sequence_length=args.sequence_length,
    penalty=args.termination_penalty,
    discount=args.discount,
)

obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
trans_dim = dataset.joined_dim
block_size = args.sequence_length * trans_dim - 1

# Model
model_config = Namespace(
    vocab_size=args.n_bins,
    block_size=block_size,
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embd=args.n_embd * args.n_head,
    observation_dim=obs_dim,
    action_dim=act_dim,
    transition_dim=trans_dim,
    action_weight=args.action_weight,
    reward_weight=args.reward_weight,
    value_weight=args.value_weight,
    embd_pdrop=args.embd_pdrop,
    resid_pdrop=args.resid_pdrop,
    attn_pdrop=args.attn_pdrop
)

renderer = Renderer(env, dataset.observation_dim, dataset.action_dim)

epoch_ranges = sorted(find_ckpt_epoch(args.checkpoint_path))
logger.debug(f"Find checkpoint epochs: {epoch_ranges}")
all_info = {}
for epoch in epoch_ranges:
    logger.info(f'Loading model epoch: {epoch}')
    state_path = os.path.join(args.checkpoint_path, f'state_{epoch}.pt')
    state = torch.load(state_path)

    model = GPT(model_config).to(args.device)
    model.load_state_dict(state, strict=True)

    info = plan(args, env, dataset, model, logger)
    for k, v in info.items():
        print(k, len(v))
    info = pd.DataFrame(info)
    info.index.name = "Timestep"
    all_info[(args.seed, epoch)] = info

    rollout_states = np.stack(info["rollout_states"], axis=0)
    predict_states = np.stack(info["predict_states"], axis=0)
    print(len(rollout_states))
    print(len(predict_states))
    print(rollout_states[:3])
    print(predict_states[:3])


all_info = pd.concat(all_info, names=["Seed", "Epoch"])
all_info.to_csv(os.path.join(args.output_dir, "reward_analysis.csv"), sep="\t")
