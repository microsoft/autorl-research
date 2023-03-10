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
from utils import AnalyzeDiscretizedDataset
from utils import Tester
from model import GPT


# python main/extract_distribution.py --dataset halfcheetah-medium-v2 --checkpoint ...


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
dataset = AnalyzeDiscretizedDataset(
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

tester_config = Namespace(
    logger=logger,
    batch_size=args.batch_size,
    num_workers=0,
)
tester = Tester(tester_config)


epoch_ranges = sorted(find_ckpt_epoch(args.checkpoint_path))
logger.debug(f"Find checkpoint epochs: {epoch_ranges}")
all_info = {}
for epoch in epoch_ranges:
    logger.info(f'Loading model epoch: {epoch}')
    state_path = os.path.join(args.checkpoint_path, f'state_{epoch}.pt')
    state = torch.load(state_path)

    model = GPT(model_config)
    model.load_state_dict(state, strict=True)

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    if args.parallel:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    generated_dataset = tester.generate_data(model, dataset, with_tqdm=args.with_tqdm)

    masks = torch.stack([generated_dataset[i]["mask"][-1] for i in range(len(generated_dataset))], dim=0)
    trans_origin = torch.stack([generated_dataset[i]["origin"][-trans_dim:] for i in range(len(generated_dataset))], dim=0)[masks]
    trans_discretized = torch.stack([generated_dataset[i]["discretized"][1][-trans_dim:] for i in range(len(generated_dataset))], dim=0)[masks]
    trans_generated_tf_realr = torch.stack([generated_dataset[i]["generated_tf_realr"][-trans_dim:] for i in range(len(generated_dataset))], dim=0)[masks]
    trans_recon_tf_realr = dataset.discretizer.reconstruct(trans_generated_tf_realr)
    trans_generated_ar_realr = torch.stack([generated_dataset[i]["generated_ar_realr"][-trans_dim:] for i in range(len(generated_dataset))], dim=0)[masks]
    trans_recon_ar_realr = dataset.discretizer.reconstruct(trans_generated_ar_realr)
    trans_generated_tf_genr = torch.stack([generated_dataset[i]["generated_tf_genr"][-trans_dim:] for i in range(len(generated_dataset))], dim=0)[masks]
    trans_recon_tf_genr = dataset.discretizer.reconstruct(trans_generated_tf_genr)
    trans_generated_ar_genr = torch.stack([generated_dataset[i]["generated_ar_genr"][-trans_dim:] for i in range(len(generated_dataset))], dim=0)[masks]
    trans_recon_ar_genr = dataset.discretizer.reconstruct(trans_generated_ar_genr)

    np.savez(
        os.path.join(args.output_dir, "transitions.npz"),
        trans_origin=trans_origin,
        trans_discretized=trans_discretized,
        trans_generated_tf_realr=trans_generated_tf_realr,
        trans_recon_tf_realr=trans_recon_tf_realr,
        trans_generated_ar_realr=trans_generated_ar_realr,
        trans_recon_ar_realr=trans_recon_ar_realr,
        trans_generated_tf_genr=trans_generated_tf_genr,
        trans_recon_tf_genr=trans_recon_tf_genr,
        trans_generated_ar_genr=trans_generated_ar_genr,
        trans_recon_ar_genr=trans_recon_ar_genr,
    )
