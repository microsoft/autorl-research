# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import json
import pandas as pd
import torch
from argparse import Namespace

sys.path.append(os.getcwd())

import utils
from utils import DiscretizedDataset
from model import GPT


def to(xs, device):
    return [x.to(device) for x in xs]


# Setup
parser = utils.ArgParser()
args = parser.parse_args()
logger = parser.get_logger()


# Resume check
resume_file = os.path.join(args.output_dir, "resume_status.json")
resume_model_state = os.path.join(args.output_dir, "resume_model_state.pt")
if args.resume == "y" and os.path.exists(resume_file):
    with open(resume_file, "r") as f:
        resume_status = json.load(f)
    resume_flag = True
    logger.info(f"Resume mode enabled, reading status from file {resume_file}")
    if resume_status["train_finished"]:
        logger.info(f"Training has finished, exiting task.")
        exit(0)
else:
    resume_flag = False
    logger.info(f"Resume mode disabled, starting a new task.")


# Dataset
dataset = DiscretizedDataset(
    logger=logger,
    env=args.dataset,
    n_bins=args.n_bins,
    sequence_length=args.sequence_length,
    penalty=args.termination_penalty,
    discount=args.discount,
)

# Model
obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
trans_dim = dataset.joined_dim
block_size = args.sequence_length * trans_dim - 1

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

device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
if args.parallel:
    logger.info(f"Using device: {device}; Enabling data parallel in torch")
    model = torch.nn.DataParallel(GPT(model_config)).to(device)
else:
    logger.info(f"Using device: {device}; Disabling data parallel in torch")
    model = GPT(model_config).to(device)
param_num = sum(p.numel() for p in model.parameters())
logger.info(f"Total number of parameters: {param_num}.")


# Trainer
trainer_config = Namespace(
    logger=logger,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    betas=(0.9, 0.95),
    grad_norm_clip=1.0,
    weight_decay=0.1,
    lr_decay=args.lr_decay,
    warmup_tokens=len(dataset)*block_size,
    final_tokens=20*len(dataset)*block_size,
    num_workers=0,
)
trainer = utils.Trainer(trainer_config)


# scale number of epochs to keep number of updates constant
n_epochs = int(1e6 / len(dataset) * args.n_epochs_ref)
# calculate epoch index of when to save model
save_interval = max(1, int(n_epochs // args.n_saves))
save_epochs = [e for e in range(n_epochs) if e % save_interval == 0 or e == n_epochs - 1]

bootstrap_kwargs = {
    "bootstrap": args.bootstrap,
    "bootstrap_type": args.bootstrap_type,
    "generation_type": args.generation_type,
    "generation_epoch_thresh": int(args.generation_epoch_thresh * n_epochs),
    "generation_len": args.generation_len,
    "generation_num": args.generation_num,
    "generation_confidence_type": args.generation_confidence_type,
    "generation_confidence_factor": args.generation_confidence_factor,
    "generation_real_r": args.generation_real_r,
    "generation_real_R": args.generation_real_R,
}

logger.info(f"Experiment: {args.exp_name} | Total epochs: {n_epochs} | Total saves: {len(save_epochs)}")
logger.info(f"Saving model at epochs: {save_epochs}")
if args.bootstrap:
    logger.info(f"Bootstrapping is enabled.")
    logger.info(f"Performing bootstrapping after epoch {bootstrap_kwargs['generation_epoch_thresh']}.")


# Resume processing
if resume_flag:
    init_epoch = resume_status["current_epoch"] + 1
    info = resume_status["info"]
    trainer.n_epochs = init_epoch
    trainer.n_tokens = resume_status["current_n_tokens"]
    timer = utils.Timer(total_num=(n_epochs - init_epoch))
    model.load_state_dict(torch.load(resume_model_state), strict=True)
else:
    init_epoch = 0
    info = {}
    timer = utils.Timer(total_num=n_epochs)


for epoch in range(init_epoch, n_epochs):
    logger.info(f"Epoch: {epoch:>3d} / {n_epochs:>3d} | {args.exp_name}")
    info_ = trainer.train(model, dataset, with_tqdm=args.with_tqdm, log_freq=100, bootstrap_kwargs=bootstrap_kwargs)

    if epoch % save_interval == 0 or epoch == n_epochs - 1:
        model_path = os.path.join(args.output_dir, f'state_{epoch}.pt')
        logger.info(f"Epoch: {epoch:>3d} / {n_epochs:>3d} | Saving model to {model_path}")
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state, model_path)
    info[epoch] = info_

    if args.resume == "y":
        resume_status = {
            "current_epoch": epoch,
            "current_n_tokens": trainer.n_tokens.item(),
            "train_finished": (epoch == n_epochs - 1),
            "info": info,
        }
        with open(resume_file, "w") as f:
            json.dump(resume_status, f)
        logger.info(f"Dumping resume status to file {resume_file} at epoch {epoch}.")
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state, resume_model_state)
        logger.info(f"Dumping resume model state to file {resume_model_state} at epoch {epoch}.")

    diff_time, total_time, eta = timer()
    logger.info(f"Epoch: {epoch:>3d} / {n_epochs:>3d} | Time: {diff_time} | Total time: {total_time} | ETA: {eta}")

df = pd.DataFrame.from_dict(info, orient="index").sort_index()
df.to_csv(os.path.join(args.output_dir, "train_epoch_info.csv"))
