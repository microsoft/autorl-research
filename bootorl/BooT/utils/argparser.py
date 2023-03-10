# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import time
import json
import logging
import argparse
import warnings

import random
import numpy as np
import torch


DEFAULT_ARGS = {
    # Miscellaneous Settings
    'with_tqdm': True,  # Show progress bar
    'seed': 0,  # Seed
    'parallel': True,

    # Sequence Configuration
    'sequence_length': 10,
    'discount': 0.99,
    'termination_penalty': -100,

    # Model Configuration
    'n_bins': 100,
    'n_layer': 4,
    'n_head': 4,
    'n_embd': 32,
    'device': 'cuda',
    'embd_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'action_weight': 5,
    'reward_weight': 1,
    'value_weight': 1,

    # Bootstrap Configuration
    'bootstrap': True,
    'bootstrap_type': 'once',
    'generation_type': "autoregressive",
    'generation_epoch_thresh': 0.4,  # perform bootstrapping after 40% of training epochs
    'generation_len': 1,
    'generation_num': 1,
    'generation_confidence_type': "ratio",
    'generation_confidence_factor': 0.04,
    'generation_real_r': False,
    'generation_real_R': False,

    # Noise Configuration for S4RL experiments, can be ignored for normal BooTORL training
    'noise': False,
    'noise_epoch_thresh': 0.4,
    'noise_length': 1,
    'noise_sigma': 3e-4,
    'noise_target': "sa",

    # Trainer Configuration
    'n_epochs_ref': 50,
    'n_saves': 10,
    'batch_size': 256,
    'learning_rate': 1e-3,
    'lr_decay': True,
    
    # Planner Configuration
    'ckpt_epoch': 'latest',
    'horizon': 10,
    'beam_width': 128,
    'n_expand': 2,
    'k_obs': 1,
    'k_act': None,
    'cdf_obs': None,
    'cdf_act': 0.6,
    'percentile': 'mean',
    'max_context_transitions': 5,
    'prefix_context': True,
}

# Following arguments are mostly the same as Trajectory Transformer,
# except for newly introduced hyperparameters `generation_confidence_factor` in BooT
OVERWRITE_ARGS = {
    'halfcheetah-medium-v2': {'k_act': 20,},
    'halfcheetah-medium-replay-v2': {'k_act': 20,},
    'halfcheetah-medium-expert-v2': {},

    'hopper-medium-v2': {'k_act': 20,},
    'hopper-medium-replay-v2': {'generation_confidence_factor': 0.06,},
    'hopper-medium-expert-v2': {'generation_confidence_factor': 0.08,},

    'walker2d-medium-v2': {'generation_epoch_thresh': 0.8,},
    'walker2d-medium-replay-v2': {'generation_epoch_thresh': 0.8, 'k_act': 20,},
    'walker2d-medium-expert-v2': {'generation_confidence_factor': 0.08, 'horizon': 5, 'k_act': 20,},

    'ant-medium-v2': {'batch_size': 128, 'horizon': 5,},
    'ant-medium-replay-v2': {'batch_size': 128, 'horizon': 5,},
    'ant-random-v2': {'batch_size': 128, 'horizon': 5,},

    'antmaze-umaze-v0': {'termination_penalty': 0, 'horizon': 5,},
    'antmaze-medium-diverse-v0': {'termination_penalty': 0, 'horizon': 5,},
    'antmaze-medium-play-v0': {'termination_penalty': 0, 'horizon': 5,},
    'antmaze-large-diverse-v0': {'termination_penalty': 0, 'horizon': 5,},
    'antmaze-large-play-v0': {'termination_penalty': 0, 'horizon': 5,},
    
    # Below is hyperparameter on Adroit domain. Note that we did not perform a careful hyperparam-tuning on Adroit.
    'pen-human-v1': {'n_epochs_ref': 5, 'horizon': 15, 'batch_size': 64, 'n_embd': 16},
    'pen-cloned-v1': {'n_epochs_ref': 5, 'horizon': 15, 'batch_size': 64, 'n_embd': 16},
    'pen-expert-v1': {'n_epochs_ref': 5, 'horizon': 15, 'batch_size': 64, 'n_embd': 16},
    'hammer-human-v1': {'n_epochs_ref': 5, 'horizon': 15, 'batch_size': 64, 'n_embd': 16},
    'hammer-cloned-v1': {'n_epochs_ref': 5, 'horizon': 15, 'batch_size': 64, 'n_embd': 16},
    'hammer-expert-v1': {'n_epochs_ref': 5, 'horizon': 15, 'batch_size': 64, 'n_embd': 16},
    'door-human-v1': {'n_epochs_ref': 5, 'horizon': 15, 'batch_size': 64, 'n_embd': 16},
    'door-cloned-v1': {'n_epochs_ref': 5, 'horizon': 15, 'batch_size': 64, 'n_embd': 16},
    'door-expert-v1': {'n_epochs_ref': 5, 'horizon': 15, 'batch_size': 64, 'n_embd': 16},
    'relocate-human-v1': {'n_epochs_ref': 5, 'horizon': 15, 'batch_size': 64, 'n_embd': 16},
    'relocate-cloned-v1': {'n_epochs_ref': 5, 'horizon': 15, 'batch_size': 64, 'n_embd': 16},
    'relocate-expert-v1': {'n_epochs_ref': 5, 'horizon': 15, 'batch_size': 64, 'n_embd': 16},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(exp_name, logger_name, output_dir, log_level=logging.DEBUG, log_to_file=True, mode="w"):
    log_file_name = os.path.join(output_dir, f"{exp_name}.log")
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(module)s - %(levelname)s:: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(log_formatter)
    logger_handlers = [ch]
    if log_to_file:
        fh = logging.FileHandler(log_file_name, mode=mode)
        fh.setLevel(log_level)
        fh.setFormatter(log_formatter)
        logger_handlers.append(fh)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    for h in logger_handlers:
        logger.addHandler(h)
    logger.propagate = False
    
    return logger, log_file_name


class ArgParser:

    def parse_args(self):
        raw_args, unknown_args = self._parse_init_args()
        assert raw_args.dataset.endswith("-v0") or raw_args.dataset.endswith("-v1") or raw_args.dataset.endswith("-v2")

        # Set experiment name as `environment-level`
        self.exp_start_time = time.strftime("%m%d%H%M", time.localtime(time.time()))
        self.exp_name = f"{raw_args.dataset[:-3]}"

        # Set output directory, default `./logs/environment-level/[suffix]`
        self.output_dir = os.path.join(raw_args.output_dir, f"{self.exp_name}/", raw_args.suffix)
        self.output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR", "."), self.output_dir)
        if os.path.exists(self.output_dir):
            warnings.warn(f"Output directory {self.output_dir} already exists!")
            # warnings.warn(f"Removing contents in {self.output_dir}!")
            # os.system(f"rm -r {self.output_dir}")
        else:
            os.makedirs(self.output_dir)

        # Initialize logger, by default logger will output to both the console and a log file
        self.logger_name = f"Transformer4RL"
        level = getattr(logging, raw_args.log_level) if isinstance(raw_args.log_level, str) else raw_args.log_level
        mode = "a" if raw_args.resume == "y" else "w"
        self.logger, self.log_file_name = setup_logger(self.exp_name, self.logger_name, self.output_dir, level, mode=mode)

        self.logger.info(f"Initialization complete.")
        if raw_args.checkpoint is not None:
            self.checkpoint_path = raw_args.checkpoint
            if raw_args.from_amlt_data == "y":
                self.checkpoint_path = os.path.join(os.getenv("AMLT_DATA_DIR", "."), self.checkpoint_path)
            else:
                self.checkpoint_path = os.path.join(os.getenv("AMLT_OUTPUT_DIR", "."), self.checkpoint_path)
            self.logger.info(f"Checkpoint path found: {self.checkpoint_path}")
        self.logger.info(f"Output directory: {self.output_dir}")

        # First load arguments from `DEFAULT_ARGS`, then overwrite with `OVERWRITE_ARGS`, 
        # finally overwrite with console input
        args = self.load_default_args(raw_args)
        args = self.add_extra_args(args, unknown_args)
        self.args = args
        self.logger.info(f"Setting seed to: {args.get('seed')}")
        set_seed(args.get("seed"))

        return self

    def __getattr__(self, name):
        # Only invoked when standard method (`__getattribute__`) has not found the attribute
        if "args" not in self.__dict__:
            raise AttributeError("Arguments are not parsed yet. Run 'parse_args()' first.")
        if name not in self.args:
            raise AttributeError(f"Args has no attribute '{name}'")
        return self.args[name]

    def _parse_init_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='hopper-medium-replay-v2')
        parser.add_argument('--checkpoint', type=str, default=None)
        parser.add_argument('--output_dir', type=str, default='./logs/')
        parser.add_argument('--suffix', type=str, default="")
        parser.add_argument('--log_level', type=str, default="DEBUG")
        parser.add_argument('--from_amlt_data', type=str, default='n')
        parser.add_argument('--resume', type=str, default='y')
        raw_args, unknown_args = parser.parse_known_args()
        return raw_args, unknown_args

    def load_default_args(self, raw_args):
        args = DEFAULT_ARGS
        args.update(OVERWRITE_ARGS.get(raw_args.dataset, {}))
        args.update(vars(raw_args))
        return args

    def add_extra_args(self, args, unknown_args):
        self.logger.debug(f"Extra arguments: {unknown_args}")
        assert len(unknown_args) % 2 == 0, f'Found odd number ({len(unknown_args)}) of extras: {unknown_args}'
        for i in range(0, len(unknown_args), 2):
            key = unknown_args[i].replace('--', '')
            val = unknown_args[i + 1]
            if args.get(key) is None:
                self.logger.warning(f"{key} not found in default arguments! Setting it to {val}")
                old_type = type(None)
            else:
                old_val = args.get(key)
                old_type = type(old_val)
                self.logger.debug(f"Overriding argument | {key}: {old_val} --> {val}")
            if val == 'None':
                val = None
            elif val == 'latest':
                val = 'latest'
            elif old_type in [bool, type(None)]:
                val = eval(val)
            else:
                val = old_type(val)
            args[key] = val
        return args

    def get_logger(self):
        return self.logger
