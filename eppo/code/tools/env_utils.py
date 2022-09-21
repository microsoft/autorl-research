# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import copy
import dataclasses
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch, gym
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy
from torch.utils.data import Dataset
from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment, use_cuda
from utilsd.experiment import print_config
from utilsd.earlystop import EarlyStop, EarlyStopStatus
from utilsd.logging import print_log

from code.env import EnvConfig, ParallelMode, Logger, FiniteDummyVectorEnv,FiniteSubprocVectorEnv,FiniteShmemVectorEnv, AtariEnvConfig, MujocoWrapper

import sys,os
cur_dir = os.getcwd()
sys.path.insert(0,cur_dir)
print(sys.path)
from code.env.atari_env import wrap_deepmind


def atari_game_env_factory(env_config: AtariEnvConfig, env_name:str, logger: Logger,seed=42, dnc=False):
    def single_env(env_config=env_config):
        env = wrap_deepmind(env_config.env_name, env_config.episode_life,env_config.clip_rewards, env_config.scale, env_config.frame_stack, dnc=dnc, env_name=env_config.env_name)
        return env
    def test_env(env_config=env_config):
        env = wrap_deepmind(env_config.env_name, episode_life=False, clip_rewards=False, dnc=False, env_name=env_config.env_name,)
        return env
    
    if env_config.parallel_mode == ParallelMode.dummy:
        venv_cls = FiniteDummyVectorEnv
    elif env_config.parallel_mode == ParallelMode.shmem:
        venv_cls = FiniteShmemVectorEnv
    elif env_config.parallel_mode == ParallelMode.subproc:
        venv_cls = FiniteSubprocVectorEnv

    envs1 = venv_cls(logger, [single_env for _ in range(env_config.concurrency)])
    envs2 = venv_cls(logger, [test_env for _ in range(env_config.concurrency)])
    envs1.seed(seed)
    envs2.seed(seed)
    return envs1,envs2
