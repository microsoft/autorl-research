# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from enum import Enum
from typing import Optional, Iterable, Callable
import gym

from tianshou.env import BaseVectorEnv
from utilsd.config import PythonConfig, configclass
from .logging import Logger
from .finite_env import FiniteDummyVectorEnv, FiniteShmemVectorEnv, FiniteSubprocVectorEnv


class ParallelMode(str, Enum):
    dummy = "dummy"
    shmem = "shmem"
    subproc = "subproc"


@configclass
class EnvConfig(PythonConfig):

    concurrency: int
    parallel_mode: ParallelMode = ParallelMode.shmem

    def post_validate(self):
        assert self.concurrency >= 1
        return True

@configclass
class DivEnvConfig(PythonConfig):
    num_skill: int
    time_per_step: int
    vol_limit: Optional[float]  # the limitation of current decision compared to volume

    concurrency: int
    parallel_mode: ParallelMode = ParallelMode.shmem

    def post_validate(self):
        assert self.vol_limit is None or self.vol_limit < 1
        assert self.concurrency >= 1
        return True

@configclass
class AtariEnvConfig(PythonConfig):
    concurrency: int
    env_name: str
    episode_life: bool=True
    clip_rewards: bool=True
    frame_stack: bool=False
    scale: bool=False
    dnc: bool=False
    max_episode_steps: Optional[int]=None
    parallel_mode: ParallelMode = ParallelMode.shmem
    


class MujocoWrapper(gym.Wrapper):
    def __init__(self,env, max_step=1000):
        super().__init__(env)
        self.max_step = max_step
        self.cur_step = 0
    
    def reset(self):
        self.cur_step = 0
        return self.env.reset()
    
    def step(self,act):
        self.cur_step += 1
        obs,rwd,done,info = self.env.step(act)
        if self.cur_step>=self.max_step:
            done = True
        return obs,rwd,done,info