# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from typing import Optional, List

from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass

from code.env import EnvConfig
from code.network import NETWORKS
from code.policy import POLICIES
from code.env import AtariEnvConfig


@configclass
class TrainerConfig(PythonConfig):
    max_epoch: int
    episode_per_collect: int
    batch_size: int
    repeat_per_collect: int
    earlystop_patience: int
    val_every_n_epoch: int
    fast_dev_run: bool = False
    buffer_size: int=200000
    save_epoch: bool=False

@configclass
class OffTrainerConfig(PythonConfig):
    max_epoch: int
    episode_per_collect: int
    steps_per_epoch: int
    update_per_step: float
    batch_size: int
    earlystop_patience: int
    val_every_n_epoch: int
    fast_dev_run: bool = False
    buffer_size: int=200000
    save_epoch: bool=False


@configclass
class BacktestConfig(PythonConfig):
    env_type: str="atari"
    eps_num: int = 20

@configclass
class GameRunConfig(PythonConfig):
    env: AtariEnvConfig
    policy: RegistryConfig[POLICIES]
    network1: Optional[RegistryConfig[NETWORKS]] = None
    network2: Optional[RegistryConfig[NETWORKS]] = None
    trainer: Optional[TrainerConfig] = None
    runtime: RuntimeConfig = RuntimeConfig()
    backtest: bool = False
    use_step: bool = False
    bk: Optional[BacktestConfig]=None

@configclass
class GameOffRunConfig(PythonConfig):
    env: AtariEnvConfig
    policy: RegistryConfig[POLICIES]
    network: Optional[List[RegistryConfig[NETWORKS]]] = None
    trainer: Optional[OffTrainerConfig] = None
    runtime: RuntimeConfig = RuntimeConfig()

@configclass
class AtariRunConfig(PythonConfig):
    env: EnvConfig
    policy: RegistryConfig[POLICIES]
    network1: Optional[RegistryConfig[NETWORKS]] = None
    network2: Optional[RegistryConfig[NETWORKS]] = None
    trainer: Optional[TrainerConfig] = None
    runtime: RuntimeConfig = RuntimeConfig()


@configclass
class GameUtilsConfig(PythonConfig):
    env: AtariEnvConfig
    network1: Optional[RegistryConfig[NETWORKS]] = None
    runtime: RuntimeConfig = RuntimeConfig()
    target: int
    pattern: int