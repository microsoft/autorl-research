# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import copy,time,gym
import dataclasses
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy
from torch.utils.data import Dataset
from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment, use_cuda
from utilsd.experiment import print_config
from utilsd.earlystop import EarlyStop, EarlyStopStatus
from utilsd.logging import print_log
from tianshou.env import BaseVectorEnv, DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv

import sys,os
cur_dir = os.getcwd()
sys.path.insert(0,cur_dir)
print(sys.path)
from code.env import EnvConfig
from code.tools.config import TrainerConfig,GameRunConfig
from code.env.game_logger import GameLogger as Logger
from code.tools.env_utils import atari_game_env_factory

class GameOnPolicyTrainer:
    def __init__(self,
                 checkpoint_dir: Optional[Path] = None,
                 metrics_dir: Optional[Path] = None,
                 save_epoch=True):
        self.checkpoint_dir = checkpoint_dir
        self.metrics_dir = metrics_dir
        self.collector = None
        self.buffer = None
        self.test_env = None
        self.test_collector = None
        self.policy = None
        self.save_epoch = save_epoch #more log info if set True,

    def _train_epoch(self, policy: BasePolicy, train_env: BaseVectorEnv, *,
                     buffer_size: int, episode_per_collect: int,
                     batch_size: int, repeat_per_collect: int) -> Dict[str, Any]:
        # 1 epoch = 1 collect
        if self.buffer is None:
            self.buffer = VectorReplayBuffer(buffer_size, len(train_env))
            self.collector = Collector(policy,train_env,self.buffer)
            #init rnd
            if hasattr(policy,'rnd') and policy.rnd:
                policy.train()
                self.collector.collect(n_step=1000, random=True)
                batch,_ = self.collector.buffer.sample(0)
                policy.rnd_model.update_obs(batch.obs)
        self.collector.reset_buffer(keep_statistics=True)
        policy.train()
        col_result = self.collector.collect(n_step=episode_per_collect)
        #update reward normalization of 
        if hasattr(policy,'rnd') and policy.rnd:
            batch,_ = self.collector.buffer.sample(0)
            policy.rnd_model.update_obs(batch.obs)
        #repeat_time = int((episode_per_collect * repeat_per_collect)/batch_size))+1
        update_result = policy.update(0, self.collector.buffer, batch_size=batch_size, repeat=repeat_per_collect)
        return {'collect/' + k: np.mean(v) for k, v in {**col_result, **update_result}.items()}

    def train(self, policy: BasePolicy,
              env_fn: Callable[[Logger, str, bool], BaseVectorEnv],
              env_name: str,
              *, max_epoch: int, repeat_per_collect: int,
              batch_size: int, episode_per_collect: int, buffer_size: int = 200000,
              earlystop_patience: int = 5, val_every_n_epoch: int = 1) -> Tuple[Logger, Logger]:
        if self.checkpoint_dir is not None:
            _resume_path = self.checkpoint_dir / 'resume.pth'
        else:
            _resume_path = Path('/tmp/resume.pth')

        def _resume():
            nonlocal best_state_dict, cur_epoch
            if _resume_path.exists():
                print_log(f'Resume from checkpoint: {_resume_path}', __name__)
                data = torch.load(_resume_path)
                logger.load_state_dict(data['logger'])
                earlystop.load_state_dict(data['earlystop'])
                policy.load_state_dict(data['policy'])
                best_state_dict = data['policy_best']
                if hasattr(policy, 'optim'):
                    policy.optim.load_state_dict(data['optim'])
                cur_epoch = data['epoch']

        def _checkpoint():
            torch.save({
                'logger': logger.state_dict(),
                'val_logger': val_logger.state_dict(),
                'earlystop': earlystop.state_dict(),
                'policy': policy.state_dict(),
                'policy_best': best_state_dict,
                'optim': policy.optim.state_dict() if hasattr(policy, 'optim') else None,
                'epoch': cur_epoch
            }, _resume_path)
            print_log(f'Checkpoint saved to {_resume_path}', __name__)

        self.policy = policy
        logger = Logger(1000, log_interval=50, tb_prefix='train', count_global='step')
        val_logger = Logger(1000, log_interval=50, tb_prefix='val')
        earlystop = EarlyStop(patience=earlystop_patience)
        cur_epoch = 0
        train_env  = best_state_dict = None

        _resume()
        save_freq = int(max_epoch/100) #decrease the frequency of save checkpoints
        save_freq = max(save_freq,1)

        while cur_epoch < max_epoch:
            cur_epoch += 1
            if train_env is None:
                train_env = env_fn(logger, env_name)
            # if val_env is None:
            #     val_env = env_fn(val_logger, env_name)

            logger.reset(f'Train Epoch [{cur_epoch}/{max_epoch}] Episode')
            val_logger.reset(f'Val Epoch [{cur_epoch}/{max_epoch}] Episode')

            collector_res = self._train_epoch(
                policy, train_env,
                buffer_size=buffer_size,
                episode_per_collect=episode_per_collect,
                batch_size=batch_size,
                repeat_per_collect=repeat_per_collect)
            
            if self.save_epoch or cur_epoch % val_every_n_epoch == 0:
                #io operation is time-consuming
                logger.write_summary(collector_res)
            

            if cur_epoch%val_every_n_epoch==0 and self.checkpoint_dir is not None:
                torch.save(policy.state_dict(), self.checkpoint_dir / 'latest.pth')
            
            if self.save_epoch and self.checkpoint_dir is not None and cur_epoch % val_every_n_epoch == 0:
                torch.save(policy.state_dict(), self.checkpoint_dir / f'epoch_{cur_epoch}.pth')

            if cur_epoch == max_epoch or cur_epoch % val_every_n_epoch == 0:
                if cur_epoch ==max_epoch:
                    val_result, _ = self.evaluate(policy, env_fn, env_name, val_logger, eps_num=100)
                else:
                    val_result, _ = self.evaluate(policy, env_fn, env_name, val_logger)
                val_logger.global_step = logger.global_step  # sync two loggers
                val_logger.write_summary()

                es = earlystop.step(val_result)
                if es == EarlyStopStatus.BEST:
                    best_state_dict = copy.deepcopy(policy.state_dict())
                    if self.checkpoint_dir is not None:
                        torch.save(best_state_dict, self.checkpoint_dir / 'best.pth')
                elif es == EarlyStopStatus.STOP:
                    break
            if cur_epoch%val_every_n_epoch==0:
                _checkpoint()

        if best_state_dict is not None:
            policy.load_state_dict(best_state_dict)

        return logger, val_logger

    def evaluate(self,
                 policy: BasePolicy,
                 env_fn: Callable[[Logger, str, bool], BaseVectorEnv],
                 env_name: str,
                 logger: Optional[Logger] = None,
                 eps_num: int = 10,
                 final_test=False):
        if logger is None:
            logger = Logger(50,log_interval=10)
        
        if final_test:
            test_env = env_fn(logger, env_name, False)
            collector = Collector(policy,test_env)
            policy.eval()
            collector.collect(n_episode=eps_num)
        else:
            if self.test_env is None:
                self.test_env = env_fn(logger, env_name, False)
                self.test_collector = Collector(policy,self.test_env)
            self.test_collector.reset()
            #self.test_collector.reset_buffer(keep_statistics=False)
            #test_collector = Collector(policy, self.test_env)
            policy.eval()
            self.test_collector.collect(n_episode=eps_num)

        return logger.summary()['reward'], None


def game_train_and_test(env_config: EnvConfig,
                   train_config: TrainerConfig,
                   policy: BasePolicy,
                   env_name: str,
                   seed: int=42):

    def env_fn(logger: Logger, env_name: str, training:bool=True) -> BaseVectorEnv:
        train_env, test_env = atari_game_env_factory(env_config,env_name,logger,seed)
        if training:
            return train_env
        else:
            return test_env

    trainer = GameOnPolicyTrainer(checkpoint_dir=get_checkpoint_dir(), metrics_dir=get_output_dir(),save_epoch=train_config.save_epoch)

    train_kwargs = dataclasses.asdict(train_config)
    train_kwargs.pop('fast_dev_run')
    train_kwargs.pop('save_epoch')
    train_logger, _ = trainer.train(policy, env_fn, env_name, **train_kwargs)



def main(config):
    setup_experiment(config.runtime)
    print_config(config)
    env_name = config.env.env_name
    env = gym.make(env_name)
    env_action_space = env.action_space
    env_observation_space = env.observation_space
    config.network1.input_dims = env_observation_space.shape[0]
    config.network2.input_dims = env_observation_space.shape[0]

    if config.network1 is not None and config.network2 is not None:
        network1 = config.network1.build()
        network2 = config.network2.build()
        policy = config.policy.build(network1 = network1,
                                     network2=network2,
                                     obs_space=env_observation_space,
                                     action_space=env_action_space)
    else:
        policy = config.policy.build(obs_space=env_observation_space,
                                     action_space=env_action_space)

    if use_cuda():
        policy.cuda()
    game_train_and_test(config.env, config.trainer, policy, env_name, seed=config.runtime.seed)


if __name__ == '__main__':
    _config = GameRunConfig.fromcli()
    main(_config)
