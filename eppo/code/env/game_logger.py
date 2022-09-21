# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import inspect
import json
from collections import defaultdict,deque
from typing import TextIO

from torch.utils.tensorboard import SummaryWriter
from utilsd import get_tb_log_dir, get_output_dir
from utilsd.avgmeter import MetricMeter
from utilsd.logging import print_log

from .finite_env import BaseLogger


__all__ = ['Logger']

_tb_logger = _json_writer = None


def _get_tb_logger() -> SummaryWriter:
    global _tb_logger
    if _tb_logger is None:
        _tb_logger = SummaryWriter(log_dir=get_tb_log_dir())
    return _tb_logger


def _get_json_writer() -> TextIO:
    global _json_writer
    if _json_writer is None:
        _json_writer = (get_output_dir() / 'summary.json').open('a')
    return _json_writer


class GameLogger(BaseLogger):

    def __init__(self, ep_total, *, log_interval=100, prefix='Episode', tb_prefix='', count_global='episode', max_len=20):
        self.meter = MetricMeter()
        self.ep_count = 0
        self.global_step = 0
        self.ep_total = ep_total
        self.log_interval = log_interval
        self.prefix = prefix
        self.active_env_ids = set()
        assert count_global in ['step', 'episode']
        self.count_global = count_global

        self.tb_writer = _get_tb_logger()
        self.tb_prefix = tb_prefix

        self.json_writer = _get_json_writer()

        self.episode_lengths = dict()
        self.episode_rewards = dict()
        self.eps_rwd = deque(maxlen=max_len)

    def log_step(self, env_id, obs, rew, done, info):
        self.active_env_ids.add(env_id)
        self.episode_lengths[env_id] += 1
        self.episode_rewards[env_id] += rew

        if self.count_global == 'step':
            self.global_step += 1

        if not done:
            return

        if self.count_global == 'episode':
            self.global_step += 1

        self.ep_count += 1
        self.eps_rwd.append(self.episode_rewards[env_id])
        logs = dict()  # deal with batch
        logs.update({
            'step_per_episode': self.episode_lengths[env_id],
            'reward': self.episode_rewards[env_id],
            'num_active_envs': len(self.active_env_ids)
        })
        # print(logs)
        # exit(1)

        self.meter.update({k: v for k, v in logs.items()})
        if self.ep_count % self.log_interval == 0 or self.ep_count >= self.ep_total:
            frm = inspect.stack()[1]
            mod = inspect.getmodule(frm[0])
            print_log(f'{self.prefix} [{self.ep_count}/{self.ep_total}]  {self.meter}', mod.__name__)
            print_log(f'{self.prefix} [{self.ep_count}/{self.ep_total}]  {sum(self.eps_rwd)/len(self.eps_rwd)}', mod.__name__)

    def log_reset(self, env_id, obs):
        self.episode_lengths[env_id] = 0
        self.episode_rewards[env_id] = 0.

    def set_prefix(self, prefix):
        self.prefix = prefix

    def write_summary(self, extra_metrics=None):
        if extra_metrics:
            self.meter.update(extra_metrics)
        summary = self.summary()
        if len(self.eps_rwd)>0:
            summary["eps_rwd_avg"] = sum(self.eps_rwd)/len(self.eps_rwd)
        else:
            summary["eps_rwd_avg"] = 0
        print_log(f'{self.prefix} Summary:\n' + '\n'.join([f'    {k}\t{v:.4f}' for k, v in summary.items()]), __name__)
        for key, value in summary.items():
            if self.tb_prefix:
                key = self.tb_prefix + '/' + key
            self.tb_writer.add_scalar(key, value, global_step=self.global_step)
        summary = {'prefix': self.tb_prefix, 'step': self.global_step, **summary}
        self.json_writer.write(json.dumps(summary) + '\n')
        self.json_writer.flush()

    def summary(self):
        return {key: self.meter[key].avg for key in self.meter}

    def reset(self, prefix=None):
        self.ep_count = self.step_count = 0
        self.meter.reset()
        self.active_env_ids = set()
        if prefix is not None:
            self.set_prefix(prefix)

    def state_dict(self):
        # logging status within epoch is not saved
        return {
            'global_step': self.global_step,
        }

    def load_state_dict(self, state_dict):
        self.global_step = state_dict['global_step']
