# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from pathlib import Path
from typing import Optional

import gym
import torch
import torch.nn as nn
from gym.spaces import Discrete
from tianshou.data import Batch
from tianshou.policy import PPOPolicy

from code.network import BaseNetwork
from .base import POLICIES
from .utils import chain_dedup, load_weight, preprocess_obs


class PPOActor(nn.Module):
    def __init__(self, extractor: BaseNetwork, action_dim: int):
        super().__init__()
        self.extractor = extractor
        self.layer_out = nn.Sequential(
            nn.Linear(extractor.output_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs, state=None, info={}):
        feature = self.extractor(preprocess_obs(obs))
        out = self.layer_out(feature)
        return out, state


class PPOCritic(nn.Module):
    def __init__(self, extractor: BaseNetwork):
        super().__init__()
        self.extractor = extractor
        self.value_out = nn.Linear(extractor.output_dim, 1)

    def forward(self, obs, state=None, info={}):
        feature = self.extractor(preprocess_obs(obs))
        return self.value_out(feature).squeeze(dim=-1)


@POLICIES.register_module()
class PPO(PPOPolicy):
    def __init__(self,
                 lr: float,
                 weight_decay: float = 0.,
                 discount_factor: float = 1.,
                 max_grad_norm: float = 100.,
                 reward_normalization: bool = True,
                 eps_clip: float = 0.3,
                 value_clip: float = True,
                 vf_coef: float = 1.,
                 gae_lambda: float = 1.,
                 network: Optional[BaseNetwork] = None,
                 obs_space: Optional[gym.Space] = None,
                 action_space: Optional[gym.Space] = None,
                 weight_file: Optional[Path] = None):
        assert network is not None and obs_space is not None
        assert isinstance(action_space, Discrete)
        actor = PPOActor(network, action_space.n)
        critic = PPOCritic(network)
        optimizer = torch.optim.Adam(
            chain_dedup(actor.parameters(), critic.parameters()),
            lr=lr, weight_decay=weight_decay)
        super().__init__(actor, critic, optimizer, torch.distributions.Categorical,
                         discount_factor=discount_factor,
                         max_grad_norm=max_grad_norm,
                         reward_normalization=reward_normalization,
                         eps_clip=eps_clip,
                         value_clip=value_clip,
                         vf_coef=vf_coef,
                         gae_lambda=gae_lambda)
        if weight_file is not None:
            load_weight(self, weight_file)

    def forward(self, batch, state=None, **kwargs):
        """
        This is already done in https://github.com/thu-ml/tianshou/pull/354
        Should be no longer needed for future release of tianshou.
        """
        logits, h = self.actor(batch.obs, state=state)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
            logits_, _ = logits
        else:
            dist = self.dist_fn(logits)
            logits_ = logits
        if self.training:
            act = dist.sample()
        else:
            act = torch.argmax(logits_, dim=-1)
        return Batch(logits=logits, act=act, state=h, dist=dist)
