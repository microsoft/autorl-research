# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from tianshou.data import to_torch
from utilsd import use_cuda
from code.network.base import load_weight

__all__ = ['chain_dedup', 'preprocess_obs', 'load_weight']


def chain_dedup(*iterables):
    seen = set()
    for iterable in iterables:
        for i in iterable:
            if i not in seen:
                seen.add(i)
                yield i


def preprocess_obs(obs):
    return dict(to_torch(obs, device='cuda' if use_cuda() else 'cpu'))
