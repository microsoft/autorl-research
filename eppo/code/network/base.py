# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import torch.nn as nn
from utilsd.config import Registry


class NETWORKS(metaclass=Registry, name='network'):
    pass

class BaseNetwork(nn.Module):
    output_dim: int

def load_weight(policy, path):
    assert isinstance(policy, nn.Module), 'Policy has to be an nn.Module to load weight.'
    policy.load_state_dict(torch.load(path, map_location='cpu'))