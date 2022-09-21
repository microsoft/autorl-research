# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from typing import Optional

import gym
from tianshou.policy import BasePolicy
from utilsd.config import Registry


class POLICIES(metaclass=Registry, name='policy'):
    pass