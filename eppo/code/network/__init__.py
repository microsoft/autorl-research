# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import torch.nn as nn
from .mlp import MLP
from .cnn import MiniCNN, AtariCNN, MiniLargeCNN
from .base import BaseNetwork, NETWORKS

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.q_net = nn.Linear(in_dim, out_dim)
        self.k_net = nn.Linear(in_dim, out_dim)
        self.v_net = nn.Linear(in_dim, out_dim)

    def forward(self, Q, K, V):
        q = self.q_net(Q)
        k = self.k_net(K)
        v = self.v_net(V)

        attn = torch.einsum("ijk,ilk->ijl", q, k)
        attn = attn.to(Q.device)
        attn_prob = torch.softmax(attn, dim=-1)

        attn_vec = torch.einsum("ijk,ikl->ijl", attn_prob, v)

        return attn_vec

class SelfAttention(Attention):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim,out_dim)

    def forward(self, X):
        return super().forward(X,X,X)