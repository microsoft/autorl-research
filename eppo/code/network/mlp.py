# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from code.network.base import NETWORKS
import torch
import torch.nn as nn

@NETWORKS.register_module()
class MLP(nn.Module):
    def __init__(self,
                 input_dims: int,
                 hidden_dim: int=64,
                 output_dim: int=32,
                 num_layers: int=2):
        super().__init__()
        self.raw_fc = nn.Sequential()
        self.output_dim = output_dim
        if num_layers==1:
            layers = [nn.Linear(input_dims, output_dim), nn.ReLU()]
        else:
            layers = [nn.Linear(input_dims, hidden_dim), nn.ReLU()]
            for i in range(num_layers-1):
                layers.append(nn.Linear(hidden_dim,hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim,output_dim))
            layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers)
    
    def forward(self,x):
        for i, l in enumerate(self.layers):
            x = self.layers[i](x) 
        return x
        
