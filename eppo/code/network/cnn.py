# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from code.network.base import NETWORKS
import torch
import torch.nn as nn
from typing import List
import numpy as np
from utilsd import print_log,use_cuda

#specially for atari
@NETWORKS.register_module()
class AtariCNN(nn.Module):
    def __init__(self,
                 input_dims: int,
                 hidden_dim: int=64,
                 output_dim: int=64,
                 num_layers: int=2,
                 features_only: bool=False):
        super().__init__()
        self.state_embed = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten())
        if not features_only:
            self.state_embed = nn.Sequential(
                self.state_embed, nn.Linear(3136, output_dim), nn.ReLU()
            )
            self.output_dim = output_dim
        else:
            self.output_dim = 3136
        
    
    def forward(self,x):
        #x = torch.transpose(x,1,3).float()
        # print(x)
        # exit(1)
        return self.state_embed(x.float())
    

@NETWORKS.register_module()
class MiniCNN(nn.Module):
    def __init__(self,
                 input_dims: List[int], #c,h,w
                 hidden_dim: int=64,
                 output_dim: int=64,
                 num_layers: int=2):
        super().__init__()
        # self.state_embed = nn.Sequential(
        #     nn.Conv2d(input_dims[0], 16, 4, stride=2, padding=0), nn.ReLU(),
        #     nn.Conv2d(16, 32, 4, stride=2, padding=0), nn.ReLU(),
        #     nn.Conv2d(32, 32, 3, stride=2, padding=0), nn.ReLU(),
        #     nn.Flatten())
        #specified for POMDP scene
        #print(f"==========={torch.__version__}==============")
        self.state_embed = nn.Sequential(
            nn.Conv2d(input_dims[0], 16, kernel_size=4, stride=2, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0), nn.ReLU(inplace=True),
            # nn.Conv2d(32, 32, 3, stride=2, padding=0), nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            tmp_data = torch.zeros(1, input_dims[0], input_dims[1], input_dims[2])
            # if use_cuda():
            #     tmp_data = tmp_data.cuda()
            self.outter_dim = np.prod(self.state_embed(tmp_data).shape[1:])
        self.net = nn.Sequential(
            self.state_embed,
            nn.Linear(self.outter_dim, output_dim), nn.ReLU(inplace=True)
        )

        self.output_dim = output_dim
    
    def forward(self,x):
        #x = torch.transpose(x,1,3).float()
        # print(x)
        # exit(1)
        return self.net(x.float())


@NETWORKS.register_module()
class MiniLargeCNN(nn.Module):
    def __init__(self,
                 input_dims: List[int], #c,h,w
                 hidden_dim: int=64,
                 output_dim: int=64,
                 num_layers: int=2):
        super().__init__()
        self.state_embed = nn.Sequential(
            nn.Conv2d(input_dims[0], 16, kernel_size=4, stride=2, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0), nn.ReLU(inplace=True),
            # nn.Conv2d(32, 32, 3, stride=2, padding=0), nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            tmp_data = torch.zeros(1, input_dims[0], input_dims[1], input_dims[2])
            # if use_cuda():
            #     tmp_data = tmp_data.cuda()
            self.outter_dim = np.prod(self.state_embed(tmp_data).shape[1:])
        self.net = nn.Sequential(
            self.state_embed,
            nn.Linear(self.outter_dim, output_dim), nn.ReLU(inplace=True)
        )

        self.output_dim = output_dim
    
    def forward(self,x):
        #x = torch.transpose(x,1,3).float()
        # print(x)
        # exit(1)
        return self.net(x.float())


@NETWORKS.register_module()
class SokoCNN(nn.Module):
    def __init__(self,
                 input_dims: List[int], #c,h,w
                 hidden_dim: int=64,
                 output_dim: int=64,
                 num_layers: int=2):
        super().__init__()
        self.state_embed = nn.Sequential(
            nn.Conv2d(input_dims[0], 16, 8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            self.outter_dim = np.prod(
                self.state_embed(torch.zeros(1, input_dims[0], input_dims[1], input_dims[2])).shape[1:])
        self.net = nn.Sequential(
            self.state_embed,
            nn.Linear(self.outter_dim, output_dim), nn.ReLU(inplace=True)
        )

        self.output_dim = output_dim
    
    def forward(self,x):
        #x = torch.transpose(x,1,3).float()
        # print(x)
        # exit(1)
        return self.net(x.float())