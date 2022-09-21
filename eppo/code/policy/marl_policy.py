# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os,json,copy,math
from pathlib import Path
from typing import Optional, List, Dict, Any

import gym
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from gym.spaces import Discrete
from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment, use_cuda, print_log
from tianshou.data import Batch,to_torch,to_torch_as, ReplayBuffer
from tianshou.policy import PPOPolicy, A2CPolicy, BasePolicy
from tianshou.utils import RunningMeanStd
from tianshou.utils.net.common import ActorCritic
from code.network import BaseNetwork, Reshape, SelfAttention
from .base import POLICIES
from .utils import chain_dedup, load_weight, preprocess_obs
from .ppo import PPO, PPOActor,PPOCritic

EPS = 1e-5

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)



class MARLPPOActor(nn.Module):
    def __init__(self, extractor: BaseNetwork, action_dim: int, num_policy: int, hidden_dim: int=64):
        super().__init__()
        self.num_policy = num_policy
        self.action_dim = action_dim
        self.feature_dim = extractor.output_dim
        
        self.extractor = [extractor]
        for i in range(self.num_policy-1):
            tmp_extractor = copy.deepcopy(extractor)
            tmp_extractor.apply(weights_init)
            self.extractor.append(tmp_extractor)
        self.extractor = nn.ModuleList(self.extractor)
        
        #build multi head module
        self.unique_head = []
        for i in range(num_policy):
            self.unique_head.append(nn.Linear(extractor.output_dim, action_dim))
        self.unique_head = nn.ModuleList(self.unique_head)      

    def forward(self, obs, state=None, info={}, ret_feature=False):
        res = []
        feature_list = []
        for i in range(self.num_policy):
            if isinstance(obs,dict):
                feature = self.extractor[i](preprocess_obs(obs))
            else:
                feature = self.extractor[i](to_torch(obs, device='cuda' if use_cuda() else 'cpu'))
            #feature = self.extractor[i](preprocess_obs(obs))
            feature_list.append(feature)
            tmp = self.unique_head[i](feature)
            res.append(tmp)
        out = torch.cat(res,dim=-1)
        concat_feature = torch.stack(feature_list,dim=1) #[batch,num_policy,feature_dim]
        out = torch.reshape(out,(-1,self.num_policy,self.action_dim)) 
        out = nn.functional.softmax(out,dim=-1)
        if ret_feature:
            return out, state,concat_feature
        else:
            return out, state # out: [B,num_policy,action_dim]


class MARLPPOCritic(nn.Module):
    def __init__(self, extractor: BaseNetwork, num_policy: int):
        super().__init__()
        self.num_policy = num_policy
        self.extractor = extractor
        self.value_out = nn.Linear(extractor.output_dim, 1)

    def forward(self, obs, state=None, info={}):
        if isinstance(obs,dict):
            feature = self.extractor(preprocess_obs(obs))
        else:
            feature = self.extractor(to_torch(obs, device='cuda' if use_cuda() else 'cpu'))
        return self.value_out(feature).squeeze(dim=-1) #[B]

@POLICIES.register_module()
class MARLPPOPolicy(PPOPolicy):
    def __init__(self,
                 num_policy: int,
                 lr: float,
                 weight_decay: float = 0.,
                 discount_factor: float = 1.,
                 max_grad_norm: float = 100.,
                 reward_normalization: bool = True,
                 eps_clip: float = 0.3,
                 value_clip: float = True,
                 vf_coef: float = 1.,
                 gae_lambda: float = 1.,
                 diverse_coef: float = 1.,
                 sub_policy_coef: float = 1.,
                 center_policy_coef: float = 1.,
                 random_sample: bool = False,
                 network1: Optional[BaseNetwork] = None, 
                 network2: Optional[BaseNetwork] = None,
                 obs_space: Optional[gym.Space] = None,
                 action_space: Optional[gym.Space] = None,
                 weight_file: Optional[Path] = None):
        """MARL policy which contains num_policy base policies
           during exploration, we use the base policies' action distribution + std of base policies' action distribution (curiosity driven exploration) to sample action
                sample distribution = (enemble_distribution + alpha*std)/(1+alpha)  (Eq. 1)
           For netwrok architecture, the K base policies share same base layers and uese K different heads, and we only train a value function for ensemble policy
           The loss = ensemble policy's actor loss + ensemble policy's critic loss + base policies' actor loss + base policies' diversity loss + entropy loss

        Args:
            num_policy (int): number of base policies
            lr (float): learning rate
            diverse_coef (float, optional): weight of diveristy loss. Defaults to 1..
            sub_policy_coef (float, optional): weight of base policies's actor loss. Defaults to 1..
            center_policy_coef (float, optional): weight of center policies's actor loss. Defaults to 1..
        """
        assert network1 is not None and obs_space is not None
        assert isinstance(action_space, Discrete)
        self.num_policy = num_policy
        self.diverse_coef = diverse_coef
        self.sub_policy_coef = sub_policy_coef
        self.center_policy_coef = center_policy_coef
        self.random_sample = random_sample

        actor = MARLPPOActor(network1, action_space.n, self.num_policy)
        critic = MARLPPOCritic(network2, self.num_policy)
        
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

        self.action_dim = action_space.n
        
        if weight_file is not None:
            load_weight(self, weight_file)
        print("init done")

    def forward(self, batch, state=None, **kwargs):
        if self.random_sample and self.training:
            logits, h = self.actor(batch.obs, state=state)
            logits = logits[np.arange(logits.shape[0]),batch.policy]
            ens_logits_ = logits
            ens_logits = logits
            dist = self.dist_fn(logits)
        else:
            logits, ens_logits,ens_logits_, dist, h, base_weight = self.getlogits(batch,state)
        
        if self.training:
            act = dist.sample()
        else:
            act = torch.argmax(ens_logits_, dim=1)
        
        return Batch(logits=ens_logits, act=act, state=h, dist=dist)
    
    def getlogits(self,batch,state=None):
        """[summary]
        get raw logits and the ensembled logits
        """
        logits, h = self.actor(batch.obs, state=state) #logits [B,num_policy,action_dim]
        concat_feature = None
        if torch.isnan(logits).any():
            print(logits)
            print(batch.obs)
            raise ValueError("Nan in raw logits")
        ens_logits,base_weight = self.ensemble_base_policies(logits,batch.obs,concat_feature) #ens_logits [B,action_dim]
        dist = self.dist_fn(ens_logits)
        ens_logits_ = ens_logits
        return logits, ens_logits, ens_logits_,dist, h, base_weight
    
    def ensemble_base_policies(self,logits,obs=None,concat_feature=None):
        """[summary]
            The way to aggregate logits of base policies into a single logits
        """
        base_weight = None
        #aggregation policy's output based on policy's output
        ens_logits = torch.mean(logits,dim=1) #just take mean, [B,num_policy,action_dim] ----> [B,action_dim]
        return ens_logits, base_weight
    
    def get_diversity(self, logits):
        """Compute the diversity loss base on the action distribution of each base policies
        Args:
            logits ([Tensor]): [B,num_policy,action_dim] action distribution of each base policies
        Returns:
            [Tensor]: diversity loss
        """
        transpose_logits = torch.transpose(logits,1,2)
        diff_vec = torch.matmul(logits,transpose_logits)
        diff_vec = torch.triu(diff_vec,diagonal=1)
        diff_vec = torch.sum(diff_vec,dim=(1,2))*2/(self.num_policy*(self.num_policy-1))
        diff_loss = diff_vec.mean() # minimize diversity
        return diff_loss


    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices

        batch = self._compute_returns(batch, buffer, indices)

        # no need to change
        batch.act = to_torch_as(batch.act, batch.v_s)
        old_log_prob = []
        with torch.no_grad():
            for b in batch.split(self._batch, shuffle=False, merge_last=True):
                old_log_prob.append(self(b).dist.log_prob(b.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        return batch

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        diff_losses,sub_clip_losses = [],[]
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indice)
            for b in batch.split(batch_size, merge_last=True):
                logits,_,__,dist,h,base_weight = self.getlogits(b)
                if self._norm_adv:
                    mean, std = b.adv.mean(), b.adv.std()
                    std+=EPS
                    b.adv = (b.adv - mean) / std  # per-batch norm
                # compute actor loss for the ensembled policy
                ratio = (dist.log_prob(b.act) - b.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                
                surr1 = ratio * b.adv
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * b.adv
                if self._dual_clip:
                    clip_loss = -torch.max(
                        torch.min(surr1, surr2), self._dual_clip * b.adv
                    ).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                
                # compute actor loss for base policies
                sub_logits = torch.reshape(logits,(-1,self.action_dim)) # [B*num_policy,action]
                sub_acts = torch.unsqueeze(b.act,dim=-1).repeat((1,self.num_policy)).reshape((-1)) #[B*num_policy]
                sub_dist = self.dist_fn(sub_logits) 
                sub_log_prob = sub_dist.log_prob(sub_acts) #[B*num_policy]
                sub_logp_old = torch.unsqueeze(b.logp_old,dim=-1).repeat((1,self.num_policy)).reshape((-1)) #[B*num_policy]
                sub_adv = torch.unsqueeze(b.adv,dim=-1).repeat((1,self.num_policy)).reshape((-1)) #[B*num_policy]
                sub_ratio = (sub_log_prob-sub_logp_old).exp().float() #[B*num_policy]
                
                
                sub_surr1 = sub_ratio*sub_adv #[B*num_policy]
                sub_surr2 = sub_ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * sub_adv
                sub_surr1 = torch.reshape(sub_surr1,(-1,self.num_policy)) # [B,num_policy]
                sub_surr2 = torch.reshape(sub_surr2,(-1,self.num_policy))
                
                if self._dual_clip:
                    sub_clip_loss = -torch.max(
                        torch.min(sub_surr1, sub_surr2), self._dual_clip * sub_adv
                    ).mean()
                else:
                    sub_clip_loss = -torch.min(sub_surr1, sub_surr2).mean()
                
                # calculate diversity loss
                if self.diverse_coef>0:
                    diff_loss = self.get_diversity(logits)
                else:
                    diff_loss = to_torch_as(torch.tensor(0.0),logits)

                # calculate loss for critic
                # whether share the same value policy? no
                value = self.critic(b.obs).flatten()
                if self._value_clip:
                    v_clip = b.v_s + (value - b.v_s).clamp(
                        -self._eps_clip, self._eps_clip)
                    vf1 = (b.returns - value).pow(2)
                    vf2 = (b.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (b.returns - value).pow(2).mean()
                
                
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = self.center_policy_coef*clip_loss + self.sub_policy_coef*sub_clip_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss \
                    + self.diverse_coef*diff_loss #_weight_ent is 0.01 by default                
                
                if torch.isnan(loss):
                    print(loss)
                    print(base_weight)
                    print(clip_loss)
                    print(sub_clip_loss)
                    print(vf_loss)
                    print(diff_loss)
                    raise ValueError("nan in loss")

                self.optim.zero_grad()
                loss.backward()

                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        set(self.actor.parameters()).union(self.critic.parameters()),
                        max_norm=self._grad_norm)
                self.optim.step()
                
                clip_losses.append(clip_loss.item())
                sub_clip_losses.append(sub_clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                diff_losses.append(diff_loss.item())
                losses.append(loss.item())
        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        res = {
            "loss": losses,
            "loss/ens_clip": clip_losses,
            "loss/sub_clip": sub_clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
            "loss/diverse": diff_losses,
            "loss/diverse/mean":np.mean(diff_losses),
            "loss/diverse/std":np.std(diff_losses),
            "loss/diverse/max":np.max(diff_losses),
            "loss/diverse/min":np.min(diff_losses)
        }

        return res