# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder import make_encoder
from encoder import ActionOrRewardEncoder

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters,
        encoder_hidden_size
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, encoder_hidden_size, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters,
        encoder_hidden_size,
        use_action_embedding_for_Q = False
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, encoder_hidden_size, output_logits=True
        )

        if use_action_embedding_for_Q == True:

            action_dim = encoder_feature_dim
            print("Adjusting the dimension of action for Q to {}".format(action_dim))
            
        else:
            action_dim =  action_shape[0]

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_dim, hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_dim, hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        
        
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class Auxi(nn.Module):
    """
    Auxi
    """

    def __init__(self, obs_shape, action_shape, z_dim, auxi_pred_horizon, batch_size, critic, critic_target, pred_input, pred_output, prediction_type='mlp', action_embedding = False, reward_embedding = False):
        super(Auxi, self).__init__()
        self.batch_size = batch_size

        self.encoder = critic.encoder

        self.encoder_target = critic_target.encoder 

        self.auxi_pred_horizon = auxi_pred_horizon

        self.pred_input = pred_input
        self.pred_output = pred_output
        self.prediction_type = prediction_type

        self.pred_input_dim = 0
        self.pred_output_dim = 0

        self.action_embedding = action_embedding
        self.reward_embedding = reward_embedding

        if self.action_embedding:
            self.action_encoder = ActionOrRewardEncoder(input_dim=action_shape[0], feature_dim=z_dim, num_layers=1)
            self.action_encoder_target = ActionOrRewardEncoder(input_dim=action_shape[0], feature_dim=z_dim, num_layers=1)
            self.action_encoder_target.load_state_dict(self.action_encoder.state_dict())
        else:
            self.action_encoder = lambda x: x
            self.action_encoder_target = lambda x: x

        if self.reward_embedding:
            self.reward_encoder = ActionOrRewardEncoder(input_dim=1, feature_dim=z_dim, num_layers=1)
            self.reward_encoder_target = ActionOrRewardEncoder(input_dim=1, feature_dim=z_dim, num_layers=1)
            self.reward_encoder_target.load_state_dict(self.reward_encoder.state_dict())
        else:
            self.reward_encoder = lambda x: x
            self.reward_encoder_target = lambda x: x
        



        # input
        for i in range(len(self.pred_input['s'])):
            if self.pred_input['s'][i] == '1':
                self.pred_input_dim += z_dim
        for i in range(len(self.pred_input['a'])):
            if self.pred_input['a'][i] == '1':
                if not self.action_embedding:
                    self.pred_input_dim += action_shape[0]
                else:
                    self.pred_input_dim += z_dim
        for i in range(len(self.pred_input['r'])):
            if self.pred_input['r'][i] == '1':
                if not self.reward_embedding:
                    self.pred_input_dim += 1
                else:
                    self.pred_input_dim += z_dim
        if self.pred_input['s_'] == '1':
            self.pred_input_dim += z_dim


        # output
        for i in range(len(self.pred_output['s'])):
            if self.pred_output['s'][i] == '1':
                self.pred_output_dim += z_dim
        for i in range(len(self.pred_output['a'])):
            if self.pred_output['a'][i] == '1':
                if not self.action_embedding:
                    self.pred_output_dim += action_shape[0]
                else:
                    self.pred_output_dim += z_dim
        for i in range(len(self.pred_output['r'])):
            if self.pred_output['r'][i] == '1':
                if not self.reward_embedding:
                    self.pred_output_dim += 1
                else:
                    self.pred_output_dim += z_dim

        if self.pred_output['s_'] == '1':
            self.pred_output_dim += z_dim

        print("[ Pred input dimension ] = {}".format(self.pred_input_dim))
        print("[ Pred output dimension ] = {}".format(self.pred_output_dim))

        self.W = nn.Parameter(torch.rand(self.pred_output_dim, self.pred_output_dim))

        if self.prediction_type == 'mlp':

            self.predictor = nn.Sequential(nn.Linear(self.pred_input_dim, 2*self.pred_input_dim),
                                                        nn.BatchNorm1d(2*self.pred_input_dim),
                                                        nn.ReLU(),
                                                        nn.Linear(2*self.pred_input_dim,
                                                            self.pred_output_dim))
        else:
            raise NotImplementedError


    def encode_input(self, x, detach=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        
        #assert self.pred_input == [*x]
        

        y_in = []

        for i in range(len(x['s'])):
            y_in.append(self.encoder(x['s'][i]))
        for i in range(len(x['a'])):
            y_in.append(self.action_encoder(x['a'][i]))
        for i in range(len(x['r'])):
            y_in.append(self.reward_encoder(x['r'][i]))

        for i in range(len(x['s_'])):
            y_in.append(self.encoder(x['s_'][i]))

        y_in = torch.cat(y_in, dim=1)     
        y_out = self.predictor(y_in)
        


        if detach:
            y_out = y_out.detach()

        
        return y_out

    def encode_output(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        # import ipdb; ipdb.set_trace()
        
        y_out = []

        for i in range(len(x['s'])):
            if ema:
                with torch.no_grad():
                    y_out.append(self.encoder_target(x['s'][i]))
            else:
                y_out.append(self.encoder(x['s'][i]))

        for i in range(len(x['a'])):
            if ema:
                with torch.no_grad():
                    y_out.append(self.action_encoder_target(x['a'][i]))
            else:
                y_out.append(self.action_encoder(x['a'][i]))
        for i in range(len(x['r'])):
            if ema:
                with torch.no_grad():
                    y_out.append(self.reward_encoder_target(x['r'][i]))
            else:
                y_out.append(self.reward_encoder(x['r'][i]))

        for i in range(len(x['s_'])):
            if ema:
                with torch.no_grad():
                    y_out.append(self.encoder_target(x['s_'][i]))
            else:
                y_out.append(self.encoder(x['s_'][i]))

        
        y_out = torch.cat(y_out, dim=1)   

        if detach:
            y_out = y_out.detach()
        
        return y_out



class AuxiSacAgent(object):
    """Auxiliary representation learning with SAC."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,

        encoder_hidden_size = 256,

        auxi_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        update_encoder_with_actorloss=False,
        auxi_batchsize = 128,
        auxi_coef = 1.0,

        auxi_pred_horizon = 1,

        auxi_pred_input_s = '1',
        auxi_pred_input_a = '1',
        auxi_pred_input_r = '1',
        auxi_pred_input_s_ = '1',

        auxi_pred_output_s = '1',
        auxi_pred_output_a = '1',
        auxi_pred_output_r = '1',
        auxi_pred_output_s_ = '1',

        augmentation_type = 'crop',
        prediction_type = 'mlp',
        auxi_ema = True,
        similarity_metric = 'mse',
        penalize_negative = False,
        action_embedding = False,
        reward_embedding = False,

        use_action_embedding_for_Q = False,
    ):

    
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.auxi_update_freq = auxi_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.detach_encoder = detach_encoder
        self.update_encoder_with_actorloss = update_encoder_with_actorloss
        print("[Detach Encoder] = ", self.detach_encoder)
        print("[Update_encoder_with_actorloss] = ", self.update_encoder_with_actorloss) 
        self.encoder_type = encoder_type
        if self.encoder_type == 'ofe': 
            encoder_feature_dim = num_layers * encoder_hidden_size + obs_shape[0]
            print("Adjusting the dimension of encoder for OFENet to {}".format(encoder_feature_dim))

        self.use_action_embedding_for_Q = use_action_embedding_for_Q

        if self.use_action_embedding_for_Q:
            assert self.encoder_type == 'ofe' and action_embedding
        print("[Using Action Embedding for Q network] = {}".format(self.use_action_embedding_for_Q)) 

            




        self.auxi_batchsize = auxi_batchsize
        
        self.auxi_coef = auxi_coef

        self.auxi_pred_horizon = auxi_pred_horizon
        # self.auxi_type = auxi_type
        
        # self.pred_input = self.auxi_type.split('---')[0].strip('[]').split(',')
        # self.pred_output = self.auxi_type.split('---')[1].strip('[]').split(',')

        assert len(auxi_pred_input_s) == auxi_pred_horizon 
        assert len(auxi_pred_input_a) == auxi_pred_horizon 
        assert len(auxi_pred_input_r) == auxi_pred_horizon 
        assert len(auxi_pred_input_s_) == 1 

        assert len(auxi_pred_output_s) == auxi_pred_horizon 
        assert len(auxi_pred_output_a) == auxi_pred_horizon 
        assert len(auxi_pred_output_r) == auxi_pred_horizon 
        assert len(auxi_pred_output_s_) == 1

        self.pred_input = {'s': auxi_pred_input_s, 'a': auxi_pred_input_a, 'r': auxi_pred_input_r, 's_': auxi_pred_input_s_}
        self.pred_output = {'s': auxi_pred_output_s, 'a': auxi_pred_output_a, 'r': auxi_pred_output_r, 's_': auxi_pred_output_s_}
        
        print("[ pred_horizon ] {}".format(self.auxi_pred_horizon))
        print("[ pred_input ] {}".format(str(self.pred_input)))
        print("[ pred_output ] {}".format(str(self.pred_output)))

        self.augmentation_type = augmentation_type
        self.prediction_type = prediction_type
        self.auxi_ema = auxi_ema
        self.similarity_metric = similarity_metric
        self.penalize_negative = penalize_negative

        self.action_embedding = action_embedding
        self.reward_embedding = reward_embedding

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, 
            encoder_hidden_size
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters,
            encoder_hidden_size,
            use_action_embedding_for_Q
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters,
            encoder_hidden_size,
            use_action_embedding_for_Q
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.encoder_type in['pixel', 'mlp', 'ofe']:
            # create CURL encoder (the 128 batch size is probably unnecessary)
            
            self.Auxi = Auxi(obs_shape, action_shape, encoder_feature_dim, 
                        self.auxi_pred_horizon, self.auxi_batchsize, 
                        self.critic,self.critic_target, pred_input=self.pred_input, pred_output=self.pred_output, prediction_type=prediction_type,
                        action_embedding=self.action_embedding, reward_embedding=self.reward_embedding).to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.auxi_optimizer = torch.optim.Adam(
                self.Auxi.parameters(), lr=encoder_lr
            )
            if self.use_action_embedding_for_Q:
                self.auxi_action_encoder_optimizer = torch.optim.Adam(
                    self.Auxi.action_encoder.parameters(), lr=encoder_lr
                )
        else:
            raise NotImplementedError
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type in ['pixel','mlp','ofe']:
            self.Auxi.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
 
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            if self.use_action_embedding_for_Q:
                policy_action_embedding = self.Auxi.action_encoder_target(policy_action)
                target_Q1, target_Q2 = self.critic_target(next_obs, policy_action_embedding)
            else:
                target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        if self.use_action_embedding_for_Q:
            if self.detach_encoder:
                with torch.no_grad():
                    action_embedding = self.Auxi.action_encoder(action)
            else:
                action_embedding = self.Auxi.action_encoder(action)
            current_Q1, current_Q2 = self.critic(
                obs, action_embedding, detach_encoder=self.detach_encoder)
        else:
            current_Q1, current_Q2 = self.critic(
                obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        if self.use_action_embedding_for_Q:
            self.auxi_action_encoder_optimizer.zero_grad()
        critic_loss.backward()

        self.critic_optimizer.step()
        if self.use_action_embedding_for_Q:
            self.auxi_action_encoder_optimizer.step()
            # print("updating action encoder")
            # for i, ap in enumerate(self.Auxi.action_encoder.parameters()): 
                
            #     print(ap)
            #     print(ap.grad)
            #     print(i)
            #     import ipdb; ipdb.set_trace()
        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        detach_encoder_from_actor_loss = not self.update_encoder_with_actorloss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=detach_encoder_from_actor_loss)
        if self.use_action_embedding_for_Q:
            if detach_encoder_from_actor_loss:
                with torch.no_grad():
                    pi_embedding = self.Auxi.action_encoder(pi)
            else:
                pi_embedding = self.Auxi.action_encoder(pi)
            actor_Q1, actor_Q2 = self.critic(obs, pi_embedding, detach_encoder=True)
        else:
            actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        if self.use_action_embedding_for_Q:
            self.auxi_action_encoder_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        if self.use_action_embedding_for_Q:
            self.auxi_action_encoder_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()

        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_auxi(self, pred_input, pred_output, auxi_kwargs, L, step):
        
        y_pred = self.Auxi.encode_input(pred_input)
        if self.auxi_ema:
            y_target = self.Auxi.encode_output(pred_output, ema=True)
        else:
            y_target = self.Auxi.encode_output(pred_output, ema=False)

        #import ipdb; ipdb.set_trace()

        if not self.penalize_negative: # NO negative samples to loss
            if self.similarity_metric == 'mse':
                loss = F.mse_loss(y_pred, y_target) * self.auxi_coef

            elif self.similarity_metric == 'nmse':
                loss = F.mse_loss(F.normalize(y_pred), F.normalize(y_target)) * self.auxi_coef
                # norm_loss = 0
                # for i in range(y_pred.shape[0]):
                #     norm_loss += (y_pred[i]-y_target[i])**2
                    # mse_loss = mean(norm_loss)/128
            elif self.similarity_metric == 'inner_product':
                inner_product = torch.matmul(y_pred, y_target.T)  # (B,B)
                # ======
                # Explode if direct optimize inner product
                # logits = inner_product - torch.max(inner_product, 1)[0][:, None]
                # loss = - torch.mean(torch.diag(logits)) * self.auxi_coef # postives, so select diag 
                # ======
                logits = torch.diag(torch.diag(inner_product)) # delete negative samples
                labels = torch.arange(logits.shape[0]).long().to(self.device)
                loss = self.cross_entropy_loss(logits, labels) * self.auxi_coef 
            elif self.similarity_metric == 'bilinear_inner_product':
                #import ipdb; ipdb.set_trace()
                W_ypred = torch.matmul(self.Auxi.W, y_pred.T)  # (z_dim,B)
                bilinear_inner_product = torch.matmul(y_target, W_ypred)  # (B,B)
                logits = torch.diag(torch.diag(bilinear_inner_product)) # delete negative samples
                labels = torch.arange(logits.shape[0]).long().to(self.device)
                loss = self.cross_entropy_loss(logits, labels) * self.auxi_coef 

            elif self.similarity_metric == 'cos_sim':
                inner_product = torch.matmul(F.normalize(y_pred), F.normalize(y_target).T)  # (B,B)
                logits = torch.diag(torch.diag(inner_product)) # delete negative samples
                labels = torch.arange(logits.shape[0]).long().to(self.device)
                loss = self.cross_entropy_loss(logits, labels) * self.auxi_coef 
            else:
                raise NotImplementedError
                
        else: # with negative samples
            if self.similarity_metric == 'mse':
                loss_positive = F.mse_loss(y_pred, y_target) 
                idx = torch.randperm(y_target.shape[0]) 
                loss_negative = - F.mse_loss(y_pred, y_target[idx]) 
                loss = (loss_positive + loss_negative) * self.auxi_coef 

            if self.similarity_metric == 'nmse':
                loss_positive = F.mse_loss(F.normalize(y_pred), F.normalize(y_target))
                idx = torch.randperm(y_target.shape[0]) 
                loss_negative = - F.mse_loss(F.normalize(y_pred), F.normalize(y_target[idx]))
                loss = (loss_positive + loss_negative) * self.auxi_coef 
                
            elif self.similarity_metric == 'inner_product':
                inner_product = torch.matmul(y_pred, y_target.T)  # (B,B)
                logits = inner_product - torch.max(inner_product, 1)[0][:, None]
                labels = torch.arange(logits.shape[0]).long().to(self.device)
                loss = self.cross_entropy_loss(logits, labels) * self.auxi_coef
            elif self.similarity_metric == 'bilinear_inner_product':
                W_ypred = torch.matmul(self.Auxi.W, y_pred.T)  # (z_dim,B)
                bilinear_inner_product = torch.matmul(y_target, W_ypred)  # (B,B)
                logits = bilinear_inner_product - torch.max(bilinear_inner_product, 1)[0][:, None]
                labels = torch.arange(logits.shape[0]).long().to(self.device)
                loss = self.cross_entropy_loss(logits, labels) * self.auxi_coef 
            elif self.similarity_metric == 'cos_sim':
                inner_product = torch.matmul(F.normalize(y_pred), F.normalize(y_target).T)  # (B,B)
                logits = inner_product - torch.max(inner_product, 1)[0][:, None]
                labels = torch.arange(logits.shape[0]).long().to(self.device)
                loss = self.cross_entropy_loss(logits, labels) * self.auxi_coef
            else:
                raise NotImplementedError
        
        self.encoder_optimizer.zero_grad()
        self.auxi_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.auxi_optimizer.step()
        if step % self.log_interval == 0:
            L.log('train/auxi_loss', loss, step)


    def update(self, replay_buffer, L, step):
        if self.encoder_type == 'pixel':

            obs, action, reward, next_obs, not_done, auxi_kwargs = replay_buffer.sample_auxi(auxi_pred_horizon=self.auxi_pred_horizon, pred_input=self.pred_input, pred_output=self.pred_output, augmentation_type=self.augmentation_type,
                                                                                                raw_state = False)
        elif self.encoder_type in ['mlp', 'ofe']:
            obs, action, reward, next_obs, not_done, auxi_kwargs = replay_buffer.sample_auxi(auxi_pred_horizon=self.auxi_pred_horizon, pred_input=self.pred_input, pred_output=self.pred_output, augmentation_type=self.augmentation_type, 
                                                                                                raw_state = True)
        else:
            raise NotImplementedError
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
            if self.Auxi.action_embedding:
                #print("soft updating action target")
                utils.soft_update_params(
                    self.Auxi.action_encoder, self.Auxi.action_encoder_target,
                    self.encoder_tau
                )
            if self.Auxi.reward_embedding:
                #print("soft updating reward target")
                utils.soft_update_params(
                    self.Auxi.reward_encoder, self.Auxi.reward_encoder_target,
                    self.encoder_tau
                )
        
        if step % self.auxi_update_freq == 0:
            if self.encoder_type in ['pixel', 'mlp', 'ofe']:
                pred_input, pred_output = auxi_kwargs['pred_input'], auxi_kwargs['pred_output']
                pred_input = pred_input
                self.update_auxi(pred_input, pred_output, auxi_kwargs, L, step)
            else:
                raise NotImplementedError

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def save_auxi(self, model_dir, step):
        torch.save(
            self.Auxi.state_dict(), '%s/auxi_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
 