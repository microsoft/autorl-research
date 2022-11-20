# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import torch.nn as nn
from torch.nn.functional import linear


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32,encoder_hidden_size=256 ,output_logits=False,*args):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM_64[num_layers] if obs_shape[-1] == 64 else OUT_DIM[num_layers] 
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]
        

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class MlpEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, encoder_hidden_size, output_logits=False,*args):
        super().__init__()

        assert len(obs_shape) == 1

        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.linears = nn.ModuleList(
            [nn.Linear(obs_shape[0], encoder_hidden_size)]
        )
        for i in range(num_layers - 1):
            self.linears.append(nn.Linear(encoder_hidden_size, encoder_hidden_size))


        self.fc = nn.Linear(encoder_hidden_size, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def forward_linear(self, obs):

        self.outputs['obs'] = obs

        linear = torch.relu(self.linears[0](obs))
        self.outputs['linear1'] = linear

        for i in range(1, self.num_layers):
            linear = torch.relu(self.linears[i](linear))
            self.outputs['linear%s' % (i + 1)] = linear

        h = linear
        return h

    def forward(self, obs, detach=False):
        h = self.forward_linear(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class OfeEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, encoder_hidden_size, output_logits=False,*args):
        super().__init__()

        assert len(obs_shape) == 1

        self.obs_shape = obs_shape
        self.feature_dim = num_layers * encoder_hidden_size + obs_shape[0]
        self.num_layers = num_layers
        self.linears = nn.ModuleList(
            [nn.Linear(obs_shape[0], encoder_hidden_size)]
        )
        for i in range(num_layers - 1):
            self.linears.append(nn.Linear( ( obs_shape[0] + (i+1) * encoder_hidden_size ), encoder_hidden_size)) # ofenet structure

        

        self.fc = nn.Linear(self.feature_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def forward_linear(self, obs):

        self.outputs['obs'] = obs

        linear = torch.cat( [obs, torch.relu(self.linears[0](obs))], axis =1)
        self.outputs['linear1'] = linear
        #import ipdb; ipdb.set_trace()
        for i in range(1, self.num_layers):
            linear = torch.cat( [linear, torch.relu(self.linears[i](linear))], axis=1)
            self.outputs['linear%s' % (i + 1)] = linear

        h = linear
        return h

    def forward(self, obs, detach=False):
        h = self.forward_linear(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class ActionOrRewardEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, num_layers, *args):
        super().__init__()
        assert type(input_dim) == int
        assert num_layers == 1
        self.feature_dim = feature_dim

        self.forward_linear_layers = nn.Sequential(nn.Linear(input_dim,feature_dim)) 

        self.fc = nn.Linear(self.feature_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()

    def forward(self, obs, detach=False):
        h = self.forward_linear_layers(obs)
        self.outputs['obs'] = obs
        self.outputs['linear'] = h

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = torch.tanh(h_norm)
        return out


    def copy_linear_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder,  'mlp': MlpEncoder, 'ofe': OfeEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, encoder_hidden_size, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, encoder_hidden_size, output_logits
    )
