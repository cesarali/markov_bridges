import math
import torch
import numpy as np
from torch import nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

import torch.nn.functional as F
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig

from markov_bridges.utils.activations import get_activation_function
from markov_bridges.models.networks.temporal.temporal_embeddings import transformer_timestep_embedding

import torch
import numpy as np
from torch import nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import math


class TemporalDeepMLP(nn.Module):

    def __init__(self,
                 config:CJBConfig,
                 device):

        super().__init__()
        self.config = config
        self.dimensions = config.data.discrete_dimensions
        self.vocab_size = config.data.vocab_size
        self.define_deep_models(config)
        self.init_weights()
        self.to(device)
        self.expected_output_shape = [self.dimensions, self.vocab_size]

    def define_deep_models(self, config):
        self.time_embed_dim = config.temporal_network.time_embed_dim
        self.hidden_layer = config.temporal_network.hidden_dim
        self.num_layers = config.temporal_network.num_layers
        self.act_fn = get_activation_function(config.temporal_network.activation)
        self.dropout_rate = config.temporal_network.dropout  # Assuming dropout rate is specified in the config

        layers = [nn.Linear(self.dimensions + self.time_embed_dim, self.hidden_layer),
                  nn.BatchNorm1d(self.hidden_layer),
                  self.act_fn]

        if self.dropout_rate: layers.append(nn.Dropout(self.dropout_rate))  # Adding dropout if specified

        for _ in range(self.num_layers - 2):
            layers.extend([nn.Linear(self.hidden_layer, self.hidden_layer),
                           nn.BatchNorm1d(self.hidden_layer),
                           self.act_fn])
            if self.dropout_rate: layers.extend([nn.Dropout(self.dropout_rate)])  # Adding dropout

        layers.append(nn.Linear(self.hidden_layer, self.dimensions * self.vocab_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x, times, databatch):
        times = times.flatten()
        batch_size = x.shape[0]
        time_embeddings = transformer_timestep_embedding(times, embedding_dim=self.time_embed_dim)
        x = torch.concat([x, time_embeddings], dim=1)
        rate_logits = self.model(x)
        rate_logits = rate_logits.reshape(batch_size, self.dimensions, self.vocab_size)

        return rate_logits

    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

class TemporalMLP(nn.Module):
    """
    """
    def __init__(self, config:CJBConfig, device):
        super().__init__()
        config_data = config.data
        self.dimensions = config_data.discrete_dimensions
        self.vocab_size = config_data.vocab_size
        self.define_deep_models(config)
        self.init_weights()
        self.to(device)
        self.expected_output_shape = [self.dimensions,self.vocab_size]

    def define_deep_models(self,config):
        self.time_embed_dim = config.temporal_network.time_embed_dim
        self.hidden_layer = config.temporal_network.hidden_dim
        self.f1 = nn.Linear(self.dimensions, self.hidden_layer)
        self.f2 = nn.Linear(self.hidden_layer + self.time_embed_dim, self.dimensions * self.vocab_size)

    def forward(self, x, times,databatch):
        times = times.flatten()
        batch_size = x.shape[0]
        time_embbedings = transformer_timestep_embedding(times,
                                                         embedding_dim=self.time_embed_dim)

        step_one = self.f1(x)
        step_two = torch.concat([step_one, time_embbedings], dim=1)
        rate_logits = self.f2(step_two)
        rate_logits = rate_logits.reshape(batch_size, self.dimensions, self.vocab_size)

        return rate_logits

    def init_weights(self):
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)

class TemporalUNet(nn.Module):

    def __init__(self, config:CJBConfig, device):
        super().__init__()

        self.dimensions = config.data.discrete_dimensions
        self.vocab_size = config.data.vocab_size
        self.dim_time_emb = config.temporal_network.time_embed_dim
        self.dim_hidden = config.temporal_network.hidden_dim
        self.dropout = config.temporal_network.dropout
        self.act = get_activation_function(config.temporal_network.activation) #nn.GELU()        
        self.Embeddings()
        self.Encoder()
        self.Decoder()
        self.to(device)
        self.expected_output_shape = [28, 28, self.vocab_size]


    def Embeddings(self):
        self.projection = nn.Conv2d(1, self.dim_hidden, kernel_size=3, stride=1, padding=1)
        self.time_embedding = Time_embedding(self.dim_time_emb, self.dim_hidden, self.act)

    def Encoder(self):
        self.down1 = Down(in_channels=self.dim_hidden, 
                          out_channels=self.dim_hidden, 
                          time_channels=self.dim_hidden, 
                          activation=self.act, 
                          use_attention_block=True,
                          dropout=self.dropout)
        
        self.down2 = Down(in_channels=self.dim_hidden, 
                          out_channels=2*self.dim_hidden, 
                          time_channels=self.dim_hidden,  
                          activation=self.act, 
                          use_attention_block=False,
                          dropout=self.dropout)
        self.pool = nn.Sequential(nn.AvgPool2d(7), self.act)

    def Decoder(self):
        self.up0 = nn.Sequential(nn.ConvTranspose2d(2 * self.dim_hidden, 2 * self.dim_hidden, 7, 7), 
                                 normalization(2 * self.dim_hidden, normalization='group', group_pref=8),
                                 self.act)

        self.up1 = Up(in_channels=4 * self.dim_hidden, 
                      out_channels=self.dim_hidden, 
                      time_channels=self.dim_hidden,  
                      activation=self.act,
                      use_attention_block=True, 
                      dropout=self.dropout)
        
        self.up2 = Up(in_channels=2 * self.dim_hidden, 
                      out_channels=self.dim_hidden, 
                      time_channels=self.dim_hidden,  
                      activation=self.act, 
                      use_attention_block=False,
                      dropout=self.dropout)
        
        self.output = nn.Sequential(nn.Conv2d(2 * self.dim_hidden, self.dim_hidden, kernel_size=3, stride=1, padding=1),
                                    normalization(self.dim_hidden, normalization='group', group_pref=8),
                                    self.act,
                                    nn.Conv2d(self.dim_hidden,  self.vocab_size, kernel_size=3, stride=1, padding=1)
                                    )

    def forward(self,  x, times):

        #...embed inputs:
        temb = self.time_embedding(times)
        x = self.projection(x)

        #...encode:
        down1 = self.down1(x, temb)
        down2 = self.down2(down1, temb)
        h = self.pool(down2)
        
        #...decode:
        up1 = self.up0(h)
        up2 = self.up1(up1, temb, down2) 
        up3 = self.up2(up2, temb, down1)
        output = self.output(torch.cat((up3, x), 1))

        return output.permute(0, 2, 3, 1) 

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, activation, use_attention_block, dropout):
        super(Down, self).__init__()
        self.conv_block = TemporalResidualConvBlock(in_channels=in_channels, 
                                                    out_channels=out_channels, 
                                                    time_channels=time_channels, 
                                                    activation=activation, 
                                                    use_attention_block=use_attention_block,
                                                    dropout=dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        x = self.conv_block(x, t)
        x = self.pool(x) 
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, activation, use_attention_block, dropout):
        super(Up, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv_block1 = TemporalResidualConvBlock(in_channels=out_channels, 
                                                     out_channels=out_channels, 
                                                     time_channels=time_channels, 
                                                     activation=activation, 
                                                     use_conv_norm='batch',
                                                     use_attention_block=use_attention_block,
                                                     dropout=dropout)
        self.conv_block2 = TemporalResidualConvBlock(in_channels=out_channels, 
                                                     out_channels=out_channels, 
                                                     time_channels=time_channels, 
                                                     activation=activation, 
                                                     dropout=dropout)

    def forward(self, x, t, skip):
        x = torch.cat((x, skip), dim=1)
        x = self.upsample(x)
        x = self.conv_block1(x, t)
        x = self.conv_block2(x, t)
        return x

class TemporalResidualConvBlock(nn.Module):
    def __init__( self, 
                 in_channels: int, 
                 out_channels: int, 
                 time_channels: int,
                 activation: nn.Module=nn.GELU(),
                 use_conv_norm: str='batch',
                 use_attention_block = False,
                 dropout: float=0.1):
        super().__init__()

        self.same_channels = in_channels==out_channels

        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                    normalization(out_channels, normalization=use_conv_norm),
                                    activation,
                                    )
        
        self.conv_2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                    normalization(out_channels, normalization=use_conv_norm),
                                    activation,
                                    )

        self.time_emb_1 = nn.Sequential(nn.Linear(time_channels, out_channels),
                                       nn.BatchNorm1d(out_channels), 
                                       activation, 
                                       nn.Dropout(dropout)) 

        self.time_emb_2 = nn.Sequential(nn.Linear(time_channels, out_channels),
                                       nn.BatchNorm1d(out_channels), 
                                       activation, 
                                       nn.Dropout(dropout)) 

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) if not self.same_channels else nn.Identity()
        self.attention = AttentionBlock(out_channels) if use_attention_block else nn.Identity()

        self.initialize()

    def forward(self, x, t):
        h = self.conv_1(x)
        h += self.time_emb_1(t).view(-1, h.shape[1], 1, 1)
        h = self.conv_2(h)
        h += self.time_emb_2(t).view(-1, h.shape[1], 1, 1)
        h += self.skip(x)
        h = self.attention(h)
        return h / (np.sqrt(2.0) if self.same_channels else 1.0) 

    def initialize(self):
        for module in self.modules():
            if isinstance(module,  (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

class AttentionBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = normalization(in_ch, normalization='group', group_pref=32)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

class Time_embedding(nn.Module):
    def __init__(self, dim_time_emb, dim_hidden, activation_fn=nn.GELU()):
        super(Time_embedding, self).__init__()

        self.dim_time_emb = dim_time_emb

        layers = [ nn.Linear(dim_time_emb, dim_hidden),
                   activation_fn,
                   nn.Linear(dim_hidden, dim_hidden),
                  ]
        self.fc = nn.Sequential(*layers)
        self.initialize()

    def forward(self, t):
        temb = transformer_timestep_embedding(t.squeeze(), self.dim_time_emb, max_positions=10000)
        return self.fc(temb)
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

def get_num_groups(channels, group_pref=8):
    while channels % group_pref != 0:
        group_pref -= 1
    return group_pref

def normalization(x, normalization='batch', group_pref=8):
    if normalization == 'batch':
        return nn.BatchNorm2d(x)
    elif normalization == 'group':
        return nn.GroupNorm(get_num_groups(x), x, group_pref)
    else:
        return nn.Identity()