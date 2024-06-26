import math
import torch
import numpy as np
from torch import nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

import torch.nn.functional as F
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig

from markov_bridges.utils.activations import get_activation_function
from markov_bridges.models.networks.temporal.temporal_embeddings import transformer_timestep_embedding

import torch
import numpy as np
from torch import nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import math

class MixedDeepMLP(nn.Module):

    def __init__(self, config:CMBConfig, device):
        super().__init__()
        self.config = config
        self.continuous_dim = config.data.continuos_dimensions
        self.discrete_dim = config.data.discrete_dimensions
        self.vocab_size = config.data.vocab_size

        self.define_deep_models(config)
        self.init_weights()
        self.to(device)

    def define_deep_models(self, config:CMBConfig):
        self.time_embed_dim = config.mixed_network.time_embed_dim
        self.hidden_layer = config.mixed_network.hidden_dim
        self.num_layers = config.mixed_network.num_layers
        self.discrete_embed_dim = config.mixed_network.discrete_embed_dim
        self.act_fn = get_activation_function(config.mixed_network.activation)
        self.dropout_rate = config.mixed_network.dropout

        self.embedding = nn.Embedding(self.vocab_size, self.discrete_embed_dim)
        layers = [nn.Linear(self.continuous_dim + self.discrete_embed_dim*self.discrete_dim + self.time_embed_dim, self.hidden_layer),
                  nn.BatchNorm1d(self.hidden_layer),
                  self.act_fn]

        if self.dropout_rate:
            layers.append(nn.Dropout(self.dropout_rate))

        for _ in range(self.num_layers - 1):
            layers.extend([nn.Linear(self.hidden_layer, self.hidden_layer),
                           nn.BatchNorm1d(self.hidden_layer),
                           self.act_fn])
            if self.dropout_rate:
                layers.extend([nn.Dropout(self.dropout_rate)])

        self.encoding_model = nn.Sequential(*layers)
        self.discrete_head = nn.Linear(self.hidden_layer, self.discrete_dim * self.vocab_size)
        self.continuous_head = nn.Linear(self.hidden_layer, self.continuous_dim)

    def forward(self,x_discrete,x_continuous, times):
        x_discrete = x_discrete.long()
        if len(times.shape) > 1:
            times = times.flatten()

        batch_size = x_continuous.shape[0]
        x_discrete_embedded = self.embedding(x_discrete)
        x_discrete_embedded = x_discrete_embedded.view(batch_size, -1)
        x_combined = torch.cat([x_continuous, x_discrete_embedded], dim=1)
        time_embeddings = transformer_timestep_embedding(times, embedding_dim=self.time_embed_dim)
        x_full = torch.cat([x_combined, time_embeddings], dim=1)
        rate_logits = self.encoding_model(x_full)

        continuous_head = self.continuous_head(rate_logits)
        discrete_head = self.discrete_head(rate_logits).reshape(batch_size,self.discrete_dim,self.vocab_size)
        return discrete_head,continuous_head

    def init_weights(self):
        for layer in self.encoding_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)