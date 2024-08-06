import torch
from torch import nn as nn
from markov_bridges.configs.config_classes.generative_models.cfm_config import CFMConfig
from markov_bridges.utils.activations import get_activation_function
from markov_bridges.models.networks.temporal.temporal_embeddings import transformer_timestep_embedding, sinusoidal_timestep_embedding


class DeepMLP(nn.Module):

    def __init__(self, config: CFMConfig, device):
        super().__init__()
        self.config = config
        self.continuous_dim = config.data.continuos_dimensions
        self.context_dim = config.data.context_discrete_dimension + config.data.context_continuous_dimension

        self.define_deep_models(config)
        self.init_weights()
        self.to(device)

    def define_deep_models(self, config:CFMConfig):
        self.time_embed_dim = config.continuous_network.time_embed_dim
        self.discrete_embed_dim = config.continuous_network.discrete_embed_dim if self.config.data.has_context_discrete else 0

        self.hidden_layer = config.continuous_network.hidden_dim
        self.num_layers = config.continuous_network.num_layers
        self.act_fn = get_activation_function(config.continuous_network.activation)
        self.dropout_rate = config.continuous_network.dropout

        if self.context_dim > 0:
            self.discrete_embedding = nn.Embedding(config.data.vocab_size, self.discrete_embed_dim)

        layers = [nn.Linear(self.continuous_dim + self.time_embed_dim + self.discrete_embed_dim, self.hidden_layer),
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
        self.continuous_head = nn.Linear(self.hidden_layer, self.continuous_dim)

    def forward(self, x_continuous, times, context_discrete=None, context_continuous=None):

        if len(times.shape) > 1:
            times = times.flatten()

        if context_discrete is not None:
            context_discrete = self.discrete_embedding(context_discrete.long())

        # t = transformer_timestep_embedding(times, embedding_dim=self.time_embed_dim)
        t = sinusoidal_timestep_embedding(times, self.time_embed_dim, max_period=10000)

        x = torch.cat([x_continuous, t], dim=1) if context_discrete is None else torch.cat([x_continuous, context_discrete.squeeze(), t], dim=1)
        x = self.encoding_model(x)

        return self.continuous_head(x)

    def init_weights(self):
        for layer in self.encoding_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)