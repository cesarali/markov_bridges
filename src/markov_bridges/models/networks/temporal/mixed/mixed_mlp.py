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
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

import torch
import numpy as np
from torch import nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import math

class MixedDeepMLP(nn.Module):

    def __init__(self, config:CMBConfig, device=None):
        super().__init__()

        self.config = config
        self.vocab_size = config.data.vocab_size
        self.set_dimensions(config)
        self.define_deep_models(config)
        self.init_weights()
        
        if device is None:
            self.to(device)

    def set_dimensions(self,config:CMBConfig):
        self.has_target_continuous = config.data.has_target_continuous
        self.has_target_discrete = config.data.has_target_discrete
        self.has_context_continuous =  config.data.has_context_continuous
        self.has_context_discrete =  config.data.has_context_discrete

        if self.has_target_discrete:
            self.discrete_dim = config.data.discrete_dimensions
        else:
            self.discrete_dim = 0

        if self.has_target_continuous:
            self.continuous_dim = config.data.continuos_dimensions
        else:
            self.continuous_dim = 0
        
        if self.has_context_continuous:
            self.context_continuous_dimension = self.config.data.context_continuous_dimension
        else:
            self.context_continuous_dimension = 0 

        if self.has_context_discrete:
            self.context_discrete_dimension = self.config.data.context_discrete_dimension
        else:
            self.context_discrete_dimension = 0 
        
        self.time_embed_dim = config.mixed_network.time_embed_dim
        self.discrete_embed_dim = config.mixed_network.discrete_embed_dim
        self.continuous_embed_dim = config.mixed_network.continuous_embed_dim

        self.total_embeddings = 0
        self.total_embeddings += self.time_embed_dim
        self.total_embeddings += int(self.has_context_continuous)*self.continuous_embed_dim
        self.total_embeddings += int(self.has_target_continuous)*self.continuous_embed_dim
        self.total_embeddings += self.discrete_dim*self.discrete_embed_dim
        self.total_embeddings += self.context_discrete_dimension*self.discrete_embed_dim

    def define_deep_models(self, config:CMBConfig):
        self.hidden_layer = config.mixed_network.hidden_dim
        self.num_layers = config.mixed_network.num_layers

        self.act_fn = get_activation_function(config.mixed_network.activation)
        self.dropout_rate = config.mixed_network.dropout

        # embeddings of data
        self.embedding_continuos = nn.Linear(self.continuous_dim, self.continuous_embed_dim)
        if self.has_target_continuous:
            self.embedding_continuos = nn.Linear(self.continuous_dim, self.continuous_embed_dim)

        if self.has_context_continuous:
            self.embedding_context_continuos = nn.Linear(self.context_continuous_dimension, self.continuous_embed_dim)

        if self.has_target_discrete or self.has_context_discrete:        
            self.embedding = nn.Embedding(self.vocab_size, self.discrete_embed_dim)

        #merging of embeddings
        layers = [nn.Linear(self.total_embeddings, self.hidden_layer),
                  nn.BatchNorm1d(self.hidden_layer),
                  self.act_fn]
        if self.dropout_rate:
            layers.append(nn.Dropout(self.dropout_rate))

        # networks layers
        for _ in range(self.num_layers - 1):
            layers.extend([nn.Linear(self.hidden_layer, self.hidden_layer),
                           nn.BatchNorm1d(self.hidden_layer),
                           self.act_fn])
            if self.dropout_rate:
                layers.extend([nn.Dropout(self.dropout_rate)])
        self.encoding_model = nn.Sequential(*layers)

        # heads 
        self.discrete_head = nn.Linear(self.hidden_layer, self.discrete_dim * self.vocab_size)
        self.continuous_head = nn.Linear(self.hidden_layer, self.continuous_dim)

    def embbed_data(self,x_discrete,x_continuous,times,databatch:MarkovBridgeDataNameTuple):
        collected_embeddings = []

        # MAKE SURE THE TIME IS 1D
        if len(times.shape) > 1:
            times = times.flatten()
        time_embeddings = transformer_timestep_embedding(times, embedding_dim=self.time_embed_dim)
        collected_embeddings.append(time_embeddings)

        if self.has_target_discrete:
            batch_size = x_discrete.shape[0]
            x_discrete = x_discrete.long()
            x_discrete_embedded = self.embedding(x_discrete)
            x_discrete_embedded = x_discrete_embedded.view(batch_size, -1)
            collected_embeddings.append(x_discrete_embedded)
        
        if self.has_context_discrete:
            batch_size = databatch.context_discrete.shape[0]
            context_discrete = databatch.context_discrete.long()
            context_discrete_embedded = self.embedding(context_discrete)
            context_discrete_embedded = context_discrete_embedded.view(batch_size, -1)
            collected_embeddings.append(context_discrete_embedded)

        if self.has_target_continuous:
            batch_size = x_continuous.shape[0]
            # Normalize or process x_continuous if necessary before embedding
            x_continuous_embedded = self.embedding_continuos(x_continuous)  # Define this method as needed
            x_continuous_embedded = x_continuous_embedded.view(batch_size, -1)
            collected_embeddings.append(x_continuous_embedded)
        
        if self.has_context_continuous:
            batch_size = databatch.context_continuous.shape[0]
            # Normalize or process context_continuous if necessary before embedding
            context_continuous_embedded = self.embedding_continuos(databatch.context_continuous)  # Define this method as needed
            context_continuous_embedded = context_continuous_embedded.view(batch_size, -1)
            collected_embeddings.append(context_continuous_embedded)

        collected_embeddings = torch.cat(collected_embeddings, dim=1)

        return batch_size,collected_embeddings

    def forward(self,x_discrete,x_continuous,times,databatch:MarkovBridgeDataNameTuple):
        batch_size,collected_embeddings = self.embbed_data(x_discrete,x_continuous,times,databatch)
        logits = self.encoding_model(collected_embeddings)

        # Obtain Head
        if self.has_target_continuous:
            continuous_head = self.continuous_head(logits)
        else:
            continuous_head = None
        
        if self.has_target_discrete:
            discrete_head = self.discrete_head(logits).reshape(batch_size,self.discrete_dim,self.vocab_size)
        else:
            discrete_head = None

        return discrete_head,continuous_head

    def init_weights(self):
        for layer in self.encoding_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)