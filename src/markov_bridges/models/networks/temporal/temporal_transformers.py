import math
import torch
from torch import nn
from dataclasses import dataclass
import os
import pytest
from pprint import pprint
from torchtyping import TensorType
from torch.nn.parallel import DistributedDataParallel as DDP

from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from .temporal_embeddings import transformer_timestep_embedding

#===================================================
# CAMBELL
#===================================================

class PositionalEncoding(nn.Module):

    def __init__(self, device, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(1, max_len, d_model, device=device)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x: TensorType["B", "L", "K"]
                ) -> TensorType["B", "L", "K"]:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        self.pe = self.pe.to(x.device)
        x = x + self.pe[:, 0:x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, temb_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads,
                                               dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.film_from_temb = nn.Linear(temb_dim, 2 * d_model)

    def forward(self,
                x: TensorType["B", "L", "K"],
                temb: TensorType["B", "temb_dim"]
                ):
        B, L, K = x.shape

        film_params = self.film_from_temb(temb)

        x = self.norm1(x + self._sa_block(x))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]
        x = self.norm2(x + self._ff_block(x))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]

        return x

    def _sa_block(self, x):
        x = self.self_attn(x, x, x)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FFResidual(nn.Module):
    def __init__(self, d_model, hidden, temb_dim):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.activation = nn.ReLU()

        self.film_from_temb = nn.Linear(temb_dim, 2 * d_model)

    def forward(self, x, temb):
        B, L, K = x.shape

        film_params = self.film_from_temb(temb)

        x = self.norm(x + self.linear2(self.activation(self.linear1(x))))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]
        return x


class TransformerEncoder(nn.Module):
    
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward,
                 dropout, num_output_FFresiduals, time_scale_factor, S, max_len,
                 temb_dim, use_one_hot_input, device):
        super().__init__()

        self.temb_dim = temb_dim
        self.use_one_hot_input = use_one_hot_input

        self.S = S

        self.pos_embed = PositionalEncoding(device, d_model, dropout, max_len)

        self.encoder_layers = []
        for i in range(num_layers):
            self.encoder_layers.append(
                TransformerEncoderLayer(d_model, num_heads, dim_feedforward,
                                        dropout, 4 * temb_dim)
            )
        self.encoder_layers = nn.ModuleList(self.encoder_layers)

        self.output_resid_layers = []
        for i in range(num_output_FFresiduals):
            self.output_resid_layers.append(
                FFResidual(d_model, dim_feedforward, 4 * temb_dim)
            )
        self.output_resid_layers = nn.ModuleList(self.output_resid_layers)

        self.output_linear = nn.Linear(d_model, self.S)

        if use_one_hot_input:
            self.input_embedding = nn.Linear(S, d_model)
        else:
            self.input_embedding = nn.Linear(1, d_model)

        self.temb_net = nn.Sequential(
            nn.Linear(temb_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 4 * temb_dim)
        )

        self.time_scale_factor = time_scale_factor

    def forward(self, x: TensorType["B", "L"],
                times: TensorType["B"]):
        B, L = x.shape

        temb = self.temb_net(
            transformer_timestep_embedding(
                times * self.time_scale_factor, self.temb_dim
            )
        )
        one_hot_x = nn.functional.one_hot(x, num_classes=self.S)  # (B, L, S)

        if self.use_one_hot_input:
            x = self.input_embedding(one_hot_x.float())  # (B, L, K)
        else:
            x = self.normalize_input(x)
            x = x.view(B, L, 1)
            x = self.input_embedding(x)  # (B, L, K)

        x = self.pos_embed(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, temb)

        # x (B, L, K)
        for resid_layer in self.output_resid_layers:
            x = resid_layer(x, temb)

        x = self.output_linear(x)  # (B, L, S)

        x = x + one_hot_x

        return x

    def normalize_input(self, x):
        x = x / self.S  # (0, 1)
        x = x * 2 - 1  # (-1, 1)
        return x

class SequenceTransformer(nn.Module):
    def __init__(self, cfg:CJBConfig, device, rank=None):
        super().__init__()

        num_layers = cfg.temporal_network.num_layers
        d_model = cfg.temporal_network.d_model
        num_heads = cfg.temporal_network.num_heads
        dim_feedforward = cfg.temporal_network.dim_feedforward
        dropout = cfg.temporal_network.dropout
        num_output_FFresiduals = cfg.temporal_network.num_output_FFresiduals
        time_scale_factor = cfg.temporal_network.time_scale_factor
        temb_dim = cfg.temporal_network.temb_dim
        use_one_hot_input = cfg.temporal_network.use_one_hot_input

        self.S = cfg.data.vocab_size
        max_len = cfg.data.discrete_dimensions

        tmp_net = TransformerEncoder(
            num_layers, d_model, num_heads, dim_feedforward, dropout,
            num_output_FFresiduals, time_scale_factor, self.S, max_len,
            temb_dim, use_one_hot_input, device
        ).to(device)

        #if cfg.distributed:
        #    self.net = DDP(tmp_net, device_ids=[rank])
        #else:
        self.net = tmp_net
        self.expected_output_shape = [max_len, self.S]
        self.to(device)

    def forward(self,
        x: TensorType["B", "D"],
        times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
            Returns logits over state space
        """
        B, D = x.shape
        S = self.S

        logits = self.net(x.long(), times.long()) # (B, D, S)

        return logits