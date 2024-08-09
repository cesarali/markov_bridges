import torch
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.configs.config_classes.networks.mixed_networks_config import(
    MixedDeepMLPConfig,
)
from markov_bridges.models.networks.temporal.mixed.mixed_mlp import (
    MixedDeepMLP
)
from markov_bridges.models.networks.temporal.temporal_transformers import SequenceTransformer


def load_mixed_network(config:CMBConfig, device=None):
    if isinstance(config.mixed_network,MixedDeepMLPConfig):
        mixed_network = MixedDeepMLP(config,device)
    else:
        raise Exception("Temporal Network not Defined")
    return mixed_network