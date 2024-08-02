import torch
from markov_bridges.configs.config_classes.generative_models.cfm_config import CFMConfig
from markov_bridges.configs.config_classes.networks.continuous_network_config import DeepMLPConfig

from markov_bridges.models.networks.temporal.cfm.mlp import DeepMLP


def load_continuous_network(config: CFMConfig, device):
    if isinstance(config.continuous_network, DeepMLPConfig):
        continuous_network = DeepMLP(config,device)
    else:
        raise Exception("Temporal Network not Defined")
    return continuous_network