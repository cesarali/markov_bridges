from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig

from markov_bridges.configs.config_classes.networks.temporal_networks_config import(
    TemporalMLPConfig,
    TemporalLeNet5Config,
    TemporalLeNet5AutoencoderConfig,
    TemporalUNetConfig,
    TemporalDeepMLPConfig,
)

from markov_bridges.models.networks.temporal.temporal_mlp import (
    TemporalDeepMLP,
    TemporalLeNet5,
    TemporalLeNet5Autoencoder,
    TemporalUNet,
    TemporalMLP
)

def load_temporal_network(config:CJBConfig, device):
    if isinstance(config.temporal_network,TemporalMLPConfig):
        temporal_network = TemporalMLP(config,device)
    elif isinstance(config.temporal_network,TemporalDeepMLPConfig):
        temporal_network = TemporalDeepMLP(config,device)
    elif isinstance(config.temporal_network,TemporalLeNet5Config):
        temporal_network = TemporalLeNet5(config,device)
    elif isinstance(config.temporal_network,TemporalLeNet5AutoencoderConfig):
        temporal_network = TemporalLeNet5Autoencoder(config,device)
    elif isinstance(config.temporal_network,TemporalUNetConfig):
        temporal_network = TemporalUNet(config,device)
    elif isinstance(config.temporal_network, TemporalLeNet5Config):
        temporal_network = TemporalLeNet5(config, device)
    else:
        raise Exception("Temporal Network not Defined")
    return temporal_network