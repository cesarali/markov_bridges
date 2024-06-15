from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.networks.temporal_networks_config import SequenceTransformerConfig

def conditional_music_experiment()->CJBConfig:
    experiment_config = CJBConfig()
    experiment_config.data = LakhPianoRollConfig(has_context_discrete=True)
    experiment_config.temporal_network = SequenceTransformerConfig()
    return experiment_config



