from markov_bridges.data.music_dataloaders import LankhPianoRollDataloader
from markov_bridges.data.graphs_dataloader import GraphDataloader
from markov_bridges.data.categorical_samples import IndependentMixDataloader
from markov_bridges.data.qm9_points_dataloader import QM9PointDataloader

from markov_bridges.configs.config_classes.data.graphs_configs import GraphDataloaderGeometricConfig
from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.data.molecules_configs import QM9Config

def get_dataloaders(config:CJBConfig):
    if isinstance(config.data,LakhPianoRollConfig):
        dataloader = LankhPianoRollDataloader(config.data)
    elif isinstance(config.data,GraphDataloaderGeometricConfig):
        dataloader = GraphDataloader(config.data)
    elif isinstance(config.data,IndependentMixConfig):
        dataloader = IndependentMixDataloader(config.data)
    elif isinstance(config.data,QM9Config):
        dataloader = QM9PointDataloader(config)
    else:
        raise Exception("Dataloader not Found!")
    return dataloader
