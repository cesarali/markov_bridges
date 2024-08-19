from markov_bridges.data.sequences.music_dataloaders import LankhPianoRollDataloader
from markov_bridges.data.graphs_dataloader import GraphDataloader
from markov_bridges.data.gaussians2D_dataloaders import IndependentMixDataloader
from markov_bridges.data.qm9.qm9_points_dataloader import QM9PointDataloader
from markov_bridges.data.gaussians2D_dataloaders import GaussiansDataloader
from markov_bridges.data.sequences.simplex_sinosoidal import SinusoidalDataloader
from markov_bridges.data.lp.LP_Dataloaders import LPDataloader

from markov_bridges.configs.config_classes.data.graphs_configs import GraphDataloaderGeometricConfig
from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.data.sequences_config import SinusoidalConfig
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig, GaussiansConfig

from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.data.molecules_configs import QM9Config, LPConfig


def get_dataloaders(config:CJBConfig):
    if isinstance(config.data,LakhPianoRollConfig):
        dataloader = LankhPianoRollDataloader(config.data)
    elif isinstance(config.data,GraphDataloaderGeometricConfig):
        dataloader = GraphDataloader(config.data)
    elif isinstance(config.data,IndependentMixConfig):
        dataloader = IndependentMixDataloader(config.data)
    elif isinstance(config.data,QM9Config):
        dataloader = QM9PointDataloader(config)
    elif isinstance(config.data,GaussiansConfig):
        dataloader = GaussiansDataloader(config.data)
    elif isinstance(config.data,SinusoidalConfig):
        dataloader = SinusoidalDataloader(config.data)
    elif isinstance(config.data,LPConfig):
        dataloader = LPDataloader(config)
    else:
        raise Exception("Dataloader not Found!")
    return dataloader
