import torch
from markov_bridges.configs.config_classes.data.molecules_configs import QM9Config

from markov_bridges.configs.config_classes.generative_models.edmg_config import (
    EDMGConfig,
    NoisingModelConfig
)
from markov_bridges.models.deprecated.generative_models.edmg import EDMG
from markov_bridges.models.networks.temporal.edmg.edmg_utils import get_edmg_model
from markov_bridges.data.dataloaders_utils import get_dataloaders


def test_equivariant_noising():
    return None

if __name__=="__main__":
    config = EDMGConfig()
    config.data = QM9Config(num_pts_train=1000,
                            num_pts_test=200,
                            num_pts_valid=200)    
    config.noising_model = NoisingModelConfig(n_layers=2)
    device = torch.device("cpu")
    edmg = EDMG(config)

