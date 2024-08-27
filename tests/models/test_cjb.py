import pytest
import torch
from markov_bridges.models.generative_models.cjb_lightning import CJBL

from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.networks.temporal_networks_config import SequenceTransformerConfig
from markov_bridges.configs.experiments_configs.graphs_experiments import get_graph_experiment

def test_cjb():
    model_config = get_graph_experiment(number_of_epochs=3)
    cjb = CJBL(config=model_config)
    databatch = cjb.dataloader.get_databatch()

if __name__=="__main__":
    test_cjb()