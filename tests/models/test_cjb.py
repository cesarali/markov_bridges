import pytest
import torch
from markov_bridges.models.generative_models.cjb import CJB
from markov_bridges.models.trainers.cjb_trainer import CJBTrainer

from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.networks.temporal_networks_config import SequenceTransformerConfig
from markov_bridges.configs.experiments_configs.music_experiments import conditional_music_experiment

def test_cjb():
    model_config = conditional_music_experiment(number_of_epochs=3,sinusoidal=True)
    model_config.temporal_network = SequenceTransformerConfig(num_heads=1,num_layers=1)
    cjb = CJB(config=model_config)

    databatch = cjb.dataloader.get_databatch()
    join_context = lambda context_discrete,data_discrete : torch.cat([context_discrete,data_discrete],dim=1)
    x = join_context(databatch.context_discrete,databatch.source_discrete)
    rates = cjb.forward_rate(x,databatch.time.flatten()) # (N, D, S)

    print(rates.shape)
    
if __name__=="__main__":
    test_cjb()