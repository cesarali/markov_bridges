import pytest
import torch
from markov_bridges.models.deprecated.generative_models.cjb import CJB
from markov_bridges.models.deprecated.trainers.cjb_trainer import CJBTrainer

from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.networks.temporal_networks_config import SequenceTransformerConfig

from markov_bridges.configs.experiments_configs.music_experiments import conditional_music_experiment

def test_cjb_bridges():
    model_config = conditional_music_experiment()
    model_config.data = LakhPianoRollConfig(has_context_discrete=True)
    model_config.temporal_network = SequenceTransformerConfig(num_heads=1,num_layers=1)
    cjb = CJB(config=model_config)

    databatch = cjb.dataloader.get_databatch()
    time = torch.rand((databatch.context_discrete.shape[0],))

    sampled_x = cjb.forward_rate.sample_x(databatch.source_discrete,databatch.target_discrete,time)
    print(sampled_x.shape)

if __name__=="__main__":
    test_cjb_bridges()