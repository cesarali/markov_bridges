from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.networks.temporal_networks_config import SequenceTransformerConfig

import os
import pytest

# configs

from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.metrics.metrics_configs import (
    MetricsAvaliable,
    HellingerMetricConfig
)

# models
from markov_bridges.models.generative_models.cjb import CJB
from markov_bridges.models.trainers.cjb_trainer import CJBTrainer
from markov_bridges.utils.experiment_files import ExperimentFiles


def conditional_music_experiment(number_of_epochs=3)->CJBConfig:
    experiment_config = CJBConfig()
    # data
    experiment_config.data = LakhPianoRollConfig(has_context_discrete=True)
    # temporal network
    experiment_config.temporal_network = SequenceTransformerConfig(num_heads=1,num_layers=1) #CHANGE
    # pipeline 
    experiment_config.pipeline.number_of_steps = 5  #CHANGE
    # trainer
    experiment_config.trainer.number_of_epochs = number_of_epochs
    experiment_config.trainer.warm_up = 5  #CHANGE
    experiment_config.trainer.learning_rate = 1e-4
    # metrics
    experiment_config.trainer.metrics = [HellingerMetricConfig()]
    
    return experiment_config


def continue_music_experiment(experiment_dir):
    experiment_files = ExperimentFiles(experiment_name="cjb",
                                       experiment_type="music")    
    
    trainer = CJBTrainer(experiment_files=experiment_files,
                         experiment_dir=experiment_dir,
                         starting_type="last") # WHERE TO START last IS LAST RECORDED MODEL
    trainer.train()


if __name__=="__main__":
    start = False # START NEW EXPERIMENT
    if start:
        experiment_config = conditional_music_experiment(number_of_epochs=3)
        experiment_config.trainer.debug = True # CHANGE

        experiment_files = ExperimentFiles(experiment_name="cjb",
                                           experiment_type="music")    
        
        trainer = CJBTrainer(config=experiment_config,
                             experiment_files=experiment_files)
        trainer.train()
    else: # CONTINUE EXPERIMENT FROM 
        # experiment dir is the experiment from were to start again
        experiment_dir = r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\results\cjb\music\1718621212"
        continue_music_experiment(experiment_dir)