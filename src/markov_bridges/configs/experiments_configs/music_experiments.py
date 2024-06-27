from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.networks.temporal_networks_config import SequenceTransformerConfig

import os
import pytest

# configs

from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig

from markov_bridges.configs.config_classes.metrics.metrics_configs import (
    MetricsAvaliable,
    HellingerMetricConfig,
    MusicPlotConfig
)

# models
from markov_bridges.configs.config_classes.trainers.trainer_config import CJBTrainerConfig
from markov_bridges.models.generative_models.cjb import CJB
from markov_bridges.models.trainers.cjb_trainer import CJBTrainer
from markov_bridges.utils.experiment_files import ExperimentFiles


def conditional_music_experiment(number_of_epochs=3)->CJBConfig:
    experiment_config = CJBConfig()
    # data
    experiment_config.data = LakhPianoRollConfig(has_context_discrete=True,
                                                 context_discrete_dimension=64) # CHANGE
    # temporal network
    experiment_config.temporal_network = SequenceTransformerConfig(num_heads=1,num_layers=1) #CHANGE
    # pipeline 
    experiment_config.pipeline.number_of_steps = 5  #CHANGE

    # trainer
    experiment_config.trainer = CJBTrainerConfig(
        number_of_epochs=number_of_epochs,
        warm_up=2,
        learning_rate=1e-4,
        scheduler="exponential",  # or "reduce", "exponential", "multi", None
        step_size=500,  # for StepLR
        gamma=0.1,  # for StepLR, MultiStepLR, ExponentialLR
        milestones=[6000, 7000, 10000],  # where to change the learning rate for MultiStepLR
        factor=0.1,  # how much to reduce lr for ReduceLROnPlateau
        patience=10  # how much to wait to judge for plateu in ReduceLROnPlateau
    )

    # metrics
    experiment_config.trainer.metrics = [HellingerMetricConfig(),
                                         MusicPlotConfig()]
    
    return experiment_config


def continue_music_experiment(experiment_dir):
    experiment_files = ExperimentFiles(experiment_name="cjb",
                                       experiment_type="music")    
    trainer = CJBTrainer(experiment_files=experiment_files, # experiment files of new experiment
                         experiment_dir=experiment_dir, # experiment dir of experiment to start with
                         starting_type="last") # WHERE TO START last IS LAST RECORDED MODEL
    trainer.train()


if __name__=="__main__":
    start = True # START NEW EXPERIMENT
    if start:
        experiment_config = conditional_music_experiment(number_of_epochs=5)
        experiment_config.trainer.debug = True # CHANGE
        experiment_config.trainer.device = "cuda:0"
        experiment_files = ExperimentFiles(experiment_name="cjb",
                                           experiment_type="music")    
        trainer = CJBTrainer(config=experiment_config,
                             experiment_files=experiment_files)
        trainer.train()

    else: # CONTINUE EXPERIMENT FROM 
        # experiment dir is the experiment from were to start again
        experiment_dir = r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\results\cjb\music\1718621212"
        continue_music_experiment(experiment_dir)