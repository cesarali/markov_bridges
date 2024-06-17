import os
import pytest

# configs
from markov_bridges.configs.config_classes.data.graphs_configs import dataset_str_to_config
from markov_bridges.configs.config_classes.data.graphs_configs import GraphDataloaderGeometricConfig,CommunitySmallGConfig

from markov_bridges.configs.config_classes.networks.temporal_networks_config import TemporalMLPConfig
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig,CJBTrainerConfig
from markov_bridges.configs.config_classes.networks.temporal_networks_config import TemporalMLPConfig
from markov_bridges.configs.config_classes.metrics.metrics_configs import MetricsAvaliable,HellingerMetricConfig

# models
from markov_bridges.models.generative_models.cjb import CJB
from markov_bridges.data.graphs_dataloader import GraphDataloader
from markov_bridges.models.trainers.cjb_trainer import CJBTrainer

from markov_bridges.utils.experiment_files import ExperimentFiles

from dataclasses import asdict

def get_graph_experiment(dataset_name="community_small",number_of_epochs=100):
    experiment_config = CJBConfig()
    #data config
    experiment_config.data = dataset_str_to_config[dataset_name](batch_size=20)
    #temporal network config
    experiment_config.temporal_network = TemporalMLPConfig(hidden_dim=150,time_embed_dim=50)
    #trainer config
    experiment_config.trainer = CJBTrainerConfig(number_of_epochs=number_of_epochs,learning_rate=1e-3)
    # define metrics
    metrics = [HellingerMetricConfig(plot_binary_histogram=True)]
    experiment_config.trainer.metrics = metrics
    return experiment_config


def continue_graph_experiment(experiment_dir):
    experiment_files = ExperimentFiles(experiment_name="cjb",
                                       experiment_type="graph")    
    
    trainer = CJBTrainer(experiment_files=experiment_files,
                         experiment_dir=experiment_dir,
                         starting_type="last")
    
    trainer.train()

if __name__=="__main__":
    start = False
    if start:
        experiment_config = get_graph_experiment(number_of_epochs=200)
        experiment_files = ExperimentFiles(experiment_name="cjb",
                                        experiment_type="graph")    
        
        
        trainer = CJBTrainer(config=experiment_config,
                            experiment_files=experiment_files)
        trainer.train()
    else:
        experiment_dir = r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\results\cjb\graph\1718616860"
        continue_graph_experiment(experiment_dir)