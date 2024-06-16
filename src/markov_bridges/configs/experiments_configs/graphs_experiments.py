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
from markov_bridges.data.graphs_dataloader import GraphDataloader
from markov_bridges.models.generative_models.cjb import CJB
from markov_bridges.models.trainers.cjb_trainer import CJBTrainer

from markov_bridges.utils.experiment_files import ExperimentFiles

from dataclasses import asdict

def get_graph_experiment(dataset_name="community_small"):
    experiment_config = CJBConfig()
    #data config
    experiment_config.data = dataset_str_to_config[dataset_name](batch_size=20)
    #temporal network config
    experiment_config.temporal_network = TemporalMLPConfig(hidden_dim=50,time_embed_dim=50)
    #trainer config
    experiment_config.trainer = CJBTrainerConfig(number_of_epochs=10,learning_rate=1e-3)
    # define metrics
    metrics = [HellingerMetricConfig()]
    experiment_config.trainer.metrics = metrics

    return experiment_config

if __name__=="__main__":
    experiment_config = get_graph_experiment()
    experiment_files = ExperimentFiles(experiment_name="cjb",
                                       experiment_type="graph")    
    
    
    trainer = CJBTrainer(config=experiment_config,
                         experiment_files=experiment_files)
    trainer.train()
