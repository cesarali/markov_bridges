import json
import torch
from pprint import pprint
from dataclasses import asdict
import lightning as L
from markov_bridges.data.dataloaders_utils import get_dataloaders
from markov_bridges.utils.experiment_files import ExperimentFiles,get_experiment_dir
from markov_bridges.configs.config_classes.trainers.trainer_config import CJBTrainerConfig
from markov_bridges.models.deprecated.trainers.cjb_trainer import CJBTrainer
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig

from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig

from markov_bridges.models.generative_models.cjb_lightning import ClassificationForwardRateL
from markov_bridges.models.generative_models.cjb_lightning import CJBL
from markov_bridges.configs.config_classes.networks.temporal_networks_config import SequenceTransformerConfig
from markov_bridges.configs.config_classes.networks.temporal_networks_config import TemporalMLPConfig
from markov_bridges.configs.config_classes.data.graphs_configs import dataset_str_to_config
from markov_bridges.models.metrics.metrics_utils import LogMetrics
from markov_bridges.models.pipelines.pipeline_cjb import CJBPipeline
from markov_bridges.configs.config_classes.metrics.metrics_configs import HellingerMetricConfig,GraphMetricsConfig
from markov_bridges.configs.config_classes.pipelines.pipeline_configs import BasicPipelineConfig

if __name__=="__main__":
    train = True
    dataset_name = "community_small"
    if train:
        model_config = CJBConfig()
        model_config.data = dataset_str_to_config[dataset_name](batch_size=20)
        model_config.temporal_network = TemporalMLPConfig(hidden_dim=150, time_embed_dim=50)
        model_config.trainer = CJBTrainerConfig(
            number_of_epochs=1000,
            learning_rate=2e-4,
            warm_up=0
        )
        model_config.pipeline =  BasicPipelineConfig(number_of_steps=200,
                                                     max_rate_at_end=True)
        metrics = [HellingerMetricConfig(plot_binary_histogram=True),
                   GraphMetricsConfig(plot_graphs=True,
                                      methods=[],
                                      windows=True)]
        model_config.trainer.metrics = metrics
        experiment_files = ExperimentFiles(experiment_name="cjb",
                                           experiment_type="graph",
                                           experiment_indentifier="lightning_test10",
                                           delete=True)
        cjb = CJBL(model_config,experiment_files)
        cjb.train()
    else:
        experiment_files = ExperimentFiles(experiment_name="cjb",experiment_type="graph",experiment_indentifier="lightning_test9",delete=False)      
        cjb = CJBL(experiment_source=experiment_files,checkpoint_type="best")
        databatch = cjb.dataloader.get_databatch()
        print(databatch._asdict().keys())


