import pytest
import torch

from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig

from markov_bridges.configs.experiments_configs.music_experiments import conditional_music_experiment
from markov_bridges.configs.config_classes.networks.temporal_networks_config import SequenceTransformerConfig

from markov_bridges.models.generative_models.cjb import CJB


from markov_bridges.models.metrics.metrics_utils import log_metrics
from markov_bridges.configs.config_classes.metrics.metrics_configs import MetricsAvaliable

def test_log_metrics():
    metrics_available = MetricsAvaliable()
    model_config:CJBConfig = conditional_music_experiment()
    model_config.temporal_network = SequenceTransformerConfig(num_heads=1,num_layers=1)

    cjb = CJB(model_config)
    log_metrics(cjb,[metrics_available.histogram_hellinger])

if __name__=="__main__":
    test_log_metrics()
    