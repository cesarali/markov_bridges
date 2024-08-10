import pytest
import torch

from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.experiments_configs.graphs_experiments import get_graph_experiment

from markov_bridges.models.deprecated.generative_models.cjb import CJB
from markov_bridges.models.metrics.metrics_utils import log_metrics
from markov_bridges.configs.config_classes.metrics.metrics_configs import (
    MetricsAvaliable,
    HellingerMetricConfig
)

def test_log_metrics():
    metrics_available = MetricsAvaliable()
    model_config:CJBConfig = get_graph_experiment()
    metrics_list = [HellingerMetricConfig(binary=True,plot_binary_histogram=True)]

    cjb = CJB(model_config)
    sample = cjb.pipeline(sample_size=10)
    print(sample.raw_sample.shape)

    log_metrics(cjb,metrics_list,debug=True)

if __name__=="__main__":
    test_log_metrics()
