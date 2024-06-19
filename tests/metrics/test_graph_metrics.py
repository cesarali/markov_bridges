import pytest

from markov_bridges.configs.config_classes.data.graphs_configs import (
    GraphDataloaderGeometricConfig,
    CommunitySmallGConfig
)

from markov_bridges.configs.experiments_configs.graphs_experiments import get_graph_experiment

from markov_bridges.models.generative_models.cjb import CJB
from markov_bridges.data.graphs_dataloader import GraphDataloader
from markov_bridges.models.metrics.metrics_utils import LogMetrics
from markov_bridges.configs.config_classes.metrics.metrics_configs import GraphMetricsConfig

def test_sample_network():
    experiment_config = get_graph_experiment()
    cjb = CJB(experiment_config)
    databatch = cjb.dataloader.get_databatch()

    data = cjb.dataloader.transform_to_native_shape(databatch)
    sample = cjb.pipeline(20)
    raw_sample = cjb.dataloader.transform_to_native_shape(sample.raw_sample)
    print(raw_sample.shape)

def test_metrics_log():
    experiment_config = get_graph_experiment()
    cjb = CJB(experiment_config)

    log_metrics = LogMetrics(cjb,[GraphMetricsConfig(plot_graphs=True,
                                                     methods=[],
                                                     windows=True)])
    all_metrics = log_metrics(cjb,None)

    print(all_metrics)

    

if __name__=="__main__":
    test_metrics_log()
    