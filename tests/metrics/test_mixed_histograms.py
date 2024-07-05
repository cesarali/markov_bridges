import torch
from markov_bridges.models.generative_models.cmb import CMB
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig

from markov_bridges.models.networks.temporal.mixed.mixed_networks_utils import load_mixed_network

from markov_bridges.models.metrics.metrics_utils import LogMetrics
from markov_bridges.configs.config_classes.metrics.metrics_configs import (
    MixedHellingerMetricConfig,
)

if __name__=="__main__":
    model_config = CMBConfig()
    model_config.data = IndependentMixConfig()
    cmb = CMB(model_config)
    log_metrics = LogMetrics(cmb,
                             metrics_configs_list=[MixedHellingerMetricConfig(plot_histogram=True,
                                                                              plot_continuous_variables=True)],
                             debug=False)
    all_metrics = log_metrics(cmb,None)
    print(all_metrics)