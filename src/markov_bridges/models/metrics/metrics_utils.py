import torch
from typing import Union,List

from markov_bridges.models.generative_models.cjb import CJB

from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig

from markov_bridges.configs.config_classes.metrics.metrics_configs import(
    MusicPlotConfig,
    HellingerMetricConfig,
    metrics_str_to_config_class
)

from markov_bridges.models.metrics.histogram_metrics import HellingerMetric

def load_metric(cjb:CJB,metric_config:Union[str]):
    # Metrics can be passed as simple strings or with the full ConfigClass
    # here we ensure the metric is a config before defining the class
    if isinstance(metric_config,str):
        metric_config = metrics_str_to_config_class[metric_config]
    if isinstance(metric_config,HellingerMetricConfig):
        metric = HellingerMetric(cjb,metric_config)
    return metric

def obtain_metrics_stats(model_config:CJBConfig,metrics_configs_list):
    """
    if at least one metric requieres paths, we store the paths in the pipeline output, same with 
    origin say in the case of completion metrics
    """
    for metric_config in metrics_configs_list:
        return_path = False
        return_origin = False
        requieres_test_loop = False

        if isinstance(metric_config,str):
            metric_config = metrics_str_to_config_class[metric_config]
    
        if metric_config.requieres_paths:
            return_path = True
        if metric_config.requieres_origin:
            return_origin = True
        if metric_config.requieres_test_loop:
            requieres_test_loop = True

    return return_path,return_origin,requieres_test_loop

def log_metrics(model:CJB,metrics_configs_list,debug=False):
    """
    In order to obtain metrics one is usually requiered to generate a sample of the size of the test set
    and obtain statistics for both the test set as well as the whole generated samples and perform distances
    e.g. one requieres the histograms of a generated sampled of the size of test set and then a histogram of the 
    test set and calculate say hellinger distance, this means that each metric must perform and operation during 
    each test set batch and then a final operation after the statistics are gathered
    """
    # Define List of Metrics
    metrics  = []
    for metric_config in metrics_configs_list:
        metric = load_metric(model,metric_config)
        metrics.append(metric)
    
    # Obtain What To Do
    return_path,return_origin,requieres_test_loop = obtain_metrics_stats(model.config,metrics_configs_list)

    # For each batch in the test set applies the operation requiered for the metrics using that batch
    for databatch in model.dataloader.test():
        for metric in metrics:
            if metric.batch_operation:
                generative_sample = model.pipeline.generate_sample(databatch,
                                                                   return_path=return_path,
                                                                   return_origin=return_origin)
                metric.batch_operation(databatch,generative_sample)
        if debug:
            break

    # Do Final Operation Per Metric
    for metric in metrics:
        metric.final_operation()