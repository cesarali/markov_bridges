import torch
import json
from typing import List,Union

from markov_bridges.models.pipelines.samplers.tau_leaping_cjb import TauLeapingOutput
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple,MarkovBridgeDataClass
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.metrics.metrics_configs import BasicMetricConfig

class BasicMetric:
    """
    In order to obtain metrics one is usually requiered to generate a sample of the size of the test set
    and obtain statistics for both the test set as well as the whole generated samples and perform distances
    e.g. one requieres the histograms of a generated sampled of the size of test set and then a histogram of the 
    test set and calculate say hellinger distance, this means that each metric must perform and operation during 
    each test set batch and then a final operation after the statistics are gathered

    this class defines the abstracts methods that each metric class should follow
    and handles the storing of the metrics

    """
    name:str 

    def __init__(self,model,metrics_config:BasicMetricConfig):
        self.metric_config = metrics_config
        self.name = metrics_config.name
        self.compute_in_gpu = metrics_config.compute_in_gpu
        self.number_of_samples_to_gather = metrics_config.number_of_samples_to_gather

        # context handling
        self.has_context_discrete = False
        if model.config.data.has_context_discrete:
            self.join_context = model.dataloader.join_context
            self.has_context_discrete = True

        # experiment files handling
        self.has_experiment_files = False
        if model.experiment_files is not None:
            self.metrics_file = model.experiment_files.metrics_file
            self.plots_path = model.experiment_files.plot_path
            self.has_experiment_files = True
        else:
            self.metrics_file = None
            self.plots_path = None


    def batch_operation(self,databatch:MarkovBridgeDataNameTuple,generative_sample:TauLeapingOutput):
        pass

    def final_operation(self,all_metrics,samples_gather:MarkovBridgeDataClass,epoch=None):
        return all_metrics

    def save_metric(self,metrics_dict,epoch="best"):
        if self.has_experiment_files:
            mse_metric_path = self.metrics_file.format(self.name + "_{0}_".format(epoch))
            with open(mse_metric_path, "w") as f:
                json.dump(metrics_dict, f)
