import torch
from typing import List,Union

from markov_bridges.models.generative_models.cjb import CJB
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.models.pipelines.pipeline_cjb import CJBPipelineOutput

class BasicMetric:
    """
    In order to obtain metrics one is usually requiered to generate a sample of the size of the test set
    and obtain statistics for both the test set as well as the whole generated samples and perform distances
    e.g. one requieres the histograms of a generated sampled of the size of test set and then a histogram of the 
    test set and calculate say hellinger distance, this means that each metric must perform and operation during 
    each test set batch and then a final operation after the statistics are gathered

    this class defines the abstracts methods that each metric class should follow
    """
    def __init__(self,model:CJB,metrics_config):
        pass

    def batch_operation(self,databatch:MarkovBridgeDataNameTuple,generative_sample:CJBPipelineOutput):
        pass

    def final_operation(self):
        pass

    def save_metric(self):
        pass
