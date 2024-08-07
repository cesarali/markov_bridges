import torch
from typing import Union,List

from markov_bridges.models.generative_models.cjb import CJB
from markov_bridges.models.pipelines.samplers.tau_leaping_cjb import TauLeapingOutput
from markov_bridges.models.metrics.abstract_metrics import BasicMetric
from markov_bridges.models.metrics.music_metrics import MusicPlots
from markov_bridges.models.metrics.histogram_metrics import HellingerMetric,MixedHellingerMetric
from markov_bridges.models.metrics.graph_metrics import GraphsMetrics
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig

from markov_bridges.configs.config_classes.metrics.metrics_configs import(
    BasicMetricConfig,
    MusicPlotConfig,
    HellingerMetricConfig,
    GraphMetricsConfig,
    MixedHellingerMetricConfig,
    metrics_config
)

class GatherSamples:
    """
    List to gather the values from the sample

    """
    def __init__(self) -> None:
        self.context_discrete: List[torch.tensor] = []    
        self.context_continuous: List[torch.tensor] = []

        self.target_discrete: List[torch.tensor] = []
        self.target_continuous: List[torch.tensor] = []

        self.sample:List[torch.tensor] = []

    def gather(self,generative_sample:TauLeapingOutput,remaining_samples:int):
        # how much to take
        raw_sample = generative_sample.discrete
        batch_size = raw_sample.size(0)
        if remaining_samples is None:
            take_size  = batch_size
        else:
            take_size = min(remaining_samples,batch_size)

        # Initialize variables to None
        context_discrete = None
        context_continuous = None
        target_discrete = None
        target_continuous = None

        # Check for attributes and assign values if they exist
        if hasattr(generative_sample.x0, 'context_discrete'):
            context_discrete = generative_sample.x0.context_discrete[:take_size]
        if hasattr(generative_sample.x0, 'context_continuous'):
            context_continuous = generative_sample.x0.context_continuous[:take_size]
        if hasattr(generative_sample.x0, 'target_discrete'):
            target_discrete = generative_sample.x0.target_discrete[:take_size]
        if hasattr(generative_sample.x0, 'target_continuous'):
            target_continuous = generative_sample.x0.target_continuous[:take_size]

        sample = generative_sample.discrete[:take_size]

        # Append to lists only if they are not None
        if context_discrete is not None:
            self.context_discrete.append(context_discrete)
        if context_continuous is not None:
            self.context_continuous.append(context_continuous)
        if target_discrete is not None:
            self.target_discrete.append(target_discrete)
        if target_continuous is not None:
            self.target_continuous.append(target_continuous)

        self.sample.append(sample)
        if remaining_samples is not None:
            remaining_samples = remaining_samples - take_size

        return remaining_samples
    
    def concatenate_bags(self):
        if len(self.context_discrete) > 0:
            self.context_discrete = torch.cat(self.context_discrete,axis=0) 
        if len(self.context_continuous) > 0:
            self.context_continuous = torch.cat(self.context_continuous,axis=0) 
        if len(self.target_discrete) > 0:
            self.target_discrete = torch.cat(self.target_discrete,axis=0) 
        if len(self.target_continuous) > 0:
            self.target_continuous = torch.cat(self.target_continuous,axis=0)
        if len(self.sample) > 0:
            self.sample = torch.cat(self.sample,axis=0)

def load_metric(cjb:CJB,metric_config:Union[str,BasicMetricConfig]):
    # Metrics can be passed as simple strings or with the full ConfigClass
    # here we ensure the metric is a config before defining the class
    if isinstance(metric_config,str):
        metric_config = metrics_config[metric_config]()
    if isinstance(metric_config,HellingerMetricConfig):
        metric = HellingerMetric(cjb,metric_config)
    elif isinstance(metric_config,MusicPlotConfig):
        metric = MusicPlots(cjb,metric_config)
    elif isinstance(metric_config,GraphMetricsConfig):
        metric = GraphsMetrics(cjb,metric_config)
    elif isinstance(metric_config,MixedHellingerMetricConfig):
        metric = MixedHellingerMetric(cjb,metric_config)
    return metric

def obtain_metrics_stats(model_config:CJBConfig,metrics_configs_list:List[BasicMetricConfig]):
    """
    if at least one metric requieres paths, we store the paths in the pipeline output, same with 
    origin say in the case of completion metrics
    """
    number_of_samples_to_gather = 0
    for metric_config in metrics_configs_list:
        return_path = False
        return_origin = False
        requieres_test_loop = False
        
        if isinstance(metric_config,str):
            metric_config = metrics_config[metric_config]
    
        if metric_config.requieres_paths:
            return_path = True
        if metric_config.requieres_origin:
            return_origin = True
        if metric_config.requieres_test_loop:
            requieres_test_loop = True
        
        # for metrics that requiere aggregates (such as plots or graph metrics) defines the maximum number of samples requiered
        if isinstance(metric_config.number_of_samples_to_gather,int):
            if isinstance(number_of_samples_to_gather,int):
                number_of_samples_to_gather = max(metric_config.number_of_samples_to_gather,number_of_samples_to_gather)
        if isinstance(metric_config.number_of_samples_to_gather,str):
            number_of_samples_to_gather = "all"

    return return_path,return_origin,requieres_test_loop,number_of_samples_to_gather

def log_metrics(model:CJB,metrics_configs_list,epoch=None,debug=False):
    """
    In order to obtain metrics one is usually requiered to generate a sample of the size of the test set
    and obtain statistics for both the test set as well as the whole generated samples and perform distances
    e.g. one requieres the histograms of a generated sampled of size of test set and then a histogram of the 
    test set and calculate say hellinger distance, this means that each metric must perform and operation during 
    each test set batch and then a final operation after the statistics are gathered.
    """


    # Define List of Metrics
    metrics  = []
    for metric_config in metrics_configs_list:
        metric = load_metric(model,metric_config)
        metrics.append(metric)
    
    # Obtain What To Do
    return_path,return_origin,requieres_test_loop,number_of_samples_to_gather = obtain_metrics_stats(model.config,metrics_configs_list)

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
        metric.final_operation(epoch)

class LogMetrics:
    """
    In order to obtain metrics one is usually requiered to generate a sample of the size of the test set
    and obtain statistics for both the test set as well as the whole generated samples and perform distances
    e.g. one requieres the histograms of a generated sampled of size of test set and then a histogram of the 
    test set and calculate say hellinger distance, this means that each metric must perform and operation during 
    each test set batch and then a final operation after the statistics are gathered.

    """
    metrics:List[BasicMetric]

    def __init__(self,model:CJB,
                 metrics_configs_list:List[BasicMetricConfig|str],
                 debug=False) -> None:
        
        self.samples_bag = None
        self.define_metrics(model,metrics_configs_list)
        self.define_metrics_stats()
        self.debug = debug

    def define_metrics(self,model,metrics_configs_list):
        """
        Initialize each of the metrics classes as specified in 
        the metrics list
        """
        # Define List of Metrics
        self.metrics  = []
        for metric_config in metrics_configs_list:
            metric = load_metric(model,metric_config)
            self.metrics.append(metric)

    def define_metrics_stats(self):
        """
        If at least one metric requieres paths, we store the paths in the pipeline output, same with 
        origin say in the case of completion metrics
        """
        self.number_of_samples_to_gather = 0
                
        self.return_path = False
        self.return_origin = False
        self.requieres_test_loop = False
        for metric in self.metrics:
            metric_config = metric.metric_config
            metric_config:BasicMetricConfig

            if metric_config.requieres_paths:
                self.return_path = True
            if metric_config.requieres_origin:
                self.return_origin = True
            if metric_config.requieres_test_loop:
                self.requieres_test_loop = True
            
            # for metrics that requiere aggregates (such as plots or graph metrics) defines the maximum number of samples requiered
            if isinstance(metric_config.number_of_samples_to_gather,int):
                if isinstance(self.number_of_samples_to_gather,int):
                    self.number_of_samples_to_gather = max(metric_config.number_of_samples_to_gather,self.number_of_samples_to_gather)
            if isinstance(metric_config.number_of_samples_to_gather,str):
                self.number_of_samples_to_gather = "all"

        if self.number_of_samples_to_gather != 0:
            self.samples_bag = GatherSamples()

    def __call__(self, model:CJB,epoch=None):
        """
        """
        # dict to gather and return all calculated metrics
        all_metrics = {}
        if len(self.metrics) > 0:

            # keeps track of how many samples one needs to gather
            remaining_samples = None
            if self.number_of_samples_to_gather != 0:
                if isinstance(self.number_of_samples_to_gather,int):
                    remaining_samples = self.number_of_samples_to_gather

            # For each batch in the test set applies the operation requiered for the metrics using that batch
            for databatch in model.dataloader.test():
                generative_sample = model.pipeline.generate_sample(databatch,
                                                                   return_path=self.return_path,
                                                                   return_origin=self.return_origin)
                
                # perform the batch operation for each metric
                for metric in self.metrics:
                    if metric.batch_operation:
                        metric.batch_operation(databatch,generative_sample)

                # gather sample
                remaining_samples = self.gather_samples(generative_sample,remaining_samples)

                if self.debug:
                    break
                
            # concatenate the samples
            if self.samples_bag is not None:
                self.samples_bag.concatenate_bags()

            # Do Final Operation Per Metric
            for metric in self.metrics:
                all_metrics = metric.final_operation(all_metrics,self.samples_bag,epoch)
            
        return all_metrics
        
    def gather_samples(self,
                       generative_sample:TauLeapingOutput,
                       remaining_samples:int):
        """
        for metrics that requiere aggregated samples we gather 
        the samples in a list, the gathering is done to the 
        maximum number requiered from the whole list of metrics.

        metrics are gathered in self.samples_bag
        """
        if self.number_of_samples_to_gather != 0:
            # if a number then there is a finite amount to gather
            if isinstance(self.number_of_samples_to_gather,int):
                remaining_samples = self.samples_bag.gather(generative_sample,remaining_samples)
                if remaining_samples == 0:
                    self.number_of_samples_to_gather = 0
            else:
                # here we gather all
                self.samples_bag.gather(generative_sample,remaining_samples)
        return remaining_samples
