

import torch
from markov_bridges.configs.config_classes.metrics.metrics_configs import BasicMetricConfig
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple
from markov_bridges.models.generative_models.cjb import CJB
from markov_bridges.models.metrics.abstract_metrics import BasicMetric
from markov_bridges.models.pipelines.pipeline_cjb import CJBPipelineOutput
from markov_bridges.utils.plots.music_plots import plot_songs

class MusicPlots(BasicMetric):
    """
    """
    def __init__(self, model: CJB, metrics_config: BasicMetricConfig):
        super().__init__(model, metrics_config)
        self.join_context:CJB.dataloader.join_context

    def batch_operation(self, databatch: MarkovBridgeDataNameTuple, generative_sample: CJBPipelineOutput):
        pass

    def final_operation(self, all_metrics,samples_gather,epoch=None):
        generative_sample = samples_gather.raw_sample
        target_discrete = samples_gather.target_discrete
        context_discrete = samples_gather.context_discrete
        original_sample = self.join_context(context_discrete,target_discrete)
        plot_songs(original_sample,generative_sample)
        return all_metrics
