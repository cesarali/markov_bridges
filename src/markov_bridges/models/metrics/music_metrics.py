

import torch
from markov_bridges.configs.config_classes.metrics.metrics_configs import BasicMetricConfig
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple
from markov_bridges.models.metrics.abstract_metrics import BasicMetric
from markov_bridges.models.pipelines.samplers.tau_leaping_cjb import TauLeapingOutput
from markov_bridges.utils.plots.music_plots import plot_songs

class MusicPlots(BasicMetric):
    """
    """
    def __init__(self, model, metrics_config: BasicMetricConfig):
        super().__init__(model, metrics_config)
        self.conditional_dimension = model.config.data.context_discrete_dimension

    def batch_operation(self, databatch: MarkovBridgeDataNameTuple, generative_sample: TauLeapingOutput):
        pass

    def final_operation(self, all_metrics,samples_gather,epoch=None):
        generative_sample = samples_gather.sample
        target_discrete = samples_gather.target_discrete
        context_discrete = samples_gather.context_discrete

        generative_sample =  torch.cat([context_discrete,generative_sample],dim=1)
        original_sample = torch.cat([context_discrete,target_discrete],dim=1)

        if self.plots_path is not None:
            plots_path = self.plots_path.format(self.name + "_{0}_".format(epoch))
        else:
            plots_path = None
        plot_songs(original_sample,generative_sample,save_path=plots_path,
                   conditional_dimension=self.conditional_dimension,
                   repeating=False)
        return all_metrics
