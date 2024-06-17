import torch
import torch.nn.functional as F
from markov_bridges.models.generative_models.cjb import CJB

from markov_bridges.configs.config_classes.metrics.metrics_configs import HellingerMetricConfig
#===============================================
# METRICS CLASSES
#===============================================

from markov_bridges.models.pipelines.pipeline_cjb import CJBPipelineOutput
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

from markov_bridges.models.metrics.abstract_metrics import BasicMetric
from markov_bridges.utils.plots.histograms_plots import plot_marginals_binary_histograms

class HellingerMetric(BasicMetric):
    """
    Calculates the Hellinger distance between the histograms of generated and real data
    since this requieres obtaining the histogram marginals, all histograms related metrics
    are contained here.

    plot_binary
    plot_histogram_colors
    """
    def __init__(self,model:CJB,metrics_config:HellingerMetricConfig):
        super().__init__(model,metrics_config)

        self.binary = metrics_config.binary
        self.dimensions = model.config.data.dimensions
        self.vocab_size = model.config.data.vocab_size
        self.plot_histogram = metrics_config.plot_histogram 
        self.plot_binary_histogram = metrics_config.plot_binary_histogram

        # variables to carry statistics
        self.sample_size = 0
        self.metrics_config = metrics_config

        self.noise_histogram = torch.zeros((self.dimensions,self.vocab_size))
        self.generative_histogram = torch.zeros((self.dimensions,self.vocab_size))
        self.real_histogram = torch.zeros((self.dimensions,self.vocab_size))

    def batch_operation(self,batch:MarkovBridgeDataNameTuple,generative_sample:CJBPipelineOutput):
        """
        aggregates the statistics to obtain histograms
        """
        batch_size = batch.target_discrete.size(0)
        self.sample_size += batch_size

        if self. has_context_discrete:
            original_sample = self.join_context(batch.context_discrete,batch.target_discrete)
            noise_sample = self.join_context(batch.context_discrete,batch.source_discrete)
        else:
            original_sample = batch.target_discrete
            noise_sample = batch.source_discrete

        # reshapce for one hot
        noise_sample = noise_sample.reshape(-1)
        original_sample = original_sample.reshape(-1)
        raw_sample = generative_sample.raw_sample.reshape(-1)

        # obtain one hot
        noise_histogram = F.one_hot(noise_sample.long(), num_classes=self.vocab_size)
        generative_histogram = F.one_hot(raw_sample.long(), num_classes=self.vocab_size)
        original_histogram = F.one_hot(original_sample.long(),  num_classes=self.vocab_size)

        # reshape for batches
        noise_histogram = noise_histogram.reshape(batch_size,self.dimensions,self.vocab_size)
        generative_histogram = generative_histogram.reshape(batch_size,self.dimensions,self.vocab_size)
        original_histogram = original_histogram.reshape(batch_size,self.dimensions,self.vocab_size)

        # sum the values
        self.noise_histogram += noise_histogram.sum(axis=0)
        self.generative_histogram += generative_histogram.sum(axis=0)
        self.real_histogram += original_histogram.sum(axis=0)
        
    def final_operation(self,epoch=None):
        #aggregates statistics
        self.noise_histogram = self.noise_histogram/self.sample_size
        self.generative_histogram = self.generative_histogram/self.sample_size
        self.real_histogram = self.real_histogram/self.sample_size

        #calculates distance
        hd = self.hellinger_distance(self.generative_histogram,self.real_histogram)

        metrics_dict = {self.name:hd.item()}

        # Plots
        if self.plot_binary_histogram:
            noise_histogram = self.noise_histogram[:,1]
            generative_histogram = self.generative_histogram[:,1]
            real_histogram = self.real_histogram[:,1]

            histograms = (noise_histogram,noise_histogram,real_histogram,generative_histogram)
            plot_path = None
            if self.has_experiment_files:
                plot_path = self.plots_path.format("binary_histograms_{0}".format(epoch))
            plot_marginals_binary_histograms(histograms,plot_path)

        if self.plot_histogram:
            pass

        if epoch is not None:
            self.save_metric(metrics_dict,epoch)

    def hellinger_distance(self,hist1,hist2):
        # Compute the square root of each bin
        sqrt_hist1 = torch.sqrt(hist1)
        sqrt_hist2 = torch.sqrt(hist2)

        # Compute the Euclidean distance between the square-rooted histograms
        euclidean_distance = torch.norm(sqrt_hist1 - sqrt_hist2, 2)

        # Normalize to get the Hellinger distance
        hellinger_distance = euclidean_distance / torch.sqrt(torch.tensor(2.0))
        return hellinger_distance
