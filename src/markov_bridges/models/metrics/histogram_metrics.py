import torch
from typing import Tuple
import torch.nn.functional as F
from dataclasses import dataclass

from markov_bridges.configs.config_classes.metrics.metrics_configs import (
    HellingerMetricConfig,
    MixedHellingerMetricConfig
)

#===============================================
# METRICS CLASSES
#===============================================

from markov_bridges.models.pipelines.samplers.tau_leaping_cjb import TauLeapingOutput
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

from markov_bridges.models.metrics.abstract_metrics import BasicMetric
from markov_bridges.utils.plots.histograms_plots import plot_marginals_binary_histograms
from markov_bridges.utils.plots.histograms_plots import plot_categorical_histogram_per_dimension
from markov_bridges.utils.plots.mix_histograms_plots import plot_scatterplot
from markov_bridges.models.pipelines.samplers.mixed_tau_diffusion import MixedTauState

class HellingerMetric(BasicMetric):
    """
    Calculates the Hellinger distance between the histograms of generated and real data
    since this requieres obtaining the histogram marginals, all histograms related metrics
    are contained here.

    plot_binary
    plot_histogram_colors
    """
    def __init__(self,model,metrics_config:HellingerMetricConfig):
        super().__init__(model,metrics_config)

        self.binary = metrics_config.binary
        self.dimensions = model.config.data.discrete_generation_dimension
        self.vocab_size = model.config.data.vocab_size
        self.plot_histogram = metrics_config.plot_histogram 
        self.plot_binary_histogram = metrics_config.plot_binary_histogram

        # variables to carry statistics
        self.sample_size = 0
        self.metrics_config = metrics_config

        self.noise_histogram = torch.zeros((self.dimensions,self.vocab_size))
        self.generative_histogram = torch.zeros((self.dimensions,self.vocab_size))
        self.real_histogram = torch.zeros((self.dimensions,self.vocab_size))

    def batch_operation(self,batch:MarkovBridgeDataNameTuple,generative_sample:TauLeapingOutput):
        """
        aggregates the statistics to obtain histograms
        """
        batch_size = batch.target_discrete.size(0)
        self.sample_size += batch_size

        original_sample = batch.target_discrete
        noise_sample = batch.source_discrete

        # reshape for one hot
        noise_sample = noise_sample.detach().cpu()
        original_sample = original_sample.detach().cpu()
        generative_sample = generative_sample.discrete.detach().cpu()

        noise_sample = noise_sample.reshape(-1)
        original_sample = original_sample.reshape(-1)
        generative_sample = generative_sample.reshape(-1)

        # obtain one hot
        noise_histogram = F.one_hot(noise_sample.long(), num_classes=self.vocab_size)
        generative_histogram = F.one_hot(generative_sample.long(), num_classes=self.vocab_size)
        original_histogram = F.one_hot(original_sample.long(),  num_classes=self.vocab_size)

        # reshape for batches
        noise_histogram = noise_histogram.reshape(batch_size,self.dimensions,self.vocab_size)
        generative_histogram = generative_histogram.reshape(batch_size,self.dimensions,self.vocab_size)
        original_histogram = original_histogram.reshape(batch_size,self.dimensions,self.vocab_size)

        # sum the values
        self.noise_histogram += noise_histogram.sum(axis=0)
        self.generative_histogram += generative_histogram.sum(axis=0)
        self.real_histogram += original_histogram.sum(axis=0)
        
    def final_operation(self,all_metrics,samples_bags,epoch=None):
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

        # Non binary plot
        if self.plot_histogram:
            pass

        if epoch is not None:
            self.save_metric(metrics_dict,epoch)

        all_metrics.update(metrics_dict)
        return all_metrics
    
    def hellinger_distance(self,hist1,hist2):
        # Compute the square root of each bin
        sqrt_hist1 = torch.sqrt(hist1)
        sqrt_hist2 = torch.sqrt(hist2)

        # Compute the Euclidean distance between the square-rooted histograms
        euclidean_distance = torch.norm(sqrt_hist1 - sqrt_hist2, 2)

        # Normalize to get the Hellinger distance
        hellinger_distance = euclidean_distance / torch.sqrt(torch.tensor(2.0))
        return hellinger_distance

@dataclass
class SamplesRequiered:
    """
    Data class to organize all variables in a sample, noise, orginal sample and generative sample
    """
    generative_sample_discrete:torch.Tensor = None
    generative_sample_continuous:torch.Tensor = None
    original_discrete_sample:torch.Tensor = None
    original_continuous_sample:torch.Tensor = None
    noise_discrete_sample:torch.Tensor = None
    noise_continuous_sample:torch.Tensor = None

    context_discrete:torch.Tensor = None
    context_continuous:torch.Tensor = None

class MixedHellingerMetric(BasicMetric):
    """
    Calculates the Hellinger distance between the histograms of generated and real data
    since this requieres obtaining the histogram marginals, all histograms related metrics
    are contained here. For mixed variables, provided the plots of the continuous variables

    plot_binary
    plot_histogram_colors
    """
    def __init__(self,model,metrics_config:MixedHellingerMetricConfig):
        super().__init__(model,metrics_config)
        
        # mixed data characteristics
        self.discrete_dimensions = model.config.data.discrete_dimensions
        self.continuous_dimensions = model.config.data.continuos_dimensions

        self.has_target_continuous = model.config.data.has_target_continuous
        self.has_target_discrete = model.config.data.has_target_discrete

        self.has_context_discrete = model.config.data.has_context_discrete
        self.has_context_continuous = model.config.data.has_context_continuous

        self.join_context = model.dataloader.join_context
        self.vocab_size = model.config.data.vocab_size
        self.plot_histogram = metrics_config.plot_histogram 
        self.plot_continuous_variables = metrics_config.plot_continuous_variables

        # variables to carry statistics
        self.sample_size = 0
        self.metrics_config = metrics_config

        # continuous variables
        self.noise_points = []
        self.generated_points = []
        self.real_points = []

        # discrete variables
        self.noise_histogram = torch.zeros((self.discrete_dimensions,self.vocab_size))
        self.generative_histogram = torch.zeros((self.discrete_dimensions,self.vocab_size))
        self.real_histogram = torch.zeros((self.discrete_dimensions,self.vocab_size))

    def organize_samples(self,batch:MarkovBridgeDataNameTuple,sample:MixedTauState)->SamplesRequiered:
        """
        This function completes the context for noise, and original sample if requiered 
        and returns a dataclass with all the different samples 

        batch: (MarkovBridgeDataNameTuple) with context
        sample: (MixedTauState)

        returns
        -------
        samples_requiered: SamplesRequiered all samples with completed context
        """
        if self.has_target_discrete:
            original_discrete_sample = batch.target_discrete
            noise_discrete_sample = batch.source_discrete
            generative_sample_discrete = sample.discrete
        else:
            original_discrete_sample = None
            noise_discrete_sample = None
            generative_sample_discrete = None

        if self.has_target_continuous:
            original_continuous_sample = batch.target_continuous 
            noise_continuous_sample = batch.source_continuous
            generative_sample_continuous = sample.continuous
        else:
            original_continuous_sample = None
            noise_continuous_sample = None
            generative_sample_continuous = None

        if self.has_context_discrete:
            context_discrete = batch.context_discrete
        else:
            context_discrete = None

        if self.has_context_continuous:
            context_continuous = batch.context_continuous
        else:
            context_continuous = None

        original_discrete_sample,original_continuous_sample = self.join_context(batch,original_discrete_sample,original_continuous_sample)
        noise_discrete_sample,noise_continuous_sample = self.join_context(batch,noise_discrete_sample,noise_continuous_sample)

        samples_requiered = SamplesRequiered(original_continuous_sample=original_continuous_sample,
                                             original_discrete_sample=original_discrete_sample,
                                             generative_sample_continuous=generative_sample_continuous,
                                             generative_sample_discrete=generative_sample_discrete,
                                             noise_discrete_sample=noise_discrete_sample,
                                             noise_continuous_sample=noise_continuous_sample,
                                             context_discrete=context_discrete,
                                             context_continuous=context_continuous)

        return samples_requiered

    def discrete_histograms(self,noise_discrete_sample,original_discrete_sample,generative_sample):
        """
        Aggregates the statistics comming from the batch for the discrete variables
        """
        #===================================================
        # HISTOGRAM STATS
        #===================================================
        if noise_discrete_sample is not None:
            batch_size = noise_discrete_sample.size(0)

            # reshape for one hot
            noise_discrete_sample = noise_discrete_sample.detach().cpu().reshape(-1)
            # obtain one hot
            noise_histogram = F.one_hot(noise_discrete_sample.long(), num_classes=self.vocab_size)
            # reshape for batches
            noise_histogram = noise_histogram.reshape(batch_size, self.discrete_dimensions, self.vocab_size)
            # sum the values
            self.noise_histogram += noise_histogram.sum(axis=0)

        if generative_sample is not None:
            batch_size = generative_sample.size(0)

            # reshape for one hot
            generative_sample = generative_sample.detach().cpu().reshape(-1)
            # obtain one hot
            generative_histogram = F.one_hot(generative_sample.long(), num_classes=self.vocab_size)
            # reshape for batches
            generative_histogram = generative_histogram.reshape(batch_size, self.discrete_dimensions, self.vocab_size)
            # sum the values
            self.generative_histogram += generative_histogram.sum(axis=0)

        if original_discrete_sample is not None:
            batch_size = original_discrete_sample.size(0)

            # reshape for one hot
            original_discrete_sample = original_discrete_sample.detach().cpu().reshape(-1)
            # obtain one hot
            original_histogram = F.one_hot(original_discrete_sample.long(), num_classes=self.vocab_size)
            # reshape for batches
            original_histogram = original_histogram.reshape(batch_size, self.discrete_dimensions, self.vocab_size)
            # sum the values
            self.real_histogram += original_histogram.sum(axis=0)

    def batch_operation(self,batch:MarkovBridgeDataNameTuple,generative_sample:Tuple[MixedTauState,torch.Tensor]):
        """
        Aggregates the statistics to obtain histograms
        """
        # Join context if necesary and defines 
        samples_requiered = self.organize_samples(batch,generative_sample)

        # Counts the discrete variables for the histograms
        if self.has_target_discrete:
            self.discrete_histograms(samples_requiered.noise_discrete_sample,
                                     samples_requiered.original_discrete_sample,
                                     samples_requiered.generative_sample_discrete)
            batch_size = samples_requiered.original_discrete_sample.size(0)

        #elif samples_requiered.original_discrete_sample is not None:
        # Gather the continuous points
        if self.has_target_continuous:
            self.noise_points.append(samples_requiered.noise_continuous_sample)
            self.real_points.append(samples_requiered.original_continuous_sample)
            self.generated_points.append(samples_requiered.generative_sample_continuous)
            batch_size = samples_requiered.original_continuous_sample.size(0)

        elif samples_requiered.original_continuous_sample is not None:
            self.real_points.append(samples_requiered.original_continuous_sample)

        self.sample_size += batch_size

    def final_operation(self,all_metrics,samples_bags,epoch=None):
        """
        After all batches are gathered
        """
        metrics_dict = {}
        # Aggregates statistics for discrete
        # Check if the sum of the histograms is greater than zero before normalization
        if torch.sum(self.noise_histogram) > 0:
            self.noise_histogram = self.noise_histogram / self.sample_size

        if torch.sum(self.generative_histogram) > 0:
            self.generative_histogram = self.generative_histogram / self.sample_size

        if torch.sum(self.real_histogram) > 0:
            self.real_histogram = self.real_histogram / self.sample_size

        if self.has_target_discrete:
            #calculates distance
            hd = self.hellinger_distance(self.generative_histogram,self.real_histogram)
            metrics_dict = {self.name:hd.item()}
        
        # Continuous variables
        if len(self.noise_points) > 0:
            self.noise_points = torch.vstack(self.noise_points)
        if len(self.real_points) > 0:
            self.real_points = torch.vstack(self.real_points)
        if len(self.generated_points) > 0:
            self.generated_points = torch.vstack(self.generated_points)

        # Histogram Plot
        if self.plot_histogram:
            plot_path = None
            if self.has_experiment_files:
                plot_path = self.plots_path.format("dimension_histograms_{0}".format(epoch))
            plot_categorical_histogram_per_dimension(self.noise_histogram,
                                                     self.real_histogram,
                                                     self.generative_histogram,
                                                     save_path=plot_path)

        # Scatter plot
        if self.plot_continuous_variables:
            plot_path = None
            if self.has_experiment_files:
                plot_path = self.plots_path.format("scatter_plot_{0}".format(epoch))
            plot_scatterplot(self.noise_points,
                             self.real_points,
                             self.generated_points,
                             save_path=plot_path)
            
        if epoch is not None:
            self.save_metric(metrics_dict,epoch)

        all_metrics.update(metrics_dict)
        return all_metrics
    
    def hellinger_distance(self,hist1,hist2):
        # Compute the square root of each bin
        sqrt_hist1 = torch.sqrt(hist1)
        sqrt_hist2 = torch.sqrt(hist2)

        # Compute the Euclidean distance between the square-rooted histograms
        euclidean_distance = torch.norm(sqrt_hist1 - sqrt_hist2, 2)

        # Normalize to get the Hellinger distance
        hellinger_distance = euclidean_distance / torch.sqrt(torch.tensor(2.0))
        return hellinger_distance
