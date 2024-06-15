import torch
import torch.nn.functional as F
from markov_bridges.models.generative_models.cjb import CJB

#===============================================
# METRICS CLASSES
#===============================================

from markov_bridges.models.pipelines.pipeline_cjb import CJBPipelineOutput
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

from markov_bridges.models.metrics.abstract_metrics import BasicMetric

class HellingerMetric(BasicMetric):
    """
    Calculates the Hellinger distance between the histograms of generated and real data
    """
    def __init__(self,model:CJB,metrics_config):
        if model.config.data.has_context_discrete:
            self.join_context = model.dataloader.join_context
            self.has_context_discrete = True

        self.dimensions = model.config.data.dimensions
        self.vocab_size = model.config.data.vocab_size

        self.sample_size = 0
        self.metrics_config = metrics_config

        self.generative_histogram = torch.zeros((self.dimensions,self.vocab_size))
        self.real_histogram = torch.zeros((self.dimensions,self.vocab_size))

    def batch_operation(self,batch:MarkovBridgeDataNameTuple,generative_sample:CJBPipelineOutput):
        batch_size = batch.target_discrete.size(0)
        self.sample_size += batch_size

        if self.has_context_discrete:
            original_sample = self.join_context(batch.context_discrete,batch.target_discrete)

        # reshapce for one hot
        original_sample = original_sample.reshape(-1)
        raw_sample = generative_sample.raw_sample.reshape(-1)

        # obtain one hot
        generative_histogram = F.one_hot(raw_sample.long(), num_classes=self.vocab_size)
        original_histogram = F.one_hot(original_sample.long(),  num_classes=self.vocab_size)

        # reshape for batches
        generative_histogram.reshape(batch_size,self.dimensions,self.vocab_size)
        original_histogram.reshape(batch_size,self.dimensions,self.vocab_size)

        # sum the values
        self.generative_histogram += generative_histogram.sum(axis=0)
        self.real_histogram += original_histogram.sum(axis=0)
        
    def final_operation(self):
        self.generative_histogram = self.generative_histogram/self.sample_size
        self.real_histogram = self.real_histogram/self.sample_size
        hd = self.hellinger_distance(self.generative_histogram,self.real_histogram)
        self.save_metric()

    def hellinger_distance(self,hist1,hist2):
        # Compute the square root of each bin
        sqrt_hist1 = torch.sqrt(hist1)
        sqrt_hist2 = torch.sqrt(hist2)

        # Compute the Euclidean distance between the square-rooted histograms
        euclidean_distance = torch.norm(sqrt_hist1 - sqrt_hist2, 2)

        # Normalize to get the Hellinger distance
        hellinger_distance = euclidean_distance / torch.sqrt(torch.tensor(2.0))
        return hellinger_distance
