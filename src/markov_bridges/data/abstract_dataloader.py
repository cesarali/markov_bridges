import os
import torch
import pickle
import numpy as np
from dataclasses import dataclass
from collections import namedtuple
from markov_bridges import data_path
from torch.utils.data import Dataset, DataLoader

import torch
from collections import namedtuple
from typing import List

MarkovBridgeDataNameTuple = namedtuple("DatabatchClass", "source_discrete source_continuous target_discrete target_continuous context_discrete context_continuous")

def create_databatch_nametuple(data):
    fields = []
    if data.source_discrete is not None:
        fields.append("source_discrete")
    if data.source_continuous is not None:
        fields.append("source_continuous")
    if data.target_discrete is not None:
        fields.append("target_discrete")
    if data.target_continuous is not None:
        fields.append("target_continuous")
    if data.context_discrete is not None:
        fields.append("context_discrete")
    if data.context_continuous is not None:
        fields.append("context_continuous")
    
    DatabatchNameTuple = namedtuple("DatabatchClass", fields)
    return DatabatchNameTuple

@dataclass
class MarkovBridgeDataClass:
    context_discrete: torch.tensor = None    
    context_continuous: torch.tensor = None

    source_discrete: torch.tensor = None
    source_continuous: torch.tensor = None

    target_discrete: torch.tensor = None
    target_continuous: torch.tensor = None
    
class MarkovBridgeDataset(Dataset):
    """
    Custom Dataset class for MarkovBridge models.
    """
    def __init__(self, data: MarkovBridgeDataClass):
        super(MarkovBridgeDataset, self).__init__()

        self.num_samples = data.target_discrete.size(0)
        self.data = data
        self.DatabatchNameTuple = create_databatch_nametuple(data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Collect data fields based on availability
        data_fields = []
        if self.data.source_discrete is not None:
            data_fields.append(self.data.source_discrete[idx])
        if self.data.source_continuous is not None:
            data_fields.append(self.data.source_continuous[idx])
        if self.data.target_discrete is not None:
            data_fields.append(self.data.target_discrete[idx])
        if self.data.target_continuous is not None:
            data_fields.append(self.data.target_continuous[idx])
        if self.data.context_discrete is not None:
            data_fields.append(self.data.context_discrete[idx])
        if self.data.context_continuous is not None:
            data_fields.append(self.data.context_continuous[idx])

        databatch = self.DatabatchNameTuple(*data_fields)
        return databatch

class MarkovBridgeDataloader:
    """
    """
    def __init__(self):
        return None
    
    def get_source_data(self):
        return None
    
    def get_target_data(self):
        return None
    
    def get_data_divisions(self)->MarkovBridgeDataClass:
        return None
    
    def get_databatch(self,train=True)->MarkovBridgeDataNameTuple:
        if train:
            databatch = next(self.train().__iter__())
        else:
            databatch = next(self.test().__iter__())
        return databatch
    
    def train(self):
        return self.train_dataloader
    
    def test(self):
        return self.test_dataloader
    
    def get_data_sample(self,sample_size:int,train:bool=True)->MarkovBridgeDataNameTuple:
        """
        Samples data from the dataloader until the sample_size is met.

        Args:
            dataloader_iterator (Iterator): The dataloader iterator to sample from.
            sample_size (int): The total number of samples to collect.

        Returns:

            namedtuple: A named tuple containing the collected samples.

            FullDatabatchNameTuple = 
            namedtuple("DatabatchClass", 
                       "source_discrete source_continuous 
                        target_discrete target_continuous 
                        context_discrete context_continuous")

        """
        def safe_append(tensor_list, tensor):
            if tensor is not None:
                tensor_list.append(tensor)
        
        context_discrete = []
        context_continuous = []
        source_discrete = []
        source_continuous = []
        target_discrete = []
        target_continuous = []

        size_left = sample_size

        for databatch in self.train():
            batch_size = databatch.context_discrete.size(0)
            take_size = min(size_left, batch_size)
            
            safe_append(context_discrete, databatch.context_discrete[:take_size] if hasattr(databatch, 'context_discrete') else None)
            safe_append(context_continuous, databatch.context_continuous[:take_size] if hasattr(databatch, 'context_continuous') else None)
            safe_append(source_discrete, databatch.source_discrete[:take_size] if hasattr(databatch, 'source_discrete') else None)
            safe_append(source_continuous, databatch.source_continuous[:take_size] if hasattr(databatch, 'source_continuous') else None)
            safe_append(target_discrete, databatch.target_discrete[:take_size] if hasattr(databatch, 'target_discrete') else None)
            safe_append(target_continuous, databatch.target_continuous[:take_size] if hasattr(databatch, 'target_continuous') else None)

            size_left -= take_size
            if size_left <= 0:
                break

        context_discrete = torch.vstack(context_discrete) if context_discrete else None
        context_continuous = torch.vstack(context_continuous) if context_continuous else None
        source_discrete = torch.vstack(source_discrete) if source_discrete else None
        source_continuous = torch.vstack(source_continuous) if source_continuous else None
        target_discrete = torch.vstack(target_discrete) if target_discrete else None
        target_continuous = torch.vstack(target_continuous) if target_continuous else None

        aggregated_batch = MarkovBridgeDataNameTuple(source_discrete, source_continuous, 
                                                  target_discrete, target_continuous,
                                                  context_discrete, context_continuous)
        return aggregated_batch
    
    def repeat_interleave_data(sample, repeat_sample=0)->MarkovBridgeDataNameTuple:
        if repeat_sample > 0:
            source_discrete = sample.source_discrete.repeat_interleave(repeat_sample, dim=0) if sample.source_discrete is not None else None
            source_continuous = sample.source_continuous.repeat_interleave(repeat_sample, dim=0) if sample.source_continuous is not None else None
            target_discrete = sample.target_discrete.repeat_interleave(repeat_sample, dim=0) if sample.target_discrete is not None else None
            target_continuous = sample.target_continuous.repeat_interleave(repeat_sample, dim=0) if sample.target_continuous is not None else None
            context_discrete = sample.context_discrete.repeat_interleave(repeat_sample, dim=0) if sample.context_discrete is not None else None
            context_continuous = sample.context_continuous.repeat_interleave(repeat_sample, dim=0) if sample.context_continuous is not None else None

            repeated_sample = MarkovBridgeDataNameTuple(source_discrete, source_continuous, 
                                                    target_discrete, target_continuous,
                                                    context_discrete, context_continuous)
            return repeated_sample
        return sample
