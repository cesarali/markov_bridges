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
from abc import ABC,abstractmethod

#MarkovBridgeDataNameTuple is the named tuple with all possible datavalues, but this changes for each dataset
MarkovBridgeDataNameTuple = namedtuple("DatabatchClass", "source_discrete source_continuous target_discrete target_continuous context_discrete context_continuous time")

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
    fields.append("time")
    DatabatchNameTuple = namedtuple("DatabatchClass", fields)
    return DatabatchNameTuple,fields

@dataclass
class MarkovBridgeDataClass:
    context_discrete: torch.tensor = None    
    context_continuous: torch.tensor = None

    source_discrete: torch.tensor = None
    source_continuous: torch.tensor = None

    target_discrete: torch.tensor = None
    target_continuous: torch.tensor = None
    
    context_discrete_dimension:int = None
    context_continuous_dimension:int = None
    discrete_dimension:int = None
    continuous_dimension:int = None

    def __post_init__(self):
        """
        We ensure that data is always saved [batch_size,dimensions]
        """
        fields = ['context_discrete', 'context_continuous', 'source_discrete', 'source_continuous', 'target_discrete', 'target_continuous']
        for field_name in fields:
            tensor = getattr(self, field_name)
            if tensor is not None:
                batch_size = tensor.shape[0]
                new_shape = (batch_size, -1)
                setattr(self, field_name, tensor.view(new_shape))

class MarkovBridgeDataset(Dataset):
    """
    Custom Dataset class for MarkovBridge models.
    """
    def __init__(self, data: MarkovBridgeDataClass):
        super(MarkovBridgeDataset, self).__init__()
        if data.target_discrete is not None:
            self.num_samples = data.target_discrete.size(0)
        if data.target_continuous is not None:
            self.num_samples = data.target_continuous.size(0)
        self.data = data
        self.DatabatchNameTuple,self.fields = create_databatch_nametuple(data)
        
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

        data_fields.append(torch.rand((1,)))

        databatch = self.DatabatchNameTuple(*data_fields)
        return databatch

class MarkovBridgeDataloader(ABC):
    """
    This is the abstract dataloader class for the markov bridge models including:

    Conditional Markov Bridge
    Conditonal Mixed Bridges
    Conditional Flow Matching

    Data should be stored in MarkovBridgeDataset as a MarkovBridgeDataClass
    
    ALL DATA SHOULD BE STORE IN SHAPE
    [data_size,dimensions]

    The function transform_to_native_shape should transform the shape to what is needed
    for postprocessing

    The MarkovBridgeDataset uses MarkovBridgeDataClass as a way of storing the whole data
    this deafults to None the non provided aspects of the data.

    The dataloaders uses a **namedtuple** for handling  the data in the batches,
    this named tuple changes if the data set in fact does not provide some elements 
    like context or continous variables, this such that we avoid a batch of
    nan tensors. 

    This behavior is different from the MarkovBridgeDataClass who defaults to None for 
    the things not provided.

    source denotes the distribution at time = 0 
    target denotes the distribution at time = 1
    """
    train_dataloader:DataLoader = None
    test_dataloader:DataLoader = None
    validation_dataloader:DataLoader = None
    
    def __init__(self,config:None):
        if config is not None:
            self.has_context_discrete = config.has_context_discrete    
            self.has_context_continuous = config.has_context_continuous
        
            self.has_target_continuous = config.has_target_continuous
            self.has_target_discrete = config.has_target_discrete

            self.discrete_dimensions = config.discrete_dimensions
            self.continuos_dimensions = config.continuos_dimensions
            self.batch_size = config.batch_size
    
    def get_databach_keys(self):
        return None
    
    def get_source_data(self):
        return None
    
    def get_target_data(self):
        return None
        
    def get_data_divisions(self)->MarkovBridgeDataClass:
        """
        This class is supposed to organize the data in the different elements once 
        target and source are either read or simulated, i.e. it defines this elements

        context_discrete: torch.tensor    
        context_continuous: torch.tensor
        source_discrete: torch.tensor
        source_continuous: torch.tensor
        target_discrete: torch.tensor
        target_continuous: torch.tensor
        """
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
    
    def validation(self):
        return self.validation_dataloader
    
    def get_data_sample(self,sample_size:int,train:bool=True)->MarkovBridgeDataNameTuple:
        """
        Aggregates data from the dataloader until the sample_size is met.

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
        if train:
            dataloader_iterator = self.train()
        else:
            dataloader_iterator = self.test()
        
        def safe_append(tensor_list, tensor):
            if tensor is not None:
                tensor_list.append(tensor)
        
        context_discrete = []
        context_continuous = []
        source_discrete = []
        source_continuous = []
        target_discrete = []
        target_continuous = []
        time = []

        size_left = sample_size

        for databatch in dataloader_iterator:
            databatch:MarkovBridgeDataNameTuple
            batch_size = self.batch_size # databatch.source_discrete.size(0)
            take_size = min(size_left, batch_size)
            
            safe_append(context_discrete, databatch.context_discrete[:take_size] if hasattr(databatch, 'context_discrete') else None)
            safe_append(context_continuous, databatch.context_continuous[:take_size] if hasattr(databatch, 'context_continuous') else None)
            safe_append(source_discrete, databatch.source_discrete[:take_size] if hasattr(databatch, 'source_discrete') else None)
            safe_append(source_continuous, databatch.source_continuous[:take_size] if hasattr(databatch, 'source_continuous') else None)
            safe_append(target_discrete, databatch.target_discrete[:take_size] if hasattr(databatch, 'target_discrete') else None)
            safe_append(target_continuous, databatch.target_continuous[:take_size] if hasattr(databatch, 'target_continuous') else None)
            safe_append(time, databatch.time[:take_size] if hasattr(databatch, 'time') else None)

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
                                                  context_discrete, context_continuous, 
                                                  time)
        return aggregated_batch
    
    def transform_to_native_shape(self)->MarkovBridgeDataNameTuple:
        """
        Remember that all data needed by the generative models requiere shape
        batch_size,dimensions

        however, for encoding or postprocessing 
        the data typically requieres being express in a different shape such as 

        graphs: [batch_size,number_of_nodes,number_of_nodes]
        images: [batch_size,number_of_channels,height,width]

        this function should transform [batch_size,dimensions] back to the requiered native shape
        """
        pass
        
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

