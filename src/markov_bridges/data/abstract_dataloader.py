import os
import torch
import pickle
import numpy as np
from dataclasses import dataclass
from collections import namedtuple
from markov_bridges import data_path
from torch.utils.data import Dataset, DataLoader


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
    
# Custom Dataset class
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
            data_fields.append(torch.tensor(self.data.source_discrete[idx]))
        if self.data.source_continuous is not None:
            data_fields.append(torch.tensor(self.data.source_continuous[idx]))
        if self.data.target_discrete is not None:
            data_fields.append(torch.tensor(self.data.target_discrete[idx]))
        if self.data.target_continuous is not None:
            data_fields.append(torch.tensor(self.data.target_continuous[idx]))
        if self.data.context_discrete is not None:
            data_fields.append(torch.tensor(self.data.context_discrete[idx]))
        if self.data.context_continuous is not None:
            data_fields.append(torch.tensor(self.data.context_continuous[idx]))

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
    
    def get_sample(self):
        return None