
import os
import torch
import pickle
import numpy as np
import networkx as nx
from torchtyping import TensorType
from typing import List,Dict,Tuple,Union
from torch.utils.data import DataLoader
from collections import namedtuple

import os
import torch
import numpy as np
from torch.distributions import Categorical
from torch.utils.data import TensorDataset,DataLoader
from markov_bridges.configs.config_classes.data.molecules_configs import QM9Config
from markov_bridges.configs.config_classes.generative_models.edmg_config import EDMGConfig

from markov_bridges.data.abstract_dataloader import (
    MarkovBridgeDataloader,
    MarkovBridgeDataClass,
    MarkovBridgeDataset,
    MarkovBridgeDataNameTuple
)

from markov_bridges.data.qm9.data.utils import initialize_datasets
from markov_bridges.data.qm9.dataset_config import get_dataset_info

from markov_bridges.utils.graphs_utils import graphs_to_tensor
from markov_bridges.data.transforms import get_transforms,get_expected_shape
from markov_bridges.data.qm9.dataset import retrieve_dataloaders
from markov_bridges.data.qm9.utils import prepare_context, compute_mean_mad

QM9PointDataNameTuple = namedtuple("DatabatchClass", "source_discrete source_continuous target_discrete target_continuous context_discrete context_continuous nodes_dist node_mask edge_mask time")

class QM9PointDataloader(MarkovBridgeDataloader):
    qm9_config : QM9Config
    name:str = "GraphDataloader"
    max_node_num:int 
    expected_shape:List[int]

    def __init__(self,config:EDMGConfig):
        """
        :param config:
        :param device:
        """
        self.qm9_config = config.data
        self.dataset = self.qm9_config.dataset
        self.conditioning =  config.noising_model.conditioning
        self.get_dataloaders(self.qm9_config)

    def get_dataloaders(self,config):
        """
        Creates the dataloaders
        """
        self.dataloaders, self.charge_scale = retrieve_dataloaders(config)
        self.dataset_info = get_dataset_info(self.qm9_config.dataset,self.qm9_config.remove_h)
        
        #self.data_dummy = next(iter(self.dataloaders['train']))
    

        #if len(self.conditioning) > 0:
        #    print(f'Conditioning on {self.conditioning}')
        #    property_norms = compute_mean_mad(self.dataloaders, self.conditioning, self.dataset)
        #    context_dummy = prepare_context(self.conditioning, self.data_dummy, property_norms)
        #    self.context_node_nf = context_dummy.size(2)
        #else:
        #    self.context_node_nf = 0
        #    property_norms = None
        
    def train(self):
        return self.dataloaders["train"]
    
    def test(self):
        return self.dataloaders["test"]
    
    def validation(self):
        return self.dataloaders["valid"]
