from pathlib import Path
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
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig

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

QM9PointDataNameTuple = namedtuple("DatabatchClass", "source_discrete source_continuous target_discrete target_continuous atom_mask edge_mask context time")

class QM9PointDataloader(MarkovBridgeDataloader):
    
    qm9_config : QM9Config
    name:str = "QM9PointDataloader"
    max_node_num:int 
    expected_shape:List[int]

    def __init__(self,config:EDMGConfig|CMBConfig):
        """
        :param config:
        :param device:
        """
        self.qm9_config = config.data
        self.dataset = self.qm9_config.dataset
        if hasattr(config,"noising_model"):
            self.conditioning =  config.noising_model.conditioning
        elif hasattr(config,"mixed_network"):
            self.conditioning =  config.mixed_network.conditioning
        else:
            raise Exception("No Noising or mixed model network")
        self.get_dataloaders(config)
    
    def get_databatch(self):
        datadir_path = Path(self.qm9_config.datadir) / self.qm9_config.dataset
        dummy_path = datadir_path / "dummy_batch.tr"
        #==================================================
        if os.path.exists(dummy_path):
            data_dummy = torch.load(dummy_path)
        else:
            data_dummy = self.get_databatch()
            torch.save(data_dummy,dummy_path)
        return data_dummy

    def get_databach_keys(self):
        return self.keys
    
    def get_dataloaders(self,config:EDMGConfig):
        """
        Creates the dataloaders
        """
        self.keys,self.dataloaders, self.charge_scale = retrieve_dataloaders(self.qm9_config)
        self.keys.extend(['one_hot', 'atom_mask', 'edge_mask'])
        
        self.dataset_info = get_dataset_info(self.qm9_config.dataset,self.qm9_config.remove_h)
        config.data.vocab_size = len(self.dataset_info['atom_decoder'])
        context_node_nf,property_norms = self.get_context_node_nf()
        if hasattr(config,"noising_model"):
            config.noising_model.context_node_nf = context_node_nf
        elif hasattr(config,"data"):
            config.data.context_node_nf = context_node_nf
            config.data.property_norms = property_norms
            config.data.discrete_dimensions = self.dataset_info["max_n_nodes"]
    
    def train(self):
        return self.dataloaders["train"]
    
    def test(self):
        return self.dataloaders["test"]
    
    def validation(self):
        return self.dataloaders["valid"]
    
    def get_context_node_nf(self):
        data_dummy = self.get_databatch()
        if len(self.conditioning) > 0:
            print(f'Conditioning on {self.conditioning}')
            self.property_norms = compute_mean_mad(self, 
                                                   self.conditioning, 
                                                   self.qm9_config.dataset)
            context_dummy = prepare_context(self.conditioning, data_dummy, self.property_norms)
            context_node_nf = context_dummy.size(2)
        else:
            context_node_nf = 0
            self.property_norms = None
        return context_node_nf,self.property_norms


QM9PointDataNameTupleCMB = namedtuple("DatabatchClass", "source_discrete source_continuous target_discrete target_continuous atom_mask edge_mask context time batch_size max_num_atoms")
        