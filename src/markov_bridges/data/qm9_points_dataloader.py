
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
from markov_bridges.configs.config_classes.data.basics_configs import MarkovBridgeDataConfig

from markov_bridges.data.abstract_dataloader import (
    MarkovBridgeDataloader,
    MarkovBridgeDataClass,
    MarkovBridgeDataset,
    MarkovBridgeDataNameTuple
)

from markov_bridges.data.qm9.data.utils import initialize_datasets
from markov_bridges.utils.graphs_utils import graphs_to_tensor
from markov_bridges.data.transforms import get_transforms,get_expected_shape

QM9PointDataNameTuple = namedtuple("DatabatchClass", "source_discrete source_continuous target_discrete target_continuous context_discrete context_continuous nodes_dist node_mask edge_mask time")

class QM9PointDataloader(MarkovBridgeDataloader):
    qm9_config : QM9Config
    name:str = "GraphDataloader"
    max_node_num:int 
    expected_shape:List[int]

    def __init__(self,qm9_config:QM9Config):
        """
        :param config:
        :param device:
        """
        self.qm9_config = qm9_config
        self.get_dataloaders()

    def transform_to_native_shape(self,data:Union[QM9PointDataNameTuple,torch.Tensor])->Union[QM9PointDataNameTuple,torch.Tensor]:
        """
        # Convert named tuple to a dictionary
        data_dict = data._asdict()
        DataNamedTuple = type(data) 

        # Update the dictionary entries
        data_dict['source_discrete'] = self.inverse_transform_list(data.source_discrete)
        data_dict['target_discrete'] = self.inverse_transform_list(data.target_discrete)

        # Convert the dictionary back to the named tuple
        updated_data = DataNamedTuple(**data_dict)
        return updated_data
        """
        #if isinstance(data,torch.Tensor):
        #    return self.inverse_transform_list(data)
        return None
    
    def get_dataloaders(self):
        """
        Creates the dataloaders
        """
        train_data,test_data = self.get_target_data(self.qm9_config)

        self.dimension,self.expected_shape = get_expected_shape(self.max_node_num,
                                                                self.qm9_config.flatten,
                                                                self.qm9_config.full_adjacency)
        
        self.qm9_config.temporal_net_expected_shape = self.expected_shape
        self.qm9_config.number_of_nodes = self.max_node_num
        self.qm9_config.discrete_dimensions = self.dimension

        train_data = self.get_data_divisions(train_data,self.qm9_config)
        train_data = MarkovBridgeDataset(train_data)

        test_data = self.get_data_divisions(test_data,self.qm9_config)
        test_data = MarkovBridgeDataset(test_data)

        self.train_dataloader = DataLoader(train_data, batch_size=self.qm9_config.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data,batch_size=self.qm9_config.batch_size, shuffle=True)

    def get_target_data(self,data_config:MarkovBridgeDataConfig):
        """
        reads the data files
        """
        train_graph_list, test_graph_list,max_number_of_nodes,min_number_of_nodes = self.read_graph_lists()
        train_data = graphs_to_tensor(train_graph_list,max_number_of_nodes)
        test_data = graphs_to_tensor(test_graph_list,max_number_of_nodes)

        self.max_node_num = max_number_of_nodes
        self.min_node_num = min_number_of_nodes

        return train_data,test_data
    
    def get_source_data(self,dataset,data_config:MarkovBridgeDataConfig):
        dataset_size = dataset.size(0)
        if data_config.source_discrete_type == "uniform":
            vocab_size = data_config.vocab_size
            NoiseDistribution = Categorical(torch.full((vocab_size,),1./vocab_size))
            noise_sample = NoiseDistribution.sample((dataset_size,self.dimension)).float()
            return noise_sample
        else:
            raise Exception("Source not Implemented")
    
    def get_data_divisions(self,dataset,data_config:MarkovBridgeDataConfig)->MarkovBridgeDataClass:
        """
        divides the data in the different context, source and target
        """
        # preprocess data
        target_discrete = self.transform_list(dataset)

        # source
        source_discrete = self.get_source_data(target_discrete,data_config)

        return MarkovBridgeDataClass(source_discrete=source_discrete,
                                     target_discrete=target_discrete)
    
    def read_molecules(self)->Tuple[List[nx.Graph]]:
        """
        :return: train_graph_list, test_graph_list
        """
        datasets, num_species, charge_scale = initialize_datasets(self.qm9_config.datadir, 
                                                                  self.qm9_config.dataset,
                                                                  subtract_thermo=self.qm9_config.subtract_thermo,
                                                                  force_download=self.qm9_config.force_download,
                                                                  remove_h=self.qm9_config.remove_h)
        
        return train_graph_list, test_graph_list,max_number_of_nodes,min_number_of_nodes
