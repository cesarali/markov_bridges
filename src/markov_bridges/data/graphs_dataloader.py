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
from markov_bridges.configs.config_classes.data.graphs_configs import GraphDataloaderGeometricConfig
from markov_bridges.configs.config_classes.data.basics_configs import MarkovBridgeDataConfig

from markov_bridges.data.abstract_dataloader import (
    MarkovBridgeDataloader,
    MarkovBridgeDataClass,
    MarkovBridgeDataset,
    MarkovBridgeDatasetNew,
    MarkovBridgeDataNameTuple
)

from markov_bridges.utils.graphs_utils import graphs_to_tensor
from markov_bridges.data.transforms import get_transforms,get_expected_shape
from markov_bridges.utils.shapes import nodes_and_edges_masks
from markov_bridges.data.utils import sample_moons

GraphDataNameTuple = namedtuple("DatabatchClass", "num_nodes source_discrete target_discrete node_mask edges_mask context_continuous time")

class GraphDataloader(MarkovBridgeDataloader):

    graph_config : GraphDataloaderGeometricConfig
    name:str = "GraphDataloader"
    max_node_num:int 
    dataset_info:dict
    expected_shape:List[int]

    def __init__(self,graph_config:GraphDataloaderGeometricConfig):
        """
        :param config:
        :param device:
        """
        MarkovBridgeDataloader.__init__(self,graph_config)
        self.graph_config = graph_config
        self.data_batch_keys = graph_config.data_batch_keys
        self.transform_list,self.inverse_transform_list = get_transforms(graph_config)
        
        self.get_dataloaders()
        if len(self.conditioning) > 0:
            self.property_norms = self.get_property_norms()

    def transform_to_native_shape(self,data:Union[GraphDataNameTuple,torch.Tensor])->Union[GraphDataNameTuple,torch.Tensor]:
        """
        """
        if isinstance(data,torch.Tensor):
            return self.inverse_transform_list(data)
        
        # Convert named tuple to a dictionary
        data_dict = data._asdict()
        DataNamedTuple = type(data) 

        # Update the dictionary entries
        data_dict['source_discrete'] = self.inverse_transform_list(data.source_discrete)
        data_dict['target_discrete'] = self.inverse_transform_list(data.target_discrete)

        # Convert the dictionary back to the named tuple
        updated_data = DataNamedTuple(**data_dict)
        return updated_data
    
    def networkx_from_sample(self,adj_matrices):
        # GET GRAPH FROM GENERATIVE MODEL
        graph_list = []
        number_of_graphs = adj_matrices.shape[0]
        adj_matrices = adj_matrices.detach().cpu().numpy()
        for graph_index in range(number_of_graphs):
            graph_ = nx.from_numpy_array(adj_matrices[graph_index])
            graph_list.append(graph_)
        return graph_list
    
    def get_dataloaders(self):
        """
        Creates the dataloaders
        """
        train_data,test_data = self.get_target_data(self.graph_config)

        self.dimension,self.expected_shape = get_expected_shape(self.max_node_num,
                                                                self.graph_config.flatten,
                                                                self.graph_config.full_adjacency)
        
        self.graph_config.temporal_net_expected_shape = self.expected_shape
        self.graph_config.number_of_nodes = self.max_node_num
        self.graph_config.discrete_dimensions = self.dimension
        self.graph_config.discrete_generation_dimension = self.dimension

        train_data = self.get_data_divisions(train_data,self.graph_config)
        train_data = MarkovBridgeDatasetNew(train_data)

        test_data = self.get_data_divisions(test_data,self.graph_config)
        test_data = MarkovBridgeDatasetNew(test_data)

        self.train_dataloader = DataLoader(train_data, batch_size=self.graph_config.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data,batch_size=self.graph_config.batch_size, shuffle=True)
        self.validation_dataloader = self.test_dataloader
        
        self.fields = test_data.data_batch_keys
        self.DatabatchNameTuple = namedtuple("DatabatchClass", self.fields)
        self.graph_config.fields = self.fields

    def get_target_data(self,data_config:MarkovBridgeDataConfig):
        """
        reads the data files
        """
        train_graph_list, test_graph_list,max_number_of_nodes,min_number_of_nodes = self.read_graph_lists()
        if "num_nodes" in self.data_batch_keys:
            number_of_node_train = torch.Tensor([graph.number_of_nodes() for graph in train_graph_list]).long()
            number_of_node_test = torch.Tensor([graph.number_of_nodes() for graph in test_graph_list]).long()
            max_ = max(number_of_node_train.max().item(),number_of_node_test.max().item())
            # Count the occurrences of each value in the tensor
            unique_values, counts = torch.unique(number_of_node_train, return_counts=True)
            # Combine unique values with their respective counts
            value_counts = dict(zip(unique_values.tolist(), counts.tolist()))
            self.dataset_info = {'n_nodes':value_counts,
                                 'max_n_nodes':max_}
        else:
            number_of_node_train = None
            number_of_node_test = None

        train_data = graphs_to_tensor(train_graph_list,max_number_of_nodes)
        test_data = graphs_to_tensor(test_graph_list,max_number_of_nodes)

        self.max_node_num = max_number_of_nodes
        self.min_node_num = min_number_of_nodes

        return (train_data,number_of_node_train),(test_data,number_of_node_test)
    
    def get_source_data(self,dataset,data_config:MarkovBridgeDataConfig):
        dataset_size = dataset.size(0)
        if data_config.source_discrete_type == "uniform":
            vocab_size = data_config.vocab_size
            NoiseDistribution = Categorical(torch.full((vocab_size,),1./vocab_size))
            noise_sample = NoiseDistribution.sample((dataset_size,self.dimension)).float()
            return noise_sample
        else:
            raise Exception("Source not Implemented")
    
    def get_data_divisions(self,dataset,data_config:MarkovBridgeDataConfig)->GraphDataNameTuple:
        """
        divides the data in the different context, source and target
        """
        graph_tensors = dataset[0]
        graph_sizes = dataset[1]

        # preprocess data
        target_discrete = self.transform_list(graph_tensors)

        # source
        source_discrete = self.get_source_data(target_discrete,data_config)

        # data dict
        data_dict = {"target_discrete":target_discrete,
                     "source_discrete":source_discrete}
        
        # add sizes
        if graph_sizes is not None:
            data_dict.update({"num_nodes":graph_sizes})

        # add masks
        if "node_mask" in self.data_batch_keys:
            max_n_nodes = graph_tensors.size(-1)
            node_mask,edge_mask = nodes_and_edges_masks(graph_sizes,max_n_nodes)
            data_dict.update({"node_mask":node_mask,
                              "edges_mask":edge_mask})
        
        # continuous context
        if "context_continuous" in self.data_batch_keys:
            sample_size = target_discrete.size(0)
            number_of_nodes = target_discrete.size(1)
            points = torch.normal(0.,1.,size=(sample_size,))
            data_dict.update({"context_continuous":points})

        return data_dict
    
    def read_graph_lists(self)->Tuple[List[nx.Graph]]:
        """
        :return: train_graph_list, test_graph_list
        """
        data_dir = self.graph_config.data_dir
        file_name = self.graph_config.dataset_name
        file_path = os.path.join(data_dir, file_name)
        with open(file_path + '.pkl', 'rb') as f:
            graph_list = pickle.load(f)
        test_size = int(self.graph_config.test_split * len(graph_list))

        all_node_numbers = list(map(lambda x: x.number_of_nodes(), graph_list))

        max_number_of_nodes = max(all_node_numbers)
        min_number_of_nodes = min(all_node_numbers)

        train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]
        return train_graph_list, test_graph_list,max_number_of_nodes,min_number_of_nodes
