import os
import torch
import pickle
import numpy as np
import networkx as nx
from torchtyping import TensorType
from typing import List,Dict,Tuple,Union
from torch.utils.data import TensorDataset, DataLoader
from conditional_rate_matching.utils.graph_utils import init_features, graphs_to_tensor
from conditional_rate_matching.data.graph_dataloaders_config import GraphDataloaderConfig
import torchvision.transforms as transforms


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
    MarkovBridgeDataNameTuple
)

from markov_bridges.utils.graphs_utils import graphs_to_tensor
from markov_bridges.data.transforms import get_transforms,get_expected_shape

class GraphDataloader(MarkovBridgeDataloader):
    """
    """
    graph_config : GraphDataloaderGeometricConfig
    name:str = "GraphDataloader"
    max_node_num:int 
    expected_shape:List[int]

    def __init__(self,graph_config:GraphDataloaderGeometricConfig):
        """
        :param config:
        :param device:
        """
        self.graph_config = graph_config
        self.transform_list,self.inverse_transform_list = get_transforms(graph_config)
        self.get_dataloaders()

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
        self.graph_config.dimensions = self.dimension

        train_data = self.get_data_divisions(train_data,self.graph_config)
        train_data = MarkovBridgeDataset(train_data)

        test_data = self.get_data_divisions(test_data,self.graph_config)
        test_data = MarkovBridgeDataset(test_data)

        self.train_dataloader = DataLoader(train_data, batch_size=self.graph_config.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data,batch_size=self.graph_config.batch_size, shuffle=True)

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
            noise_sample = NoiseDistribution.sample((dataset_size,self.dimension))
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

class GraphDataloaders:
    """

    """
    graph_data_config : GraphDataloaderConfig
    name:str = "GraphDataloaders"

    def __init__(self,graph_data_config,bridge_data=None):
        """
        :param config:
        :param device:
        """
        self.graph_data_config = graph_data_config
        self.number_of_spins = self.graph_data_config.dimensions

        transform_list,inverse_transform_list = get_transforms(self.graph_data_config)
        self.composed_transform = transforms.Compose(transform_list)
        self.transform_to_graph = transforms.Compose(inverse_transform_list)

        train_graph_list, test_graph_list,max_node_num,min_number_of_nodes = self.read_graph_lists(bridge_data)
        if graph_data_config.max_training_size is not None:
            train_graph_list = [train_graph_list[i] for i in range(min(graph_data_config.max_training_size,len(train_graph_list)))]

        if graph_data_config.max_test_size is not None:
            test_graph_list = [test_graph_list[i] for i in range(min(graph_data_config.max_test_size,len(test_graph_list)))]

        self.training_data_size = len(train_graph_list)
        self.test_data_size = len(test_graph_list)
        self.total_data_size = self.training_data_size + self.test_data_size

        self.graph_data_config.training_size = self.training_data_size
        self.graph_data_config.test_size = self.test_data_size
        self.graph_data_config.total_data_size = self.total_data_size
        self.graph_data_config.training_proportion = float(self.training_data_size)/self.total_data_size

        train_adjs_tensor,train_x_tensor = self.graph_to_tensor_and_features(train_graph_list,
                                                                             self.graph_data_config.init,
                                                                             max_node_num,
                                                                             self.graph_data_config.max_feat_num)
        test_adjs_tensor, test_x_tensor = self.graph_to_tensor_and_features(test_graph_list,
                                                                            self.graph_data_config.init,
                                                                            max_node_num,
                                                                            self.graph_data_config.max_feat_num)


        train_adjs_tensor = self.composed_transform(train_adjs_tensor)
        self.train_dataloader_ = self.create_dataloaders(train_adjs_tensor,train_x_tensor)

        test_adjs_tensor = self.composed_transform(test_adjs_tensor)
        self.test_dataloader_ = self.create_dataloaders(test_adjs_tensor,test_x_tensor)

        self.fake_time_ = torch.rand(self.graph_data_config.batch_size)

    def train(self):
        return self.train_dataloader_

    def test(self):
        return self.test_dataloader_

    def create_dataloaders(self,x_tensor, adjs_tensor):
        train_ds = TensorDataset(x_tensor, adjs_tensor)
        train_dl = DataLoader(train_ds,
                              batch_size=self.graph_data_config.batch_size,
                              shuffle=True)
        return train_dl

    def sample_to_graph(self, x_sample):
        """
        undo the transformations

        :param x_sample:
        :return:
        """
        sample_size = x_sample.size(0)
        temporal_net_expected_shape = self.graph_data_config.temporal_net_expected_shape
        expected_shape = [sample_size] + temporal_net_expected_shape
        expected_shape = torch.Size(expected_shape)
        if x_sample.shape != expected_shape:
            x_sample = x_sample.reshape(expected_shape)
        adj_matrices = self.transform_to_graph(x_sample)

        # GET GRAPH FROM GENERATIVE MODEL
        graph_list = []
        number_of_graphs = adj_matrices.shape[0]
        adj_matrices = adj_matrices.detach().cpu().numpy()
        for graph_index in range(number_of_graphs):
            graph_ = nx.from_numpy_array(adj_matrices[graph_index])
            graph_list.append(graph_)

        return graph_list


    def graph_to_tensor_and_features(self,
                                     graph_list:List[nx.Graph],
                                     init:str="zeros",
                                     max_node_num:int=None,
                                     max_feat_num:int=10):
        """
        :return:adjs_tensor,x_tensor
        """
        if max_node_num is None:
            max_node_num = max([g.number_of_nodes() for g in graph_list])
        adjs_tensor = graphs_to_tensor(graph_list,max_node_num)
        x_tensor = init_features(init,adjs_tensor,max_feat_num)
        return adjs_tensor,x_tensor

 

