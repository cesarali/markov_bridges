import os
from conditional_rate_matching import data_path
from dataclasses import dataclass,asdict,field
from typing import List, Union, Optional, Tuple, Dict
from markov_bridges.configs.config_classes.data.basics_configs import MarkovBridgeDataConfig

graph_data_path = os.path.join(data_path,"raw","graph")

#=============================================================================
# graph geometric
#=============================================================================

@dataclass
class GraphDataloaderGeometricConfig(MarkovBridgeDataConfig):
    name:str = "GraphDataloaderGeometric"
    data_dir:str = graph_data_path

    dataset_name: str =None
    batch_size: int=None
    data_dir:str = graph_data_path
    
    full_adjacency:bool = False
    flatten:bool = True

    max_node_num: int=None
    max_feat_num: int=None

    dimensions: int = None
    vocab_size: int = 2

    total_data_size:int = None
    training_size:int = None
    test_size:int = None
    test_split: float=None

    max_training_size:int =None
    max_test_size:int=None

    temporal_net_expected_shape : List[int] = None
    data_min_max: List[float] = field(default_factory=lambda:[0.,1.])


@dataclass
class EgoGConfig(GraphDataloaderGeometricConfig):
    dataset_name: str = "ego_small"
    batch_size: int = 20
    test_split: float = 0.2
    max_node_num: int = 17
    max_feat_num: int = 17
    total_data_size:int = 200

@dataclass
class CommunitySmallGConfig(GraphDataloaderGeometricConfig):
    dataset_name: str = 'community_small'
    batch_size: int = 20
    test_split: float = 0.2
    max_node_num: int = 20
    max_feat_num: int = 10
    total_data_size:int = 200

@dataclass
class CommunityGConfig(GraphDataloaderGeometricConfig):
    dataset_name: str = 'community'
    batch_size: int = 32
    test_split: float = 0.2
    max_node_num: int = 11
    max_feat_num: int = 10
    total_data_size:int = 1000

@dataclass
class GridGConfig(GraphDataloaderGeometricConfig):
    dataset_name: str = 'grid'
    batch_size: int = 32
    test_split: float = 0.2
    max_node_num: int = 361
    max_feat_num: int = 5
    total_data_size:int = 200

@dataclass
class EnzymesGConfig(GraphDataloaderGeometricConfig):
    dataset_name: str = 'ENZYMES'
    batch_size: int = 64
    test_split: float = 0.2
    max_node_num: int = 125
    max_feat_num: int = 10
    total_data_size:int = 587

dataset_str_to_config = {'ENZYMES':EnzymesGConfig,
                         "grid":GridGConfig,
                         "community_small":CommunitySmallGConfig}