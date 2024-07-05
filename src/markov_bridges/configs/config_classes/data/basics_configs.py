import os
from typing import List
from dataclasses import dataclass,asdict,field
from  markov_bridges import data_path

image_data_path = os.path.join(data_path,"raw")

@dataclass
class MarkovBridgeDataConfig:
    #names 
    name:str = "MarkovBridgeData"
    dataset_name:str = "tensors" 

    # variables model
    has_context_discrete: bool = False    
    has_context_continuous: bool = False

    has_target_discrete: bool = True
    has_target_continuous: bool = False

    source_discrete_type: str = "uniform"
    source_continuous_type: str = None

    # dimensions
    continuos_dimensions: int = 0
    discrete_dimensions: int = 256
    
    context_discrete_dimension:int = 0
    context_continuous_dimension:int = 0

    vocab_size: int = 129
    max_training_size: int = None
    max_test_size:int = None

@dataclass
class IndependentMixConfig(MarkovBridgeDataConfig):
    name:str = "IndependentMix"
    dataset_name:str = "independent_mix" 
    vocab_size: int = 3

    # variables model
    has_context_discrete: bool = False    
    has_context_continuous: bool = False

    has_target_continuous:bool = True
    has_target_discrete:bool = True

    discrete_dimensions:int = 2
    continuos_dimensions:int = 2

    source_discrete_type: str = "uniform"
    source_continuous_type: str = "gaussian"

    target_dirichlet: float = 0.1
    target_probability:list = field(default_factory=lambda:[])
    target_continuous_type: str = "moons" # gaussian,moons,8gaussian

    total_data_size: int = 2500
    train_data_size: int = 2000
    test_data_size: int = 500

    batch_size:int = 32

    def __post_init__(self):
        if self.has_context_continuous:
            self.context_continuous_dimension = 2
            self.has_target_continuous = False
        if self.has_context_discrete:
            self.context_discrete_dimension = 2
            self.has_target_discrete = False
        if self.has_context_continuous and self.has_context_discrete:
            raise ValueError("Both has_context_continuous and has_context_discrete cannot be true at the same time.")

    
@dataclass
class ColorMoonsConfig(MarkovBridgeDataConfig):
    name:str = "colors_moons"
    total_data_size: int = 2000

    vocab_size: int = 4
    continuos_dimensions:int = 2
    discrete_dimensions:int = 1

    has_target_discrete: bool = True
    has_target_continuous: bool = True

    source_dirichlet: float = 0.1 #label,dirichlet
    source_discrete_type: str = "uniform"
    source_continuous_type: str = "8gaussian" # gaussian,moons,8gaussian

    target_dirichlet: float = 0.1
    target_discrete_type: str = "label" #label,dirichlet
    target_continuous_type: str = "moons" # gaussian,moons,8gaussian
    

@dataclass
class CategoricalDataloaderConfig(MarkovBridgeDataConfig):
    name:str = "CategoricalDataloader"
    dataset_name:str = "categorical_dirichlet" # categorical_dirichlet
    data_dir:str = image_data_path

    dirichlet_alpha:float = None
    categorical_probability:float = None

    dimensions: int = 4
    vocab_size: int = 2

    batch_size: int = 23
    test_split: float = 0.2
    total_data_size: int = 60000
    
    temporal_net_expected_shape : List[int] = None
    data_min_max: List[float] = field(default_factory=lambda:[0.,1.])

    def __post_init__(self):
        self.temporal_net_expected_shape =  [self.dimensions]

