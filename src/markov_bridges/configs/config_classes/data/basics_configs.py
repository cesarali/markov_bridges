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
    has_target_discrete: bool = True
    has_target_continuous: bool = False

    source_discrete_type: str = "uniform"
    source_continuous_type: str = None

    # dimensions
    discrete_dimensions: int = 256
    continuos_dimensions: int = 0

    vocab_size: int = 129
    max_training_size: int = None
    max_test_size:int = None

    fields:list = field(default_factory=lambda:[])

    num_workers: int = 0
    pin_memory: bool = False

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

    context_discrete_dimension:int=0
    context_continuous_dimension:int=0

    source_discrete_type: str = "uniform"
    source_continuous_type: str = "gaussian"

    target_dirichlet: float = 0.5
    target_probability:list = field(default_factory=lambda:[])
    target_continuous_type: str = "moons" # gaussian,moons,8gaussian

    total_data_size: int = 3500
    train_data_size: int = 2000
    test_data_size: int = 1500

    batch_size:int = 64
     
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
class GaussiansConfig(MarkovBridgeDataConfig):
    name:str = "Gaussians"
    dataset_name:str = "Ngaussians" 
    vocab_size: int = 0

    # variables model
    has_context_discrete: bool = False    
    has_context_continuous: bool = False
    has_target_continuous:bool = True
    has_target_discrete: bool = False
    discrete_dimensions: int = 0
    continuos_dimensions: int = 2

    context_discrete_dimension: int = 0
    context_continuous_dimension: int = 0

    source_continuous_type: str = "gaussian"
    target_continuous_type: str = "Ngaussians"
    number_of_gaussians: int = 8
    gauss_std: float = 0.1

    total_data_size: int = 22000
    train_data_size: int = 20000
    test_data_size: int = 2000

    batch_size: int = 256

    def __post_init__(self):
        self.vocab_size = self.number_of_gaussians
        if self.has_target_discrete: 
            self.discrete_dimensions = 1
        if self.has_context_discrete: 
            self.context_discrete_dimension = 1
            self.has_target_discrete = False