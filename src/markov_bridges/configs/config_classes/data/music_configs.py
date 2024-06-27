import os
from typing import List
from markov_bridges import data_path
from dataclasses import dataclass,field

from markov_bridges.configs.config_classes.data.basics_configs import MarkovBridgeDataConfig

data_path = str(os.path.join(data_path,"raw"))


@dataclass
class LakhPianoRollConfig(MarkovBridgeDataConfig):
    name:str = "LakhPianoRoll"
    dataset_name:str = "lakh_roll" # emnist, fashion, mnist

    batch_size: int= 32
    data_dir:str = data_path

    discrete_dimensions: int = 256
    vocab_size: int = 129

    context_discrete_dimension:int = 32
    context_continuous_dimension:int = 32

    discrete_generation_dimension:int = 224
    
    total_data_size: int = 6973
    training_size: int = 6000
    test_size: int = 973

    temporal_net_expected_shape : List[int] = None
    data_min_max: List[float] = field(default_factory=lambda:[0.,128.])

    def __post_init__(self):
        self.discrete_dimensions, self.temporal_net_expected_shape = self.discrete_dimensions, [self.discrete_dimensions]
        self.number_of_labels = None
        self.test_split = self.test_size/float(self.total_data_size)
        self.discrete_generation_dimension = self.discrete_dimensions - self.context_discrete_dimension
