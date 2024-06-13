import os
from typing import List
from dataclasses import dataclass,asdict,field
from conditional_rate_matching import data_path

image_data_path = os.path.join(data_path,"raw")

@dataclass
class CategoricalDataloaderConfig:
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

