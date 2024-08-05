

import torch
import numpy as np
from torch.distributions import Normal
from torch.utils.data import DataLoader

from markov_bridges.configs.config_classes.data.basics_configs import GaussiansConfig
from torch.utils.data import DataLoader

from markov_bridges.data.abstract_dataloader import (
    MarkovBridgeDataClass,
    MarkovBridgeDataNameTuple,
    MarkovBridgeDataloader,
    MarkovBridgeDataset
)


class GaussiansDataloader(MarkovBridgeDataloader):

    """ Creates the dataloaders for N 2D Gaussians on the unit circle
    """

    def __init__(self, config: GaussiansConfig):
        super().__init__(config)
        self.data_config = config
        self.has_context_discrete = config.has_context_discrete
        self.has_target_discrete = config.has_target_discrete
        self.get_dataloaders()

    def get_source_data(self, data_size):
        gaussian_probability = Normal(0., self.data_config.gauss_std) 
        x_0 = gaussian_probability.sample((data_size, self.data_config.continuos_dimensions))
        labels = torch.randint(low=0, high=self.data_config.number_of_gaussians, size=(data_size,))
        source_continuous = x_0
        source_discrete = labels if self.has_target_discrete else None
        return source_continuous, source_discrete
    
    def get_target_data(self, data_size):
        N = self.data_config.number_of_gaussians
        x_1, labels = generate_N_gaussians(num_gaussians=N, 
                                           num_points_per_gaussian=data_size/N,  
                                           std_dev=self.data_config.gauss_std)        
        
        target_continuous = x_1
        target_discrete = labels if self.has_target_discrete else None
        context_continuous = None
        context_discrete = labels if self.has_context_discrete else None

        return target_continuous, target_discrete, context_continuous, context_discrete
    
    def get_data_divisions(self, type="train") -> MarkovBridgeDataClass:
        """
        This class is supposed to organize the data in the different elements once 
        target and source are either read or simulated, i.e. it defines this elements

        context_discrete: torch.tensor    
        context_continuous: torch.tensor
        source_discrete: torch.tensor
        source_continuous: torch.tensor
        target_discrete: torch.tensor
        target_continuous: torch.tensor
        """
        if type == "train":
            target_continuous_train, target_discrete_train, context_continuous_train, context_discrete_train = self.get_target_data(self.data_config.train_data_size)
            source_continuous_train, source_discrete_train = self.get_source_data(self.data_config.train_data_size)
            
            data = MarkovBridgeDataClass(source_continuous=source_continuous_train,
                                         target_continuous=target_continuous_train, 
                                         source_discrete=source_discrete_train, 
                                         target_discrete=target_discrete_train,
                                         context_continuous=context_continuous_train,
                                         context_discrete=context_discrete_train)

            return data
        
        elif type == "test":
            target_continuous_test, target_discrete_test, context_continuous_test, context_discrete_test = self.get_target_data(self.data_config.test_data_size)
            source_continuous_test, source_discrete_test = self.get_source_data(self.data_config.test_data_size)
            
            data = MarkovBridgeDataClass(source_continuous=source_continuous_test,
                                         target_continuous=target_continuous_test, 
                                         source_discrete=source_discrete_test, 
                                         target_discrete=target_discrete_test,
                                         context_continuous=context_continuous_test,
                                         context_discrete=context_discrete_test)

            return data
        
    def get_dataloaders(self):
        """
        Creates the dataloaders
        """
        train_data = self.get_data_divisions(type="train")
        train_data = MarkovBridgeDataset(train_data)

        test_data = self.get_data_divisions(type="test")
        test_data = MarkovBridgeDataset(test_data)

        self.fields = test_data.fields
        self.train_dataloader = DataLoader(train_data, batch_size=self.data_config.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=self.data_config.batch_size, shuffle=False)

    def join_context(self, databatch: MarkovBridgeDataNameTuple, discrete_data=None, continuous_data=None):
        full_continuous = databatch.context_continuous if continuous_data is None else continuous_data 
        full_discrete = databatch.context_discrete if discrete_data is None else discrete_data 
        return full_discrete, full_continuous

def generate_N_gaussians(num_gaussians=8, num_points_per_gaussian=1000, std_dev=0.1):

    num_points_per_gaussian = int(num_points_per_gaussian)
    angle_step = 2 * np.pi / num_gaussians
    data = []
    labels = []

    for i in range(num_gaussians):
        angle = i * angle_step
        center_x = np.cos(angle)
        center_y = np.sin(angle)
        points = np.random.randn(num_points_per_gaussian, 2) * std_dev
        points += np.array([center_x, center_y])
        data.append(points)
        labels += [i] * num_points_per_gaussian

    data = np.concatenate(data, axis=0)
    data = torch.tensor(data, dtype=torch.float32)
    labels = np.array(labels)
    labels = torch.tensor(labels, dtype=torch.long)

    return shuffle_data(data, labels)

def shuffle_data(pos, labels):
    combined = list(zip(pos, labels))
    np.random.shuffle(combined)
    pos, labels = zip(*combined)
    pos = torch.stack(pos)
    labels = torch.tensor(labels, dtype=torch.long).unsqueeze(1) 
    return pos, labels