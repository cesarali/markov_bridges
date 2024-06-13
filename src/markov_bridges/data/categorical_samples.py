import torch
from pprint import pprint
from dataclasses import dataclass,asdict
from torch.distributions import Categorical
from torch.utils.data import TensorDataset,DataLoader
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

def set_probabilities(config:StatesDataloaderConfig,return_tensor_samples=False):
    """

    :param probs:
    :param alpha:
    :param sample_size:
    :param dimensions:
    :param vocab_size:
    :param test_split:
    :return:
    """
    probs = config.bernoulli_probability
    alpha = config.dirichlet_alpha
    sample_size = config.sample_size
    dimensions = config.dimensions
    vocab_size = config.vocab_size

    # ensure we have the probabilites
    if probs is None:
        if alpha is not None:
            if isinstance(alpha, float):
                alpha = torch.full((vocab_size,), alpha)
            else:
                assert len(alpha.shape) == 1
                assert alpha.size(0) == vocab_size
                # Sample from the Dirichlet distribution
            probs = torch.distributions.Dirichlet(alpha).sample([dimensions])
        else:
            probs = torch.ones((dimensions,vocab_size))*1./vocab_size
    else:
        if isinstance(probs,(np.ndarray,list)):
            probs = torch.Tensor(probs)
        probs = probs.squeeze()
        assert probs.max() <= 1.
        assert probs.max() >= 0.

    return probs

class CategoricalDataset(Dataset):
    def __init__(self, total_samples, probabilities):
        """
        Initializes the dataset object with total number of samples and category probabilities.
        
        Args:
            total_samples (int): Total number of samples to generate.
            probabilities (list): List of probabilities for each category.
        """
        super().__init__()
        self.total_samples = total_samples
        self.probabilities = probabilities
        self.distribution_per_dimension = Categorical(self.probabilities)
        
    def __len__(self):
        """ Returns the total number of samples in the dataset. """
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        Generates a random sample from the categorical distribution based on defined probabilities.
        
        Args:
            idx (int): Index of the sample (unused, as samples are i.i.d)
        
        Returns:
            int: A randomly selected category index.
        """
        sample = torch.multinomial(self.probabilities, 1).squeeze(1).float()
        return [sample]

class StatesDataloader:
    """

    """
    def __init__(self, config:StatesDataloaderConfig):
        """
        Initializes the StatesDataloaders2 object with configuration settings.
        
        Args:
            config (object): Configuration object containing dataset parameters.
        """
        self.config = config
        self.batch_size = config.batch_size
        self.test_split = config.test_split
        self.max_test_size = config.max_test_size
        total_samples = config.total_data_size

        # Define probabilities
        self.probs = set_probabilities(config)
        self.config.bernoulli_probability = self.probs.numpy().tolist()

        # Splitting dataset into train and test
        test_size = int(total_samples * self.test_split)
        if self.max_test_size is not None:
            test_size = min(test_size, self.max_test_size)
        train_size = total_samples - test_size
        
        train_dataset = CategoricalDataset(train_size, self.probs)
        test_dataset = CategoricalDataset(test_size, self.probs)

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,drop_last=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,drop_last=True)

    def train(self):
        """ Returns a DataLoader for the training dataset. """
        return self.train_dataloader 
    
    def test(self):
        """ Returns a DataLoader for the testing dataset. """
        return self.test_dataloader

if __name__=="__main__":
    from  conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
    data_config = StatesDataloaderConfig(bernoulli_probability=None,dirichlet_alpha=None)
    dataloader = StatesDataloader(data_config)
    databatch = next(dataloader.train().__iter__())
    x_ = databatch
    print(x_[0].shape)
