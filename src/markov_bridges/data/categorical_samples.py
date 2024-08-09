import torch
from pprint import pprint
from dataclasses import dataclass,asdict
from torch.distributions import Categorical
from torch.utils.data import TensorDataset,DataLoader
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig
from collections import namedtuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from markov_bridges.data.abstract_dataloader import (
    MarkovBridgeDataClass,
    MarkovBridgeDataNameTuple,
    MarkovBridgeDataloader,
    MarkovBridgeDataset
)

from torch.utils.data import DataLoader
from markov_bridges.data.utils import sample_8gaussians, sample_moons
from torch.distributions import Categorical,Normal,Dirichlet
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig

def set_probabilities(config:IndependentMixConfig,return_tensor_samples=False):
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
    def __init__(self, config):
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

class IndependentMixDataloader(MarkovBridgeDataloader):

    def __init__(self,config:IndependentMixConfig):
        super().__init__(config)
        self.data_config = config
        self.has_context_continuous = self.data_config.has_context_continuous
        self.has_context_discrete = self.data_config.has_context_discrete
        self.get_dataloaders()

    def get_source_data(self,data_size):
        #Discrete
        uniform_probability = torch.full((self.data_config.vocab_size,),1./self.data_config.vocab_size)
        source_discrete = Categorical(uniform_probability).sample((data_size,self.data_config.discrete_dimensions))

        #Continuous
        gaussian_probability = Normal(0.,1.)
        source_continuous = gaussian_probability.sample((data_size,self.data_config.continuos_dimensions))
        return source_discrete,source_continuous
    
    def get_target_data(self,total_size):
        # Continuous
        if self.data_config.target_continuous_type == "8gaussian":
            X_continuous,_ = sample_8gaussians(total_size)
        elif self.data_config.target_continuous_type == "moons":
            X_continuous,_ = sample_moons(total_size)

        # Discrete
        if len(self.data_config.target_probability) == 0:
            dirichlet_alpha = torch.full((self.data_config.vocab_size,),self.data_config.target_dirichlet)
            probability_per_dimension = Dirichlet(dirichlet_alpha).sample((self.data_config.discrete_dimensions,))    
            self.data_config.target_probability = probability_per_dimension.tolist()
        else:
            probability_per_dimension = torch.Tensor(self.data_config.target_probability)
            
        X_discrete  = Categorical(probability_per_dimension).sample((total_size,))
        return X_continuous,X_discrete
    
    def get_data_divisions(self,type="train")->MarkovBridgeDataClass:
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
            X_continuous_train,X_discrete_train = self.get_target_data(self.data_config.train_data_size)
            source_discrete_train,source_continuous_train = self.get_source_data(self.data_config.train_data_size)
            if self.has_context_continuous:
                data = MarkovBridgeDataClass(context_continuous=X_continuous_train,
                                            source_discrete=source_discrete_train,
                                             target_discrete=X_discrete_train)
                return data
            if self.has_context_discrete:
                data = MarkovBridgeDataClass(context_discrete=X_discrete_train,
                                             source_continuous=source_continuous_train,
                                             target_continuous=X_continuous_train)
                return data
            data = MarkovBridgeDataClass(source_discrete=source_discrete_train,
                                         source_continuous=source_continuous_train,
                                         target_discrete=X_discrete_train,
                                         target_continuous=X_continuous_train)
            return data
        
        elif type == "test":
            X_continuous_test,X_discrete_test = self.get_target_data(self.data_config.test_data_size)
            source_discrete_test,source_continuous_test = self.get_source_data(self.data_config.test_data_size)

            if self.has_context_continuous:
                data = MarkovBridgeDataClass(context_continuous=X_continuous_test,
                                             source_discrete=source_discrete_test,
                                             target_discrete=X_discrete_test)
                return data
            if self.has_context_discrete:
                data = MarkovBridgeDataClass(context_discrete=X_discrete_test,
                                             source_continuous=source_continuous_test,
                                             target_continuous=X_continuous_test)
                return data
            data = MarkovBridgeDataClass(source_discrete=source_discrete_test,
                                         source_continuous=source_continuous_test,
                                         target_discrete=X_discrete_test,
                                         target_continuous=X_continuous_test)
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
        self.DatabatchNameTuple = namedtuple("DatabatchClass", self.fields)

        self.train_dataloader = DataLoader(train_data, batch_size=self.data_config.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data,batch_size=self.data_config.batch_size, shuffle=True)

        self.data_config.fields = self.fields

    def join_context(self,databatch:MarkovBridgeDataNameTuple,discrete_data=None,continuous_data=None):
        if self.has_context_continuous:
            context_continuous = databatch.context_continuous
            full_continuous = context_continuous
        else:
            full_continuous = continuous_data

        if self.has_context_discrete:
            context_discrete = databatch.context_discrete
            full_discrete = context_discrete
        else:
            full_discrete = discrete_data
        return full_discrete,full_continuous
    
if __name__=="__main__":
    data_config = IndependentMixConfig(has_context_discrete=True)
    dataloader = IndependentMixDataloader(data_config)

    databatch = dataloader.get_databatch()
    print(databatch.context_discrete.shape)
    
