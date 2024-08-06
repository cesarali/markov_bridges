import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import os
import torch
import numpy as np
from torch.distributions import Categorical
from torch.utils.data import TensorDataset,DataLoader
from markov_bridges.configs.config_classes.data.basics_configs import MarkovBridgeDataConfig

from markov_bridges.data.abstract_dataloader import (
    MarkovBridgeDataClass,
    MarkovBridgeDataNameTuple,
    MarkovBridgeDataloader,
    MarkovBridgeDataset
)

from markov_bridges.configs.config_classes.data.sequences_config import SinusoidalConfig

def create_dataset(sample_size, time_steps, K, seed=42):
    # Create a local generator and set the fixed seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    
    # Step 1: Generate Gaussian vectors with the local generator
    gaussian_vectors = torch.randn(sample_size, K, generator=generator)
    
    # Step 2: Define the sine functions
    time = torch.linspace(0, 1, time_steps).unsqueeze(0).unsqueeze(2)
    frequencies = torch.arange(1, K+1).unsqueeze(0).unsqueeze(0)
    sine_functions = torch.sin(2 * torch.pi * frequencies * time)
    
    # Step 3: Multiply Gaussian vectors by sine functions
    data = gaussian_vectors.unsqueeze(1) * sine_functions
    
    # Step 4: Normalize to obtain distributions in the K-simplex
    data = F.softmax(data, dim=2)
    
    return data

def sample_categorical(data,seed=72):
    generator = torch.Generator().manual_seed(seed)

    # Sample from the categorical distribution at each time step for each sample
    sample_size, time_steps, num_states = data.shape
    # Flatten the data to shape (sample_size * time_steps, num_states) for batch sampling
    flat_data = data.view(-1, num_states)
    
    # Perform the sampling in batch
    categorical_samples_flat = torch.multinomial(flat_data, 1,generator=generator).squeeze(1)
    
    # Reshape back to (sample_size, time_steps)
    categorical_samples = categorical_samples_flat.view(sample_size, time_steps)
    
    return categorical_samples

def plot_dataset(data, sample_index=0):
    time_steps = data.shape[1]
    num_states = data.shape[2]
    
    plt.figure(figsize=(14, 7))
    for k in range(num_states):
        plt.plot(range(time_steps), data[sample_index, :, k].numpy(), label=f'State {k+1}')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Softmax Value')
    plt.title('Evolution of each k state over time')
    plt.legend()
    plt.show()

def plot_categorical_samples(categorical_samples, sample_index=0):
    plt.figure(figsize=(14, 3))
    plt.plot(range(categorical_samples.shape[1]), categorical_samples[sample_index].numpy(), marker='o', linestyle='-')
    plt.xlabel('Time Steps')
    plt.ylabel('Sampled State')
    plt.title('Categorical Samples over Time')
    plt.show()

class SinusoidalDataloader(MarkovBridgeDataloader):
    """
    """
    music_config : SinusoidalConfig
    name:str = "SinusoidalDataloader"

    def __init__(self,sinusoidal_config:SinusoidalConfig):
        """
        :param config:
        :param device:
        """
        self.music_config = sinusoidal_config
        self.number_of_spins = self.music_config.discrete_dimensions
        self.get_dataloaders()
        self.define_functions()

    def get_dataloaders(self):
        train_data,test_data = self.get_target_data(self.music_config)

        train_data = self.get_data_divisions(train_data,self.music_config)
        train_data = MarkovBridgeDataset(train_data)

        test_data = self.get_data_divisions(test_data,self.music_config)
        test_data = MarkovBridgeDataset(test_data)


        self.train_dataloader = DataLoader(train_data, batch_size=self.music_config.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data,batch_size=self.music_config.batch_size, shuffle=True)

    def get_target_data(self,data_config:SinusoidalConfig):
        """
        reads the piano data
        """

        train_data = create_dataset(data_config.training_size, data_config.discrete_dimensions, data_config.vocab_size, seed=42)
        test_data = create_dataset(data_config.test_size, data_config.discrete_dimensions, data_config.vocab_size, seed=52)
         
        if data_config.max_training_size is not None:
            train_data = train_data[:min(data_config.max_training_size, len(train_data))]

        if data_config.max_test_size is not None:
            test_data = test_data[:min(data_config.max_test_size, len(test_data))]

        return train_data,test_data
    
    def get_source_data(self,dataset,data_config:MarkovBridgeDataConfig):
        dataset_size = dataset.size(0)
        generation_dimension = data_config.discrete_dimensions - data_config.context_discrete_dimension
        if data_config.source_discrete_type == "uniform":
            vocab_size = data_config.vocab_size
            NoiseDistribution = Categorical(torch.full((vocab_size,),1./vocab_size))
            noise_sample = NoiseDistribution.sample((dataset_size,generation_dimension))
            return noise_sample
        else:
            raise Exception("Source not Implemented")
    
    def get_data_divisions(self,dataset,data_config:MarkovBridgeDataConfig)->MarkovBridgeDataClass:
        """
        divides the data in the different context, source and target
        """
        # source
        source_discrete = self.get_source_data(dataset,data_config)
        target_discrete = sample_categorical(dataset)
        context_continuous = dataset

        # context
        if data_config.context_discrete_dimension > 0:
            context_discrete = target_discrete[:,:data_config.context_discrete_dimension]
        else:
            context_discrete = None
        # target
        target_discrete = target_discrete[:,data_config.context_discrete_dimension:]

        return MarkovBridgeDataClass(source_discrete=source_discrete,
                                     context_continuous=context_continuous,
                                     context_discrete=context_discrete,
                                     target_discrete=target_discrete)

    def descramble(self,samples):
        return samples
    
    def define_functions(self):
        context_dimension = self.music_config.context_discrete_dimension
        self.join_context = lambda context_discrete,data_discrete : torch.cat([context_discrete,data_discrete],dim=1)
        self.remove_context = lambda full_data_discrete : full_data_discrete[:,context_dimension:]
    
if __name__=="__main__":        
    # Parameters
    sample_size = 100  # Number of samples
    time_steps = 50    # Number of time steps
    num_states = 4    # Number of states (K)

    # Create the dataset with a fixed seed
    dataset = create_dataset(sample_size, time_steps, num_states, seed=42)

    # Sample categorical values from the dataset
    categorical_samples = sample_categorical(dataset)

    # Plot the dataset for the first sample
    plot_dataset(dataset, sample_index=0)

    # Plot the categorical samples for the first sample
    plot_categorical_samples(categorical_samples, sample_index=0)
    """
    data_config = SinusoidalConfig()
    dataloader = SinusoidalDataloader(data_config)
    databatch = dataloader.get_databatch()
    print(databatch)
    """
