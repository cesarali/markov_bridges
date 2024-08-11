import os
import torch
import numpy as np
from torch.distributions import Categorical
from torch.utils.data import TensorDataset,DataLoader
from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.data.basics_configs import MarkovBridgeDataConfig
from collections import namedtuple

from markov_bridges.data.abstract_dataloader import (
    MarkovBridgeDataloader,
    MarkovBridgeDataClass,
    MarkovBridgeDataset,
    MarkovBridgeDataNameTuple
)

class LankhPianoRollDataloader(MarkovBridgeDataloader):
    """
    """
    music_config : LakhPianoRollConfig
    name:str = "LankhPianoRollDataloader"

    def __init__(self,music_config:LakhPianoRollConfig):
        """
        :param config:
        :param device:
        """
        self.music_config = music_config
        self.number_of_spins = self.music_config.discrete_dimensions
        self.get_dataloaders()
        self.define_functions()

    def get_dataloaders(self):
        train_data,test_data,descramble_key = self.get_target_data(self.music_config)

        train_data = self.get_data_divisions(train_data,self.music_config)
        train_data = MarkovBridgeDataset(train_data)

        test_data = self.get_data_divisions(test_data,self.music_config)
        test_data = MarkovBridgeDataset(test_data)

        self.fields = test_data.fields
        self.DatabatchNameTuple = namedtuple("DatabatchClass", self.fields)
        self.music_config.fields = self.fields

        self.descramble_key = descramble_key

        self.train_dataloader = DataLoader(train_data, batch_size=self.music_config.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data,batch_size=self.music_config.batch_size, shuffle=True)
        self.validation_dataloader = self.test_dataloader

    def get_target_data(self,data_config:MarkovBridgeDataConfig):
        """
        reads the piano data
        """
        data_path = data_config.data_dir
        train_datafile = os.path.join(data_path , "pianoroll_dataset", "train.npy")
        test_datafile = os.path.join(data_path, "pianoroll_dataset", "test.npy")
        train_data = np.load(train_datafile)
        test_data = np.load(test_datafile)

        if data_config.max_training_size is not None:
            train_data = train_data[:min(data_config.max_training_size, len(train_data))]

        if data_config.max_test_size is not None:
            test_data = test_data[:min(data_config.max_test_size, len(test_data))]

        descramble_datafile = os.path.join(data_path, "pianoroll_dataset", "descramble_key.txt")
        descramble_key = np.loadtxt(descramble_datafile)
        return torch.Tensor(train_data),torch.Tensor(test_data),descramble_key
    
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

        # context
        if data_config.context_discrete_dimension > 0:
            context_discrete = dataset[:,:data_config.context_discrete_dimension]
        else:
            context_discrete = None
        # target
        target_discrete = dataset[:,data_config.context_discrete_dimension:]

        return MarkovBridgeDataClass(source_discrete=source_discrete,
                                    context_discrete=context_discrete,
                                    target_discrete=target_discrete)

    def descramble(self,samples):
        return self.descramble_key[samples.flatten().astype(int)].reshape(*samples.shape)
    
    def define_functions(self):
        context_dimension = self.music_config.context_discrete_dimension
        self.join_context = lambda context_discrete,data_discrete : torch.cat([context_discrete,data_discrete],dim=1)
        self.remove_context = lambda full_data_discrete : full_data_discrete[:,context_dimension:]

if __name__=="__main__":

    music_config = LakhPianoRollConfig()
    music_dataloader = LankhPianoRollDataloader(music_config)
    
    databatch = music_dataloader.get_databatch()
    data_sample = music_dataloader.get_data_sample(47)
    
