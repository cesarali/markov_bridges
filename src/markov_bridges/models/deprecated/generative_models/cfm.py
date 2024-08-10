import json
import torch
from torch import nn
from dataclasses import dataclass
from torch.utils.data import DataLoader

from typing import Union
from dataclasses import asdict

import numpy as np
from torch.nn.functional import softmax
from markov_bridges.utils.experiment_files import ExperimentFiles

from markov_bridges.data.dataloaders_utils import get_dataloaders
from markov_bridges.data.gaussians2D_dataloaders import GaussiansDataloader
from markov_bridges.models.pipelines.pipeline_cfm import CFMPipeline

from markov_bridges.models.metrics.optimal_transport import OTPlanSampler
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataloader
from markov_bridges.models.deprecated.generative_models.cfm_forward import ContinuousForwardMap
from markov_bridges.configs.config_classes.generative_models.cfm_config import CFMConfig

from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

@dataclass
class CFM:
    """
    This class contains all elements to sample and train a conditional flow-matching model
    
    if DEVICE is not provided it is obtained from the trainer config 

    the actual torch model that contains the networks for sampling is specified in forward map
    and contains all the mathematical elements.

    the experiment folder is created in experiment files and has to be provided, 
    currently it is passed to the model by the trainer, it is only needed during 
    training
    """
    config: CFMConfig = None
    experiment_dir:str = None

    experiment_files: ExperimentFiles = None
    dataloader: Union[MarkovBridgeDataloader|GaussiansDataloader] = None
    forward_map: ContinuousForwardMap = None
    op_sampler: OTPlanSampler = None
    pipeline: CFMPipeline = None
    device: torch.device = None
    image_data_path: str = None
    type_of_load: Union[str,int] = "last"

    def __post_init__(self):
        if self.experiment_dir is not None:
            self.load_from_experiment(self.experiment_dir,self.device,self.image_data_path)
        elif self.config is not None:
            self.initialize_from_config(config=self.config,device=self.device)

    def initialize_from_config(self, config, device):
        # =====================================================
        # DATA STUFF
        # =====================================================
        self.dataloader = get_dataloaders(config)
        # =========================================================
        # Initialize
        # =========================================================
        if device is None:
            self.device = torch.device(self.config.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        
        self.forward_map = ContinuousForwardMap(self.config, self.device).to(self.device)
        self.pipeline = CFMPipeline(self.config,self.forward_map,self.dataloader)
        self.op_sampler = OTPlanSampler(**asdict(self.config.optimal_transport))

        if self.config.trainer.paralellize_gpu:
            self.forward_map.continuous_network = self.forward_map.continuous_network.to("cuda:0")
            self.forward_map.continuous_network = torch.nn.DataParallel(self.forward_map.continuous_network)
        else:
            self.forward_map.continuous_network = self.forward_map.continuous_network.to(self.device)


    def load_from_experiment(self, experiment_dir, device=None, set_data_path=None):
        self.experiment_files = ExperimentFiles(experiment_dir=experiment_dir)

        # get config and device
        config_path_json = json.load(open(self.experiment_files.config_path, "r"))
        if hasattr(config_path_json, "delete"):
            config_path_json["delete"] = False
        self.config = CFMConfig(**config_path_json)
        if device is None:
            self.device = torch.device(self.config.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        # get dataloader
        if set_data_path is not None:
            self.config.data.data_dir = set_data_path
        self.dataloader = get_dataloaders(self.config)

        # load model
        results_ = self.experiment_files.load_results(self.type_of_load, device=torch.device("cpu"))
        number_of_test_step = results_["number_of_test_step"]
        number_of_training_steps = results_["number_of_training_steps"]
        epoch  = results_["epoch"]

        self.config.trainer.epoch = epoch + 1
        self.config.trainer.number_of_test_step = number_of_test_step + 1
        self.config.trainer.number_of_training_steps = number_of_training_steps + 1

        # set forward model
        self.forward_map = ContinuousForwardMap(self.config, self.device).to(self.device)
        self.forward_map.continuous_network = results_["model"]
        self.forward_map = self.forward_map.to(self.device)
        
        if self.config.trainer.paralellize_gpu:
            self.forward_map.continuous_network = self.forward_map.continuous_network.to("cuda:0")
            self.forward_map.continuous_network = torch.nn.DataParallel(self.forward_map.continuous_network)
        else:
            self.forward_map.continuous_network = self.forward_map.continuous_network.to(self.device)

        # define pipeline
        self.pipeline = CFMPipeline(self.config,self.forward_map,self.dataloader)

        # define ot sampler
        self.op_sampler = OTPlanSampler(**asdict(self.config.optimal_transport))

        return epoch,number_of_training_steps,number_of_test_step
    
    
    def start_new_experiment(self):
        #create directories
        self.experiment_files.create_directories()

        #align configs
        self.align_configs()

        #save config
        config_as_dict = asdict(self.config)
        with open(self.experiment_files.config_path, "w") as file:
            json.dump(config_as_dict, file, indent=4)

    def align_configs(self):
        pass

    def sample_pair(self,databatch,seed=None):
        """
        data is returned with shape [batch_size,dimension]
        """
        x1,x0 = uniform_pair_x0_x1(databatch)

        if self.config.optimal_transport.name == "OTPlanSampler":
            cost=None
            if self.config.optimal_transport.cost == "log":
                with torch.no_grad():
                    cost = self.forward_map.log_cost(x0,x1)

            x0, x1 = self.op_sampler.sample_plan(x0, x1, replace=False,cost=cost)

        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        return x1,x0

def uniform_pair_x0_x1(databatch:MarkovBridgeDataNameTuple):
    """
    Most simple Z sampler

    :param batch_1:
    :param batch_0:

    :return:x_1, x_0
    """
    x_0 = databatch.source_continuous
    x_1 =  databatch.target_continuous

    batch_size_0 = x_0.size(0)
    batch_size_1 = x_1.size(0)

    batch_size = min(batch_size_0, batch_size_1)

    x_0 = x_0[:batch_size]
    x_1 = x_1[:batch_size]

    x_1 = x_1.float()
    x_0 = x_0.float()
    
    batch_size = x_0.shape[0]
    x_0 = x_0.reshape(batch_size,-1)
    x_1 = x_1.reshape(batch_size,-1)

    return x_1, x_0
