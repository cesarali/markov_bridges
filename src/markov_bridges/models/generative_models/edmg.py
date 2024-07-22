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
from markov_bridges.data.qm9_points_dataloader import QM9PointDataloader
from markov_bridges.models.pipelines.pipeline_cmb import CMBPipeline

from markov_bridges.data.abstract_dataloader import MarkovBridgeDataloader
from markov_bridges.models.generative_models.edmg_noising import EquivariantDiffussionNoising
from markov_bridges.configs.config_classes.generative_models.edmg_config import EDMGConfig

from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple
from markov_bridges.models.networks.temporal.edmg.edmg_utils import get_edmg_model
from markov_bridges.models.networks.temporal.edmg.en_diffusion import EnVariationalDiffusion

@dataclass
class EDMG:
    """
    This class contains all elements to sample and train a conditional jump bridge model
    
    if DEVICE is not provided it is obtained from the trainer config 

    the actual torch model that contains the networks for sampling is specified in forward map
    and contains all the mathematical elements.

    the experiment folder is created in experiment files and has to be provided, 
    currently it is passed to the model by the trainer, it is only needed during 
    training
    """
    config: EDMGConfig = None
    experiment_dir:str = None

    experiment_files: ExperimentFiles = None
    dataloader: Union[QM9PointDataloader] = None
    noising_model: Union[EnVariationalDiffusion] = None
    pipeline:CMBPipeline = None
    device: torch.device = None
    image_data_path: str = None
    type_of_load:Union[str,int] = "last"

    def __post_init__(self):
        if self.experiment_dir is not None:
            self.load_from_experiment(self.experiment_dir,self.device,self.image_data_path)
        elif self.config is not None:
            self.initialize_from_config(config=self.config,device=self.device)

    def initialize_from_config(self,config,device):
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
        
        self.noising_model,self.nodes_dist, self.prop_dist = get_edmg_model(config,
                                                                            self.dataloader.dataset_info,
                                                                            self.dataloader.dataloaders['train'],
                                                                            device)

        self.pipeline = None

    def load_from_experiment(self,experiment_dir,device=None,set_data_path=None):
        self.experiment_files = ExperimentFiles(experiment_dir=experiment_dir)

        # get config and device
        config_path_json = json.load(open(self.experiment_files.config_path, "r"))
        if hasattr(config_path_json,"delete"):
            config_path_json["delete"] = False
        self.config = EDMGConfig (**config_path_json)
        if device is None:
            self.device = torch.device(self.config.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        # get dataloader
        if set_data_path is not None:
            self.config.data.data_dir = set_data_path
        self.dataloader = get_dataloaders(self.config)

        # load model
        results_ = self.experiment_files.load_results(self.type_of_load,device=torch.device("cpu"))
        number_of_test_step = results_["number_of_test_step"]
        number_of_training_steps = results_["number_of_training_steps"]
        epoch  = results_["epoch"]

        self.config.trainer.epoch = epoch + 1
        self.config.trainer.number_of_test_step = number_of_test_step + 1
        self.config.trainer.number_of_training_steps = number_of_training_steps + 1

        # set forward model
        self.noising_model = results_["model"]
        self.noising_model = self.noising_model.to(self.device)

        # define pipeline
        self.pipeline = None

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