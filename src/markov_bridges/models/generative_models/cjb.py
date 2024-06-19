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

from markov_bridges.data.utils import get_dataloaders
from markov_bridges.data.graphs_dataloader import GraphDataloader
from markov_bridges.models.pipelines.pipeline_cjb import CJBPipeline
from markov_bridges.models.metrics.optimal_transport import OTPlanSampler
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataloader
from markov_bridges.data.music_dataloaders import LankhPianoRollDataloader
from markov_bridges.models.generative_models.cjb_rate import ClassificationForwardRate
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig

from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

@dataclass
class CJB:
    """
    This class contains all elements to sample and train a conditional jump bridge model
    
    if DEVICE is not provided it is obtained from the trainer config 

    the actual torch model that contains the networks for sampling is specified in forward rate
    and contains all the mathematical elements.

    the experiment folder is created in experiment files and has to be provided by hand, 
    currently it is passed to the model by the trainer, it is only needed during training
    """
    config: CJBConfig = None
    experiment_dir:str = None

    experiment_files: ExperimentFiles = None
    dataloader: Union[MarkovBridgeDataloader|GraphDataloader|LankhPianoRollDataloader] = None
    forward_rate: Union[ClassificationForwardRate] = None
    op_sampler: OTPlanSampler = None
    pipeline:CJBPipeline = None
    device: torch.device = None
    image_data_path: str = None
    type_of_load:Union[str,int] = "best"

    def __post_init__(self):
        self.loss = nn.CrossEntropyLoss(reduction='none')
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

        self.forward_rate = ClassificationForwardRate(self.config, self.device).to(self.device)
        self.pipeline = CJBPipeline(self.config,self.forward_rate,self.dataloader)

        if self.config.optimal_transport.cost == "log":
            B = self.forward_rate.log_cost_regularizer()
            B = B.item() if isinstance(B,torch.Tensor) else B
            self.config.optimal_transport.method = "sinkhorn"
            self.config.optimal_transport.normalize_cost = True
            self.config.optimal_transport.normalize_cost_constant = float(self.config.data.dimensions)
            reg = 1./B
            print("OT regularizer for Schrodinger Plan {0}".format(reg))
            self.config.optimal_transport.reg = reg

        self.op_sampler = OTPlanSampler(**asdict(self.config.optimal_transport))

    def load_from_experiment(self,experiment_dir,device=None,set_data_path=None):
        self.experiment_files = ExperimentFiles(experiment_dir=experiment_dir)

        results_ = self.experiment_files.load_results(self.type_of_load,
                                                      device=torch.device("cpu"))
        self.forward_rate = results_["model"]
        number_of_test_step = results_["number_of_test_step"]
        number_of_training_steps = results_["number_of_training_steps"]
        epoch  = results_["epoch"]

        config_path_json = json.load(open(self.experiment_files.config_path, "r"))

        if hasattr(config_path_json,"delete"):
            config_path_json["delete"] = False
        self.config = CJBConfig(**config_path_json)
        if set_data_path is not None:
            self.config.data.data_dir = set_data_path
        if device is None:
            self.device = torch.device(self.config.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.config.trainer.epoch = epoch + 1
        self.config.trainer.number_of_test_step = number_of_test_step + 1
        self.config.trainer.number_of_training_steps = number_of_training_steps + 1
        
        self.forward_rate.to(self.device)
        self.dataloader = get_dataloaders(self.config)
        self.pipeline = CJBPipeline(self.config,self.forward_rate,self.dataloader)
        if self.config.optimal_transport.cost == "log":
            B = self.forward_rate.log_cost_regularizer()
            B = B.item() if isinstance(B,torch.Tensor) else B
            self.config.optimal_transport.method = "sinkhorn"
            self.config.optimal_transport.normalize_cost = True
            self.config.optimal_transport.normalize_cost_constant = float(self.config.data.dimensions)
            reg = 1./B
            print("OT regularizer for Schrodinger Plan {0}".format(reg))
            self.config.optimal_transport.reg = reg        
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
                    cost = self.forward_rate.log_cost(x0,x1)

            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)        
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
    x_0 = databatch.source_discrete
    x_1 =  databatch.target_discrete

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
