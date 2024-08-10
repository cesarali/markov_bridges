import numpy as np
import torch
from tqdm import tqdm
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter

from typing import List
from dataclasses import dataclass,field
from markov_bridges.utils.experiment_files import ExperimentFiles

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR


from markov_bridges.models.deprecated.generative_models.cfm import CFM

import torch
import numpy as np
from torch.optim.adam import Adam
from markov_bridges.models.networks.utils.ema import EMA
from markov_bridges.configs.config_classes.generative_models.cfm_config import CFMConfig
from markov_bridges.models.deprecated.trainers.abstract_trainer import TrainerState,Trainer
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple
from markov_bridges.utils.paralellism import nametuple_to_device

class CFMTrainer(Trainer):

    config: CFMConfig
    generative_model_class = CFM
    name_ = "conditional_flow_matching_trainer"

    def __init__(self, config=None, experiment_files=None, cfm=None, experiment_dir=None, starting_type="last"):
        """
        If experiment dir is provided, he loads the model from that folder and then creates
        a new folder 

        config: configuration file to start model
        cfm: model to train 
        experiment_files: files where to store the experiment
        experiment_dir: if provided experiment dir of model to load to continue training
        starting_type (str,int): for model in experiment_dir, defines which model to load, best, last or checkpoint if int provided

        if experiment_dir is provided, it will ignore config
        """
        if experiment_dir is not None:
            print("Starting Training from Model Provided in Experiment Dirs")
            self.generative_model = CFM(experiment_dir=experiment_dir,type_of_load=starting_type)
            self.generative_model.experiment_files = experiment_files
            self.config = self.generative_model.config
            self.number_of_epochs = self.config.trainer.number_of_epochs
            device_str = self.config.trainer.device
            self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        else:
            self.config = config
            self.number_of_epochs = self.config.trainer.number_of_epochs
            device_str = self.config.trainer.device
            self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
            if cfm is None:
                self.generative_model = CFM(self.config, experiment_files=experiment_files, device=self.device)
            else:
                self.generative_model = cfm
                self.dataloader = self.generative_model.dataloader

    def preprocess_data(self, databatch):    
        return databatch

    def paralellize_model(self):
        pass
    
    def get_model(self):
        return self.generative_model.forward_map.continuous_network
    
    def define_scheduler(self):
        # Check if a scheduler is defined in the configuration
        if self.config.trainer.scheduler is not None:
            if self.config.trainer.scheduler == "step":
                # Define StepLR scheduler
                self.scheduler = StepLR(self.optimizer, 
                                        step_size=self.config.trainer.step_size, 
                                        gamma=self.config.trainer.gamma)
            elif self.config.trainer.scheduler == "multi":
                # Define MultiStepLR scheduler
                self.scheduler = MultiStepLR(self.optimizer, 
                                            milestones=self.config.trainer.milestones, 
                                            gamma=self.config.trainer.gamma)
            elif self.config.trainer.scheduler == "exponential":
                # Define ExponentialLR scheduler
                self.scheduler = ExponentialLR(self.optimizer, 
                                            gamma=self.config.trainer.gamma)
            elif self.config.trainer.scheduler == "reduce":
                # Define ReduceLROnPlateau scheduler
                self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                                mode='min', 
                                                factor=self.config.trainer.factor, 
                                                patience=self.config.trainer.patience)
        
    def initialize(self):
        """
        Obtains initial loss to know when to save, restart the optimizer
        :return:
        """
        if isinstance(self.generative_model.forward_map, EMA) and self.config.trainer.do_ema:
            self.do_ema = True

        self.generative_model.start_new_experiment()

        #DEFINE OPTIMIZERS
        self.optimizer = Adam(self.generative_model.forward_map.parameters(),
                              lr=self.config.trainer.learning_rate,
                              weight_decay=self.config.trainer.weight_decay)
        
        self.define_scheduler()
        self.lr = self.config.trainer.learning_rate

        return np.inf

    def train_step(self,databatch:MarkovBridgeDataNameTuple, number_of_training_step, epoch):
        # time selection
        databatch = nametuple_to_device(databatch, self.device)

        # sample bridge
        continuous_sample = self.generative_model.forward_map.sample_bridge(databatch)

        # loss
        loss_ = self.generative_model.forward_map.loss(databatch, continuous_sample)
    
        # optimization
        self.optimizer.zero_grad()
        loss_.backward()

        # clip grad norm
        if self.config.trainer.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.generative_model.forward_map.parameters(), self.config.trainer.clip_max_norm)

        self.optimizer.step()

        # learning rate schedulers
        self.handle_scheduler(number_of_training_step,loss_)

        if self.do_ema:
            self.generative_model.forward_map.update_ema()

        self.writer.add_scalar('training loss', loss_.item(), number_of_training_step)
        return loss_

    def test_step(self,databatch:MarkovBridgeDataNameTuple, number_of_test_step,epoch):
        with torch.no_grad():
            # gpu handling
            databatch = nametuple_to_device(databatch, self.device)

            # data pair and time sample
            continuous_sample = self.generative_model.forward_map.sample_bridge(databatch)

            # sample x from z
            loss_ = self.generative_model.forward_map.loss(databatch, continuous_sample)
        
            self.writer.add_scalar('test loss', loss_.item(), number_of_test_step)

        return loss_

    def handle_scheduler(self,number_of_training_step,loss_):
        # this is the Cambel solution to the learning rate
        if self.config.trainer.warm_up > 0:
            for g in self.optimizer.param_groups:
                new_lr = self.lr * np.minimum(float(number_of_training_step+1) / self.config.trainer.warm_up, 1.0)
                g['lr'] = new_lr
        
        # after warm up finish start scheduler with last learning rate 
        if number_of_training_step == self.config.trainer.warm_up:
            # Capture the learning rate at the end of warm-up
            base_lr = self.optimizer.param_groups[0]['lr']
            # Reinitialize the optimizer and scheduler with the new base_lr
            self.optimizer = Adam(self.generative_model.forward_map.parameters(),
                                  lr=base_lr,
                                  weight_decay=self.config.trainer.weight_decay)
            self.define_scheduler()

        #after warm up call schedulers
        if self.config.trainer.warm_up > 0:
            if number_of_training_step > self.config.trainer.warm_up:
                # Update the learning rate scheduler based on its type
                if self.config.trainer.scheduler in ["step", "multi", "exponential"]:
                    self.scheduler.step()
                elif self.config.trainer.scheduler == "reduce":
                    self.scheduler.step(loss_)
            
