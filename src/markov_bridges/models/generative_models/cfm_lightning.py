import torch
import numpy as np
from torch import nn
from torch.nn.functional import softmax
from typing import Union
from torch.distributions import Categorical
import lightning as L
from dataclasses import asdict
from markov_bridges.data.dataloaders_utils import get_dataloaders
from markov_bridges.models.generative_models.generative_models_lightning import AbstractGenerativeModelL
from markov_bridges.models.metrics.metrics_utils import LogMetrics
from markov_bridges.models.metrics.optimal_transport import uniform_pair_x0_x1

from markov_bridges.models.networks.utils.ema import EMA
from torch.optim import Adam

from markov_bridges.models.pipelines.pipeline_cfm import CFMPipeline
from markov_bridges.models.pipelines.thermostats import ConstantThermostat
from markov_bridges.configs.config_classes.generative_models.cfm_config import CFMConfig

from markov_bridges.utils.numerics.integration import integrate_quad_tensor_vec
from markov_bridges.models.pipelines.thermostat_utils import load_thermostat
from markov_bridges.models.networks.temporal.cfm.continuous_networks_utils import load_continuous_network
from markov_bridges.models.metrics.optimal_transport import OTPlanSampler
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple
from collections import namedtuple
from markov_bridges.utils.experiment_files import ExperimentFiles


from torch.optim.lr_scheduler import(
    ReduceLROnPlateau,
    ExponentialLR,
    MultiStepLR,
    StepLR
)

class ConditionalFlowMatchingL(EMA,L.LightningModule):
    
    def __init__(self, config:CFMConfig):
        EMA.__init__(self,config)
        L.LightningModule.__init__(self)
        self.define_deep_models(config)
        self.define_ot(config)
        self.define_thermostat(config)
        self.init_ema()

        self.save_hyperparameters()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        self.config = config
        config_data = config.data

        self.dimensions = config_data.discrete_dimensions
        self.DatabatchNameTuple = namedtuple("DatabatchClass", self.config.data.fields)

    def define_deep_models(self,config):
        self.nn_loss = nn.MSELoss(reduction='none')
        self.continuous_network = load_continuous_network(config)

    def define_thermostat(self,config):
        self.thermostat = load_thermostat(config)

    #====================================================================
    # SAMPLE BRIDGE
    #====================================================================
    
    def conditional_drift(self, x, x1, x0, t):
        """ conditional vector field (drift) u_t(x|x_0,x_1)
        """

        if self.config.trainer.conditional_bridge_type == 'linear':
            A = 0.
            B = 1.
            C = -1.

        elif self.config.trainer.conditional_bridge_type == 'schrodinger':
            A = (1. - 2. * t) / (t * (1. - t))
            B = t**2 / (t * (1. - t))
            C = -1. * (1. - t)**2 / (t * (1. - t))

        return A * x + B * x1 + C * x0 

    def sample_bridge(self, x1, x0, time):
        """
        simple bridge. Equivalent to a linear interpolant x_t 
        """
        device = x1.device
        original_shape = x0.shape
        continuous_dimensions = x1.size(1)
        time_ = time[:,None].repeat((1,continuous_dimensions))

        t = time_.flatten()
        x1 = x1.flatten()
        x0 = x0.flatten()

        mean = x0 * (1.-t) + x1 * t

        if self.config.trainer.conditional_bridge_type == 'linear':
            std = self.config.thermostat.gamma

        elif self.config.trainer.conditional_bridge_type == 'schrodinger':
            std = self.config.thermostat.gamma * torch.sqrt(t * (1.-t))

        x = mean + std * torch.randn_like(mean)
        x = x.to(device)
        x = x.reshape(original_shape)
        return x
    
    def sample_x(self, databatch):
        time = databatch.time.flatten()
        source = databatch.source_continuous
        target = databatch.target_continuous     
        continuous_sample = self.sample_bridge(target, source, time)
        return continuous_sample

    def sample_pair(self,databatch,seed=None):
        """
        data is returned with shape [batch_size,dimension]
        """
        x1,x0 = uniform_pair_x0_x1(databatch)
        if self.config.optimal_transport.name == "OTPlanSampler":
            cost=None
            if self.config.optimal_transport.cost == "log":
                with torch.no_grad():
                    cost = self.log_cost(x0,x1)
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)        
            x0, x1 = self.ot_sampler.sample_plan(x0, x1, replace=False,cost=cost)
        return x1,x0

    #====================================================================
    # LOSS
    #====================================================================
    
    def loss(self, databatch: MarkovBridgeDataNameTuple, continuous_sample):
        continuous_head = self.continuous_network(x_continuous=continuous_sample,  
                                                  context_discrete=databatch.context_discrete if self.has_context_discrete else None, 
                                                  context_continuous=databatch.context_continuous if self.has_context_continuous else None,
                                                  times=databatch.time)
        
        ut = self.conditional_drift(x=continuous_sample,
                                    x1=databatch.target_continuous, 
                                    x0=databatch.source_continuous,
                                    t=databatch.time) 
        
        full_loss = torch.Tensor([0.]).to(continuous_sample.device)
        full_loss += self.continuous_loss_nn(continuous_head, ut).mean() 
        return full_loss


    def log_cost_regularizer(self):
        S = self.vocab_size
        beta_integral_ = self.beta_integral(torch.Tensor([1.]), torch.Tensor([0.]))
        w_10 = torch.exp(- S* beta_integral_)
        A = torch.log(1./S + w_10*(-1./S))
        B = torch.log(1./S + w_10*(-1./S + 1.)) - A
        return B

    def log_cost(self,x0,x1):
        """
        Schrodinger transport cost 

        params
        ------
        x0,x1: torch.Tensor(batch_size,dimensions) 

        returns
        -------
        cost: torch.Tensor(batch_size,batch_size)
        """
        batch_size = x0.shape[0]
        x0 = x0.repeat_interleave(batch_size,0)
        x1 = x1.repeat((batch_size,1))
        cost = (x1 == x0).sum(axis=1).reshape(batch_size,batch_size).float()
        return -cost
    
    def define_ot(self,config:CFMConfig):
        if config.optimal_transport.cost == "log":
            B = self.log_cost_regularizer()
            B = B.item() if isinstance(B,torch.Tensor) else B
            config.optimal_transport.method = "sinkhorn"
            config.optimal_transport.normalize_cost = True
            config.optimal_transport.normalize_cost_constant = float(config.data.discrete_dimensions)
            reg = 1./B
            print("OT regularizer for Schrodinger Plan {0}".format(reg))
            config.optimal_transport.reg = reg
        self.ot_sampler = OTPlanSampler(**asdict(config.optimal_transport))
        
    #=====================================================================
    # TRAINING
    #=====================================================================
    
    def training_step(self,batch, batch_idx):
        optimizer = self.optimizers()
        # obtain loss
        databatch = self.DatabatchNameTuple(*batch)
        # data pair and time sample
        target_continuous, source_continuous = self.sample_pair(databatch)
        # sample x from z
        sampled_x = self.sample_x(target_continuous, source_continuous, databatch.time).float()
        # loss
        loss = self.loss(databatch,sampled_x)
        # optimization
        optimizer.zero_grad()
        self.manual_backward(loss)
        # clip grad norm
        if self.config.trainer.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.trainer.clip_max_norm)
        # cambell scheduler
        if self.config.trainer.warm_up > 0:
            for g in optimizer.param_groups:
                new_lr = self.lr * np.minimum(float(self.number_of_training_step+1) / self.config.trainer.warm_up, 1.0)
                g['lr'] = new_lr
        optimizer.step()
        self.number_of_training_step += 1
        if self.do_ema:
            self.update_ema()
        self.log('train_loss', loss, on_epoch=True,on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # obtain loss
        databatch = self.DatabatchNameTuple(*batch)
        # data pair and time sample
        target_continuous, source_continuous = self.sample_pair(databatch)
        # sample x from z
        sampled_x = self.sample_x(target_continuous, source_continuous, databatch.time).float()
        # loss
        loss = self.loss(databatch,sampled_x)
        self.log('val_loss', loss, on_epoch=True,on_step=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        """
        Sets up the optimizer and learning rate scheduler for PyTorch Lightning.
        The optimizer setup here is consistent with the `initialize` method.
        """
        self.number_of_training_step = 0
        if self.config.trainer.do_ema:
            self.do_ema = True

        #DEFINE OPTIMIZERS
        optimizer = Adam(self.parameters(),
                         lr=self.config.trainer.learning_rate,
                         weight_decay=self.config.trainer.weight_decay)
        
        scheduler = self.define_scheduler(optimizer)
        self.lr = self.config.trainer.learning_rate

        if scheduler is None:
            return optimizer
        else:
            return [optimizer],[scheduler]
        
    def define_scheduler(self,optimizer):
        scheduler = None
        # Check if a scheduler is defined in the configuration
        if self.config.trainer.scheduler is not None:
            if self.config.trainer.scheduler == "step":
                # Define StepLR scheduler
                scheduler = StepLR(optimizer, 
                                   step_size=self.config.trainer.step_size, 
                                   gamma=self.config.trainer.gamma)
            elif self.config.trainer.scheduler == "multi":
                # Define MultiStepLR scheduler
                scheduler = MultiStepLR(optimizer, 
                                            milestones=self.config.trainer.milestones, 
                                            gamma=self.config.trainer.gamma)
            elif self.config.trainer.scheduler == "exponential":
                # Define ExponentialLR scheduler
                scheduler = ExponentialLR(optimizer, 
                                          gamma=self.config.trainer.gamma)
            elif self.config.trainer.scheduler == "reduce":
                # Define ReduceLROnPlateau scheduler
                scheduler = ReduceLROnPlateau(optimizer, 
                                              mode='min', 
                                              factor=self.config.trainer.factor, 
                                              patience=self.config.trainer.patience)
        return scheduler

class CFML(AbstractGenerativeModelL):

    config_type = CFMConfig

    def define_from_config(self,config:CFMConfig):
        self.config = config
        self.dataloader = get_dataloaders(self.config)
        self.model = ConditionalFlowMatchingL(self.config)
        self.pipeline = CFMPipeline(self.config,self.model,self.dataloader)
        self.log_metrics = LogMetrics(self,metrics_configs_list=self.config.trainer.metrics)

    def define_from_dir(self, experiment_dir:str|ExperimentFiles=None, checkpoint_type: str = "best"):
        # define experiments files
        if isinstance(experiment_dir,str):
            self.experiment_files = ExperimentFiles(experiment_dir=experiment_dir)
        else:
            self.experiment_files = experiment_dir
        # read config
        self.config = self.read_config(self.experiment_files)
        # obtain dataloader
        self.dataloader = get_dataloaders(self.config)
        # obtain checkpoint path
        CKPT_PATH = self.experiment_files.get_lightning_checkpoint_path(checkpoint_type)
        # load model
        self.model = ConditionalFlowMatchingL.load_from_checkpoint(CKPT_PATH, config=self.config)
        self.pipeline = CFMPipeline(self.config,self.model,self.dataloader)
        self.log_metrics = LogMetrics(self,metrics_configs_list=self.config.trainer.metrics)

        return self.config

    def test_evaluation(self) -> dict:
        all_metrics = self.log_metrics(self,"best")
        return all_metrics
    