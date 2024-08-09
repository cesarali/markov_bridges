import torch
import torch.nn as nn
import lightning as L
from torch.optim import Adam
import numpy as np

import numpy as np
import torch
from torch.optim.adam import Adam

from typing import List
from dataclasses import dataclass,field
from collections import namedtuple

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import softmax


from torch.distributions import Categorical,Normal
import torch
import numpy as np
from torch.optim.adam import Adam
from markov_bridges.models.networks.utils.ema import EMA
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

import torch
from torch import nn

from markov_bridges.utils.shapes import right_shape,right_time_size,where_to_go_x
from markov_bridges.models.networks.utils.ema import EMA
from markov_bridges.models.pipelines.thermostat_utils import load_thermostat
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.utils.shapes import right_shape,right_time_size
from markov_bridges.models.pipelines.thermostats import Thermostat
from markov_bridges.models.networks.temporal.mixed.mixed_networks_utils import load_mixed_network
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

class MixedForwardMapL(EMA,L.LightningModule):

    def __init__(self,config:CMBConfig):
        self.automatic_optimization = False
        EMA.__init__(self,config)
        L.LightningModule.__init__(self)

        self.config = config
        self.do_ema = False
        self.lr = None

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        self.has_target_discrete = config.data.has_target_discrete 
        self.has_target_continuous = config.data.has_target_continuous 

        self.define_deep_models(config)
        self.define_bridge_parameters(config)
        self.DatabatchNameTuple = namedtuple("DatabatchClass", self.config.data.fields)
        self.init_ema()

    def define_deep_models(self,  config: CMBConfig):
        self.mixed_network = load_mixed_network(config)

        self.discrete_loss_nn = nn.CrossEntropyLoss(reduction='none')
        self.continuous_loss_nn = nn.MSELoss(reduction='none')

    def define_bridge_parameters(self,  config: CMBConfig):
        self.continuous_loss_type = config.continuous_loss_type
        self.discrete_bridge_: Thermostat = load_thermostat(config)
        self.continuous_bridge_ = None
    #====================================================================
    # INTERPOLATIONS AND/OR BRIDGES
    #====================================================================
    def sample_discrete_bridge(self,x_1,x_0,time):
        x_to_go = where_to_go_x(x_0,self.vocab_size)
        transition_probs = self.telegram_bridge_probability(x_to_go, x_1, x_0, time)
        sampled_x = Categorical(transition_probs).sample()
        return sampled_x
    
    def sample_continuous_bridge(self,x_1,x_0,time):
        """
        simple brownian bridge
        """
        original_shape = x_0.shape
        continuous_dimensions = x_1.size(1)
        time_ = time[:,None].repeat((1,continuous_dimensions))

        t = time_.flatten()
        x_1 = x_1.flatten()
        x_0 = x_0.flatten()

        x_m = x_0*(1.-t) + x_1*t
        variance = t*(1. - t)

        x = Normal(x_m,variance).sample()
        x = x.reshape(original_shape)
        return x
    
    def sample_bridge(self,databatch):
        time = databatch.time.flatten()

        if self.has_target_discrete:
            source_discrete = databatch.source_discrete.float()
            target_discrete = databatch.target_discrete.float()
            discrete_sample = self.sample_discrete_bridge(target_discrete,source_discrete,time)
        else:
            discrete_sample = None

        if self.has_target_continuous:
            source_continuous = databatch.source_continuous
            target_continuous = databatch.target_continuous     
            continuous_sample = self.sample_continuous_bridge(target_continuous,source_continuous,time)
        else:
            continuous_sample = None
        return discrete_sample,continuous_sample    
    #====================================================================
    # RATES,FLOWS AND DRIFT for GENERATION
    #====================================================================
    def discrete_rate(self,change_logits, x, time):
        """
        RATE

        :param x: [batch_size,dimensions]
        :param time:
        :return:[batch_size,dimensions,vocabulary_size]
        """
        batch_size = x.size(0)
        if len(x.shape) != 2:
            x = x.reshape(batch_size,-1)

        t_1 = right_time_size(1.,x)
        time_ = right_time_size(time,x)

        beta_integral_ = self.discrete_bridge_.beta_integral(t_1,time_)
        w_1t = torch.exp(-self.vocab_size * beta_integral_)
        A = 1.
        B = (w_1t * self.vocab_size) / (1. - w_1t)
        C = w_1t

        change_classifier = softmax(change_logits, dim=2)

        where_iam_classifier = torch.gather(change_classifier, 2, x.long().unsqueeze(2))

        rates = A + B[:,None,None]*change_classifier + C[:,None,None]*where_iam_classifier
        return rates
    
    def continuous_drift(self, x, x1, t):
        if len(t.shape) == 1:
            t = t[:,None]
        drift = (x1 - x)/(1.-t)
        return drift
    
    def continuous_flow(self, x, x1, x0, t):
        A = (1 - 2 * t) / (t * (1 - t))
        B = t**2 / (t * (1 - t))
        C = -1 * (1 - t)**2 / (t * (1 - t))
        return A * x + B * x1 + C * x0 
   
    def forward_map(self,discrete_sample,continuous_sample,time,databatch):
        if len(time.shape) > 1:
            time = time.flatten()
        discrete_head,continuous_head = self.mixed_network(discrete_sample, continuous_sample, time,databatch)
        if self.has_target_discrete:
            rate = self.discrete_rate(discrete_head, discrete_sample, time)
        else:
            rate = None
        if self.has_target_continuous:
            if self.continuous_loss_type == "regression":
                vector_field = self.continuous_drift(continuous_head, continuous_sample, time)
            elif self.continuous_loss_type == "flow":
                vector_field = continuous_head
            elif self.continuous_loss_type == "drift":
                vector_field = continuous_head
        else:
            vector_field = None
        return rate, vector_field
    #====================================================================
    # LOSS
    #====================================================================
    def loss(self,databatch:MarkovBridgeDataNameTuple,discrete_sample,continuous_sample):
        # Calculate Heads For Classifier or Mean Average
        discrete_head,continuous_head = self.mixed_network(discrete_sample,
                                                           continuous_sample,
                                                           databatch.time,
                                                           databatch)
        # Train What is Needed
        full_loss = torch.Tensor([0.])
        if self.has_target_discrete:
            full_loss += self.discrete_loss(databatch,discrete_head,discrete_sample).mean()
        if self.has_target_continuous:
            full_loss += self.continuous_loss(databatch,continuous_head,continuous_sample).mean()
        return full_loss
    
    def discrete_loss(self, databatch:MarkovBridgeDataNameTuple, discrete_head, discrete_sample=None): 
        # reshape for cross logits
        discrete_head = discrete_head.reshape(-1, self.config.data.vocab_size)
        target_discrete = databatch.target_discrete.reshape(-1)
        discrete_loss = self.discrete_loss_nn(discrete_head,target_discrete.long())
        return discrete_loss
    
    def continuous_loss(self, databatch:MarkovBridgeDataNameTuple, continuous_head, continuous_sample=None):
        # pick loss
        if self.continuous_loss_type == "flow":
            conditional_flow = self.continuous_flow(continuous_sample, databatch.target_continuous, databatch.source_continuous, databatch.time)
            mse = self.continuous_loss_nn(continuous_head, conditional_flow)
        elif self.continuous_loss_type == "drift":
            conditional_drift = self.continuous_drift(continuous_head, databatch.target_continuous, databatch.time)
            mse = self.continuous_loss_nn(continuous_head, conditional_drift)
        elif self.continuous_loss_type == "regression":
            mse = self.continuous_loss_nn(continuous_head, databatch.target_continuous)

        return mse    
    #====================================================================
    # DISCRETE BRIDGE FUNCTIONS
    #====================================================================
    def multivariate_telegram_conditional(self,x, x0, t, t0):
        """
        \begin{equation}
        P(x(t) = i|x(t_0)) = \frac{1}{s} + w_{t,t_0}\left(-\frac{1}{s} + \delta_{i,x(t_0)}\right)
        \end{equation}

        \begin{equation}
        w_{t,t_0} = e^{-S \int_{t_0}^{t} \beta(r)dr}
        \end{equation}

        """
        t = right_time_size(t,x)
        t0 = right_time_size(t0,x)

        integral_t0 = self.discrete_bridge_.beta_integral(t, t0)
        w_t0 = torch.exp(-self.vocab_size * integral_t0)

        x = right_shape(x)
        x0 = right_shape(x0)

        delta_x = (x == x0).float()
        probability = 1. / self.vocab_size + w_t0[:, None, None] * ((-1. / self.vocab_size) + delta_x)
        return probability

    def telegram_bridge_probability(self,x,x1,x0,t):
        """
        \begin{equation}
        P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
        \end{equation}
        """
        P_x_to_x1 = self.multivariate_telegram_conditional(x1, x, t=1., t0=t)
        P_x0_to_x = self.multivariate_telegram_conditional(x, x0, t=t, t0=0.)
        P_x0_to_x1 = self.multivariate_telegram_conditional(x1, x0, t=1., t0=0.)
        conditional_transition_probability = (P_x_to_x1 * P_x0_to_x) / P_x0_to_x1
        return conditional_transition_probability
    #============================================================================
    # TRAINING
    #============================================================================
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        # obtain loss
        databatach = self.DatabatchNameTuple(*batch)
        discrete_sample, continuous_sample = self.sample_bridge(databatach)
        loss = self.loss(databatach, discrete_sample, continuous_sample)
        # optimization
        optimizer.zero_grad()
        self.manual_backward(loss)
        # clip grad norm
        if self.config.trainer.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.trainer.clip_max_norm)
        optimizer.step()
        # handle schedulers
        sch = self.lr_schedulers()
        self.handle_scheduler(sch,self.number_of_training_step,loss)
        # ema
        if self.do_ema:
            self.update_ema()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.number_of_training_step += 1
        return loss

    def validation_step(self, batch, batch_idx):
        databatach = self.DatabatchNameTuple(*batch)
        discrete_sample, continuous_sample = self.sample_bridge(databatach)
        loss = self.loss(databatach, discrete_sample, continuous_sample)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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

    def handle_scheduler(self,scheduler,number_of_training_step,loss_):
        #after warm up call schedulers
        if self.config.trainer.warm_up > 0:
            if number_of_training_step > self.config.trainer.warm_up:
                # Update the learning rate scheduler based on its type
                if self.config.trainer.scheduler in ["step", "multi", "exponential"]:
                    scheduler.step()
                elif self.config.trainer.scheduler == "reduce":
                    scheduler.step(loss_)