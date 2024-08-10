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

from markov_bridges.models.pipelines.pipeline_cjb import CJBPipeline
from markov_bridges.models.pipelines.thermostats import ConstantThermostat
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig

from markov_bridges.utils.numerics.integration import integrate_quad_tensor_vec
from markov_bridges.models.pipelines.thermostat_utils import load_thermostat
from markov_bridges.models.networks.temporal.temporal_networks_utils import load_temporal_network
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

class ClassificationForwardRateL(EMA,L.LightningModule):
    
    def __init__(self, config:CJBConfig):
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

        self.vocab_size = config_data.vocab_size
        self.dimensions = config_data.discrete_dimensions
        self.temporal_network_to_rate = config.temporal_network_to_rate
        self.DatabatchNameTuple = namedtuple("DatabatchClass", self.config.data.fields)




    def define_deep_models(self,config):
        self.nn_loss = nn.CrossEntropyLoss(reduction='none')
        self.temporal_network = load_temporal_network(config)

    def define_thermostat(self,config):
        self.thermostat = load_thermostat(config)

    def classify(self,x,time,databatch,sample=False):
        """
        this function takes the shape [batch_size,dimension,vocab_size] 
        

        :param x: [batch_size,dimension,vocab_size]
        :param times:
        :return:
        """
        change_logits = self.temporal_network(x,time,databatch)
        return change_logits

    def forward(self, x, time, databatch):
        """
        RATE

        :param x: [batch_size,dimensions]
        :param time:
        :return:[batch_size,dimensions,vocabulary_size]
        """
        batch_size = x.size(0)
        if len(x.shape) != 2:
            x = x.reshape(batch_size,-1)
        right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t).to(x.device)

        beta_integral_ = self.beta_integral(right_time_size(1.), right_time_size(time))
        w_1t = torch.exp(-self.vocab_size * beta_integral_)
        A = 1.
        B = (w_1t * self.vocab_size) / (1. - w_1t)
        C = w_1t

        change_logits = self.classify(x,time,databatch,sample=True)
        change_classifier = softmax(change_logits, dim=2)

        #x = x.reshape(batch_size,self.dimensions)
        where_iam_classifier = torch.gather(change_classifier, 2, x.long().unsqueeze(2))

        rates = A + B[:,None,None]*change_classifier + C[:,None,None]*where_iam_classifier
        return rates
    #====================================================================
    # CONDITIONAL AND TRANSITIONS RATES INVOLVED
    #====================================================================
    def conditional_probability(self, x, x0, t, t0):
        """

        \begin{equation}
        P(x(t) = i|x(t_0)) = \frac{1}{s} + w_{t,t_0}\left(-\frac{1}{s} + \delta_{i,x(t_0)}\right)
        \end{equation}

        \begin{equation}
        w_{t,t_0} = e^{-S \int_{t_0}^{t} \beta(r)dr}
        \end{equation}

        """
        right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
        right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t).to(x.device)

        t = right_time_size(t).to(x0.device)
        t0 = right_time_size(t0).to(x0.device)

        S = self.vocab_size
        integral_t0 = self.beta_integral(t, t0)
        w_t0 = torch.exp(-S * integral_t0)

        x = right_shape(x)
        x0 = right_shape(x0)

        delta_x = (x == x0).float()
        probability = 1. / S + w_t0[:, None, None] * ((-1. / S) + delta_x)

        return probability

    def telegram_bridge_probability(self, x, x1, x0, t):
        """
        \begin{equation}
        P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
        \end{equation}
        """

        P_x_to_x1 = self.conditional_probability(x1, x, t=1., t0=t)
        P_x0_to_x = self.conditional_probability(x, x0, t=t, t0=0.)
        P_x0_to_x1 = self.conditional_probability(x1, x0, t=1., t0=0.)

        conditional_transition_probability = (P_x_to_x1 * P_x0_to_x) / P_x0_to_x1
        return conditional_transition_probability

    def conditional_transition_rate(self, x, x1, t):
        """
        \begin{equation}
        f_t(\*x'|\*x,\*x_1) = \frac{p(\*x_1|x_t=\*x')}{p(\*x_1|x_t=\*x)}f_t(\*x'|\*x)
        \end{equation}
        """
        right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t).to(x.device)
        x_to_go = self.where_to_go_x(x)

        P_xp_to_x1 = self.conditional_probability(x1, x_to_go, t=1., t0=t)
        P_x_to_x1 = self.conditional_probability(x1, x, t=1., t0=t)

        forward_rate = self.thermostat(t)[:,None,None]
        rate_transition = (P_xp_to_x1 / P_x_to_x1) * forward_rate

        return rate_transition

    def sample_x(self, x_1, x_0, time):
        if len(time.shape) > 1:
            time = time.flatten()
        device = x_1.device
        x_to_go = self.where_to_go_x(x_0)
        transition_probs = self.telegram_bridge_probability(x_to_go, x_1, x_0, time)
        sampled_x = Categorical(transition_probs).sample().to(device)
        return sampled_x

    def beta_integral(self, t1, t0):
        """
        Dummy integral for constant rate
        """
        if isinstance(self.thermostat,ConstantThermostat):
            integral = (t1 - t0)*self.thermostat.gamma
        else:
            integral = integrate_quad_tensor_vec(self.thermostat, t0, t1, 100)
        return integral

    def where_to_go_x(self, x):
        x_to_go = torch.arange(0, self.vocab_size)
        x_to_go = x_to_go[None, None, :].repeat((x.size(0), x.size(1), 1)).float()
        x_to_go = x_to_go.to(x.device)
        return x_to_go
    
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
    
    def define_ot(self,config:CJBConfig):
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
    
    def loss(self, databatch:MarkovBridgeDataNameTuple,discrete_sample):
        discrete_head = self.classify(discrete_sample, databatch.time, databatch)
        # reshape for cross logits
        discrete_head = discrete_head.reshape(-1, self.config.data.vocab_size)
        target_discrete = databatch.target_discrete.reshape(-1)
        discrete_loss = self.nn_loss(discrete_head,target_discrete.long())
        discrete_loss = discrete_loss.mean()
        return discrete_loss
    #=====================================================================
    # TRAINING
    #=====================================================================
    def training_step(self,batch, batch_idx):
        optimizer = self.optimizers()
        # obtain loss
        databatch = self.DatabatchNameTuple(*batch)
        # data pair and time sample
        target_discrete, source_discrete = self.sample_pair(databatch)
        # sample x from z
        sampled_x = self.sample_x(target_discrete, source_discrete, databatch.time).float()
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # obtain loss
        databatch = self.DatabatchNameTuple(*batch)
        # data pair and time sample
        target_discrete, source_discrete = self.sample_pair(databatch)
        # sample x from z
        sampled_x = self.sample_x(target_discrete, source_discrete, databatch.time).float()
        # loss
        loss = self.loss(databatch,sampled_x)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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

class CJBL(AbstractGenerativeModelL):

    def define_from_config(self,config:CJBConfig):
        self.config = config
        self.dataloader = get_dataloaders(self.config)
        self.model = ClassificationForwardRateL(self.config)
        self.pipeline = CJBPipeline(self.config,self.model,self.dataloader)
        self.log_metrics = LogMetrics(self,metrics_configs_list=self.config.trainer.metrics)

    def read_config(self,experiment_files):
        config_json = super().read_config(experiment_files)
        config = CJBConfig(**config_json)   
        return config
     
    def define_from_dir(self, experiment_dir:str|ExperimentFiles=None, checkpoint_type: str = "best"):
        if isinstance(experiment_dir,str):
            self.experiment_files = ExperimentFiles(experiment_dir=experiment_dir)
        else:
            self.experiment_files = experiment_dir

        self.config = self.read_config(self.experiment_files)
        self.dataloader = get_dataloaders(self.config)

        CKPT_PATH = self.experiment_files.get_lightning_checkpoint_path(checkpoint_type)
        print(CKPT_PATH)

        self.model = ClassificationForwardRateL.load_from_checkpoint(CKPT_PATH, config=self.config)
        self.model = None
        self.pipeline = None
        self.log_metrics = None

        return self.config

    def test_evaluation(self) -> dict:
        all_metrics = self.log_metrics(self,"best")
        return all_metrics
    