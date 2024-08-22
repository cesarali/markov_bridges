from typing import Tuple
from collections import namedtuple

import torch
from torch import nn
import lightning as L
from torch.optim.adam import Adam
from torch.distributions import Categorical,Normal
from torch.nn.functional import softmax

from torch.optim.lr_scheduler import(
    ReduceLROnPlateau,
    ExponentialLR,
    MultiStepLR,
    StepLR
)

from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig

from markov_bridges.data.dataloaders_utils import get_dataloaders
from markov_bridges.data.qm9.qm9_points_dataloader import QM9PointDataNameTupleCMB
from markov_bridges.models.pipelines.thermostat_utils import load_thermostat
from markov_bridges.models.networks.temporal.mixed.mixed_networks_utils import load_mixed_network

from markov_bridges.models.generative_models.generative_models_lightning import AbstractGenerativeModelL
from markov_bridges.models.networks.utils.ema import EMA
from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataloader
from markov_bridges.models.pipelines.pipeline_cmb import CMBPipeline
from markov_bridges.models.pipelines.thermostats import Thermostat

from markov_bridges.models.networks.temporal.edmg.helper_distributions import (
    DistributionNodes,
    DistributionProperty
)

from markov_bridges.utils import equivariant_diffusion as diffusion_utils
from markov_bridges.models.metrics.metrics_utils import LogMetrics
from markov_bridges.utils.shapes import nodes_and_edges_masks

from markov_bridges.utils.shapes import right_shape,right_time_size,where_to_go_x
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

from markov_bridges.utils.equivariant_diffusion import (
    assert_mean_zero_with_mask, 
    remove_mean_with_mask,
    check_mask_correct, 
    sample_center_gravity_zero_gaussian_with_mask,
    random_rotation,
    assert_correctly_masked
)
from markov_bridges.data.qm9.utils import prepare_context
from markov_bridges.configs.config_classes.networks.mixed_networks_config import (
    MixedEGNN_dynamics_QM9Config,
)

class MixedForwardMapL(EMA,L.LightningModule):

    nodes_dist:DistributionNodes
    prop_dist:DistributionProperty

    def __init__(self,config:CMBConfig,dataloader:MarkovBridgeDataloader,save=True):
        self.automatic_optimization = False
        EMA.__init__(self,config)
        L.LightningModule.__init__(self)
        if save:
            self.save_hyperparameters()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.config = config

        self.vocab_size = self.config.data.vocab_size
        self.continuos_dimensions = config.data.continuos_dimensions

        self.in_node_nf = 1 # just the categories
        self.property_norms = dataloader.property_norms

        self.has_target_discrete = config.data.has_target_discrete 
        self.has_target_continuous = config.data.has_target_continuous 

        self.dataset_info = dataloader.dataset_info
        self.databatch_keys = dataloader.get_databach_keys()

        self.define_deep_models()
        self.define_bridge_parameters()
        self.define_sample_distributions(dataloader)
        
        self.DatabatchNameTuple = namedtuple("DatabatchClass",
                                             self.databatch_keys)
        self.init_ema()

    def define_deep_models(self):
        self.mixed_network = load_mixed_network(self.config)
        self.discrete_loss_nn = nn.CrossEntropyLoss(reduction='none')
        self.continuous_loss_nn = nn.MSELoss(reduction='none')

    def define_sample_distributions(self,dataloader:MarkovBridgeDataloader):
        self.nodes_dist = None
        if hasattr(dataloader,"dataset_info"):
            histogram = dataloader.dataset_info['n_nodes']
            self.nodes_dist = DistributionNodes(histogram)

        self.prop_dist = None
        if len(self.config.mixed_network.conditioning) > 0:
            self.prop_dist = DistributionProperty(dataloader.train(), 
                                                  self.config.mixed_network.conditioning,
                                                  number_string=self.config.mixed_network.number_string)
            
        if self.prop_dist is not None:
            self.prop_dist.set_normalizer(self.property_norms)

    def define_bridge_parameters(self):
        self.continuous_loss_type = self.config.continuous_loss_type
        self.discrete_bridge_: Thermostat = load_thermostat(self.config)
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
        if self.has_target_continuous:
            full_loss = torch.Tensor([0.]).to(continuous_sample.device)
        if self.has_target_discrete:
            full_loss = torch.Tensor([0.]).to(discrete_sample.device)

        # Train What is Needed
        if self.has_target_discrete:
            discrete_loss_ = self.discrete_loss(databatch,discrete_head,discrete_sample)
            discrete_loss_ = discrete_loss_.reshape(discrete_sample.size(0),discrete_sample.size(1))
            full_loss += discrete_loss_.sum(axis=1).mean()
        if self.has_target_continuous:
            continuous_loss_ = self.continuous_loss(databatch,continuous_head,continuous_sample)
            continuous_loss_ = continuous_loss_.reshape(continuous_sample.size(0),continuous_sample.size(1))
            full_loss += continuous_loss_.sum(axis=1).mean()

        return full_loss,discrete_loss_,continuous_loss_
    
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
    def prepare_batch(self,batch):
        databatch = self.DatabatchNameTuple(*batch)
        return databatch
    
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        # obtain loss
        databatach = self.prepare_batch(batch)
        # sample bridge
        discrete_sample, continuous_sample = self.sample_bridge(databatach)
        loss,discrete_loss_,continuous_loss_ = self.loss(databatach, discrete_sample, continuous_sample)
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
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log('discrete_training_loss', discrete_loss_.mean(), on_step=True, prog_bar=True, logger=True)
        self.log('continuous_training_loss', continuous_loss_.mean(), on_step=True, prog_bar=True, logger=True)
        self.number_of_training_step += 1
        return loss

    def validation_step(self, batch, batch_idx):
        databatach = self.prepare_batch(batch)
        discrete_sample, continuous_sample = self.sample_bridge(databatach)
        loss,discrete_loss_,continuous_loss_ = self.loss(databatach, discrete_sample, continuous_sample)
        self.log('val_loss', loss, on_step=False, prog_bar=True, logger=True)
        self.log('discrete_val_loss', discrete_loss_.mean(), on_step=True, prog_bar=True, logger=True)
        self.log('continuous_val_loss', continuous_loss_.mean(), on_step=True, prog_bar=True, logger=True)
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

class MixedForwardMapMoleculesL(MixedForwardMapL):

    def __init__(self,config:CMBConfig,dataloader:MarkovBridgeDataloader,save=True):
        MixedForwardMapL.__init__(self,config,dataloader,save)

    def prepare_batch(self,databatch)->QM9PointDataNameTupleCMB:
        """
        cmb requieres as batch a name tuble that contains the noise sample 
        and where all the samples are of size batch_size,dimensions

        for molecules the continuos_variables dimensions = number_of_atoms*3 (positions)
                          discrete_variables dimensions = number_of_atoms*1 (discrete category)

        """
        dtype = torch.float32
        x = databatch['positions'].to(dtype)
        node_mask = databatch['atom_mask'].to(dtype).unsqueeze(2)
        edge_mask = databatch['edge_mask'].to(dtype)
        one_hot = databatch['one_hot'].to(dtype)
        charges = databatch['charges'].to(x.device, dtype)

        x,h = self.augment_noise(x,one_hot,node_mask,charges)
        databatch_nametuple = self.cmb_source_and_nametuple(databatch,x,h,self.config)
        return databatch_nametuple

    def loss(self,databatch:QM9PointDataNameTupleCMB,discrete_sample,continuous_sample):
        full_loss,discrete_loss_,continuous_loss_ = super().loss(databatch,discrete_sample,continuous_sample)
        continuous_loss_ = continuous_loss_.reshape(databatch.batch_size,databatch.max_num_atoms,self.continuos_dimensions)
        continuous_loss_ = continuous_loss_.sum(axis=-1)

        # masks
        continuous_loss_ = continuous_loss_*databatch.atom_mask
        discrete_loss_ = discrete_loss_*databatch.atom_mask

        if self.has_target_continuous:
            full_loss = torch.Tensor([0.]).to(continuous_sample.device)
        if self.has_target_discrete:
            full_loss = torch.Tensor([0.]).to(discrete_sample.device)

        # Train What is Needed
        if self.has_target_discrete:
            full_loss += discrete_loss_.sum(axis=1).mean()
        if self.has_target_continuous:
            full_loss += continuous_loss_.sum(axis=1).mean()

        return full_loss,discrete_loss_,continuous_loss_

    def augment_noise(self,x,one_hot,node_mask,charges,augment_noise=0.,data_augmentation=False):
        """
        """
        # add noise 
        x = remove_mean_with_mask(x, node_mask)
        if augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * augment_noise
        x = remove_mean_with_mask(x, node_mask)
        if data_augmentation:
            x = random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)
        h = {'categorical': one_hot, 'integer': charges}
        return x, h

    def cmb_source_and_nametuple(self,databatch,x,h,config)->Tuple[int,int,QM9PointDataNameTupleCMB]:
        """
        creates the source data and defines the name tuple
        creates the context
        takes only the properties requiered in the conditioning
        """
        # Keys and databatch info
        node_mask = databatch['atom_mask'].to(self.dtype).unsqueeze(2)
        conditioning = config.mixed_network.conditioning
        basic_key_strings = "num_atoms source_discrete source_continuous target_discrete target_continuous"
        mask_key_strings = "atom_mask edge_mask context time batch_size max_num_atoms"
        if len(conditioning) > 0:
            condition_key_strings = " ".join(conditioning)
            all_key_strings = basic_key_strings+" "+condition_key_strings+" "+mask_key_strings
        else:
            all_key_strings = basic_key_strings+" "+mask_key_strings
        DatabatchNametuple = namedtuple("DatabatchClass", all_key_strings)

        # CMB only handles shape of lenght 2
        target_discrete = torch.argmax(h["categorical"],dim=2) # NEEDED ONLY TO REMOVES ONE HOT
        vocab_size = self.vocab_size
        data_size = target_discrete.size(0)
        discrete_dimensions = target_discrete.size(1)
        target_continuous = x.reshape(data_size,-1)
        continuous_dimensions = target_continuous.size(1)

        #Discrete SOURCE
        uniform_probability = torch.full((vocab_size,),1./vocab_size)
        source_discrete = Categorical(uniform_probability).sample((data_size,discrete_dimensions)).to(x.device)

        #Continuous SOURCE
        gaussian_probability = Normal(0.,1.)
        source_continuous = gaussian_probability.sample((data_size,continuous_dimensions)).to(x.device)

        # Create Time
        time = torch.rand((data_size,1)).to(x.device,self.dtype)

        # CONTEXT AS TENSOR
        context = None
        if len(conditioning) > 0:
            context = prepare_context(conditioning, 
                                      databatch, 
                                      self.property_norms).to(x.device, self.dtype)
            assert_correctly_masked(context, node_mask)
            
        items_ = [databatch["num_atoms"],source_discrete,source_continuous,target_discrete,target_continuous]
        for prop in conditioning:
            items_.append(databatch[prop])
        items_.extend([databatch["atom_mask"].to(self.dtype),
                       databatch["edge_mask"].to(self.dtype),
                       context,
                       time,
                       data_size,
                       discrete_dimensions])
        databatch_nametuple = DatabatchNametuple(*items_)
        return databatch_nametuple
    
    def sample_sizes_and_masks(self,sample_size,device,context=None):
        max_n_nodes = self.dataset_info['max_n_nodes']
        nodesxsample = self.nodes_dist.sample(sample_size)
        node_mask,edge_mask= nodes_and_edges_masks(nodesxsample,max_n_nodes,device)
        batch_size = node_mask.size(0)
        
        # TODO FIX: This conditioning just zeros.
        if len(self.config.mixed_network.conditioning) > 0:
            if context is None:
                context = self.prop_dist.sample_batch(nodesxsample)
            context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
        else:
            context = None
        return max_n_nodes,nodesxsample,node_mask,edge_mask,context
    
    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = diffusion_utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.continuos_dimensions), device=node_mask.device,
            node_mask=node_mask)
        z_h = diffusion_utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z
    
class CMBL(AbstractGenerativeModelL):

    config_type = CMBConfig

    def define_from_config(self,config:CMBConfig):
        self.config = config
        self.dataloader = get_dataloaders(self.config)
        if isinstance(self.config.mixed_network,MixedEGNN_dynamics_QM9Config):
            self.model = MixedForwardMapMoleculesL(self.config,self.dataloader)
        else:
            self.model = MixedForwardMapL(self.config,self.dataloader)
        self.pipeline = CMBPipeline(self.config,self.model,self.dataloader)
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
        if isinstance(self.config.mixed_network,MixedEGNN_dynamics_QM9Config):
            self.model = MixedForwardMapMoleculesL.load_from_checkpoint(CKPT_PATH,
                                                                        config=self.config,
                                                                        dataloader=self.dataloader)
        else:
            self.model = MixedForwardMapL.load_from_checkpoint(CKPT_PATH,
                                                               config=self.config,
                                                               dataloader=self.dataloader)
        
        self.pipeline = CMBPipeline(self.config,self.model,self.dataloader)
        self.log_metrics = LogMetrics(self,metrics_configs_list=self.config.trainer.metrics)
        return self.config

    def test_evaluation(self) -> dict:
        #all_metrics = self.log_metrics(self,"last")
        self.define_from_dir(self.experiment_files.experiment_dir,checkpoint_type="best")
        all_metrics = self.log_metrics(self,"best")
        return all_metrics
    