# general
from collections import namedtuple
import lightning as L

# torch 
import torch
from torch import nn
from dataclasses import dataclass,asdict,field
from torch.distributions import Categorical,Normal,Dirichlet

# EDMG model
from markov_bridges.models.networks.utils.ema import EMA
from markov_bridges.models.generative_models.generative_models_lightning import AbstractGenerativeModelL
from markov_bridges.configs.config_classes.generative_models.edmg_config import EDMGConfig
from markov_bridges.models.pipelines.pipeline_edmg import EDGMPipeline
from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.data.qm9.qm9_points_dataloader import QM9PointDataloader
from markov_bridges.models.networks.temporal.edmg.helper_distributions import (
    DistributionNodes,
    DistributionProperty
)

from markov_bridges.models.networks.temporal.edmg.en_diffusion import EnVariationalDiffusion
from markov_bridges.utils.shapes import nodes_and_edges_masks

from markov_bridges.models.pipelines.pipeline_edmg import (
    save_and_sample_chain,
    sample_different_sizes_and_save,
    save_and_sample_conditional
)

# loaders
from markov_bridges.data.dataloaders_utils import get_dataloaders
from markov_bridges.models.networks.temporal.edmg.edmg_utils import get_edmg_model

# data
from torch.optim import Adam

from markov_bridges.utils.equivariant_diffusion import (
    assert_mean_zero_with_mask, 
    remove_mean_with_mask,
    assert_correctly_masked, 
    sample_center_gravity_zero_gaussian_with_mask,
    random_rotation,
    Queue
)

from markov_bridges.data.qm9.utils import prepare_context


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

class EquivariantDiffussionNoisingL(EMA,L.LightningModule):
    """
    This corresponds to the torch module which contains all the elements requiered to 
    sample and train a Mixed Variable Bridge

    """
    noising_model:EnVariationalDiffusion 
    nodes_dist:DistributionNodes
    prop_dist:DistributionProperty

    def __init__(self,config:EDMGConfig,dataloader:QM9PointDataloader):
        """
        this function should allow us to create a full discrete and continuous vector from the context and data
        """
        EMA.__init__(self,config)
        L.LightningModule.__init__(self)
        self.automatic_optimization = False

        # different config
        self.config = config
        self.data_config =  config.data
        self.noising_config = config.noising_model

        # dataset information for sampling
        self.dataset_info = dataloader.dataset_info
        self.property_norms = dataloader.property_norms
        self.conditioning = self.noising_config.conditioning
        self.DatabatchNameTuple = namedtuple("DatabatchClass", dataloader.get_databach_keys())

        self.define_deep_models(config,dataloader)
        self.define_bridge_parameters(config)
        self.init_ema()

    def define_deep_models(self,config,dataloader:QM9PointDataloader):
        self.noising_model,self.nodes_dist, self.prop_dist = get_edmg_model(config,
                                                                            dataloader.dataset_info,
                                                                            dataloader.train())
        if self.prop_dist is not None:
            self.prop_dist.set_normalizer(self.property_norms)
        
    def define_bridge_parameters(self,config):
        pass
    #====================================================================
    # RATES AND DRIFT for GENERATION
    #====================================================================
    def forward_map(self,discrete_sample,continuous_sample,time):
        return None
    
    def sample_sizes_and_masks(self,sample_size,device,context=None):
        max_n_nodes = self.dataset_info['max_n_nodes']
        nodesxsample = self.nodes_dist.sample(sample_size)

        node_mask,edge_mask= nodes_and_edges_masks(nodesxsample,max_n_nodes,device)
        batch_size = node_mask.size(0)
        
        # TODO FIX: This conditioning just zeros.
        if self.config.noising_model.context_node_nf > 0:
            if context is None:
                context = self.prop_dist.sample_batch(nodesxsample)
            context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
        else:
            context = None

        return max_n_nodes,nodesxsample,node_mask,edge_mask,context
    
    #====================================================================
    # TRAINING
    #====================================================================
    def loss(self,x, h, node_mask, edge_mask, context):
        """
        """
        bs, n_nodes, n_dims = x.size()

        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        nll = self.noising_model(x, h, node_mask, edge_mask, context)
        N = node_mask.squeeze(2).sum(1).long()
        log_pN = self.nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)
        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.

        return nll, reg_term, mean_abs_z
    
    def augment_noise(self,x,one_hot,node_mask,charges):
        """
        """
        # add noise 
        x = remove_mean_with_mask(x, node_mask)
        if self.noising_config.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * self.noising_config.augment_noise
        x = remove_mean_with_mask(x, node_mask)
        if self.noising_config.data_augmentation:
            x = random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)
        h = {'categorical': one_hot, 'integer': charges}
        return x, h
    
    def training_step(self,databatch, batch_idx):
        optimizer = self.optimizers()
        #organize data
        #databatch = self.DatabatchNameTuple(*batch)._asdict()
        x = databatch['positions'].to(self.dtype)
        node_mask = databatch['atom_mask'].to(self.dtype).unsqueeze(2)
        edge_mask = databatch['edge_mask'].to(self.dtype)
        one_hot = databatch['one_hot'].to(self.dtype)
        charges = (databatch['charges'] if self.data_config.include_charges else torch.zeros(0)).to(x.device, self.dtype)
        # noise handling
        x,h = self.augment_noise(x,one_hot,node_mask,charges)
        if len(self.conditioning) > 0:
            context = prepare_context(self.conditioning, 
                                      databatch, 
                                      self.property_norms).to(x.device, self.dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None
        # transform batch through flow
        nll, reg_term, mean_abs_z = self.loss(x, h, node_mask,edge_mask, context)
        # standard nll from forward KL
        loss = nll + self.noising_config.ode_regularization * reg_term
        # optimization
        optimizer.zero_grad()
        self.manual_backward(loss)
        # clip grad norm
        if self.config.trainer.clip_grad: grad_norm = self.gradient_clipping(self.gradnorm_queue) 
        else: grad_norm = 0.
        optimizer.step()
        # Update EMA if enabled.
        if self.do_ema:
            self.update_ema()
        self.log('test_loss', loss, on_epoch=True,on_step=True, prog_bar=True, logger=True)
        return nll
    
    def validation_step(self,databatch, batch_idx):
        #organize data
        #databatch = self.DatabatchNameTuple(*batch)._asdict()
        x = databatch['positions'].to(self.dtype)
        node_mask = databatch['atom_mask'].to(self.dtype).unsqueeze(2)
        edge_mask = databatch['edge_mask'].to(self.dtype)
        one_hot = databatch['one_hot'].to(self.dtype)
        charges = (databatch['charges'] if self.data_config.include_charges else torch.zeros(0)).to(x.device, self.dtype)
        # noise handling
        x,h = self.augment_noise(x,one_hot,node_mask,charges)
        if len(self.conditioning) > 0:
            context = prepare_context(self.conditioning, 
                                      databatch, 
                                      self.property_norms).to(self.device, self.dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None
        # transform batch through flow
        nll, reg_term, mean_abs_z = self.loss(x, h, node_mask,edge_mask, context)
                                              
        # standard nll from forward KL
        loss = nll + self.noising_config.ode_regularization * reg_term
        self.log('val_loss', loss, on_epoch=True,on_step=True, prog_bar=True, logger=True)
        return nll

    def gradient_clipping(self, gradnorm_queue):
        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

        # Clips gradient and returns the norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=max_grad_norm, norm_type=2.0)

        if float(grad_norm) > max_grad_norm:
            gradnorm_queue.add(float(max_grad_norm))
        else:
            gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                f'while allowed {max_grad_norm:.1f}')
        return grad_norm
    
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
        
        self.lr = self.config.trainer.learning_rate

        # GRADIENT CLIPPING QUEUE
        self.gradnorm_queue = Queue()
        self.gradnorm_queue.add(3000)  # Add large value that will be flushed.

        return optimizer


class EDGML(AbstractGenerativeModelL):

    config_type = EDMGConfig
    model:EquivariantDiffussionNoisingL=None
    dataloader:QM9PointDataloader=None
    pipeline:EDGMPipeline=None
    
    def define_from_config(self,config:EDMGConfig):
        self.config = config
        self.dataloader = get_dataloaders(self.config)
        self.model = EquivariantDiffussionNoisingL(self.config,self.dataloader)
        self.pipeline = EDGMPipeline(self.config,self.model,self.dataloader)

    def define_from_dir(self, experiment_dir:str|ExperimentFiles=None, checkpoint_type: str = "best"):
        # define experiments files
        if isinstance(experiment_dir,str):
            self.experiment_files = ExperimentFiles(experiment_dir=experiment_dir)
        else:
            self.experiment_files = experiment_dir
        # read config
        self.config = self.read_config(self.experiment_files)
        # obtain dataloader
        self.dataloader = get_dataloaders(self.config,self.dataloader)
        # obtain checkpoint path
        CKPT_PATH = self.experiment_files.get_lightning_checkpoint_path(checkpoint_type)
        # load model
        self.model = EquivariantDiffussionNoisingL.load_from_checkpoint(CKPT_PATH, config=self.config)
        self.pipeline = EDGMPipeline(self.config,self.model,self.dataloader)
        return self.config
    
    def test_evaluation(self) -> dict:
        return {}
        #if len(self.config.noising_model.conditioning) > 0:
        #        save_and_sample_conditional(self.pipeline,epoch=1000)
        #save_and_sample_chain(self.pipeline,epoch=epoch,batch_id=str(i))
        #sample_different_sizes_and_save(self.pipeline, epoch=epoch)
        
    