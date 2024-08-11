import torch
from torch import nn

from dataclasses import dataclass,asdict,field
from torch.distributions import Categorical,Normal,Dirichlet
import markov_bridges.data.qm9.utils as qm9utils
from markov_bridges.models.networks.utils.ema import EMA
from markov_bridges.models.pipelines.thermostat_utils import load_thermostat
from markov_bridges.configs.config_classes.generative_models.edmg_config import EDMGConfig

from torch.nn.functional import softmax
from markov_bridges.utils.shapes import right_shape,right_time_size,where_to_go_x
from markov_bridges.models.pipelines.thermostats import Thermostat
from markov_bridges.models.networks.temporal.mixed.mixed_networks_utils import load_mixed_network
from markov_bridges.data.qm9.qm9_points_dataloader import QM9PointDataNameTuple
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple


from markov_bridges.utils.equivariant_diffusion import (
    assert_mean_zero_with_mask, 
    remove_mean_with_mask,
    assert_correctly_masked, 
    sample_center_gravity_zero_gaussian_with_mask,
    random_rotation,
    gradient_clipping,
    Queue
)

from markov_bridges.data.qm9.utils import prepare_context, compute_mean_mad

def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

class EquivariantDiffussionNoisingL(EMA,nn.Module):
    """
    This corresponds to the torch module which contains all the elements requiered to 
    sample and train a Mixed Variable Bridge

    """
    def __init__(self, config:EDMGConfig,device,join_context=None):
        """
        this function should allow us to create a full discrete and continuous vector from the context and data

        """
        EMA.__init__(self,config)
        nn.Module.__init__(self)

        self.config = config
        config_data = config.data
        
        self.noising_config = config.noising_model
        self.data_config = config_data

        self.vocab_size = config_data.vocab_size

        self.has_target_discrete = config_data.has_target_discrete 
        self.has_target_continuous = config_data.has_target_continuous 

        self.continuos_dimensions = config_data.continuos_dimensions
        self.discrete_dimensions = config_data.discrete_dimensions
    
        self.context_discrete_dimension = config_data.context_discrete_dimension
        self.context_continuous_dimension = config_data.context_continuous_dimension

        self.define_deep_models(config,device)
        self.define_bridge_parameters(config)
        
        self.nodes_dist = Normal(0.,1.)

        self.device = device
        self.to(device)
        self.init_ema()

    def to(self,device):
        self.device = device 
        return super().to(device)

    def define_deep_models(self,config,device):
        self.mixed_network = load_mixed_network(config,device=device)
        
    def define_bridge_parameters(self,config):
        self.discrete_bridge_:Thermostat = load_thermostat(config)
        self.continuous_bridge_ = None
            
    #====================================================================
    # RATES AND DRIFT for GENERATION
    #====================================================================
    
    def forward_map(self,discrete_sample,continuous_sample,time):
        return None
    
    #====================================================================
    # LOSS
    #====================================================================
    def loss(self,x, h, node_mask, edge_mask, context):
        """
        """
        bs, n_nodes, n_dims = x.size()

        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        nll = self.mixed_network(x, h, node_mask, edge_mask, context)

        N = node_mask.squeeze(2).sum(1).long()

        log_pN = self.nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.

        return nll, reg_term, mean_abs_z
    
    def train_step(self,databatch:MarkovBridgeDataNameTuple, number_of_training_step,  epoch):
        x = databatch['positions'].to(self.device, self.dtype)
        node_mask = databatch['atom_mask'].to(self.device, self.dtype).unsqueeze(2)
        edge_mask = databatch['edge_mask'].to(self.device, self.dtype)
        one_hot = databatch['one_hot'].to(self.device, self.dtype)
        charges = (databatch['charges'] if self.data_config.include_charges else torch.zeros(0)).to(self.device, self.dtype)

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

        if len(self.noising_config.conditioning) > 0:
            context = qm9utils.prepare_context(self.noising_config.conditioning, 
                                               databatch, 
                                               self.property_norms).to(self.device, self.dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        self.optimizer.zero_grad()

        # transform batch through flow
        nll, reg_term, mean_abs_z = self.loss(x, 
                                              h, 
                                              node_mask, 
                                              edge_mask, 
                                              context,
                                              self.nodes_dist)
        
        # standard nll from forward KL
        loss = nll + self.noising_config.ode_regularization * reg_term
        loss.backward()

        if self.config.trainer.clip_grad:
            grad_norm = gradient_clipping(self.generative_model.noising_model, 
                                          self.gradnorm_queue)
        else:
            grad_norm = 0.

        self.optimizer.step()

        # Update EMA if enabled.
        #if args.ema_decay > 0:
        #    ema.update_model_average(model_ema, model)

        return nll

    def test_step(self,databatch:MarkovBridgeDataNameTuple, number_of_test_step,epoch):
        self.generative_model.noising_model.eval()
        with torch.no_grad():
            x = databatch['positions'].to(self.device, self.dtype)
            node_mask = databatch['atom_mask'].to(self.device, self.dtype).unsqueeze(2)
            edge_mask = databatch['edge_mask'].to(self.device, self.dtype)
            one_hot = databatch['one_hot'].to(self.device, self.dtype)
            charges = (databatch['charges'] if self.data_config.include_charges else torch.zeros(0)).to(self.device, self.dtype)

            if self.noising_model_config.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * self.noising_model_config.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(self.noising_model_config.conditioning) > 0:
                context = qm9utils.prepare_context(self.noising_model_config.conditioning, 
                                                   databatch, 
                                                   self.property_norms).to(self.device, self.dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            nll, _, _ = self.generative_model.noising_model.loss(x,
                                                                 h, 
                                                                 node_mask, 
                                                                 edge_mask, 
                                                                 context,
                                                                 self.generative_model.nodes_dist)

        return nll
