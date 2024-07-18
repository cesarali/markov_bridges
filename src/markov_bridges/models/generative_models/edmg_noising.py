import torch
from torch import nn

from dataclasses import dataclass,asdict,field
from torch.distributions import Categorical,Normal,Dirichlet

from markov_bridges.models.networks.utils.ema import EMA
from markov_bridges.models.pipelines.thermostat_utils import load_thermostat
from markov_bridges.configs.config_classes.generative_models.edmg_config import EDMGConfig

from markov_bridges.utils.shapes import right_shape,right_time_size,where_to_go_x
from markov_bridges.models.pipelines.thermostats import Thermostat
from markov_bridges.models.networks.temporal.mixed.mixed_networks_utils import load_mixed_network
from markov_bridges.data.qm9_points_dataloader import QM9PointDataNameTuple
from torch.nn.functional import softmax

def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


class EquivariantDiffussionNoising(EMA,nn.Module):
    """
    This corresponds to the torch module which contains all the elements requiered to 
    sample and train a Mixed Variable Bridge

    """
    def __init__(self, config:EDMGConfig,device,join_context=None):
        """
        join_context(context_discrete,discrete_data,context_continuous,continuuous_data)->full_discrete,full_continuous: 
        this function should allow us to create a full discrete and continuous vector from the context and data

        """
        EMA.__init__(self,config)
        nn.Module.__init__(self)

        self.config = config
        config_data = config.data

        self.vocab_size = config_data.vocab_size

        self.has_context_discrete = config_data.has_context_discrete     
        self.has_context_continuous = config_data.has_context_continuous 

        self.has_target_discrete = config_data.has_target_discrete 
        self.has_target_continuous = config_data.has_target_continuous 

        self.continuos_dimensions = config_data.continuos_dimensions
        self.discrete_dimensions = config_data.discrete_dimensions
    
        self.context_discrete_dimension = config_data.context_discrete_dimension
        self.context_continuous_dimension = config_data.context_continuous_dimension

        self.join_context = join_context

        self.define_deep_models(config,device)
        self.define_bridge_parameters(config)
        
        self.nodes_dist = Normal(0.,1.)

        self.to(device)
        self.init_ema()

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
    def loss(self,databatch:QM9PointDataNameTuple,discrete_sample,continuous_sample):
        """
        """
        x = databatch.target_continuous
        h = databatch.target_discrete
        node_mask = databatch.node_mask
        edge_mask = databatch.edge_mask
        context = databatch.context_discrete
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
