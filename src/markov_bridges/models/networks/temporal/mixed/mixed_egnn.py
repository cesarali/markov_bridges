import torch
from torch import nn

from markov_bridges.data.qm9.qm9_points_dataloader import QM9PointDataNameTupleCMB
from markov_bridges.models.networks.temporal.edmg.egnn_dynamics import EGNN_dynamics_QM9
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.data.qm9.utils import prepare_context

from markov_bridges.utils.equivariant_diffusion import (
    assert_correctly_masked
)

class MixedEGNN_dynamics_QM9(nn.Module):
    """
    Wrapper for QM9 EGNN dynamics
    """
    def __init__(self, config:CMBConfig, device=None):
        super().__init__()
        self.config = config
        self.vocab_size = config.data.vocab_size
        self.continuos_dimensions = config.data.continuos_dimensions

        # node features
        #self.dynamics_in_node_nf = self.vocab_size + 1 + int(config.data.include_charges)
        # categories plus time
        self.discrete_emb_dim = config.mixed_network.discrete_emb_dim
        self.dynamics_in_node_nf = self.discrete_emb_dim + 1

        # context features dimensions
        self.context_node_nf = config.data.context_node_nf

        self.conditioning = config.mixed_network.conditioning
        self.property_norms = config.data.property_norms
        self.dtype = torch.float32
        self.define_deep_models(config)
        
    def define_deep_models(self,config:CMBConfig,device=None):
        self.net_dynamics = EGNN_dynamics_QM9(
            in_node_nf=self.dynamics_in_node_nf, 
            context_node_nf=self.context_node_nf,
            n_dims=3, 
            device=device, 
            hidden_nf=config.mixed_network.nf,
            act_fn=torch.nn.SiLU(), 
            n_layers=config.mixed_network.n_layers,
            attention=config.mixed_network.attention, 
            tanh=config.mixed_network.tanh, 
            mode=config.mixed_network.mode, 
            norm_constant=config.mixed_network.norm_constant,
            inv_sublayers=config.mixed_network.inv_sublayers, 
            sin_embedding=config.mixed_network.sin_embedding,
            normalization_factor=config.mixed_network.normalization_factor, 
            aggregation_method=config.mixed_network.aggregation_method)
        
        self.embedding = nn.Embedding(self.vocab_size,self.discrete_emb_dim)
        self.embedding_to_rate = nn.Linear(self.discrete_emb_dim,self.vocab_size)

        if device is not None:
            self.to(device)

    def forward(self,x_discrete,x_continuous,times,databatch:QM9PointDataNameTupleCMB):
        """
        """
        # shapes of egnn requieres batch_size,num_atoms,dim
        bs = databatch.batch_size
        num_atoms = databatch.max_num_atoms
        x_discrete = x_discrete.reshape(bs,num_atoms,1)
        x_continuous = x_continuous.reshape(bs,num_atoms,self.continuos_dimensions)

        #embed
        embbeded_discrete = self.embedding(x_discrete).squeeze()
        xh = torch.cat([x_continuous,embbeded_discrete],dim=2)

        # encode
        head = self.net_dynamics._forward(times, 
                                          xh, 
                                          databatch.atom_mask, 
                                          databatch.edge_mask, 
                                          databatch.context)
        
        continuous_head = head[:,:,:self.continuos_dimensions]
        continuous_head = continuous_head.reshape(databatch.batch_size,-1)
        discrete_head = head[:,:,self.continuos_dimensions:]
        discrete_head_ = self.embedding_to_rate(discrete_head)

        return discrete_head,continuous_head
    
    def phi(self,t,zt, node_mask, edge_mask, context):
        return self.net_dynamics._forward(t,zt, node_mask, edge_mask, context)