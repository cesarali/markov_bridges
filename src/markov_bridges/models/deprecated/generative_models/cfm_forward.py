import torch
from torch import nn
from markov_bridges.models.networks.utils.ema import EMA
from markov_bridges.configs.config_classes.generative_models.cfm_config import CFMConfig
from markov_bridges.models.networks.temporal.cfm.continuous_networks_utils import load_continuous_network
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

class ContinuousForwardMap(EMA, nn.Module):
    """
    This corresponds to the torch module which contains all the elements requiered to 
    sample and train a Continuous Variable Bridge

    """
    def __init__(self, config: CFMConfig, device):
        """
        join_context(context_discrete,discrete_data,context_continuous,continuuous_data)->full_discrete,full_continuous: 
        this function should allow us to create a full discrete and continuous vector from the context and data

        """
        EMA.__init__(self, config)
        nn.Module.__init__(self)

        self.config = config
        config_data = config.data

        self.vocab_size = 0

        self.has_context_discrete = config_data.has_context_discrete     
        self.has_context_continuous = config_data.has_context_continuous 

        self.has_target_discrete = False
        self.has_target_continuous = True 

        self.continuos_dimensions = config_data.continuos_dimensions
        self.discrete_dimensions = 0
    
        self.context_discrete_dimension = config_data.context_discrete_dimension
        self.context_continuous_dimension = config_data.context_continuous_dimension

        self.define_deep_models(config, device)
        self.define_bridge_parameters(config)
        
        self.discrete_loss_nn = None
        self.continuous_loss_nn = nn.MSELoss(reduction='none')

        self.to(device)
        self.init_ema()


    def define_deep_models(self,config,device):
        self.continuous_network = load_continuous_network(config, device=device)
        
    def define_bridge_parameters(self,config):
        self.continuous_bridge_ = None
        
    #====================================================================
    # SAMPLE BRIDGE
    #====================================================================


    def sample_continuous_bridge(self, x1, x0, time):
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
    
    def sample_bridge(self, databatch):
        time = databatch.time.flatten()
        source = databatch.source_continuous
        target = databatch.target_continuous     
        continuous_sample = self.sample_continuous_bridge(target, source, time)
        return continuous_sample
    
    #====================================================================
    # LOSS
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
