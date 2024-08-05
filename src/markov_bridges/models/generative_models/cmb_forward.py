import torch
from torch import nn

from dataclasses import dataclass,asdict,field
from torch.distributions import Categorical,Normal,Dirichlet

from markov_bridges.models.networks.utils.ema import EMA
from markov_bridges.models.pipelines.thermostat_utils import load_thermostat
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.utils.shapes import right_shape,right_time_size,where_to_go_x
from markov_bridges.models.pipelines.thermostats import Thermostat
from markov_bridges.models.networks.temporal.mixed.mixed_networks_utils import load_mixed_network
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple
from torch.nn.functional import softmax

class MixedForwardMap(EMA,nn.Module):
    """
    This corresponds to the torch module which contains all the elements requiered to 
    sample and train a Mixed Variable Bridge

    """
    def __init__(self, config:CMBConfig,device,join_context=None):
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
        self.continuous_loss_type = config.continuous_loss_type

        self.join_context = join_context

        self.define_deep_models(config,device)
        self.define_bridge_parameters(config)
        
        self.discrete_loss_nn = nn.CrossEntropyLoss(reduction='none')
        self.continuous_loss_nn = nn.MSELoss(reduction='none')

        self.to(device)
        self.init_ema()

    def define_deep_models(self,config,device):
        self.mixed_network = load_mixed_network(config,device=device)
        
    def define_bridge_parameters(self,config):
        self.discrete_bridge_:Thermostat = load_thermostat(config)
        self.continuous_bridge_ = None
        
    #====================================================================
    # SAMPLE BRIDGE
    #====================================================================
    def sample_discrete_bridge(self,x_1,x_0,time):
        device = x_1.device
        x_to_go = where_to_go_x(x_0,self.vocab_size)
        transition_probs = self.telegram_bridge_probability(x_to_go, x_1, x_0, time)
        sampled_x = Categorical(transition_probs).sample().to(device)
        return sampled_x
    
    def sample_continuous_bridge(self,x_1,x_0,time):
        """
        simple brownian bridge
        """
        device = x_1.device
        original_shape = x_0.shape
        continuous_dimensions = x_1.size(1)
        time_ = time[:,None].repeat((1,continuous_dimensions))

        t = time_.flatten()
        x_1 = x_1.flatten()
        x_0 = x_0.flatten()

        x_m = x_0*(1.-t) + x_1*t
        variance = t*(1. - t)

        x = Normal(x_m,variance).sample().to(device)
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
    # RATES AND DRIFT for GENERATION
    #====================================================================
    def discrete_rate(self,change_logits,x,time):
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
    
    def continuous_drift(self,x1,x,time):
        if len(time.shape) == 1:
            time = time[:,None]
        drift = (x1 - x)/(1.-time)
        return drift
    
    def continuous_flow(self,x,x1,x0,time):
        if len(time.shape) == 1:
            time = time[:,None]
        A = (1.-2*time)/(time*(1.-time))
        x_m = x0*(1.-time) + x1*time
        flow = A*(x - x_m) + (x1- x0)
        return flow
    
    def forward_map(self,discrete_sample,continuous_sample,time):
        if len(time.shape) > 1:
            time = time.flatten()

        discrete_head,continuous_head = self.mixed_network(discrete_sample,continuous_sample,time)

        if self.has_target_discrete:
            rate = self.discrete_rate(discrete_head,discrete_sample,time)
        else:
            rate = None
            
        if self.has_target_continuous:
            if self.continuous_loss_type == "regression":
                drift = self.continuous_drift(continuous_head,continuous_sample,time)
            elif self.continuous_loss_type == "flow":
                drift = continuous_head
            elif self.continuous_loss_type == "drift":
                drift = continuous_head
        else:
            drift = None
        return rate,drift
    
    #====================================================================
    # LOSS
    #====================================================================
    def loss(self,databatch:MarkovBridgeDataNameTuple,discrete_sample,continuous_sample):
        # IF WE HAVE CONTEXT JOIN FOR FULL DATA
        if self.has_context_continuous or self.has_context_discrete:
            discrete_sample,continuous_sample = self.join_context(databatch,
                                                                  discrete_sample,
                                                                  continuous_sample)
        
        # Calculate Heads For Classifier or Mean Average
        discrete_head,continuous_head = self.mixed_network(discrete_sample,continuous_sample,databatch.time)
        
        # Train What is Needed
        full_loss = torch.Tensor([0.]).to(discrete_sample.device if discrete_sample is not None else continuous_sample.device)
        
        if self.has_target_discrete:
            full_loss += self.discrete_loss(databatch,discrete_head,discrete_sample).mean()

        if self.has_target_continuous:
            full_loss += self.continuous_loss(databatch,continuous_head,continuous_sample).mean()

        return full_loss
    
    def discrete_loss(self,databatch:MarkovBridgeDataNameTuple,discrete_head,discrete_sample=None):
        # If has context remove the part predicting context
        if self.has_context_discrete:
            discrete_head = discrete_head[:, self.context_discrete_dimension:,:]
        
        # reshape for cross logits
        discrete_head = discrete_head.reshape(-1, self.config.data.vocab_size)
        target_discrete = databatch.target_discrete.reshape(-1)
        discrete_loss = self.discrete_loss_nn(discrete_head,target_discrete.long())
        return discrete_loss
    
    def continuous_loss(self,databatch:MarkovBridgeDataNameTuple,continuous_head,continuous_sample=None):
        # If has context continuous
        if self.has_context_continuous:
            continuous_head = continuous_head[:, self.context_continuous_dimension:,:]
        # pick loss
        if self.continuous_loss_type == "flow":
            conditional_flow = self.continuous_flow(continuous_sample,databatch.target_continuous,databatch.source_continuous,databatch.time)
            mse = self.continuous_loss_nn(conditional_flow,databatch.target_continuous)
        elif self.continuous_loss_type == "drift":
            conditional_drift = self.continuous_drift(databatch.target_continuous,continuous_sample,databatch.time)
            mse = self.continuous_loss_nn(continuous_head,conditional_drift)
        elif self.continuous_loss_type == "regression":
            mse = self.continuous_loss_nn(continuous_head,databatch.target_continuous)
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
        t = right_time_size(t,x).to(x0.device)
        t0 = right_time_size(t0,x).to(x0.device)

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