from torch import nn
import torch.nn.functional as F
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig, TemporalNetworkToRateConfig, nn


import torch


from functools import reduce


class TemporalToRateLogistic(nn.Module):
    """
    # Truncated logistic output from https://arxiv.org/pdf/2107.03006.pdf
    """
    def __init__(self, config:CJBConfig,temporal_output_total,device):
        nn.Module.__init__(self)
        self.D = config.data.discrete_dimensions
        self.S = config.data.vocab_size
        self.device = device
        self.fix_logistic = config.temporal_network_to_rate.fix_logistic

    def forward(self,net_out):
        B = net_out.shape[0]
        D = self.D
        C = 3
        S = self.S
        net_out = net_out.view(B,2*C,32,32)

        mu = net_out[:, 0:C, :, :].unsqueeze(-1)
        log_scale = net_out[:, C:, :, :].unsqueeze(-1)

        inv_scale = torch.exp(- (log_scale - 2))

        bin_width = 2. / self.S
        bin_centers = torch.linspace(start=-1. + bin_width/2,
            end=1. - bin_width/2,
            steps=self.S,
            device=self.device).view(1, 1, 1, 1, self.S)

        sig_in_left = (bin_centers - bin_width/2 - mu) * inv_scale
        bin_left_logcdf = F.logsigmoid(sig_in_left)
        sig_in_right = (bin_centers + bin_width/2 - mu) * inv_scale
        bin_right_logcdf = F.logsigmoid(sig_in_right)

        logits_1 = self._log_minus_exp(bin_right_logcdf, bin_left_logcdf)
        logits_2 = self._log_minus_exp(-sig_in_left + bin_left_logcdf, -sig_in_right + bin_right_logcdf)
        if self.fix_logistic:
            logits = torch.min(logits_1, logits_2)
        else:
            logits = logits_1
        logits = logits.view(B,D,S)

        return logits

    def _log_minus_exp(self, a, b, eps=1e-6):
        """ 
            Compute log (exp(a) - exp(b)) for (b<a)
            From https://arxiv.org/pdf/2107.03006.pdf
        """
        return a + torch.log1p(-torch.exp(b-a) + eps)


class TemporalToRateEmpty(nn.Module):
    """
    Directly Takes the Output and converts into a rate
    """
    def __init__(self,  config:CJBConfig,temporal_output_total,device):
        nn.Module.__init__(self)
        self.device = device

    def forward(self,x):
        return x


class TemporalToRateBernoulli(nn.Module):
    """
    Takes the output of the temporal rate as bernoulli probabilities completing 
    with 1 - p
    """
    def __init__(self, config:CJBConfig, temporal_output_total,device):
        nn.Module.__init__(self)
        self.device = device

    def forward(self,x):
        #here we expect len(x.shape) == 2
        x_ = torch.zeros_like(x)
        x = torch.cat([x[:,:,None],x_[:,:,None]],dim=2)
        return x


class TemporalToRateLinear(nn.Module):
    """
    Assigns a linear Layer
    """
    def __init__(self, config:CJBConfig, temporal_output_total,device):
        nn.Module.__init__(self)
        self.vocab_size = config.data.vocab_size
        self.dimensions = config.data.discrete_dimensions
        self.temporal_output_total = temporal_output_total
        self.device = device

        if isinstance(config.temporal_network_to_rate,TemporalNetworkToRateConfig):
            intermediate_to_rate = config.temporal_network_to_rate.linear_reduction
        else:
            intermediate_to_rate = config.temporal_network_to_rate

        if intermediate_to_rate is None:
            self.temporal_to_rate = nn.Linear(temporal_output_total,self.dimensions*self.vocab_size)
        else:

            if isinstance(intermediate_to_rate,float):
                assert intermediate_to_rate < 1.
                intermediate_to_rate = int(self.dimensions * self.vocab_size * intermediate_to_rate)
            self.temporal_to_rate = nn.Sequential(
                nn.Linear(temporal_output_total, intermediate_to_rate),
                nn.Linear(intermediate_to_rate, self.dimensions * self.vocab_size)
            )

    def forward(self,x):
        return self.temporal_to_rate(x)


def select_temporal_to_rate(config:CJBConfig, expected_temporal_output_shape,device=torch.device("cpu")):

    temporal_output_total = reduce(lambda x, y: x * y,expected_temporal_output_shape)
    temporal_network_to_rate = config.temporal_network_to_rate

    if isinstance(temporal_network_to_rate,TemporalNetworkToRateConfig):
        type_of = temporal_network_to_rate.type_of
        if type_of == "bernoulli":
             temporal_to_rate = TemporalToRateBernoulli(config,temporal_output_total,device)
        elif type_of == "empty":
            temporal_to_rate = TemporalToRateEmpty(config,temporal_output_total,device)
        elif type_of == "linear":
            temporal_to_rate = TemporalToRateLinear(config,temporal_output_total,device)
        elif type_of == "logistic":
            temporal_to_rate = TemporalToRateLogistic(config,temporal_output_total,device)
        elif type_of is None:
            config.temporal_network_to_rate.linear_reduction = None
            temporal_to_rate = TemporalToRateLinear(config,temporal_output_total,device)
    else:
        temporal_to_rate = TemporalToRateLinear(config,temporal_output_total,device)

    temporal_to_rate.device = device
    temporal_to_rate = temporal_to_rate.to(device)

    return temporal_to_rate