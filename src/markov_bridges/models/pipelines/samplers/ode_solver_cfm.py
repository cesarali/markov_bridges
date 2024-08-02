import torch
from typing import Union
from torch import functional as F
from torchdyn.core import NeuralODE

from markov_bridges.configs.config_classes.generative_models.cfm_config import CFMConfig
from markov_bridges.models.generative_models.cfm_forward import ContinuousForwardMap
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple


class MixedTauState:
    """
    Dataclass that defines the output and state of for generating a mix variables model

    The generation of mixed variables requieres handling of the discrete as well as continuous variables
    this class defines variables for each type of variable and defines list to store the paths of each
    of the variables if requiered
    """
    discrete: torch.Tensor = None
    continuous: torch.Tensor = None
    save_ts: torch.Tensor = None
    number_of_paths: int = 0 

def ODESamplerCFM(config: CFMConfig,
                  drift_model: ContinuousForwardMap,
                  x_0: MarkovBridgeDataNameTuple,
                  return_path=False):
    """
    :param drift_model:
    :param x_0:
    :param N:
    :return:
    """

    num_steps = config.pipeline.number_of_steps
    ode_solver = config.pipeline.ode_solver
    atol = config.pipeline.atol
    rtol = config.pipeline.rtol
    sensitivity = config.pipeline.sensitivity
    device = x_0.source_continuous.device
    time_steps = torch.linspace(0, 1, num_steps, device=device)

    node = NeuralODE(vector_field=TorchdynWrapper(drift_model.continuous_network, 
                                                  context_discrete=x_0.context_discrete if config.data.has_context_discrete else None, 
                                                  context_continuous=x_0.context_continuous if config.data.has_context_continuous else None), 
                     solver=ode_solver, 
                     sensitivity=sensitivity, 
                     seminorm=True if ode_solver=='dopri5' else False,
                     atol=atol if ode_solver=='dopri5' else None,
                     rtol=rtol if ode_solver=='dopri5' else None)
    
    trajectories = node.trajectory(x=x_0.source_continuous, t_span=time_steps).detach().cpu()
    trajectories = trajectories.detach().float()

    return trajectories


#----------------------------------------------
# utils for pipelines            
#----------------------------------------------

class TorchdynWrapper(torch.nn.Module):
    """ Wraps model to torchdyn compatible format.
    """
    def __init__(self, net, context_discrete=None, context_continuous=None):
        super().__init__()
        self.nn = net
        self.context_discrete = context_discrete
        self.context_continuous = context_continuous

    def forward(self, t, x, args):
        t = t.repeat(x.shape[0])
        t = reshape_time_like(t, x)
        return self.nn(x_continuous=x, times=t, context_discrete=self.context_discrete, context_continuous=self.context_continuous)

def reshape_time_like(t, x):
	if isinstance(t, (float, int)): return t
	else: return t.reshape(-1, *([1] * (x.dim() - 1)))
     
