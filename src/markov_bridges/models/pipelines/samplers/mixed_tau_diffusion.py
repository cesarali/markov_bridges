import torch
import numpy as np
from tqdm import tqdm
from typing import Union, Tuple
from torch import functional as F
from torch.distributions import Normal

from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple
from markov_bridges.models.deprecated.generative_models.cmb_forward import MixedForwardMap
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig

class MixedTauState:
    """
    Dataclass that defines the output and state of for generating a mix variables model

    The generation of mixed variables requieres handling of the discrete as well as continuous variables
    this class defines variables for each type of variable and defines list to store the paths of each
    of the variables if requiered
    """
    discrete: torch.Tensor = None
    continuous: torch.Tensor = None
    state_0:MarkovBridgeDataNameTuple=None
    save_ts: torch.Tensor = None
    number_of_paths: int = 0 

    def __init__(self, config:CMBConfig, state: MarkovBridgeDataNameTuple,save_ts=None):
        self.has_target_continuous = config.data.has_target_continuous
        self.has_target_discrete =  config.data.has_target_discrete

        self.discrete_paths: torch.Tensor = []
        self.continuous_paths: torch.Tensor = []
        
        if config.data.has_target_discrete:
            self.discrete = state.source_discrete
            self.device = self.discrete.device
            self.number_of_paths = self.discrete.size(0)

        if config.data.has_target_continuous:
            self.continuous = state.source_continuous
            self.device = self.continuous.device
            self.number_of_paths = self.continuous.size(0)

        self.state_0 = state
        self.save_ts = save_ts

    def update_state(self, new_discrete, new_continuos):
        self.discrete = new_discrete
        self.continuous = new_continuos
    
    def update_paths(self, new_discrete, new_continuos):
        if self.has_target_discrete:
            self.discrete_paths.append(new_discrete.clone().detach().unsqueeze(1))
        if self.has_target_continuous:
            self.continuous_paths.append(new_continuos.clone().detach().unsqueeze(1))
    
    def concatenate_paths(self):
        if len(self.discrete_paths) > 0:
            self.discrete_paths = torch.cat(self.discrete_paths, dim=1).float()
        if len(self.continuous_paths) > 0:
            self.continuous_paths = torch.cat(self.continuous_paths, dim=1).float()

class TauDiffusion:
    """
    This class performs the sample for mixed variables, combines a tau leaping step 
    for discrete variables and an euler mayorama step for continuous variables

    If the return_path is set to True during sampling, the states at save_ts times will be stored.
    save_ts is defined for the whole path if number_of_intermediaries is set to None
    """
    def __init__(self, config: CMBConfig):
        self.config = config
        self.D = config.data.discrete_dimensions
        self.S = config.data.vocab_size
        self.num_steps = config.pipeline.number_of_steps
        self.num_intermediates = config.pipeline.num_intermediates
        self.max_rate_at_end = config.pipeline.max_rate_at_end
        self.brownian_sigma = config.brownian_sigma
        
        self.solver_type = config.pipeline.solver
        self.time_epsilon = config.pipeline.time_epsilon
        self.min_t = 1./self.num_steps

        self.has_target_continuous = config.data.has_target_continuous
        self.has_target_discrete = config.data.has_target_discrete

    def define_time(self, return_path):
        """
        If the return_path is set to True during sampling, the states at save_ts times will be stored.
        save_ts is defined for the whole path if number_of_intermediaries is set to None
        """
        # define time steps as well as the time where to save the paths
        ts = np.concatenate((np.linspace(1.0 - self.time_epsilon, self.min_t, self.num_steps), np.array([0])))
        if return_path:
            if self.num_intermediates is None:
                save_ts = np.concatenate((np.linspace(1.0 - self.time_epsilon, self.min_t, self.num_steps), np.array([0])))
            else:
                save_ts = ts[np.linspace(0, len(ts)-2, self.num_intermediates, dtype=int)]
            save_ts = save_ts[::-1]
        else:
            save_ts = None
        ts = ts[::-1]

        return ts, save_ts

    def TauStep(self, rates, h, state: MixedTauState, end: bool = False):
        x = state.discrete
        number_of_paths = state.number_of_paths
        device = x.device
        # TAU LEAPING
        if not end:
            diffs = torch.arange(self.S, device=device).view(1, 1, self.S) - x.view(number_of_paths, self.D, 1)
            poisson_dist = torch.distributions.poisson.Poisson(rates * h)
            jump_nums = poisson_dist.sample().to(device)
            adj_diffs = jump_nums * diffs
            overall_jump = torch.sum(adj_diffs, dim=2)
            xp = x + overall_jump
            x_new = torch.clamp(xp, min=0, max=self.S-1)
        # END
        if end:
            x_new = torch.max(rates, dim=2)[1]
        return x_new
    
    def EulerStep(self,vector_field,dt,state: MixedTauState,end:bool=False):
        x = state.continuous
        x = x + dt * vector_field.to(x.device)
        return x
    
    def DiffusionStep(self, drift, dt, state: MixedTauState):
        #x_mean = x + drift * dt
        #x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        x_mean = state.continuous + drift*dt
        z = torch.randn_like(x_mean)
        x = x_mean + self.brownian_sigma* np.sqrt(dt) * z
        return x

    def sample(self,
               forward_model: Union[MixedForwardMap],
               state_0: MarkovBridgeDataNameTuple,
               return_path=False) -> MixedTauState:
        """
        """
        #==========================================
        # CONDITIONAL SAMPLING
        #==========================================
        with torch.no_grad():
            # Define Simulation Times
            ts, save_ts = self.define_time(return_path)
            # Define States 
            state = MixedTauState(self.config,state_0,save_ts)
            
            new_discrete = state.discrete
            new_continuous = state.continuous

            for idx, t in tqdm(enumerate(ts[0:-1])):
                # handles current time
                h = ts[idx+1] - ts[idx]
                times = self.get_time_vector(t, state.number_of_paths, state.device)

                rates, drift = forward_model.forward_map(state.discrete,
                                                         state.continuous,
                                                         times,
                                                         state_0)
                # moves the variables forward
                if self.has_target_discrete:
                    new_discrete = self.TauStep(rates, h, state)
                if self.has_target_continuous:
                    if self.solver_type == 'ode_tau':
                        new_continuous = self.EulerStep(drift,h,state)
                    if self.solver_type == 'sde_tau':
                        new_continuous = self.DiffusionStep(drift, h, state)

                # store the paths if requiered
                if save_ts is not None:
                    if t in save_ts:
                        state.update_paths(new_discrete, new_continuous)

                state.update_state(new_discrete, new_continuous)

            # LAST STEP
            last_time = self.min_t * torch.ones((state.number_of_paths,), device=state.device)
            rates, drift = forward_model.forward_map(state.discrete,
                                                     state.continuous,
                                                     last_time,
                                                     state_0)
            if self.has_target_discrete:
                new_discrete = self.TauStep(rates, h, state, end=self.max_rate_at_end)
            if self.has_target_continuous:
                if self.solver_type == 'ode_tau':
                    new_continuous = self.EulerStep(drift,h,state)
                if self.solver_type == 'sde_tau':
                    new_continuous = self.DiffusionStep(drift, h, state)
                    
            state.update_state(new_discrete, new_continuous)

            # SAVE LAST STEP
            if save_ts is not None:
                state.update_paths(new_discrete, new_continuous)
            state.concatenate_paths()
            return state
                
    def get_time_vector(self, t, number_of_paths, device):
        """
        Simply repeats the time value to obtain a tensor in the right device
        """
        times = t * torch.ones(number_of_paths,).to(device)
        return times
