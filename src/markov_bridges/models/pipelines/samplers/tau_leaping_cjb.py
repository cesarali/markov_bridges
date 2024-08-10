import torch
import numpy as np
from tqdm import tqdm
from typing import Union
from torch import functional as F

from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.models.deprecated.generative_models.cjb_rate import ClassificationForwardRate
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple
from dataclasses import dataclass

class TauLeapingOutput:
    """
    Dataclass that defines the output and state of for generating a mix variables model

    The generation of mixed variables requieres handling of the discrete as well as continuous variables
    this class defines variables for each type of variable and defines list to store the paths of each
    of the variables if requiered
    """
    discrete: torch.Tensor = None
    save_ts: torch.Tensor = None
    number_of_paths: int = 0
    x0:MarkovBridgeDataNameTuple = None

    def __init__(self, state: MarkovBridgeDataNameTuple,save_ts=None):

        self.discrete_paths: torch.Tensor = []
        self.discrete = state.source_discrete
        self.device = self.discrete.device
        self.number_of_paths = self.discrete.size(0)
        self.save_ts = save_ts
        self.x0 = state

    def update_state(self, new_discrete):
        self.discrete = new_discrete
    
    def update_paths(self, new_discrete):
        self.discrete_paths.append(new_discrete.clone().detach().unsqueeze(1))

    def concatenate_paths(self):
        self.discrete_paths = torch.cat(self.discrete_paths, dim=1).float()

class TauLeaping:
    """
    This class performs the sample for mixed variables, combines a tau leaping step 
    for discrete variables and an euler mayorama step for continuous variables

    If the return_path is set to True during sampling, the states at save_ts times will be stored.
    save_ts is defined for the whole path if number_of_intermediaries is set to None
    """
    def __init__(self, config: CJBConfig):
        self.config = config
        self.D = config.data.discrete_generation_dimension
        self.S = config.data.vocab_size
        self.num_steps = config.pipeline.number_of_steps
        self.num_intermediates = config.pipeline.num_intermediates
        self.max_rate_at_end = config.pipeline.max_rate_at_end

        self.time_epsilon = config.pipeline.time_epsilon
        self.min_t = 1./self.num_steps

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

    def TauStep(self, rates, h, state: TauLeapingOutput, end: bool = False):
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

    def sample(self,
               forward_model: Union[ClassificationForwardRate],
               state_0: MarkovBridgeDataNameTuple,
               return_path=False) -> TauLeapingOutput:
        """
        """
        #==========================================
        # CONDITIONAL SAMPLING
        #==========================================
        with torch.no_grad():
            # Define Simulation Times
            ts, save_ts = self.define_time(return_path)
            # Define States 
            state = TauLeapingOutput(state_0,save_ts)
            new_discrete = state.discrete

            for idx, t in tqdm(enumerate(ts[0:-1])):
                # handles current time
                h = ts[idx+1] - ts[idx]
                times = self.get_time_vector(t, state.number_of_paths, state.device)
                rates = forward_model.forward(state.discrete,
                                              times,
                                              state.x0)
                # moves the variables forward
                new_discrete = self.TauStep(rates, h, state)

                # store the paths if requiered
                if save_ts is not None:
                    if t in save_ts:
                        state.update_paths(new_discrete)

                state.update_state(new_discrete)

            # LAST STEP
            last_time = self.min_t * torch.ones((state.number_of_paths,), device=state.device)
            rates = forward_model.forward(state.discrete,
                                          last_time,
                                          state.x0)
            
            new_discrete = self.TauStep(rates, h, state, end=self.max_rate_at_end)
            state.update_state(new_discrete)

            # SAVE LAST STEP
            if save_ts is not None:
                state.update_paths(new_discrete)
                state.concatenate_paths()
            return state
                
    def get_time_vector(self, t, number_of_paths, device):
        """
        Simply repeats the time value to obtain a tensor in the right device
        """
        times = t * torch.ones(number_of_paths,).to(device)
        return times
