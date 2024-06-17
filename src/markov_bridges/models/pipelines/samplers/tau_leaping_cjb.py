import torch
import numpy as np
from tqdm import tqdm
from typing import Union
from torch import functional as F

from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.models.generative_models.cjb_rate import ClassificationForwardRate
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

def TauLeapingCJB(config:Union[CJBConfig],
                  rate_model:Union[ClassificationForwardRate],
                  x_0:MarkovBridgeDataNameTuple,
                  return_path=False):
    """
    :param rate_model:
    :param x_0:
    :param N:
    :param num_intermediates:

    :return:
    """
    D = config.data.dimensions
    S = config.data.vocab_size
    conditional_dimension = config.data.context_dimension

    if config.data.has_context_discrete:
        join_context = lambda context_discrete,data_discrete : torch.cat([context_discrete,data_discrete],dim=1)
        remove_context = lambda full_data_discrete : full_data_discrete[:,conditional_dimension:]

    number_of_paths = x_0.source_discrete.size(0)

    num_steps = config.pipeline.number_of_steps
    time_epsilon = config.pipeline.time_epsilon

    min_t = 1./num_steps
    device = x_0.source_discrete.device

    #==========================================
    # CONDITIONAL SAMPLING
    #==========================================
    with torch.no_grad():
        if config.data.has_context_discrete:
            x = join_context(x_0.context_discrete,x_0.source_discrete).clone()
        else:
            x = x_0.source_discrete
            
        # define time
        ts = np.concatenate((np.linspace(1.0 - time_epsilon, min_t, num_steps), np.array([0])))
        if return_path:
            save_ts = np.concatenate((np.linspace(1.0 - time_epsilon, min_t, num_steps), np.array([0])))
        else:
            save_ts = ts[np.linspace(0, len(ts)-2, config.pipeline.num_intermediates, dtype=int)]
        ts = ts[::-1]
        save_ts = save_ts[::-1]

        x_hist = []
        x0_hist = []
        for idx, t in tqdm(enumerate(ts[0:-1])):

            h = min_t
            times = t * torch.ones(number_of_paths,).to(device)
            rates = rate_model(x,times) # (N, D, S)
            x_0max = torch.max(rates, dim=2)[1]

            if t in save_ts:
                x_hist.append(x.clone().detach().unsqueeze(1))
                x0_hist.append(x_0max.clone().detach().unsqueeze(1))

            #TAU LEAPING
            diffs = torch.arange(S, device=device).view(1,1,S) - x.view(number_of_paths,D,1)
            poisson_dist = torch.distributions.poisson.Poisson(rates*h)
            jump_nums = poisson_dist.sample().to(device)
            adj_diffs = jump_nums * diffs
            overall_jump = torch.sum(adj_diffs, dim=2)
            xp = x + overall_jump
            x_new = torch.clamp(xp, min=0, max=S-1)
            x = x_new

            #if config.data.has_context_discrete:
            #    x = remove_context(x)
            #    x = join_context(x_0.context_discrete,x).clone()

        # last step ------------------------------------------------
        if config.data.has_context_discrete:
            x = remove_context(x)
            x = join_context(x_0.context_discrete,x).clone()

        p_0gt = rate_model(x, min_t * torch.ones((number_of_paths,), device=device)) # (N, D, S)
        x_0max = torch.max(p_0gt, dim=2)[1]
        
        if config.data.has_context_discrete:
            x_0max = remove_context(x_0max)
            x_0max = join_context(x_0.context_discrete,x_0max).clone()

        # save last step
        x_hist.append(x.clone().detach().unsqueeze(1))
        x0_hist.append(x_0max.clone().detach().unsqueeze(1))
        if len(x_hist) > 0:
            x_hist = torch.cat(x_hist,dim=1).float()
            x0_hist = torch.cat(x0_hist,dim=1).float()

        return x_0max.detach().float(), x_hist, x0_hist, torch.Tensor(save_ts.copy()).to(device)
