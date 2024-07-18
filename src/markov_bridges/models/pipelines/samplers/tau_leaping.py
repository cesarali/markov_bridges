import torch
import numpy as np
from tqdm import tqdm
from typing import Union
from torch import functional as F

from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.models.generative_models.cjb_rate import ClassificationForwardRate
def set_diagonal_rate(rates,x):
    """
    Ensures that we have the right diagonal rate
    """
    batch_size = rates.shape[0]
    dimensions = rates.shape[1]

    #set diagonal to sum of other values
    batch_index = torch.arange(batch_size).repeat_interleave((dimensions))
    dimension_index = torch.arange(dimensions).repeat((batch_size))
    rates[batch_index,dimension_index,x.long().view(-1)] = 0.

    #rate_diagonal = -rates.sum(axis=-1)
    #rates[batch_index,dimension_index,x.long().view(-1)] = rate_diagonal[batch_index,dimension_index]
    x_0max = torch.max(rates, dim=2)[1]
    #rates = rates * h
    #rates[batch_index,dimension_index,x.long().view(-1)] = 1. - rates[batch_index,dimension_index,x.long().view(-1)]

    #removes negatives
    #rates[torch.where(rates < 0.)]  = 0.

    return  x_0max,rates

def TauLeaping(config:Union[CJBConfig],
               rate_model:Union[ClassificationForwardRate],
               x_0:torch.Tensor,
               forward=True,
               return_path=False):
    """
    :param rate_model:
    :param x_0:
    :param N:
    :return:
    """

    number_of_paths = x_0.size(0)
    D = x_0.size(1)
    S = config.data0.vocab_size
    num_steps = config.pipeline.number_of_steps
    time_epsilon = config.pipeline.time_epsilon
    min_t = 1./num_steps
    device = x_0.device

    #==========================================
    # CONDITIONAL SAMPLING
    #==========================================
    conditional_tau_leaping = False
    conditional_model = False
    bridge_conditional = False
    if hasattr(config.data1,"conditional_model"):
        conditional_model = config.data1.conditional_model
        conditional_dimension = config.data1.conditional_dimension
        bridge_conditional = config.data1.bridge_conditional

    if conditional_model and not bridge_conditional:
        conditional_tau_leaping = True

    if conditional_tau_leaping:
        conditioner = x_0[:,0:conditional_dimension]

    with torch.no_grad():
        x = x_0

        ts = np.concatenate((np.linspace(1.0 - time_epsilon, min_t, num_steps), np.array([0])))

        if return_path:
            save_ts = np.concatenate((np.linspace(1.0 - time_epsilon, min_t, num_steps), np.array([0])))
        else:
            save_ts = ts[np.linspace(0, len(ts)-2, config.pipeline.num_intermediates, dtype=int)]

        if forward:
            ts = ts[::-1]
            save_ts = save_ts[::-1]

        x_hist = []
        x0_hist = []

        counter = 0
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

        # last step ------------------------------------------------
        if conditional_tau_leaping:
            x[:,0:conditional_dimension] = conditioner

        p_0gt = rate_model(x, min_t * torch.ones((number_of_paths,), device=device)) # (N, D, S)
        x_0max = torch.max(p_0gt, dim=2)[1]
        if conditional_tau_leaping:
            x_0max[:,0:conditional_dimension] = conditioner

        # save last step
        x_hist.append(x.clone().detach().unsqueeze(1))
        x0_hist.append(x_0max.clone().detach().unsqueeze(1))
        if len(x_hist) > 0:
            x_hist = torch.cat(x_hist,dim=1).float()
            x0_hist = torch.cat(x0_hist,dim=1).float()

        return x_0max.detach().float(), x_hist, x0_hist, torch.Tensor(save_ts.copy()).to(device)

def TauLeapingRates(config:Union[CJBConfig],
                    rate_model:Union[ClassificationForwardRate],
                    x_0:torch.Tensor,
                    forward=True,
                    return_path=False):
    """
    :param rate_model:
    :param x_0:
    :param N:
    :return:
    """
    number_of_paths = x_0.size(0)
    D = x_0.size(1)
    S = config.data0.vocab_size
    num_steps = config.pipeline.number_of_steps
    min_t = 1./num_steps
    device = x_0.device

    with torch.no_grad():
        x = x_0
        ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))

        if return_path:
            save_ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))
        else:
            save_ts = ts[np.linspace(0, len(ts)-2, config.pipeline.num_intermediates, dtype=int)]

        if forward:
            ts = ts[::-1]
            save_ts = save_ts[::-1]

        x_hist = []
        x0_hist = []
        rates_histogram = []

        counter = 0
        for idx, t in tqdm(enumerate(ts[0:-1])):

            h = min_t
            times = t * torch.ones(number_of_paths,).to(device)
            rates = rate_model(x,times) # (N, D, S)
            x_0max = torch.max(rates, dim=2)[1]

            if t in save_ts:
                x_hist.append(x.clone().detach().unsqueeze(1))
                x0_hist.append(x_0max.clone().detach().unsqueeze(1))
                rates_histogram.append(rates.clone().detach().unsqueeze(1))

            #TAU LEAPING
            diffs = torch.arange(S, device=device).view(1,1,S) - x.view(number_of_paths,D,1)
            poisson_dist = torch.distributions.poisson.Poisson(rates * h)
            jump_nums = poisson_dist.sample().to(device)
            adj_diffs = jump_nums * diffs
            overall_jump = torch.sum(adj_diffs, dim=2)
            xp = x + overall_jump
            x_new = torch.clamp(xp, min=0, max=S-1)

            x = x_new

        # last step
        p_0gt = rate_model(x, min_t * torch.ones((number_of_paths,), device=device)) # (N, D, S)
        x_0max = torch.max(p_0gt, dim=2)[1]

        # save last step
        x_hist.append(x.clone().detach().unsqueeze(1))
        x0_hist.append(x_0max.clone().detach().unsqueeze(1))
        rates_histogram.append(rates.clone().detach().unsqueeze(1))
        if len(x_hist) > 0:
            x_hist = torch.cat(x_hist,dim=1).float()
            x0_hist = torch.cat(x0_hist,dim=1).float()
            rates_histogram = torch.cat(rates_histogram, dim=1).float()

        return x_0max.detach().float(), x_hist, x0_hist, rates_histogram, torch.Tensor(save_ts.copy()).to(device)
