import torch
import matplotlib.pyplot as plt
import torch
from pprint import pprint
from dataclasses import dataclass,asdict
from torch.distributions import Categorical
from torch.utils.data import TensorDataset,DataLoader
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from markov_bridges.data.abstract_dataloader import (
    MarkovBridgeDataClass,
    MarkovBridgeDataNameTuple,
    MarkovBridgeDataloader,
    MarkovBridgeDataset
)

EPSILON = 1e-3

# Define the SDE solver using the Euler-Maruyama method
def sde_solver(x0, s, a, b, dt, T, device='cpu'):
    B, D, K = x0.shape
    N = int(T / dt)  # number of time steps
    
    # Initialize the solution tensor
    x = torch.zeros((N, B, D, K), device=device)
    x[0] = x0
    
    # Precompute constants
    s_half = s / 2.0
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))
    
    for n in range(1, N):
        # Compute the drift and diffusion terms
        drift = s_half[:,None,None] * (a[None,None,:] * (1 - x[n-1]) - b[None,None,:] * x[n-1])
        diffusion = torch.sqrt(s[:,None,None] * x[n-1] * (1 - x[n-1]))
        
        # Generate the Wiener process increments
        dW = torch.randn((B, D, K), device=device) * sqrt_dt
        
        # Update the solution
        x[n] = x[n-1] + drift * dt + diffusion * dW

        # Ensure values are within [0, 1]
        x[n] = torch.clamp(x[n], EPSILON, 1.0)
        # print(x)
        
    return x

# Stick-breaking function to map K-dimensional values to the simplex
def stick_breaking_process(x):
    x = torch.sigmoid(x)  # Ensure values are between 0 and 1
    remaining_stick = 1 - x
    remaining_stick = torch.cat([torch.ones_like(remaining_stick[:, :, :, :1]), remaining_stick], dim=-1)
    remaining_stick = torch.cumprod(remaining_stick[:, :, :, :-1], dim=-1)
    simplex_path = x * remaining_stick
    simplex_path[:, :, :, -1] = 1.0 - torch.sum(simplex_path[:, :, :, :-1], dim=-1)
    return simplex_path
