import torch

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchdyn
from torchdyn.datasets import generate_moons

# Implement some helper functions

def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    labels = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[labels[i]] + noise[i])
    data = torch.stack(data)
    return data,labels

def sample_moons(n,noise=0.2):
    x0, labels = generate_moons(n, noise=noise)
    return x0,labels

def sample_8gaussians(n):
    XY, labels = eight_normal_sample(n, 2, scale=5, var=0.1)
    return XY.float(),labels

def sample_from_dataloader_iterator(dataloder_iterator,sample_size,flatten=True):
    """
    Samples data from the dataloader until the sample_size is met.

    Args:
        dataloader (DataLoader): The dataloader to sample from.
        sample_size (int): The total number of samples to collect.
        flatten (bool): Whether to flatten the samples or not.

    Returns:
        torch.Tensor: A tensor containing the collected samples.
    """
    size_left = sample_size
    x_0 = []
    while size_left > 0:
        dataloader_iterator = iter(dataloder_iterator)
        for databatch in dataloader_iterator:
            x_ = databatch[0]
            batch_size = x_.size(0)
            take_size = min(size_left, batch_size)
            x_0.append(x_[:take_size])
            size_left -= take_size
            if size_left == 0:
                break

    x_0 = torch.vstack(x_0)
    actual_sample_size = x_0.size(0)
    if flatten:
        x_0 = x_0.reshape(actual_sample_size, -1)
    return x_0

def sample_discrete_target(dataloader,train=True,number_of_batches = 20):
    data_sample = []
    batch_index  = 0
    for databatch in dataloader.train():
        data_sample.append(databatch.target_discrete)
        batch_index += 1
        if batch_index >= number_of_batches:
            break
    data_sample = torch.cat(data_sample,dim=0).unsqueeze(1)
    return data_sample
    
