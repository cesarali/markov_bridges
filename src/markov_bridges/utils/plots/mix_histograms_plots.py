import os
import sys
import torch
import numpy as np
from typing import List,Tuple
from matplotlib import pyplot as plt


def plot_scatterplot(noise_points=None,
                     real_points=None,
                     generated_points=None,
                     save_path=None):
    """
    Simple scatter plot for the 2D variables (swiss rolll)
    """

    # send everything to cpu for plotting
    if isinstance(noise_points, torch.Tensor):
        noise_points = noise_points.detach().cpu()
    if isinstance(real_points, torch.Tensor):
        real_points = real_points.detach().cpu()
    if isinstance(generated_points, torch.Tensor):
        generated_points = generated_points.detach().cpu()

    if real_points is not None and len(real_points) > 0:
        plt.plot(real_points[:,0],real_points[:,1],"x",label="real")

    if generated_points is not None and len(generated_points) > 0:
        plt.plot(generated_points[:,0],generated_points[:,1],"+",label="generation")
    plt.legend(loc="best")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
        
