import torch
from typing import Tuple
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def plot_histograms_one_time(sample_histogram,vocab_size):
    dimensions = sample_histogram.size(0)
    fig, axes = plt.subplots(nrows=1, ncols=dimensions, figsize=(20, 6))
    for dimension_index in range(dimensions):
        axes[dimension_index].bar(range(vocab_size), sample_histogram[dimension_index].numpy(), color='red',alpha=0.4)
    plt.show()

def plot_marginals_binary_histograms(marginal_histograms:Tuple[torch.Tensor], plots_path=None):
    """

    :param marginal_histograms: List[] marginal_0,marginal_generated_0,marginal_1,marginal_noising_1
    :param plots_path:
    :return:
    """
    marginal_0,marginal_generated_0,marginal_1,marginal_noising_1 = marginal_histograms
    marginal_0,marginal_generated_0,marginal_1,marginal_noising_1 = marginal_0.detach().cpu().numpy(),marginal_generated_0.detach().cpu().numpy(),marginal_1.detach().cpu().numpy(),marginal_noising_1.detach().cpu().numpy()

    fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(12,4))

    bin_edges = range(len(marginal_0))
    ax1.bar(bin_edges, marginal_generated_0, align='edge', width=1.0,alpha=0.2,label="generated_0")
    ax1.bar(bin_edges, marginal_0, align='edge', width=1.0,alpha=0.2,label="data ")

    ax1.set_title(r'Time 0')
    ax1.set_xlabel('Bins')
    ax1.set_ylabel('Counts')
    ax1.legend(loc="upper right")

    ax2.bar(bin_edges,marginal_noising_1 , align='edge', width=1.0,alpha=0.2,label="generated_0")
    ax2.bar(bin_edges, marginal_1, align='edge', width=1.0,alpha=0.2,label="data ")

    ax2.set_title(r'Time 1')
    ax2.set_xlabel('Bins')
    ax2.set_ylabel('Counts')
    ax2.legend(loc="upper right")

    ax2.set_ylim(0, 1.)
    ax1.set_ylim(0, 1.)

    if plots_path is None:
        plt.show()
    else:
        plt.savefig(plots_path)

def plot_time_series_histograms(data, data2=None,num_timesteps_to_plot=10):
    """
    Plots a series of histograms from a 3D PyTorch tensor, ensuring the first
    and last timesteps are always included in the plot.

    Parameters:
    data (torch.Tensor): A tensor of shape (number_of_steps, dimensions, vocab_size)
    num_timesteps_to_plot (int): Number of timesteps to plot, including the first and last timesteps

    """
    number_of_steps, dimensions, vocab_size = data.shape
    
    # Generate indices for the timesteps to plot
    if num_timesteps_to_plot >= number_of_steps:
        indices = range(number_of_steps)
    else:
        indices = np.linspace(0, number_of_steps - 1, num=num_timesteps_to_plot, dtype=int)
    
    # Set up the matplotlib figure and axes
    fig, axes = plt.subplots(nrows=dimensions, ncols=len(indices), figsize=(20, 6))
    
    # Make sure axes is always 2D
    if dimensions == 1:
        axes = axes.reshape((1, -1))
    if len(indices) == 1:
        axes = axes.reshape((-1, 1))
    
    # Plot each dimension and timestep
    for i in range(dimensions):
        for j, idx in enumerate(indices):
            ax = axes[i, j]
            ax.bar(range(vocab_size), data[idx, i].numpy(), color='blue',alpha=0.4)
            if data2 is not None:
                ax.bar(range(vocab_size), data2[j, i].numpy(), color='red',alpha=0.4)
            ax.set_title(f"t = {idx+1}")
            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_yticks([])  # Remove y-axis ticks
            ax.set_ylim(0,1)
    plt.tight_layout()
    plt.show()
    return indices

def plot_categorical_histogram_per_dimension(states_histogram_at_0,
                                             states_histogram_at_1,
                                             generative_at_1,
                                             states_legends=None,
                                             save_path=None,
                                             remove_ticks=True):
    """
    Forward is the direction of the past model

    :param is_past_forward:
    :param time_:
    :param states_histogram_at_0:
    :param states_histogram_at_1:
    :param histogram_from_rate:
    :param states_legends:
    :return:
    """

    if isinstance(states_histogram_at_0, torch.Tensor):
        states_histogram_at_0 = states_histogram_at_0.detach().cpu()
    if isinstance(states_histogram_at_1, torch.Tensor):
        states_histogram_at_1 = states_histogram_at_1.detach().cpu()
    if isinstance(generative_at_1, torch.Tensor):
        generative_at_1 = generative_at_1.detach().cpu()

    number_of_dimensions = states_histogram_at_0.size(0)
    number_of_total_states = states_histogram_at_0.size(1)
    if states_legends is None:
        states_legends = [str(a) for a in range(number_of_total_states)]

    # Create a GridSpec object
    fig, axs = plt.subplots(figsize=(12, 6))
    outer_ax = fig.axes[0]
    outer_ax.set_axis_off()

    gs = GridSpec(nrows=number_of_dimensions, ncols=2,
                  width_ratios=[1, 1],
                  hspace=.6,
                  left=0.05, right=0.95, bottom=0.1, top=0.9)  # Adjust hspace for vertical spacing

    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.35, top=0.8, hspace=0.1)

    for dimension_index in range(number_of_dimensions):
        ax1 = fig.add_subplot(gs[dimension_index, 0])
        ax3 = fig.add_subplot(gs[dimension_index, 1])

        if remove_ticks:
            ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                            labelbottom=False,
                            labelleft=False)

            ax3.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                            labelbottom=False,
                            labelleft=False)

        if dimension_index == 0:
            ax1.set_title(r"$P_0(x)$")
            ax3.set_title(r"$P_1(x)$")
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # ax2.legend(loc='upper center',bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)

        ax1.bar(range(number_of_total_states), states_histogram_at_0[dimension_index, :].tolist(),
                alpha=0.3, label="Data 0", color=colors[0])
        #ax1.bar(range(number_of_total_states), start_target[dimension_index, :].tolist(),
        #        alpha=0.3, label="Backward", color=colors[1])
        # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)

        ax3.bar(range(number_of_total_states), states_histogram_at_1[dimension_index, :].tolist(), alpha=0.3,
                label="Data 1", color=colors[0])
        ax3.bar(range(number_of_total_states), generative_at_1[dimension_index, :].tolist(), alpha=0.3,
                label="Generative", color=colors[1])
        ax3.legend(loc='best') #, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)

        ax1.set_ylim(0., 1.)
        ax3.set_ylim(0., 1.)

    # Remove ticks from the figure
    # plt.tick_params(axis='both', which='both', bottom=False, top=False,
    #                labelbottom=False, right=False, left=False, labelleft=False)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()