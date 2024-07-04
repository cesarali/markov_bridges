import os
import sys
import torch
import numpy as np
from typing import List,Tuple
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


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
                alpha=0.3, label="Data 0 ", color=colors[0])
        #ax1.bar(range(number_of_total_states), start_target[dimension_index, :].tolist(),
        #        alpha=0.3, label="Backward", color=colors[1])
        # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)

        ax3.bar(range(number_of_total_states), states_histogram_at_1[dimension_index, :].tolist(), alpha=0.3,
                label="Target T", color=colors[0])
        ax3.bar(range(number_of_total_states), generative_at_1[dimension_index, :].tolist(), alpha=0.3,
                label="Forward", color=colors[1])
        # ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)

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