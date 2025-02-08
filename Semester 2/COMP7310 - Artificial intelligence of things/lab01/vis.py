from matplotlib import pyplot as plt
import numpy as np
import os

SAVE_ROOT = "./fig/"
os.makedirs(SAVE_ROOT, exist_ok=True)

def plot1D_single(t, x, title, xlabel, ylabel, grid=True):
    """
    Plot a single signal.

    Args:
        t (numpy.array): Array of timestamps in seconds.
        x (numpy.array): Array of signal values.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        grid (bool): Enable or disable the grid. Defaults to True.
    """
    plt.figure(figsize=(6, 3))
    plt.plot(t, x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    # save the plot
    plt.savefig(os.path.join(SAVE_ROOT, f'{title}.pdf'), format='pdf', bbox_inches='tight')


def plot1D_multiple(t, x, labels, title, xlabel, ylabel, grid=True):
    """
    Plot multiple signals.

    Args:
        t (numpy.array): Array of timestamps in seconds.
        x (list): List of signal arrays.
        labels (list): List of labels for each signal.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        grid (bool): Enable or disable the grid. Defaults to True.
    """
    num_plots = len(x)
    fig, axs = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots), squeeze=False)
    
    for i in range(num_plots):
        axs[i, 0].plot(t, x[i], label=labels[i])
        axs[i, 0].set_title(labels[i])
        axs[i, 0].set_xlabel(xlabel)
        axs[i, 0].set_ylabel(ylabel)
        axs[i, 0].grid(grid)
        axs[i, 0].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, f'{title}.pdf'), format='pdf', bbox_inches='tight')