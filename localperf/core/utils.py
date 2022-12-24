from argparse import ArgumentParser
from time import time, perf_counter
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Any, Tuple
import os


def build_plt_figure(list_inputs, list_mean_time, list_std_time):
    """Build the matplotlib figure for the CPU performance."""
    plt.errorbar(list_inputs, list_mean_time, yerr=list_std_time)
    plt.xlabel("n_data")
    plt.ylabel("Time (s)")
    plt.xscale("log")
    plt.yscale("log")
    
def get_results_as_string(list_inputs, list_mean_time, list_std_time):
    """Generate results as string."""
    string = "input\tmean_time\tstd_time\n"
    for x_input, mean_time, std_time in zip(list_inputs, list_mean_time, list_std_time):
        string += f"{x_input}\t{mean_time}\t{std_time}\n"
    return string

def create_dir(directory : str = None):
    """Create a directory if it does not exist."""
    if directory is not None and not os.path.exists(directory):
        os.makedirs(directory)