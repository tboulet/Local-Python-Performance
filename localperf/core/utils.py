from argparse import ArgumentParser
from time import time, perf_counter
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Any, Tuple
import os


def build_plt_figure(list_inputs, list_mean_time, list_std_time, title : str):
    """Build the matplotlib figure for the CPU performance."""
    plt.errorbar(list_inputs, list_mean_time, yerr=list_std_time, label=title)
    plt.xlabel("n_data")
    plt.ylabel("Time (s)")
    plt.legend(loc = "upper left")
    plt.xscale("log")
    plt.yscale("log")
    
def get_results_as_string(list_inputs, list_mean_time, list_std_time, list_speed_up = None):
    """Generate results as string."""
    if list_speed_up is not None:
        string = "n_data\tmean_time\tstd_time\tspeed_up (time taken divided by basic_python_time)\n"
        for x_input, mean_time, std_time, speed_up in zip(list_inputs, list_mean_time, list_std_time, list_speed_up):
            string += f"{x_input}\t{mean_time:.2e}\t{std_time:.2e}\t{speed_up}\n"
    else:
        string = "n_data\tmean_time\tstd_time\n"
        for x_input, mean_time, std_time in zip(list_inputs, list_mean_time, list_std_time):
            string += f"{x_input}\t{mean_time:.2e}\t{std_time:.2e}\n"
    return string

def create_dir(directory : str = None):
    """Create a directory if it does not exist."""
    if directory is not None and not os.path.exists(directory):
        os.makedirs(directory)
        
def remove_file(filename : str = None):
    """Remove a file if it exists."""
    if filename is not None and os.path.exists(filename):
        os.remove(filename)