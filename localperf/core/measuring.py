from argparse import ArgumentParser
from time import time, perf_counter
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Any, Tuple
from tqdm import tqdm

from localperf.core.utils import build_plt_figure, get_results_as_string
from localperf.core.utils import create_dir, remove_file

def measure_time(
        func : Callable, 
        list_inputs : List[Any], 
        n_measures : int = 10,
        show_progress_bar : bool = False,
        ) -> Tuple[List[float], List[float]]:
    """Measure the mean and std of the time taken by a function, for each input in list_input.

    Args:
        func (Callable): the function we want to measure
        list_input (List[Any]): the list of inputs for the function func
        n_measures (int, optional): The number of measures that will be made for evaluating the mean and std. Defaults to 10.
        show_progress_bar (bool, optional): Whether to show a progress bar. Defaults to False.
        
    Returns:
        Tuple[List[float], List[float]]: The list of mean and std of the time taken by the function func, for each input in list_input.
    """
    list_mean_time = []
    list_std_time = []
    for x_input in list_inputs:
        list_time = []
        
        if show_progress_bar:
            iterable = tqdm(range(n_measures), desc=f"Measuring time for {x_input} data")
        else:
            iterable = range(n_measures)
        
        for _ in iterable:
            t_start = perf_counter()
            func(x_input)
            t_end = perf_counter()
            list_time.append(t_end - t_start)
        list_mean_time.append(np.mean(list_time))
        list_std_time.append(np.std(list_time))
    return list_mean_time, list_std_time
    
    

def deal_with_results(
        list_inputs : List[Any], 
        list_mean_time : List[float], 
        list_std_time : List[float], 
        list_speed_up : List[float] = None,
        do_print : bool = False, do_plot : bool = False, 
        log_filename : str = None, image_filename : str = None,
        title : str = None,
        ) -> None:
    """Deals with the results obtained, by (eventually) printing, plotting and saving logs and images.

    Args:
        list_inputs (List[Any]): the list of inputs for the function func. If can't be print, you can put None.
        list_mean_time (List[float]): the list of mean of the time taken by the function func, for each input in list_input.
        list_std_time (List[float]): the list of std of the time taken by the function func, for each input in list_input.
        list_speed_up (List[float], optional): the list of speed up obtained. Defaults to None.
        do_print (bool): whether to print the results on the terminal.
        do_plot (bool): whether to plot the results.
        log_filename (str): filename for the log file.
        image_filename (str): filename for the image file.
    """
    if list_inputs is None:
        list_inputs = ["-" for _ in list_mean_time]
        
    if do_print:
        string = string = get_results_as_string(list_inputs, list_mean_time, list_std_time, list_speed_up = list_speed_up)
        print(string)
    
    if log_filename is not None:
        string = get_results_as_string(list_inputs, list_mean_time, list_std_time, list_speed_up = list_speed_up)
        if title is not None:
            string = title + "\n" + string + "\n"
        with open(log_filename, "a") as f:
            f.write(string)
                
    if do_plot:
        build_plt_figure(list_inputs, list_mean_time, list_std_time, title)
        
    if image_filename is not None:
        build_plt_figure(list_inputs, list_mean_time, list_std_time, title)
        plt.savefig(image_filename)