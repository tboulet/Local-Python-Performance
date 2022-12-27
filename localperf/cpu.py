
from argparse import ArgumentParser
from time import time, perf_counter
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Any, Tuple

# Local imports
from localperf.core.measuring import measure_time, deal_with_results
from localperf.core.utils import create_dir, remove_file
from localperf.core.compute import compute
from localperf.core.config import default_log_n_data_cpu, default_n_measures_cpu            

      
if __name__ == "__main__":
    # Parser
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None, help="Directory where to save the logs. No saving by default")
    parser.add_argument("--plot", action="store_true", default=False, help="Plot the results. No plotting by default")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory where to save the images. No saving by default")
    parser.add_argument("--no-progress", action="store_true", help="Hide the progress bar.")
    
    parser.add_argument("--log_n_data", type=int, default=default_log_n_data_cpu, help=f"Value (in log scale) of the maximum n_data to be tested. Default: {default_log_n_data_cpu} (10^{default_log_n_data_cpu})")
    parser.add_argument("--n_measures", type=int, default=default_n_measures_cpu, help=f"Number of measures to be made for each n_data. Default: {default_n_measures_cpu}")
                        
    args = parser.parse_args()

    image_dir = args.image_dir
    log_dir = args.log_dir
    do_plot = args.plot
    log_n_data_max = args.log_n_data
    n_measures = args.n_measures
    show_progress_bar = not args.no_progress


    # Setup
    print(
f"===== CPU measurement ===== \n\
CPU speed will be measured for data in range [1, 10^{log_n_data_max}] \n\
and for {n_measures} measures for each data. \n\
===========================\n\
        ")
    list_n_data = [10**k for k in range(0, log_n_data_max + 1)]
    log_filename = log_dir + "/cpu.txt" if log_dir is not None else None
    image_filename = image_dir + "/cpu.png" if image_dir is not None else None
    create_dir(log_dir)
    create_dir(image_dir)
    remove_file(log_filename)
    
    
    
    # Measure CPU time
    list_mean_time, list_std_time = measure_time(
        func = compute, 
        list_inputs = list_n_data, 
        n_measures = n_measures,
        show_progress_bar = show_progress_bar,
        )
        
    deal_with_results(
        list_inputs=list_n_data,
        list_mean_time=list_mean_time,
        list_std_time=list_std_time,
        do_print=True,
        do_plot=do_plot,
        log_filename=log_dir + "/cpu.txt" if log_dir is not None else None,
        image_filename=image_dir + "/cpu.png" if image_dir is not None else None,
        title = "CPU",
    )        