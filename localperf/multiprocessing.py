
from argparse import ArgumentParser
from time import time, perf_counter
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Any, Tuple
from joblib import Parallel, delayed

# Local imports
from localperf.core.measuring import measure_time, deal_with_results
from localperf.core.utils import create_dir, remove_file
from localperf.core.compute import treat_one_data
from localperf.core.config import default_log_n_data_mp, default_log2_n_process_mp, default_n_measures_mp


            
            
if __name__ == "__main__":
    # Parser
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None, help="Directory where to save the logs. No saving by default")
    parser.add_argument("--plot", action="store_true", default=False, help="Plot the results. No plotting by default")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory where to save the images. No saving by default")
    parser.add_argument("--no-progress", action="store_true", help="Hide the progress bar.")
    
    parser.add_argument("--log_n_data", type=int, default=default_log_n_data_mp, help=f"Value (in log10 scale) of the maximum n_data to be tested. Default: {default_log_n_data_mp} (10^{default_log_n_data_mp} data max)")
    parser.add_argument("--log2_n_process", type=int, default=default_log2_n_process_mp, help=f"Value (in log2 scale) of the maximum n_process to be tested. Default: {default_log2_n_process_mp} ({2**default_log2_n_process_mp} process max)")
    parser.add_argument("--n_measures", type=int, default=default_n_measures_mp, help=f"Number of measures to be made for each n_data. Default: {default_n_measures_mp}")
    
    args = parser.parse_args()
    
    image_dir = args.image_dir
    log_dir = args.log_dir
    do_plot = args.plot
    log_n_data_max = args.log_n_data
    log2_n_process_max = args.log2_n_process
    n_measures = args.n_measures    
    show_progress_bar = not args.no_progress

    

    # Setup
    print(
f"===== Parallelization speed-up measurement ===== \n\
Speed up with parallelization with Joblib will be measured \n\
For data in range [1, 10^{log_n_data_max}] \n\
For n_process in range [1, 2^{log2_n_process_max}] = [1, {2**log2_n_process_max}] \n\
With {n_measures} measures for each data. \n\
================================================\n\
        ")
    list_n_data = [10**k for k in range(0, log_n_data_max + 1)]
    list_n_process = [2**k for k in range(0, log2_n_process_max + 1)]
    log_filename = log_dir + "/multiprocessing.txt" if log_dir is not None else None
    image_filename = image_dir + "/multiprocessing.png" if image_dir is not None else None
    create_dir(log_dir)
    create_dir(image_dir)
    remove_file(log_filename)
    
    
    
    # Measure with no parallelization
    title = f"No parallelization"
    print(title)
        
    def for_loop_computing(n_data: int):
        "Compute data sequentially."
        for _ in range(n_data):
            treat_one_data()
            
    list_mean_time_no_parallelization, list_std_time = measure_time(
        func = for_loop_computing, 
        list_inputs = list_n_data, 
        n_measures = n_measures,
        show_progress_bar = show_progress_bar,
        )

    deal_with_results(
        list_inputs=list_n_data,
        list_mean_time=list_mean_time_no_parallelization,
        list_std_time=list_std_time,
        do_print=True,
        do_plot=do_plot,
        log_filename=log_filename,
        image_filename=image_filename,
        title=title,
    ) 
        
        
        
    for n_process in list_n_process:
        
        # Measure with joblib with n_process processes
        title=f"Joblib parallel° (n_process={n_process})"
        print(title)
        
        def parallel_computing_with_joblib(n_data: int):
            """Compute data in parallel with joblib."""
            Parallel(n_jobs=n_process)(delayed(treat_one_data)() for _ in range(n_data))
    
        list_mean_time, list_std_time = measure_time(
            func = parallel_computing_with_joblib, 
            list_inputs = list_n_data, 
            n_measures = n_measures,
            show_progress_bar = show_progress_bar,
            )
        list_speed_up = [list_mean_time_no_parallelization[i] / list_mean_time[i] for i in range(len(list_mean_time))]
        
        deal_with_results(
            list_inputs=list_n_data,
            list_mean_time=list_mean_time,
            list_std_time=list_std_time,
            list_speed_up=list_speed_up,
            do_print=True,
            do_plot=do_plot,
            log_filename=log_filename,
            image_filename=image_filename,
            title=title,
        )        