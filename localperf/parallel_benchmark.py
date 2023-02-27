import multiprocessing as mp
from argparse import ArgumentParser
from time import time, perf_counter
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Any, Tuple


# Local imports
from localperf.core.measuring import measure_time, deal_with_results
from localperf.core.parallel_func import get_parallel_function
from localperf.core.utils import create_dir, remove_file
from localperf.core.compute import treat_one_data
from localperf.core.config import default_log_n_data_parallel, default_log2_n_process_parallel, default_n_measures_parallel, supported_libs


num_cores = mp.cpu_count()

            
if __name__ == "__main__":
    # Parser
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None, help="Directory where to save the logs. No saving by default")
    parser.add_argument("--plot", action="store_true", default=False, help="Plot the results. No plotting by default")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory where to save the images. No saving by default")
    parser.add_argument("--no-progress", action="store_true", help="Hide the progress bar.")
    
    parser.add_argument("--log_n_data", type=int, default=default_log_n_data_parallel, help=f"Value (in log10 scale) of the maximum n_data to be tested. Default: {default_log_n_data_parallel} (10^{default_log_n_data_parallel} data max)")
    parser.add_argument("--n_process", type=int, default=num_cores, help=f"Value of the number of process used. Default is your number of detected cores: {num_cores}")
    parser.add_argument("--n_measures", type=int, default=default_n_measures_parallel, help=f"Number of measures to be made for each n_data. Default: {default_n_measures_parallel}")

    args = parser.parse_args()
    
    image_dir = args.image_dir
    log_dir = args.log_dir
    do_plot = args.plot
    log_n_data_max = args.log_n_data
    n_process = args.n_process
    n_measures = args.n_measures    
    show_progress_bar = not args.no_progress
    

    

    # Setup
    print(f"Number of cores: {num_cores}")
    print(
f"===== Parallelization speed-up measurement library benchmark ===== \n\
Benchmark of the speed up with parallelization with {supported_libs}. \n\
For data in range [1, 10^{log_n_data_max}] \n\
With n_process = {n_process} \n\
With {n_measures} measures for each data size. \n\
================================================\n\
        ")
    list_n_data = [10**k for k in range(0, log_n_data_max + 1)]

    log_filename = log_dir + f"/benchmark_parallel.txt" if log_dir is not None else None
    image_filename = image_dir + f"/benchmark_parallel.png" if image_dir is not None else None
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
        do_plot=False,
        log_filename=log_filename,
        image_filename=image_filename,
        title=title,
    ) 
        
        
        
    for i, lib_name in enumerate(supported_libs):
        
        # Measure with the parallelization lib with n_process processes
        title=f"Parallel° with {lib_name} (n_process={n_process})"
        print(title)
        
        parallel_computing = get_parallel_function(lib_name, n_process)
        list_mean_time, list_std_time = measure_time(
            func = parallel_computing, 
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
            do_plot=do_plot if i == len(supported_libs) - 1 else False,
            log_filename=log_filename,
            image_filename=image_filename,
            title=title,
        )        