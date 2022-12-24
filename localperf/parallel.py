
from argparse import ArgumentParser
from time import time, perf_counter
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Any, Tuple

# Local imports
from localperf.core.measuring import measure_time, deal_with_results
from localperf.core.utils import create_dir
from localperf.core.compute import sum_integers, compute_iteratively_for_each_data




            
            
if __name__ == "__main__":
    # Parser
    parser = ArgumentParser()
    parser.add_argument("--print", action="store_true", default=True, help="Print the results on the terminal")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory where to save the logs. No saving by default")
    parser.add_argument("--plot", action="store_true", default=False, help="Plot the results. No plotting by default")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory where to save the images. No saving by default")
    parser.add_argument("--log_n_data", type=int, default=6, help="Value (in log scale) of the maximum n_data to be tested. Default: 6 (10^6)")
    parser.add_argument("--n_measures", type=int, default=10, help="Number of measures to be made for each n_data. Default: 10")
    
    # Get attributes
    args = parser.parse_args()
    image_dir = args.image_dir
    log_dir = args.log_dir
    do_plot = args.plot
    do_print = args.print
    log_n_data_max = args.log_n_data
    n_measures = args.n_measures

    
    # Parallelization
    from joblib import Parallel, delayed
    
    

    # Measure
    list_n_data = [10**k for k in range(0, log_n_data_max + 1)]
    list_n_process = [2, 4]
    
    for n_process in list_n_process:
        
        def parallel_computing_with_joblib(n_data: int):
            Parallel(n_jobs=n_process)(delayed(sum_integers)(100) for _ in range(n_data))
    
        list_mean_time, list_std_time = measure_time(
            func = parallel_computing_with_joblib, 
            list_inputs = list_n_data, 
            n_measures = n_measures,
            show_progress_bar = False,
            )
        
        
        
        # Deal with results
        create_dir(log_dir)
        create_dir(image_dir)
        
        deal_with_results(
            list_inputs=list_n_data,
            list_mean_time=list_mean_time,
            list_std_time=list_std_time,
            do_print=do_print,
            do_plot=do_plot,
            log_filename=log_dir + "/parallel.txt" if log_dir is not None else None,
            image_filename=image_dir + "/parallel.png" if image_dir is not None else None,
        )        