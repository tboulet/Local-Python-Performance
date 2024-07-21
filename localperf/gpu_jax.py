
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import time

# Local imports
from localperf.core.measuring import measure_time, deal_with_results
from localperf.core.utils import create_dir, remove_file
from localperf.core.compute import compute, treat_batch
from localperf.core.config import default_n_measures_jax, default_log_n_data_jax, n_data_jax
    
    
if __name__ == "__main__":
    # Parser
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None, help="Directory where to save the logs. No saving by default")
    parser.add_argument("--plot", action="store_true", default=False, help="Plot the results. No plotting by default")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory where to save the images. No saving by default")
    parser.add_argument("--no-progress", action="store_true", help="Hide the progress bar.")
    
    parser.add_argument("--log_n_data", type=int, default=default_log_n_data_jax, help=f"Value (in log scale) of the maximum n_data to be tested. Default: {default_log_n_data_jax} (10^{default_log_n_data_jax})")
    parser.add_argument("--n_measures", type=int, default=default_n_measures_jax, help=f"Number of measures to be made for each n_data. Default: {default_n_measures_jax}")
    parser.add_argument("--n_measures_gpu", type=int, default=None, help=f"Number of measures to be made for each n_data for the GPU. Default: same as --n_measures")                    
    args = parser.parse_args()

    image_dir = args.image_dir
    log_dir = args.log_dir
    do_plot = args.plot
    log_n_data_max = args.log_n_data
    n_measures = args.n_measures
    n_measures_gpu = args.n_measures_gpu if args.n_measures_gpu is not None else n_measures
    show_progress_bar = not args.no_progress



    # Setup
    print(
f"===== GPU measurement ===== \n\
Speed will be measured for data in range [1, 10^{log_n_data_max}] \n\
and for {n_measures} measures for each data. \n\
===========================\n\
        ")
    list_n_data = [10**k for k in range(0, log_n_data_max + 1)]
    log_filename = log_dir + "/gpu_jax.txt" if log_dir is not None else None
    image_filename = image_dir + "/gpu_jax.png" if image_dir is not None else None
    create_dir(log_dir)
    create_dir(image_dir)
    remove_file(log_filename)
    
    # Measure CPU_only jax performance
    device = "cpu"
    title = f"JAX with {device}"
    print(title)
    
    key = jax.random.PRNGKey(0)
    list_keys = jax.random.split(key, len(list_n_data))
    
    def cpu_only_jax_compute(x):
        n_data, key = x
        X = jax.random.normal(key, (int(jnp.sqrt(n_data)), int(jnp.sqrt(n_data))))
        return jnp.dot(X, X)
        
    list_mean_time_cpu, list_std_time = measure_time(
        func = cpu_only_jax_compute, 
        list_inputs = zip(list_n_data, list_keys),
        n_measures = n_measures,
        show_progress_bar = show_progress_bar,
        )
            
    deal_with_results(
        list_inputs=list_n_data,
        list_mean_time=list_mean_time_cpu,
        list_std_time=list_std_time,
        do_print=True,
        do_plot=False,
        log_filename=log_filename,
        image_filename=image_filename,
        title = title,
    )        
    
    
    # Measure GPU JAX performance
    if len(jax.devices("gpu")) == 0:
        print("WARNING : No GPU recognized by jax.devices(). Skipping GPU jax performance measurement.")
    else:
        title = f"JAX with GPU"
        print(title)
        
        @jax.jit
        def compute(X):
            print(f"Compiling for shape {X.shape}")
            return jnp.dot(X, X)
            
        def gpu_jax_compute(n_data : int):
            key = jax.random.PRNGKey(0)
            X = jax.random.normal(key, (int(jnp.sqrt(n_data)), int(jnp.sqrt(n_data))))
            return compute(X)
        
        list_mean_time_gpu, list_std_time = measure_time(
            func = gpu_jax_compute, 
            list_inputs = list_n_data, 
            n_measures = n_measures_gpu,
            show_progress_bar = show_progress_bar,
            )
        list_speed_up = [list_mean_time_cpu[i] / list_mean_time_gpu[i] for i in range(len(list_mean_time_cpu))]
                
        deal_with_results(
            list_inputs=list_n_data,
            list_mean_time=list_mean_time_gpu,
            list_std_time=list_std_time,
            list_speed_up=list_speed_up,
            do_print=True,
            do_plot=do_plot,
            log_filename=log_filename,
            image_filename=image_filename,
            title = title,
        )