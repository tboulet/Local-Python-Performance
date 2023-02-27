from typing import Callable, List, Any, Tuple

from localperf.core.compute import treat_one_data
from localperf.core.config import supported_libs


def get_parallel_function(lib_name : str, n_process : int) -> Callable[[int], Any]:
    """Return a function that will compute data in parallel with the given library.
    
    Args:
        lib_name (str): Name of the library to use for parallelization. Available: {supported_libs}
        n_process (int): Number of process to use for parallelization.
    
    Returns:
        parallel_computing (Callable[[int], None]): Function that will compute data in parallel.
    """
    
    if lib_name == "joblib":
        try:
            from joblib import Parallel, delayed
        except ImportError:
            raise ImportError("Please install joblib with: pip install joblib")
        def parallel_computing(n_data: int):
            """Compute data in parallel with joblib."""
            Parallel(n_jobs=n_process)(delayed(treat_one_data)() for _ in range(n_data))
    elif lib_name == "mp":
        from multiprocessing import Pool
        def parallel_computing(n_data: int):
            """Compute data in parallel with multiprocessing."""
            with Pool(n_process) as p:
                p.starmap(treat_one_data, [() for _ in range(n_data)])
    elif lib_name == "ray":
        try:
            import ray
            if not ray.is_initialized():
                ray.init()
        except ImportError:
            raise ImportError("Please install ray with: pip install ray")
        def parallel_computing(n_data: int):
            """Compute data in parallel with ray."""
            @ray.remote
            def treat_n_data_ray(n_data : int):
                for _ in range(n_data):
                    treat_one_data()
            
            n_data_per_process_list = [n_data // n_process] * n_process
            futures = [treat_n_data_ray.remote(n_data = n_data_per_process) 
                       for n_data_per_process in n_data_per_process_list]
            ray.get(futures)
    
    else:
        raise ValueError(f"Unknown lib: {lib_name}. Please choose one of {supported_libs}")
    return parallel_computing
