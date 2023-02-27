# Common config
data_size = 1000

# CPU config
default_n_measures_cpu = 10
default_log_n_data_cpu = 4

# Multiprocessing config
default_n_measures_parallel = 10
default_log_n_data_parallel = 4
default_log2_n_process_parallel = 3
supported_libs = ["joblib", "mp", "ray"] 

# Torch config
default_n_measures_torch = 10
default_log_n_data_torch = 6
n_neurons_torch_model = 10