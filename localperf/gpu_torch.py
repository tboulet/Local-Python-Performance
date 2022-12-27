
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch

# Local imports
from localperf.core.measuring import measure_time, deal_with_results
from localperf.core.utils import create_dir, remove_file
from localperf.core.compute import compute, treat_batch
from localperf.core.config import default_log_n_data_torch, default_n_measures_torch, n_neurons_torch_model
plt.show()
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(n_neurons_torch_model, n_neurons_torch_model)
        self.fc2 = torch.nn.Linear(n_neurons_torch_model, n_neurons_torch_model)
        self.fc3 = torch.nn.Linear(n_neurons_torch_model, n_neurons_torch_model)
        self.fc4 = torch.nn.Linear(n_neurons_torch_model, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

def get_model(device : torch.device):
    """Create a model and move it to the device.

    Args:
        device (torch.device): the device on which the model will be moved.

    Returns:
        torch.nn.Module: the torch model.
    """
    model = Net()
    model.to(device = device)
    batch = torch.rand(size = (2, n_neurons_torch_model), device=device)
    treat_batch(model=model, batch=batch, device=device)
    return model
    
    
    
if __name__ == "__main__":
    # Parser
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None, help="Directory where to save the logs. No saving by default")
    parser.add_argument("--plot", action="store_true", default=False, help="Plot the results. No plotting by default")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory where to save the images. No saving by default")
    parser.add_argument("--no-progress", action="store_true", help="Hide the progress bar.")
    
    parser.add_argument("--log_n_data", type=int, default=default_log_n_data_torch, help=f"Value (in log scale) of the maximum n_data to be tested. Default: {default_log_n_data_torch} (10^{default_log_n_data_torch})")
    parser.add_argument("--n_measures", type=int, default=default_n_measures_torch, help=f"Number of measures to be made for each n_data. Default: {default_n_measures_torch}")
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
    log_filename = log_dir + "/gpu_torch.txt" if log_dir is not None else None
    image_filename = image_dir + "/gpu_torch.png" if image_dir is not None else None
    create_dir(log_dir)
    create_dir(image_dir)
    remove_file(log_filename)
    
    
    
    # Measure CPU_only torch performance
    device = "cpu"
    model = get_model(device = device)
    title = f"Torch with {device}"
    print(title)
    
    def cpu_only_torch_compute(n_data : int):
        batch = torch.rand(size = (n_data, n_neurons_torch_model), device = device)
        treat_batch(model=model, batch=batch, device=device)
        
        
    list_mean_time_cpu, list_std_time = measure_time(
        func = cpu_only_torch_compute, 
        list_inputs = list_n_data, 
        n_measures = n_measures,
        show_progress_bar = show_progress_bar,
        )
            
    deal_with_results(
        list_inputs=list_n_data,
        list_mean_time=list_mean_time_cpu,
        list_std_time=list_std_time,
        do_print=True,
        do_plot=do_plot,
        log_filename=log_dir + "/gpu_torch.txt" if log_dir is not None else None,
        image_filename=image_dir + "/gpu_torch.png" if image_dir is not None else None,
        title = title,
    )        
    
    
    # Measure GPU torch performance
    if not torch.cuda.is_available():
        print("No GPU available. Skipping GPU torch performance measurement")
    else:
        device = torch.device("cuda")
        model = get_model(device = device)
        title = f"Torch with {device}"
        print(title)
        
        def gpu_torch_compute(n_data : int):
            batch = torch.rand(size = (n_data, n_neurons_torch_model), device = device)
            treat_batch(model=model, batch=batch, device=device)
            
            
        list_mean_time_gpu, list_std_time = measure_time(
            func = gpu_torch_compute, 
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
            log_filename=log_dir + "/gpu_torch.txt" if log_dir is not None else None,
            image_filename=image_dir + "/gpu_torch.png" if image_dir is not None else None,
            title = title,
        )
        
    if do_plot:
        plt.show()