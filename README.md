# Local Python performance measurement

This project aims to create an easy-to-use package for measuring the performance of python in any machine, in terms of CPU, multiprocessing and GPU (pytorch with CUDA), and also verify that the GPU is used.

# Installation

Install the requirements in requirements.txt, then install the package:

```bash
pip install -r requirements.txt
pip install localperf
```


# Usage

You can measure the performance of your machine in terms of CPU, multiprocessing and GPU (pytorch only for now) by running the commands below.

Relevant arguments for visualization are:
- `--plot` : plot the results (default False)
- `--log_dir` [log directory] : directory where to save the results (default no logging)
- `--image_dir` [image directory]: directory where to save the images (default no image saving)
- `--no-progess` : do not show progress bar (default behavior is to show)

# CPU

<p align="center">
  <img src="assets/cpu.png" alt="CPU performances" width="60%"/>
</p>

To measure the performance of your machine in terms of CPU, run the following command:
```bash
python -m localperf.cpu
```

Relevant arguments for the benchmark are:
- `log_n_data` [log n data] : maximum number of data to do the benchmark (in log10 scale). The treatment of 1 data is defined as the sum of integers from 1 to 1000 (with a for loop), it is used as a base unit of computation.
- `n_measures` [n measures] : number of measures to do for each data size

# Parallelization

<p align="center">
  <img src="assets/multiprocessing.png" alt="Parallelization performances" width="60%"/>
</p>

To measure the performance of your machine in terms of parallelization, run the following command:
```bash
python -m localperf.parallel
```

Relevant arguments for the benchmark are:
- `log_n_data` [log n data] : maximum number of data to do the benchmark (in log10 scale)
- `log2_n_process` [log2 n process] : maximum number of processes to do the benchmark (in log2 scale)
- `n_measures` [n measures] : number of measures to do for each data size
- `lib` [lib] : library to use for parallelization. Default is joblib. Currently supported libraries are multiprocessing (`mp`), joblib (`joblib`) and ray (`ray`). For ray you will need to install it with pip before running the benchmark.

## Compare parallelization libraries

To compare the performances of the different libraries, run the following command:
```bash
python -m localperf.parallel_benchmark
```
This will compare the performances of multiprocessing, joblib and ray. Relevant arguments are:
- `log_n_data` [log n data] : maximum number of data to do the benchmark (in log10 scale)
- `n_process` [n process] : number of processes to do the benchmark. Default behavior is to use your number of CPUs, given by `multiprocessing.cpu_count()`
- `n_measures` [n measures] : number of measures to do for each data size


# GPU (pytorch)

## Install CUDA for pytorch

<p align="center">
  <img src="assets/gpu_torch.png" alt="GPU with torch performances" width="60%"/>
</p>

First, install pytorch with CUDA following the instructions on the [pytorch website](https://pytorch.org/get-started/locally/). 
If this code returns `True`, it means pytorch and CUDA are installed and its a good sign your GPU is used but this is not a guarantee:
```python
import torch
print(torch.cuda.is_available())
```
You may use the command `nvidia-smi` to check if your GPU is recognized by the system.

    nvidia-smi
<p align="center">
  <img src="assets/nvidia_smi.png" alt="nvidia-smi sceenshot" width="60%"/>
</p>
This give you information about each GPU recognized by the system : the name, the VRAM used. You can run this command in a separate terminal to see the GPU usage during your code :

    watch -n 0.1 nvidia-smi

## Measure performance

To measure the performance of your machine in terms of GPU, run the following command:
```bash
python -m localperf.gpu_torch
```
Relevant arguments for the benchmark are:
- `log_n_data` [log n data] : maximum number of data to do the benchmark (in log10 scale)
- `n_measures` [n measures] : number of measures to do for each data size
- `n_measures_gpu` [n measures gpu] : number of measures to do for each data size, on the GPU. If not specified, the same number of measures as on the CPU is done.




# JAX

## Install JAX

On linux (note your CUDA version may vary) :
```bash
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

On Windows (you won't be able to use GPU) :
```bash
pip install jax[cpu]
```

<p align="center">
  <img src="assets/gpu_jax.png" alt="GPU with JAX performances" width="60%"/>
</p>

Run this code to check if JAX is installed and if it can use the GPU:
```python
import jax
from jax.lib import xla_bridge
print(f"Available devices: {jax.devices()}")
print(f"Platform: {xla_bridge.get_backend().platform}")
```

## Measure performance

To measure the performance of your machine in terms of GPU, run the following command:
```bash
python -m localperf.gpu_jax
```