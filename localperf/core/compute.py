"""This module contains functions aimed at simulating intensive computations.
"""

from argparse import ArgumentParser
from time import time, perf_counter
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Any, Tuple

import torch
from localperf.core.config import data_size

def treat_one_data():
    """Compute the sum of the first n integers.
    This is the most basic function to compute. Its computation correspond to the treatment of a single data.
    Complexity is O(1) = BaseComplexity. This is the base unit of complexity in our package."""
    sum_x = 0
    for i in range(data_size):
        sum_x += i
    return sum_x

def compute(n_data : int):
    """Compute iteratively.
    Complexity is O(n_data) = BaseComplexity * n_data"""
    for _ in range(n_data):
        treat_one_data()


def treat_batch(model : torch.nn.Module, batch : torch.Tensor, device : torch.cuda.device):
    """Realize a forward pass on a batch.

    Args:
        model (torch.nn.Module): a neural network
        batch (torch.Tensor): the batch to be treated
        device (torch.device): the device on which the computation is performed
    """
    # print("Device:", torch.cuda.current_device())
    model(batch)