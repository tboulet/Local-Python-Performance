"""This module contains functions aimed at simulating intensive computations.
"""

from argparse import ArgumentParser
from time import time, perf_counter
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Any, Tuple



def sum_integers(n_data : int):
    """Compute the sum of the first n_data integers.
    Complexity is O(n_data)."""
    sum_x = 0
    for i in range(n_data):
        sum_x += i
    return sum_x


def compute_iteratively_for_each_data(list_input : List[Any]):
    """Compute iteratively and independantly a sum of integers for each input.
    This function is easily parallelizable. 
    Complexity is O(len(list_input))"""
    for x_input in list_input:
        sum_integers(1000)