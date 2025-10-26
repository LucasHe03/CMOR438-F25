import numpy as np

__all__ = ['euclidean_distance', 'manhattan_distance']

"""
Computes the Euclidean between two NumPy arrays.
Inputs:
    - a: a NumPy array of any shape
    - b: a NumPy array of any shape
"""
def euclidean_distance(a, b):
    a = np.array(a, np.float64)
    b = np.array(b, np.float64)
    return np.sqrt(np.sum((a - b) ** 2))

"""
Computes the Manhattan distance between two NumPy arrays.
Inputs:
    - a: a NumPy array of any shape
    - b: a NumPy array of any shape
"""
def manhattan_distance(a, b):
    a = np.array(a, np.float64)
    b = np.array(b, np.float64)
    return np.sum(np.abs(a - b))