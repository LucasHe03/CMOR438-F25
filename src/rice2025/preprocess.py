import numpy as np

"""
Normalizes the input data to be between 0 and 1. 
Inputs:
    - data: a NumPy array of any shape
"""
def normalize(data):
    # check for empty data
    if data.size == 0:
        return data
    # handle 1D array
    if len(data.shape) == 1:
        min, max = np.min(data), np.max(data)
        return (data - min) / (max - min)
    # handle other shapes
    else:
        normalized = np.zeros_like(data, np.float64)
        for i in range(data.shape[1]):
            min, max = np.min(data[:, i]), np.max(data[:, i])
            normalized[:, i] = (data[:, i] - min) / (max - min)
        return normalized

def scale():
    return

def train_test_split():
    return