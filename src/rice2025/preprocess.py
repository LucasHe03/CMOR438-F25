import numpy as np

"""
Scales the input data to be between 0 and 1. 
Inputs:
    - data: a NumPy array of any shape
"""
def scale(data):
    # check for empty data
    if data.size == 0:
        return data
    # handle 1D array
    if len(data.shape) == 1:
        min, max = np.min(data), np.max(data)
        return (data - min) / (max - min)
    # handle other shapes
    else:
        scaled = np.zeros_like(data, np.float64)
        for i in range(data.shape[1]):
            min, max = np.min(data[:, i]), np.max(data[:, i])
            scaled[:, i] = (data[:, i] - min) / (max - min)
        return scaled

def normalize():
    return

def train_test_split():
    return