import numpy as np

"""
Scales the input data to be between 0 and 1. 
Inputs:
    - data: a NumPy array of any shape
"""
def scale(data):
    # convert to np array
    data = np.array(data, np.float64)
    # check for empty data
    if data.size == 0:
        return data
    # handle 1D array
    if len(data.shape) == 1:
        min, max = np.min(data), np.max(data)
        if min - max == 0: # handle div by 0
            return np.zeros_like(data)
        else:
            return (data - min) / (max - min)
    # handle other shapes
    else:
        scaled = np.zeros_like(data, np.float64)
        for i in range(data.shape[1]):
            min, max = np.min(data[:, i]), np.max(data[:, i])
            if min - max == 0: # handle div by 0
                scaled[:, i] = np.zeros_like(data[:, i])
            else:
                scaled[:, i] = (data[:, i] - min) / (max - min)
        return scaled


"""
Normalizes the input data using z-score standardization. 
Inputs:
    - data: a NumPy array of any shape
"""
def normalize(data):
    # convert to np array
    data = np.array(data, np.float64)
    # check for empty data
    if data.size == 0:
        return data
    # handle 1D array
    if len(data.shape) == 1:
        mean, std = np.mean(data), np.std(data)
        if std == 0: # handle div by 0
            std = 1
        return (data - mean) / std
    # handle other shapes
    else:
        normalized = np.zeros_like(data, np.float64)
        for i in range(data.shape[1]):
            mean, std = np.mean(data[:, i]), np.std(data[:, i])
            if std == 0: # handle div by 0
                std = 1
            normalized[:, i] = (data[:, i] - mean) / std
        return normalized
    
def train_test_split():
    return