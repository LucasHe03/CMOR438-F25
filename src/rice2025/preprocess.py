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

"""
Splits the input data into training and testing sets. 
Inputs:
    - data: a NumPy array of any shape
    - test_size: the proportion of the data to be used for testing (default = .25)
"""
def train_test_split(data, test_size = .25):
    # convert to np array
    data = np.array(data, np.float64)
    # check for empty data
    if data.size == 0:
        return np.array([]), np.array([])
    # check test_size
    if test_size < 0 or test_size > 1:
        raise ValueError("test_size must be between 0 and 1")
    
    # obtain random indicies
    n = data.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)

    # find split 
    split = n - max(1, int(round(n * test_size)))
    train_idx, test_idx = indices[:split], indices[split:]

    train = data[train_idx]
    test = data[test_idx]

    return train, test