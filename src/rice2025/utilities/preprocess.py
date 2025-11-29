import numpy as np

__all__ = ['scale', 'normalize', 'train_test_split']

def scale(data):
    """
    Scales the input data to be between 0 and 1. 
    Inputs:
        - data: a NumPy array of any shape
    """
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

def normalize(data):
    """
    Normalizes the input data using z-score standardization. 
    Inputs:
        - data: a NumPy array of any shape
    """
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

def train_test_split(X, y, test_size=0.25, random_seed=None):
    """
    Splits features and labels into training and testing sets.
    
    Inputs:
        - X: NumPy array of features, shape (n_samples, n_features)
        - y: NumPy array of labels, shape (n_samples,) or (n_samples, n_outputs)
        - test_size: fraction of data to use as test set (0 < test_size < 1)
        - random_seed: optional integer for reproducibility
    
    Returns:
        - X_train, X_test, y_train, y_test
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Convert to NumPy arrays
    X = np.array(X, np.float64)
    y = np.array(y, np.float64)
    
    # Check sizes
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if test_size < 0 or test_size > 1:
        raise ValueError("test_size must be between 0 and 1")
    
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    split = n - max(1, int(round(n * test_size)))
    train_idx, test_idx = indices[:split], indices[split:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test