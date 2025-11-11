import numpy as np

__all__ = ['LinearRegression']

"""
Implements linear regression.
"""
class LinearRegression:
    """
    Initializes the LinearRegression Class.
    """
    def __init__(self):
        self.weights = None
        self.bias = None
    
    """
    Fits the model to the training data.
    Inputs:
        - x: the training data
        - y: the training target values
    """
    def fit(self, x, y):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # check for errors
        if len(x) != len(y):
            raise ValueError("x and y must have equal lengths")
        if x.size == 0:
            raise ValueError("x and y must be non-empty")

        n_samples, n_features = x.shape

        # add column of ones for the bias
        x_b = np.c_[np.ones((n_samples, 1)), x]

        # compute weights
        theta = np.linalg.pinv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

        self.bias = theta[0]
        self.weights = theta[1:]
        
    """
    Predicts the labels of the test data.
    Inputs:
        - x: the test data
    """
    def predict(self, x):
        # ensure model has been fitted
        if self.weights is None or self.bias is None:
            raise ValueError("model must be fit before predicting")
        
        x = np.array(x, dtype=np.float64)
        if x.ndim == 1 and self.weights.shape[0] == 1:
            x = x.reshape(-1, 1)
        elif x.ndim == 1:
            x = x.reshape(1, -1) 
        
        # predict
        return np.dot(x, self.weights) + self.bias