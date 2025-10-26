import numpy as np
from . import metrics 
from . import postprocess as postp

__all__ = ['KNN']
"""
Implements the KNN algorithm.
"""
class KNN:
    """
    Initializes the KNN class. 
    Inputs:
        - k: the number of neighbors to use (default = 3)
    """
    def __init__(self, k = 3):
        self.k = k
        self.x_train = None
        self.y_train = None

        # check for invalid k
        if k <= 0:
            raise ValueError("k must be a positive integer")
    
    """
    Fits the model to the training data.
    Inputs:
        - x: the training data
        - y: the training labels
    """
    def fit(self, x, y):
        self.x_train = np.array(x, np.float64)
        self.y_train = np.array(y)

        # check for errors
        if len(self.x_train) != len(self.y_train):
            raise ValueError("x and y must have equal lengths")
        if self.x_train.size == 0:
            raise ValueError("x and y must be non-empty")
        if self.k > len(self.x_train):
            raise ValueError("k must be less than the number of training samples")
    
    """
    Predicts the labels of the test data.
    Inputs:
        - x: the test data
    """
    def predict(self, x):
        # ensure model has been fitted
        if self.x_train is None or self.y_train is None:
            raise ValueError("model most be fit before predicting")

        # format input data
        x = np.array(x, np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # predict labels
        predictions = []
        for dp in x:
            distances = [metrics.euclidean_distance(dp, x) for x in self.x_train]
            indices = np.argsort(distances)[:self.k]
            candidates = self.y_train[indices]
            # check for type of label
            if np.issubdtype(self.y_train.dtype, np.number):
                predictions.append(postp.average_label(candidates))
            else:
                predictions.append(postp.majority_label(candidates))
        return np.array(predictions)