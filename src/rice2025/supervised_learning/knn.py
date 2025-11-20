import numpy as np
from ..utilities import metrics 
from ..utilities import postprocess as postp

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
        # check for invalid k
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be a positive integer")

        self.k = k
        self.x_train = None
        self.y_train = None
        self._is_classifier = None
        self._n_features = None

    """
    Fits the model to the training data.
    Inputs:
        - x: the training data
        - y: the training labels
    """
    def fit(self, x, y):
        self.x_train = np.array(x, np.float64)
        self.y_train = np.array(y)

        if self.x_train.ndim == 1:
            self.x_train = self.x_train.reshape(-1, 1)

        # check for errors
        if len(self.x_train) != len(self.y_train):
            raise ValueError("x and y must have equal lengths")
        if self.x_train.size == 0:
            raise ValueError("x and y must be non-empty")
        if self.k > len(self.x_train):
            raise ValueError("k must be less than the number of training samples")
        
        self._is_classifier = not np.issubdtype(self.y_train.dtype, np.number) or len(np.unique(self.y_train)) < 5
        self._n_features = self.x_train.shape[1]
    
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
        if x.size == 0:
            return np.array([])
        
        is_1d_input = x.ndim == 1
        if x.ndim == 1:
            if self._n_features == 1:
                x = x.reshape(-1, 1)
            else:
                x = x.reshape(1, -1)

        if x.shape[1] != self._n_features:
            raise ValueError("Number of features of the model must match the input")
        
        # predict labels
        predictions = []
        for dp in x:
            distances = [metrics.euclidean_distance(dp, train_x) for train_x in self.x_train]
            indices = np.argsort(distances)[:self.k]
            candidates = self.y_train[indices]
            # check for type of label
            if self._is_classifier:
                predictions.append(postp.majority_label(candidates))
            else:
                predictions.append(postp.average_label(candidates))
        return np.array(predictions)