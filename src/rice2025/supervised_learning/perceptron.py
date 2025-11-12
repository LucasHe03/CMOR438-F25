import numpy as np

__all__ = ['Perceptron']

"""
Implements the Perceptron algorithm.
"""
class Perceptron:
    """
    Initializes the Perceptron class.
    Inputs:
        - learning_rate: The learning rate (default = .01)
        - n_iter: The number of passes over the training dataset (default = 1000)
    """
    def __init__(self, lr = 0.01, n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    """
    Fits the model to the training data.
    Inputs:
        - x: the training data
        - y: the training labels
    """
    def fit(self, x, y):
        x = np.array(x, dtype = np.float64)
        y = np.array(y, dtype = np.int_)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # check for errors
        if len(x) != len(y):
            raise ValueError("x and y must have equal lengths")
        if x.size == 0:
            raise ValueError("x and y must be non-empty")

        n_samples, n_features = x.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(x):
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Predict
                y_predicted = np.where(linear_output >= 0.0, 1, -1)
                
                # Update
                if y_predicted != y[idx]:
                    self.weights += self.lr * y[idx] * x_i
                    self.bias += self.lr * y[idx]
        return self

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
                
        # Calculate net input
        linear_output = np.dot(x, self.weights) + self.bias
        # Predict
        y_predicted = np.where(linear_output >= 0.0, 1, -1)
        return y_predicted