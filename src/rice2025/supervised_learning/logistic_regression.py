import numpy as np

__all__ = ['LogisticRegression']

"""
Implements logistic regression.
"""
class LogisticRegression:
    """
    Initializes the LogisticRegression Class.
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
    Private helper for sigmoid function. 
    """
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    """
    Fits the model to the training data.
    Inputs:
        - x: the training data
        - y: the training target values (0 or 1)
    """
    def fit(self, x, y):
        x = np.array(x, dtype = np.float64)
        y = np.array(y, dtype = np.float64)

        if len(x) != len(y):
            raise ValueError("x and y must have equal lengths")
        if x.size == 0:
            raise ValueError("x and y must be non-empty")
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n_samples, n_features = x.shape
        
        # initialize params
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # gradient descent
        for _ in range(self.n_iter):
            linear_model = np.dot(x, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    """
    Predicts the labels of the test data.
    Inputs:
        - x: the test data
    """
    def predict(self, x):
        if self.weights is None or self.bias is None:
            raise ValueError("model must be fit before predicting")
        
        x = np.array(x, dtype = np.float64)
        if x.size == 0:
            return np.array([])
        if x.ndim == 1:
            if x.shape[0] == self.weights.shape[0]:
                x = x.reshape(1, -1)
            else:
                x = x.reshape(-1, 1)
        
        linear_model = np.dot(x, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred)