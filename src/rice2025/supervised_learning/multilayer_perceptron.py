import numpy as np

__all__ = ['MultilayerPerceptron']

"""
Implements the multilayer perceptron algorithm.
"""
class MultilayerPerceptron:
   
    """
    Initializes the MultilayerPerceptron class.
    Inputs:
        - learning_rate: The learning rate (default = .01)
        - n_iter: The number of passes over the training dataset (default = 1000)
    """
    def __init__(self, n_hidden=100, lr = 0.01, n_iter = 1000):
        self.n_hidden = n_hidden
        self.lr = lr
        self.n_iter = n_iter
        self.w1, self.b1 = None, None
        self.w2, self.b2 = None, None

    """
    Private helper for sigmoid function. 
    """
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    """
    Private helper for RELU function.
    """
    def _relu(self, z):
        return np.maximum(0, z)

    """
    Private helper for derivative of RELU function.
    """
    def _relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    """
    Fits the model to the training data.
    Inputs:
        - x: the training data
        - y: the training target values
    """
    def fit(self, x, y):
        x = np.array(x, dtype = np.float64)
        y = np.array(y, dtype = np.float64).reshape(-1, 1)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if len(x) != len(y):
            raise ValueError("x and y must have equal lengths")
        if x.size == 0:
            raise ValueError("x and y must be non-empty")

        n_samples, n_features = x.shape
        
        # init weights and biases
        self.w1 = np.random.randn(n_features, self.n_hidden) * 0.01
        self.b1 = np.zeros((1, self.n_hidden))
        self.w2 = np.random.randn(self.n_hidden, 1) * 0.01
        self.b2 = np.zeros((1, 1))

        # gradient descent
        losses = []
        for _ in range(self.n_iter):
            # forward propagation
            z1 = np.dot(x, self.w1) + self.b1
            a1 = self._relu(z1)
            z2 = np.dot(a1, self.w2) + self.b2
            a2 = self._sigmoid(z2)

            # backward
            d_z2 = a2 - y
            # gradients
            d_w2 = (1 / n_samples) * np.dot(a1.T, d_z2)
            d_b2 = (1 / n_samples) * np.sum(d_z2, axis = 0, keepdims = True)

            # hidden layer
            d_a1 = np.dot(d_z2, self.w2.T)
            d_z1 = d_a1 * self._relu_derivative(z1)
            # gradients for w1 and b1
            d_w1 = (1 / n_samples) * np.dot(x.T, d_z1)
            d_b1 = (1 / n_samples) * np.sum(d_z1, axis = 0, keepdims = True)

            # update
            self.w1 -= self.lr * d_w1
            self.b1 -= self.lr * d_b1
            self.w2 -= self.lr * d_w2
            self.b2 -= self.lr * d_b2

            loss = -np.mean(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))
            losses.append(loss)
        return losses
        

    """
    Predicts the labels of the test data.
    Inputs:
        - x: the test data
    """
    def predict(self, x):
        if self.w1 is None or self.w2 is None:
            raise ValueError("model must be fit before predicting")

        z1 = np.dot(x, self.w1) + self.b1
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self._sigmoid(z2)
        
        predictions = (a2 > 0.5).astype(int)
        return predictions.flatten()