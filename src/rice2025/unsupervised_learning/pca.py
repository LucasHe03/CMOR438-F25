import numpy as np

__all__ = ['PCA']


class PCA:
    """
    Principal Component Analysis (PCA).
    """

    def __init__(self, n_components):
        """
        Initializes the PCA class.

        Inputs:
            n_components: The number of principal components to keep.
        """
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer")

        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, x):
        """
        Fit the model with X by computing the principal components.

        Inputs:
            : Training data, shape (n_samples, n_features).
        """
        if len(x) == 0:
            raise ValueError("Input data x cannot be empty")

        x = np.array(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if self.n_components > x.shape[1]:
            raise ValueError("n_components cannot be greater than the number of features")

        # center data
        self.mean_ = np.mean(x, axis=0)
        x_centered = x - self.mean_

        # compute covariance matrix
        cov_matrix = np.cov(x_centered, rowvar=False)

        cov_matrix = np.atleast_2d(cov_matrix)

        # compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # sort eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # store the principal components
        self.components_ = sorted_eigenvectors[:, :self.n_components]

        return self

    def transform(self, x):
        """
        Apply dimensionality reduction to X.

        Inputs:
            x: Data to transform, shape (n_samples, n_features).
        """
        if self.components_ is None:
            raise ValueError("PCA model must be fit before transforming data.")

        x = np.array(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # center the data
        x_centered = x - self.mean_

        # project the data
        return np.dot(x_centered, self.components_)

    def fit_transform(self, x):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        """
        self.fit(x)
        return self.transform(x)