import numpy as np
from ..utilities import metrics

__all__ = ['KMeansClustering']


class KMeansClustering:
    """
    Implements the K-Means clustering algorithm.
    """

    def __init__(self, k=3, max_iter=100):
        """
        Initializes the KMeansClustering class.

        Args:
            k (int): The number of clusters to form.
            max_iter (int): The maximum number of iterations for the algorithm to run.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")

        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def _assign_clusters(self, x):
        """Assigns each data point to the closest centroid."""
        labels = np.zeros(len(x))
        for i, point in enumerate(x):
            distances = [metrics.euclidean_distance(point, centroid) for centroid in self.centroids]
            labels[i] = np.argmin(distances)
        return labels.astype(int)

    def _update_centroids(self, x, labels):
        """Updates centroids to be the mean of the points in their cluster."""
        new_centroids = np.zeros((self.k, x.shape[1]))
        for i in range(self.k):
            cluster_points = x[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = self.centroids[i]
        return new_centroids

    def fit(self, x):
        """
        Computes K-Means clustering.

        Args:
            x (array-like): The training data of shape (n_samples, n_features).
        """
        x = np.array(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if len(x) < self.k:
            raise ValueError("Number of samples must be greater than or equal to k")

        # Initialize centroids by randomly selecting k points from the data
        random_indices = np.random.choice(len(x), self.k, replace=False)
        self.centroids = x[random_indices]

        for _ in range(self.max_iter):
            # Assign points to the closest centroid
            self.labels = self._assign_clusters(x)

            # Update centroids
            new_centroids = self._update_centroids(x, self.labels)

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self

    def predict(self, x):
        """
        Predicts the closest cluster for each sample in x.

        Args:
            x (array-like): New data to predict of shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted cluster labels.
        """
        if self.centroids is None:
            raise ValueError("Model must be fit before predicting")

        x = np.array(x, dtype=np.float64)
        if x.ndim == 0:
            x = x.reshape(1, 1)

        if x.ndim == 1:
            if self.centroids.shape[1] == 1:
                x = x.reshape(-1, 1)
            else:
                x = x.reshape(1, -1)

        if x.shape[1] != self.centroids.shape[1]:
            raise ValueError(f"Number of features of the model must "
                             f"match the input. Model n_features is {self.centroids.shape[1]} and "
                             f"input n_features is {x.shape[1]}")

        return self._assign_clusters(x)