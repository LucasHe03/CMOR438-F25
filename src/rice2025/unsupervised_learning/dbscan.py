import numpy as np
from ..utilities import metrics

__all__ = ['DBSCAN']


class DBSCAN:
    """
    Implements the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm.
    """

    def __init__(self, eps=0.5, min_samples=5):
        """
        Initializes the DBSCAN class.
        Inputs:
            eps: The maximum distance between two samples for one to be
                         considered as in the neighborhood of the other.
            min_samples: The number of samples in a neighborhood for a
                               point to be considered as a core point.
        """
        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError("eps must be a positive number")
        if not isinstance(min_samples, int) or min_samples <= 0:
            raise ValueError("min_samples must be a positive integer")

        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _region_query(self, x, point_idx):
        """Finds all points within eps distance of a given point."""
        neighbors = []
        for i in range(len(x)):
            if metrics.euclidean_distance(x[point_idx], x[i]) < self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, x, labels, point_idx, neighbors, cluster_id):
        """Expands a cluster from a core point."""
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if labels[neighbor_idx] == -1:  # if point was labeled as noise
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == 0:  # if point is unvisited
                labels[neighbor_idx] = cluster_id
                new_neighbors = self._region_query(x, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)
            i += 1

    def fit(self, x):
        """
        Performs DBSCAN clustering.

        Inputs:
            x: The training data of shape (n_samples, n_features).
        """
        x = np.array(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n_samples = len(x)
        if n_samples == 0:
            self.labels_ = np.array([])
            return self

        # 0: unvisited, -1: noise
        self.labels_ = np.zeros(n_samples, dtype=int)
        cluster_id = 0

        for i in range(n_samples):
            if self.labels_[i] != 0:  # if already visited
                continue

            neighbors = self._region_query(x, i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1 # mark as noise
            else:
                cluster_id += 1
                self._expand_cluster(x, self.labels_, i, neighbors, cluster_id)

        self.labels_[self.labels_ > 0] -= 1

        return self

    def fit_predict(self, x):
        """
        Fits the model and returns the cluster labels.

        Inputs:
            x: The training data of shape (n_samples, n_features).
        """
        self.fit(x)
        return self.labels_