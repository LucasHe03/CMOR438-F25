import numpy as np

__all__ = ['CommunityDetection']


class CommunityDetection:
    """
    Community Detection Algorithm using label propagation.
    """

    def __init__(self, max_iter=100):
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")

        self.max_iter = max_iter
        self.labels_ = None

    def fit(self, x):
        """
        Fits the model by detecting communities via label propagation.

        Args:
            x (array-like): adjacency matrix

        Returns:
            self
        """
        # handle empty input
        if isinstance(x, list) and len(x) == 0:
            self.labels_ = np.array([])
            return self

        # convert to numpy array
        adj_matrix = np.array(x, dtype=np.float64)

        if adj_matrix.size == 0:
            self.labels_ = np.array([])
            return self

        # must be square
        if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Input must be a square adjacency matrix.")

        n = adj_matrix.shape[0]

        if n == 0:
            self.labels_ = np.array([])
            return self

        # if no edges, every node is own community
        if np.sum(adj_matrix) == 0:
            self.labels_ = np.arange(n)
            return self

        labels = np.arange(n)

        # Iterative label propagation
        for _ in range(self.max_iter):
            # shuffle node order each iteration for randomness
            nodes = np.arange(n)
            np.random.shuffle(nodes)
            for i in nodes:
                neighbors = np.where(adj_matrix[i] > 0)[0]
                if len(neighbors) == 0:
                    continue
                # assign node i the most common label among neighbors
                neighbor_labels = labels[neighbors]
                labels[i] = np.bincount(neighbor_labels).argmax()

        self.labels_ = labels
        return self

    def fit_predict(self, x):
        self.fit(x)
        return self.labels_