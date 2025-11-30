import numpy as np

__all__ = ['CommunityDetection']


class CommunityDetection:
    """
    Community Detection Algorithm.
    """

    def __init__(self, max_iter=100):
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")

        self.max_iter = max_iter
        self.labels_ = None

    def fit(self, x):
        """
        Fits the model by detecting connected components.

        Args:
            x (array-like): adjacency matrix

        Returns:
            self
        """

        # handle of empty input
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

        # If no edges, every node is own community
        if np.sum(adj_matrix) == 0:
            self.labels_ = np.arange(n)
            return self

        # Connected components
        visited = np.zeros(n, dtype=bool)
        labels = np.full(n, -1)
        current_label = 0

        for i in range(n):
            if not visited[i]:
                stack = [i]
                visited[i] = True
                labels[i] = current_label

                while stack:
                    node = stack.pop()
                    neighbors = np.where(adj_matrix[node] > 0)[0]
                    for nb in neighbors:
                        if not visited[nb]:
                            visited[nb] = True
                            labels[nb] = current_label
                            stack.append(nb)

                current_label += 1

        self.labels_ = labels
        return self

    def fit_predict(self, x):
        self.fit(x)
        return self.labels_