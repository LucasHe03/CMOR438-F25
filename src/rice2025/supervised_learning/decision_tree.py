import numpy as np
from ..utilities import postprocess as postp

__all__ = ['DecisionTree']


"""
Helper class to represent a node in the decision tree.
"""
class _Node:
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, *, value = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

"""
Implements the decision tree algorithm.
"""
class DecisionTree:
    """
    Initializes the DecisionTree class.
    Inputs:
        - min_samples_split: The minimum number of samples required to split a node (default = 2).
        - max_depth: The maximum depth of the tree (default = 100).
    """
    def __init__(self, min_samples_split = 2, max_depth = 100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self._is_classifier = None
        self._n_features = None

    def fit(self, x, y):
        """
        Fits the model to the training data.
        Inputs:
            - x: the training data
            - y: the training target values
        """
        x = np.array(x, dtype = np.float64)
        y = np.array(y)

        if len(x) != len(y):
            raise ValueError("x and y must have equal lengths")
        if x.size == 0:
            raise ValueError("x and y must be non-empty")

        self._is_classifier = not np.issubdtype(y.dtype, np.number) or len(np.unique(y)) < 20
        self._n_features = x.shape[1] if x.ndim > 1 else 1
        self.root = self._build_tree(x, y)

    """
    Private helper to recursively build the tree.
    """
    def _build_tree(self, x, y, depth = 0):
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))

        # check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._leaf_value(y)
            return _Node(value=leaf_value)

        # find the best split
        feat_idxs = np.random.choice(n_features, n_features, replace = False)
        best_split = self._best_split(x, y, n_features)
        if best_split is None or best_split['threshold'] is None or best_split['gain'] <= 0:
            leaf_value = self._leaf_value(y)
            return _Node(value = leaf_value)
        
        # recurse on children
        left_idxs, right_idxs = self._split(x[:, best_split['feature_index']], best_split['threshold'])
        left = self._build_tree(x[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(x[right_idxs, :], y[right_idxs], depth + 1)
        return _Node(best_split['feature_index'], best_split['threshold'], left, right)

    """
    Private helper to find the best split for a node.
    """
    def _best_split(self, x, y, n_features):
        best_gain = -1
        best_split = None

        for feat_idx in range(n_features):
            x_column = x[:, feat_idx]
            unique_values = np.unique(x_column)
            if len(unique_values) == 1:
                continue
            for threshold in unique_values:
                gain = self._information_gain(y, x_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_split = {'feature_index': feat_idx, 'threshold': threshold, 'gain': gain}
        return best_split

    """
    Private helper to calculate information gain.
    """
    def _information_gain(self, y, x_column, threshold):
        if self._is_classifier:
            parent_impurity = self._gini(y)
        else:
            parent_impurity = np.var(y)

        # generate split
        left_idxs, right_idxs = self._split(x_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute weighted average of children impurity
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        if self._is_classifier:
            e_l, e_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        else:
            e_l, e_r = np.var(y[left_idxs]), np.var(y[right_idxs])

        child_impurity = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is parent impurity - weighted child impurity
        return parent_impurity - child_impurity

    """
    Private helper to split a column by a threshold.
    """
    def _split(self, x_column, split_thresh):
        left_idxs = np.argwhere(x_column <= split_thresh).flatten()
        right_idxs = np.argwhere(x_column > split_thresh).flatten()
        return left_idxs, right_idxs

    """
    Private helper to calculate Gini impurity.
    """
    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)

    """
    Private helper to determine the value of a leaf node.
    """
    def _leaf_value(self, y):
        if self._is_classifier:
            return postp.majority_label(y)
        return np.mean(y)

    def predict(self, x):
        """
        Predicts the labels of the test data.
        Inputs:
            - x: the test data
        """
        if self.root is None:
            raise ValueError("model must be fit before predicting")

        x = np.array(x, dtype = np.float64)
        if x.size == 0:
            return np.array([])
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != self._n_features:
            raise ValueError(f"Number of features of the model must "
                             f"match the input. Model n_features is {self._n_features} and "
                             f"input n_features is {x.shape[1]}")
        return np.array([self._traverse_tree(sample, self.root) for sample in x])

    """
    Private helper to traverse the tree for a single data point.
    """
    def _traverse_tree(self, x, node):
        if node.is_leaf_node() or node.threshold is None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)