import numpy as np
from .decision_tree import DecisionTree
from ..utilities import postprocess as postp

__all__ = ['RandomForest']

"""
Implements the Random Forest algorithm.
"""
class RandomForest:
    """
    Initializes the RandomForest class.
    Inputs:
        - n_estimators: The number of trees in the forest (default = 100).
        - min_samples_split: The minimum number of samples required to split a node (default = 2).
        - max_depth: The maximum depth of the tree (default = 100).
    """
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=100):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []
        self._is_classifier = None

    """
    Fits the model to the training data.
    Inputs:
        - x: the training data
        - y: the training target values
    """
    def fit(self, x, y):
        x = np.array(x, dtype=np.float64)
        y = np.array(y)

        if len(x) != len(y):
            raise ValueError("x and y must have equal lengths")
        if x.size == 0:
            raise ValueError("x and y must be non-empty")

        self._is_classifier = not np.issubdtype(y.dtype, np.number) or len(np.unique(y)) < 20
        self.trees = []
        n_samples = x.shape[0]

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            x_sample, y_sample = x[idxs], y[idxs]

            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)
            
    """ 
    Predicts the labels of the test data.
    Inputs:
        - x: the test data
    """
    def predict(self, x):
        if not self.trees:
            raise ValueError("model must be fit before predicting")

        x = np.array(x, dtype=np.float64)
        if x.size == 0:
            return np.array([])
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Get predictions from all trees
        tree_preds = np.array([tree.predict(x) for tree in self.trees])

        # For each sample, aggregate the predictions from all trees
        # Transpose so that rows are samples and columns are tree predictions
        predictions = []
        for sample_preds in tree_preds.T:
            if self._is_classifier:
                predictions.append(postp.majority_label(sample_preds))
            else:
                predictions.append(np.mean(sample_preds))
        
        return np.array(predictions)