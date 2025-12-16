import numpy as np
from ..utilities import postprocess as postp

class EnsembleVotingClassifier:
    """
    Ensemble classifier that combines three different models.

    Parameters
    ----------
    model1, model2, model3 : models that implement:
        - fit(X, y)
        - predict(X)
    """

    def __init__(self, model1, model2, model3):
        """
        Initialize the ensemble with three models and voting type.
        """
        self.models = [model1, model2, model3]

    def fit(self, X, y):
        """
        Fit all three models on the training data.
        """
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X, f=postp.majority_label):
        """
        Predict class labels for X using ensemble voting.
        """
        # collect predictions from all 3 models
        predictions = np.array([model.predict(X) for model in self.models])

        return np.array([f(sample_preds) for sample_preds in predictions.T])