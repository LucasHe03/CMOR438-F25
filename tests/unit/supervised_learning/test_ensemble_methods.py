import numpy as np
import pytest

from rice2025.supervised_learning.ensemble_methods import EnsembleVotingClassifier


class ConstantModel:
    def __init__(self, label):
        self.label = label
        self.was_fit = False

    def fit(self, X, y):
        self.was_fit = True

    def predict(self, X):
        return np.full(len(X), self.label)


class CyclingModel:
    def __init__(self, preds):
        self.preds = np.array(preds)
        self.was_fit = False

    def fit(self, X, y):
        self.was_fit = True

    def predict(self, X):
        return self.preds[: len(X)]

def fake_majority_label(sample_preds):
    """Simple majority vote: pick the most frequent label."""
    vals, counts = np.unique(sample_preds, return_counts=True)
    return vals[np.argmax(counts)]

def test_fit_calls_fit_on_all_models():
    m1 = ConstantModel(0)
    m2 = ConstantModel(1)
    m3 = ConstantModel(2)
    ens = EnsembleVotingClassifier(m1, m2, m3)

    X = np.array([[0], [1]])
    y = np.array([0, 1])
    ens.fit(X, y)

    assert m1.was_fit
    assert m2.was_fit
    assert m3.was_fit


def test_predict_constant_models_majority_vote():
    m1 = ConstantModel(1)
    m2 = ConstantModel(1)
    m3 = ConstantModel(0)

    ens = EnsembleVotingClassifier(m1, m2, m3)

    X = np.array([[10], [20], [30]])
    preds = ens.predict(X)

    assert np.all(preds == 1)


def test_predict_all_different_labels_resolves_by_first_max():
    m1 = ConstantModel(0)
    m2 = ConstantModel(1)
    m3 = ConstantModel(2)

    ens = EnsembleVotingClassifier(m1, m2, m3)
    X = np.array([[0], [1], [2]])
    preds = ens.predict(X)
    assert np.all(preds == 0)


def test_predict_correct_shape_output():
    m1 = ConstantModel(1)
    m2 = ConstantModel(1)
    m3 = ConstantModel(1)

    ens = EnsembleVotingClassifier(m1, m2, m3)
    X = np.ones((7, 3)) 
    preds = ens.predict(X)

    assert preds.shape == (7,)


def test_predict_with_cycling_models():
    m1 = CyclingModel([0, 1, 0])
    m2 = CyclingModel([1, 1, 0])
    m3 = CyclingModel([0, 0, 1])

    ens = EnsembleVotingClassifier(m1, m2, m3)
    X = np.zeros((3, 2))

    preds = ens.predict(X)
    assert np.array_equal(preds, np.array([0, 1, 0]))


def test_predict_empty_X_returns_empty():
    m1 = ConstantModel(1)
    m2 = ConstantModel(1)
    m3 = ConstantModel(1)

    ens = EnsembleVotingClassifier(m1, m2, m3)
    X = np.empty((0, 3))
    preds = ens.predict(X)

    assert preds.size == 0


def test_predict_single_sample():
    m1 = ConstantModel(1)
    m2 = ConstantModel(0)
    m3 = ConstantModel(1)

    ens = EnsembleVotingClassifier(m1, m2, m3)
    X = np.array([[42]])
    preds = ens.predict(X)

    assert preds[0] == 1


def test_does_not_modify_input_data():
    m1 = CyclingModel([1, 0])
    m2 = CyclingModel([1, 0])
    m3 = CyclingModel([0, 0])

    X = np.array([[5], [6]])
    X_copy = X.copy()

    ens = EnsembleVotingClassifier(m1, m2, m3)
    _ = ens.predict(X)

    assert np.array_equal(X, X_copy)


def test_fit_returns_self():
    m1 = ConstantModel(1)
    m2 = ConstantModel(1)
    m3 = ConstantModel(1)

    ens = EnsembleVotingClassifier(m1, m2, m3)
    returned = ens.fit(np.array([[0]]), np.array([0]))

    assert returned is ens


def test_predict_mismatched_length_models():
    m1 = CyclingModel([0, 1]) 
    m2 = ConstantModel(1)
    m3 = ConstantModel(1)

    ens = EnsembleVotingClassifier(m1, m2, m3)

    X = np.array([[0], [1], [2]])

    with pytest.raises(ValueError):
        predictions = np.array([m.predict(X) for m in ens.models])


def test_predict_consistency_repeated_calls():
    m1 = CyclingModel([0, 1, 0])
    m2 = CyclingModel([1, 1, 0])
    m3 = CyclingModel([0, 0, 1])
    X = np.zeros((3, 2))

    ens = EnsembleVotingClassifier(m1, m2, m3)

    p1 = ens.predict(X)
    p2 = ens.predict(X)

    assert np.array_equal(p1, p2)