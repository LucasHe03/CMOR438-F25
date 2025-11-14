import numpy as np
import pytest
import rice2025.supervised_learning.decision_tree as dt

"""
Test __init__ function
"""
def test_init_default():
    model = dt.DecisionTree()
    assert model.min_samples_split == 2
    assert model.max_depth == 100
    assert model.root is None

def test_init_custom():
    model = dt.DecisionTree(min_samples_split=5, max_depth=10)
    assert model.min_samples_split == 5
    assert model.max_depth == 10

"""
Test fit function
"""
def test_fit_success():
    model = dt.DecisionTree()
    x = [[1], [2]]
    y = [0, 1]
    model.fit(x, y)
    assert model.root is not None

def test_fit_mismatched_lengths():
    model = dt.DecisionTree()
    x = [[1], [2]]
    y = [0]
    with pytest.raises(ValueError, match="x and y must have equal lengths"):
        model.fit(x, y)

def test_fit_empty():
    model = dt.DecisionTree()
    x = []
    y = []
    with pytest.raises(ValueError, match="x and y must be non-empty"):
        model.fit(x, y)

"""
Test predict function
"""
def test_predict_not_fitted():
    model = dt.DecisionTree()
    with pytest.raises(ValueError, match="model must be fit before predicting"):
        model.predict([[1]])

def test_predict_classification():
    model = dt.DecisionTree(max_depth=5)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)
    
    x_test = [[1.5], [9.5]]
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_predict_regression():
    model = dt.DecisionTree(max_depth=5)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    model.fit(x_train, y_train)
    
    x_test = [[2.5], [8.5]]
    predictions = model.predict(x_test)
    np.testing.assert_array_almost_equal(predictions, np.array([10.0, 20.0]))

def test_predict_single_point():
    model = dt.DecisionTree()
    x_train = [[1], [10]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    
    prediction = model.predict([11])
    np.testing.assert_array_equal(prediction, np.array([1]))

def test_predict_max_depth():
    model = dt.DecisionTree(max_depth=0)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 1, 1, 1, 1]
    model.fit(x_train, y_train)
    
    predictions = model.predict([[1], [9]])
    np.testing.assert_array_equal(predictions, np.array([1, 1]))

def test_predict_min_samples_split():
    model = dt.DecisionTree(min_samples_split=10)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)

    assert model.root.is_leaf_node()

def test_predict_min_samples_split_tie_classification():
    model = dt.DecisionTree(min_samples_split=10)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 0, 1, 1, 1] # Tie in classes
    model.fit(x_train, y_train)

    predictions = model.predict([[5]])
    np.testing.assert_array_equal(predictions, np.array([0]))

def test_pure_node_becomes_leaf():
    model = dt.DecisionTree()
    x_train = [[1], [2], [3], [4]]
    y_train = [1, 1, 1, 1] # All same class
    model.fit(x_train, y_train)
    assert model.root.is_leaf_node()
    assert model.root.value == 1

def test_no_information_gain():
    model = dt.DecisionTree()
    x_train = [[1], [1], [1], [1]]
    y_train = [0, 1, 0, 1]
    model.fit(x_train, y_train)
    assert model.root.is_leaf_node()

def test_predict_multi_feature_classification():
    model = dt.DecisionTree(max_depth=5)
    x_train = [[1, 10], [2, 11], [8, 1], [9, 2]]
    y_train = [0, 0, 1, 1]
    model.fit(x_train, y_train)
    predictions = model.predict([[1.5, 20], [8.5, 0]])
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_predict_empty_input():
    model = dt.DecisionTree()
    model.fit([[1]], [1])
    predictions = model.predict([])
    assert predictions.shape == (0,)

def test_predict_mismatched_features():
    model = dt.DecisionTree()
    model.fit([[1, 2]], [1])
    with pytest.raises(ValueError, match="Number of features of the model must match the input"):
        model.predict([[1, 2, 3]])