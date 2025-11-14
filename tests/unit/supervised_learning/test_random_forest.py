import numpy as np
import pytest
import rice2025.supervised_learning.random_forest as rf

"""
Test __init__ function
"""
def test_init_default():
    model = rf.RandomForest()
    assert model.n_estimators == 100
    assert model.min_samples_split == 2
    assert model.max_depth == 100
    assert model.trees == []

def test_init_custom():
    model = rf.RandomForest(n_estimators=50, min_samples_split=5, max_depth=10)
    assert model.n_estimators == 50
    assert model.min_samples_split == 5
    assert model.max_depth == 10

"""
Test fit function
"""
def test_fit_success():
    model = rf.RandomForest(n_estimators=5)
    x = [[1], [2]]
    y = [0, 1]
    model.fit(x, y)
    assert model.trees is not None
    assert len(model.trees) == 5

def test_fit_mismatched_lengths():
    model = rf.RandomForest()
    x = [[1], [2]]
    y = [0]
    with pytest.raises(ValueError, match="x and y must have equal lengths"):
        model.fit(x, y)

def test_fit_empty():
    model = rf.RandomForest()
    x = []
    y = []
    with pytest.raises(ValueError, match="x and y must be non-empty"):
        model.fit(x, y)

"""
Test predict function
"""
def test_predict_not_fitted():
    model = rf.RandomForest()
    with pytest.raises(ValueError, match="model must be fit before predicting"):
        model.predict([[1]])

def test_predict_classification():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=10, max_depth=5)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)
    
    x_test = [[1.5], [9.5]]
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_predict_regression():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=10, max_depth=5)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    model.fit(x_train, y_train)
    
    x_test = [[2.5], [8.5]]
    predictions = model.predict(x_test)
    np.testing.assert_array_almost_equal(predictions, np.array([10.0, 20.0]), decimal=1)

def test_predict_single_point_classification():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=10)
    x_train = [[1], [10]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    
    prediction = model.predict([11])
    np.testing.assert_array_equal(prediction, np.array([1]))

def test_predict_single_point_regression():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=10)
    x_train = [[1], [10]]
    y_train = [10.0, 20.0]
    model.fit(x_train, y_train)
    
    prediction = model.predict([11])
    assert prediction[0] > 15.0

def test_predict_empty_input():
    model = rf.RandomForest()
    x_train = [[0, 0], [1, 1]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    predictions = model.predict([])
    assert predictions.shape == (0,)