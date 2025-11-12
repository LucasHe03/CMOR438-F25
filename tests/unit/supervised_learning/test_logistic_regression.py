import numpy as np
import pytest
import rice2025.supervised_learning.logistic_regression as logr

"""
Test __init__ function
"""
def test_init_default():
    model = logr.LogisticRegression()
    assert model.lr == 0.01
    assert model.n_iter == 1000
    assert model.weights is None
    assert model.bias is None

def test_init_custom():
    model = logr.LogisticRegression(lr = 0.1, n_iter = 2000)
    assert model.lr == 0.1
    assert model.n_iter == 2000

"""
Test fit function
"""
def test_fit_success():
    model = logr.LogisticRegression()
    x = [[1, 2], [3, 4]]
    y = [0, 1]
    model.fit(x, y)
    assert model.weights is not None
    assert model.bias is not None

def test_fit_1d():
    model = logr.LogisticRegression()
    x = [1, 5]
    y = [0, 1]
    model.fit(x, y)
    assert model.weights is not None
    assert model.bias is not None

def test_fit_mismatched_lengths():
    model = logr.LogisticRegression()
    x = [[1, 2], [3, 4]]
    y = [0]
    with pytest.raises(ValueError, match="x and y must have equal lengths"):
        model.fit(x, y)

def test_fit_empty():
    model = logr.LogisticRegression()
    x = []
    y = []
    with pytest.raises(ValueError, match="x and y must be non-empty"):
        model.fit(x, y)

"""
Test predict function
"""
def test_predict_not_fitted():
    model = logr.LogisticRegression()
    with pytest.raises(ValueError, match="model must be fit before predicting"):
        model.predict([[1, 1]])

def test_predict_simple_separable():
    model = logr.LogisticRegression(lr=0.1, n_iter=1000)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)
    
    x_test = [[-1], [1.5], [5], [12]]
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([0, 0, 1, 1]))

def test_predict_1d():
    model = logr.LogisticRegression(lr=0.1, n_iter=1000)
    x_train = [-1, -2, -3, 1, 2, 3]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)
    
    x_test = [-1.5, -2.5, 1.5, 2.5]
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([0, 0, 1, 1]))

def test_predict_single_point():
    model = logr.LogisticRegression(lr=0.1, n_iter=1000)
    x_train = [[-2, -1], [-1, -1], [1, 1], [2, 1]]
    y_train = [0, 0, 1, 1]
    model.fit(x_train, y_train)
    
    prediction = model.predict([1.5, 1])
    np.testing.assert_array_equal(prediction, np.array([1]))

def test_predict_mismatched_features():
    model = logr.LogisticRegression()
    x_train = [[1, 1], [2, 2]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    
    x_test_mismatched = [[1, 1, 1]]
    with pytest.raises(ValueError):
        model.predict(x_test_mismatched)

