import numpy as np
import pytest
import rice2025.supervised_learning.perceptron as p

"""
Test __init__ function
"""
def test_init_default():
    model = p.Perceptron()
    assert model.lr == 0.01
    assert model.n_iter == 1000
    assert model.weights is None
    assert model.bias is None

def test_init_custom():
    model = p.Perceptron(lr = 0.1, n_iter = 100)
    assert model.lr == 0.1
    assert model.n_iter == 100

"""
Test fit function
"""
def test_fit_success():
    model = p.Perceptron()
    x = [[1, 2], [3, 4]]
    y = [1, -1]
    model.fit(x, y)
    assert model.weights is not None
    assert model.bias is not None

def test_fit_mismatched_lengths():
    model = p.Perceptron()
    x = [[1, 2], [3, 4]]
    y = [1]
    with pytest.raises(ValueError, match="x and y must have equal lengths"):
        model.fit(x, y)

def test_fit_empty():
    model = p.Perceptron()
    x = []
    y = []
    with pytest.raises(ValueError, match="x and y must be non-empty"):
        model.fit(x, y)

"""
Test predict function
"""
def test_predict_not_fitted():
    model = p.Perceptron()
    with pytest.raises(ValueError, match="model must be fit before predicting"):
        model.predict([[1, 1]])

def test_predict_simple_separable():
    model = p.Perceptron(lr = 0.1, n_iter = 10)
    x_train = [[2, 1], [3, 4], [-1, -2], [-3, -1]]
    y_train = [1, 1, -1, -1]
    model.fit(x_train, y_train)
    
    x_test = [[1, 1], [-1, -1]]
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([1, -1]))

def test_predict_single_point():
    model = p.Perceptron(lr = 0.1, n_iter = 10)
    x_train = [[2, 1], [3, 4], [-1, -2], [-3, -1]]
    y_train = [1, 1, -1, -1]
    model.fit(x_train, y_train)
    
    prediction = model.predict([1, 1])
    np.testing.assert_array_equal(prediction, np.array([1]))

def test_predict_multiple_points():
    model = p.Perceptron(lr = 0.1, n_iter = 10)
    x_train = [[2, 1], [3, 4], [-1, -2], [-3, -1]]
    y_train = [1, 1, -1, -1]
    model.fit(x_train, y_train)
    
    x_test = [[1, 1], [-1, -1], [5, 5], [-5, -5]]
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([1, -1, 1, -1]))

def test_predict_mismatched_features():
    model = p.Perceptron()
    x_train = [[1, 1], [2, 2]]
    y_train = [1, -1]
    model.fit(x_train, y_train)
    
    x_test_mismatched = [[1, 1, 1]]
    with pytest.raises(ValueError):
        model.predict(x_test_mismatched)

def test_predict_1d_data():
    model = p.Perceptron(lr = 0.1, n_iter = 100)
    x_train = [[1], [2], [-1], [-2]]
    y_train = [1, 1, -1, -1]
    model.fit(x_train, y_train)
    
    x_test = [[1], [-1]]
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([1, -1]))

def test_predict_1d_full():
    model = p.Perceptron(lr = 0.1, n_iter = 100)
    x_train = [.1, .2, .3, -.1, -.2, -.3]
    y_train = [1, 1, 1, -1, -1, -1]
    model.fit(x_train, y_train)

    x_test = [.15, -.15]
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([1, -1]))