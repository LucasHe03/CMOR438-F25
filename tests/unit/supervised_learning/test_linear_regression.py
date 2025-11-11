import numpy as np
import pytest
import rice2025.supervised_learning.linear_regression as lr

"""
Test __init__ function
"""
def test_init_default():
    model = lr.LinearRegression()
    assert model.weights is None
    assert model.bias is None

"""
Test fit function
"""
def test_fit_success():
    model = lr.LinearRegression()
    x = [[1, 2], [3, 4]]
    y = [1, 2]
    model.fit(x, y)
    assert model.weights is not None
    assert model.bias is not None
    assert model.weights.shape == (2,)

def test_fit_mismatched_lengths():
    model = lr.LinearRegression()
    x = [[1, 2], [3, 4]]
    y = [1]
    with pytest.raises(ValueError, match="x and y must have equal lengths"):
        model.fit(x, y)

def test_fit_empty():
    model = lr.LinearRegression()
    x = []
    y = []
    with pytest.raises(ValueError, match="x and y must be non-empty"):
        model.fit(x, y)

"""
Test predict function
"""
def test_predict_not_fitted():
    model = lr.LinearRegression()
    with pytest.raises(ValueError, match="model must be fit before predicting"):
        model.predict([[1, 1]])

def test_predict_simple_1d():
    model = lr.LinearRegression()
    # y = 2x + 1
    x_train = np.array([1, 2, 3, 4])
    y_train = np.array([3, 5, 7, 9])
    model.fit(x_train, y_train)

    x_test = np.array([5, 6]).reshape(-1, 1)
    predictions = model.predict(x_test)

    assert model.weights is not None
    assert model.bias is not None
    np.testing.assert_almost_equal(model.weights[0], 2.0, decimal=5)
    np.testing.assert_almost_equal(model.bias, 1.0, decimal=5)
    np.testing.assert_array_almost_equal(predictions, np.array([11, 13]), decimal=5)

def test_predict_simple_2d():
    model = lr.LinearRegression()
    # y = 3*x1 + 2*x2 + 5
    x_train = np.array([[1, 1], [2, 2], [3, 1], [1, 3]])
    y_train = np.array([10, 15, 16, 14])
    model.fit(x_train, y_train)

    x_test = np.array([[2, 3], [4, 1]])
    predictions = model.predict(x_test)

    assert model.weights is not None
    assert model.bias is not None
    np.testing.assert_array_almost_equal(model.weights, np.array([3.0, 2.0]), decimal=5)
    np.testing.assert_almost_equal(model.bias, 5.0, decimal=5)
    np.testing.assert_array_almost_equal(predictions, np.array([17, 19]), decimal=5)

def test_predict_single_point():
    model = lr.LinearRegression()
    x_train = [[1], [2]]
    y_train = [1, 2]
    model.fit(x_train, y_train)
    
    prediction = model.predict([3])
    np.testing.assert_array_almost_equal(prediction, np.array([3.0]))

def test_predict_mismatched_features():
    model = lr.LinearRegression()
    x_train = [[1, 1], [2, 2]]
    y_train = [1, 2]
    model.fit(x_train, y_train)
    
    x_test_mismatched = [[1, 1, 1]]
    with pytest.raises(ValueError):
        model.predict(x_test_mismatched)