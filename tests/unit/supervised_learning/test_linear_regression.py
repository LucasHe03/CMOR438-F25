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

def test_fit_numpy_input():
    model = lr.LinearRegression()
    x = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    model.fit(x, y)
    assert model.weights is not None
    assert model.bias is not None

def test_fit_with_bias_only():
    model = lr.LinearRegression()
    x = np.array([[1], [2], [3], [4]])
    y = np.array([5, 5, 5, 5])
    model.fit(x, y)
    np.testing.assert_almost_equal(model.bias, 5.0, decimal=5)
    np.testing.assert_almost_equal(model.weights[0], 0.0, decimal=5)

def test_fit_perfect_collinearity():
    model = lr.LinearRegression()
    x = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([2, 4, 6]) # y = 2 * (x1 or x2)
    model.fit(x, y)
    assert model.weights is not None
    assert model.bias is not None
    np.testing.assert_almost_equal(np.sum(model.weights), 2.0, decimal=5)

def test_fit_data_with_negative_values():
    model = lr.LinearRegression()
    x = np.array([[-1], [-2], [-3]])
    y = np.array([-2, -4, -6]) # y = 2x
    model.fit(x, y)
    np.testing.assert_almost_equal(model.weights[0], 2.0, decimal=5)
    np.testing.assert_almost_equal(model.bias, 0.0, decimal=5)

def test_fit_more_features_than_samples():
    model = lr.LinearRegression()
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    model.fit(x, y)
    assert model.weights.shape == (3,)

def test_fit_float_y_values():
    model = lr.LinearRegression()
    x = np.array([[1], [2]])
    y = np.array([1.5, 2.5])
    model.fit(x, y)
    np.testing.assert_almost_equal(model.weights[0], 1.0, decimal=5)
    np.testing.assert_almost_equal(model.bias, 0.5, decimal=5)

def test_predict_on_training_data():
    model = lr.LinearRegression()
    x_train = np.array([[1], [2], [3]])
    y_train = np.array([2, 4, 6])
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)
    np.testing.assert_array_almost_equal(predictions, y_train, decimal=5)

def test_predict_empty_input():
    model = lr.LinearRegression()
    model.fit([[1]], [1])
    predictions = model.predict([])
    assert predictions.shape == (0,)

def test_predict_single_1d_feature_input_scalar():
    model = lr.LinearRegression()
    model.fit([1, 2], [1, 2])
    prediction = model.predict(3)
    np.testing.assert_array_almost_equal(prediction, [3.0])

def test_predict_multiple_1d_feature_inputs_list():
    model = lr.LinearRegression()
    model.fit([1, 2], [1, 2])
    prediction = model.predict([3, 4])
    np.testing.assert_array_almost_equal(prediction, [3.0, 4.0])

def test_predict_single_2d_feature_input_list():
    model = lr.LinearRegression()
    model.fit([[1, 1], [2, 2]], [2, 4])
    prediction = model.predict([3, 3])
    np.testing.assert_array_almost_equal(prediction, [6.0])

def test_predict_extrapolation_far_out():
    model = lr.LinearRegression()
    model.fit([0, 1], [0, 1])
    prediction = model.predict([100])
    np.testing.assert_array_almost_equal(prediction, [100.0])

def test_predict_interpolation():
    model = lr.LinearRegression()
    model.fit([0, 10], [0, 10])
    prediction = model.predict([5])
    np.testing.assert_array_almost_equal(prediction, [5.0])

def test_predict_zero_input():
    model = lr.LinearRegression()
    model.fit([[1], [2]], [3, 5]) # y = 2x + 1
    prediction = model.predict([[0]])
    np.testing.assert_array_almost_equal(prediction, [1.0])

def test_predict_negative_input():
    model = lr.LinearRegression()
    model.fit([[1], [2]], [3, 5]) # y = 2x + 1
    prediction = model.predict([[-1]])
    np.testing.assert_array_almost_equal(prediction, [-1.0])

def test_model_with_intercept_at_zero():
    model = lr.LinearRegression()
    x = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6]) # y = 2x
    model.fit(x, y)
    np.testing.assert_almost_equal(model.bias, 0.0, decimal=5)
    np.testing.assert_almost_equal(model.weights[0], 2.0, decimal=5)

def test_model_with_negative_slope():
    model = lr.LinearRegression()
    x = np.array([[1], [2], [3]])
    y = np.array([3, 1, -1]) # y = -2x + 5
    model.fit(x, y)
    np.testing.assert_almost_equal(model.weights[0], -2.0, decimal=5)
    np.testing.assert_almost_equal(model.bias, 5.0, decimal=5)

def test_predict_with_list_input_for_2d_model():
    model = lr.LinearRegression()
    model.fit([[1, 1], [2, 2]], [2, 4])
    prediction = model.predict([[3, 3], [4, 4]])
    np.testing.assert_array_almost_equal(prediction, [6.0, 8.0])

def test_predict_with_single_sample_2d_numpy():
    model = lr.LinearRegression()
    model.fit([[1, 1], [2, 2]], [2, 4])
    prediction = model.predict(np.array([[3, 3]]))
    np.testing.assert_array_almost_equal(prediction, [6.0])

def test_predict_with_single_sample_1d_numpy():
    model = lr.LinearRegression()
    model.fit([1, 2], [1, 2])
    prediction = model.predict(np.array([3]))
    np.testing.assert_array_almost_equal(prediction, [3.0])

def test_fit_with_small_values():
    model = lr.LinearRegression()
    x = np.array([[1e-6], [2e-6]])
    y = np.array([1e-6, 2e-6])
    model.fit(x, y)
    np.testing.assert_almost_equal(model.weights[0], 1.0, decimal=5)
    np.testing.assert_almost_equal(model.bias, 0.0, decimal=5)

def test_predict_returns_float_array():
    model = lr.LinearRegression()
    model.fit([1], [1])
    prediction = model.predict([1])
    assert prediction.dtype == np.float64

def test_fit_single_sample():
    model = lr.LinearRegression()
    model.fit([[1, 2]], [3])
    assert model.weights is not None
    assert model.bias is not None

def test_predict_on_collinear_model():
    model = lr.LinearRegression()
    x = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([2, 4, 6])
    model.fit(x, y)
    prediction = model.predict([[4, 4]])
    np.testing.assert_almost_equal(prediction[0], 8.0, decimal=5)

def test_predict_on_zero_variance_feature_model():
    model = lr.LinearRegression()
    x = np.array([[1, 5], [2, 5], [3, 5]])
    y = np.array([3, 5, 7])
    model.fit(x, y)
    prediction = model.predict([[4, 100]])
    np.testing.assert_almost_equal(prediction[0], 9.0, decimal=5)