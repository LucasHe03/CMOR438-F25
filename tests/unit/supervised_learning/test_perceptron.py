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

def test_predict_empty_input():
    model = p.Perceptron()
    model.fit([[1]], [1])
    predictions = model.predict([])
    assert predictions.shape == (0,)

def test_fit_numpy_input():
    model = p.Perceptron()
    x = np.array([[1, 2], [3, 4]])
    y = np.array([1, -1])
    model.fit(x, y)
    assert model.weights is not None
    assert model.bias is not None

def test_predict_numpy_input():
    model = p.Perceptron(lr=0.1, n_iter=10)
    x_train = np.array([[1], [10]])
    y_train = np.array([1, -1])
    model.fit(x_train, y_train)
    x_test = np.array([0, 11])
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([1, -1]))

def test_refit_model():
    model = p.Perceptron(lr=0.1, n_iter=100)
    x1, y1 = [[1], [2]], [1, 1]
    model.fit(x1, y1)
    pred1 = model.predict([1.5])
    np.testing.assert_array_equal(pred1, np.array([1]))

    x2, y2 = [[-10], [-11]], [-1, -1]
    model.fit(x2, y2)
    pred2 = model.predict([-10.5])
    np.testing.assert_array_equal(pred2, np.array([-1]))

def test_predict_on_training_data():
    model = p.Perceptron(lr=0.1, n_iter=100)
    x_train = [[-2], [-1], [1], [2]]
    y_train = [-1, -1, 1, 1]
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)
    np.testing.assert_array_equal(predictions, y_train)

def test_predict_returns_numpy_array():
    model = p.Perceptron()
    model.fit([1], [1])
    prediction = model.predict([1])
    assert isinstance(prediction, np.ndarray)

def test_predict_output_dtype_is_int():
    model = p.Perceptron()
    model.fit([1], [1])
    prediction = model.predict([1])
    assert np.issubdtype(prediction.dtype, np.integer)

def test_fit_with_all_ones_label():
    model = p.Perceptron(lr=0.1, n_iter=100)
    x = [[1], [2], [3]]
    y = [1, 1, 1]
    model.fit(x, y)
    prediction = model.predict([[1.5], [2.5]])
    np.testing.assert_array_equal(prediction, np.array([1, 1]))

def test_fit_with_all_minus_ones_label():
    model = p.Perceptron(lr=0.1, n_iter=100)
    x = [[1], [2], [3]]
    y = [-1, -1, -1]
    model.fit(x, y)
    prediction = model.predict([[1.5], [2.5]])
    np.testing.assert_array_equal(prediction, np.array([-1, -1]))

def test_predict_with_single_training_sample():
    model = p.Perceptron()
    model.fit([[1]], [1])
    prediction = model.predict([[10]])
    np.testing.assert_array_equal(prediction, np.array([1]))

def test_predict_with_few_iterations():
    model = p.Perceptron(n_iter=1)
    model.fit([[1], [10]], [1, -1])
    model.predict([5])
    assert True  # Test passes if it runs without error

def test_predict_with_zero_learning_rate():
    model = p.Perceptron(lr=0)
    x_train = [[-1], [1]]
    y_train = [-1, 1]
    model.fit(x_train, y_train)
    assert model.weights[0] == 0
    assert model.bias == 0
    # With zero weights and bias, prediction is always 1
    predictions = model.predict([[-2], [2]])
    np.testing.assert_array_equal(predictions, np.array([1, 1]))

def test_predict_far_point():
    model = p.Perceptron(lr=0.1, n_iter=100)
    model.fit([1, 2], [1, 1])
    prediction = model.predict([100])
    np.testing.assert_array_equal(prediction, np.array([1]))

def test_predict_with_large_feature_values():
    model = p.Perceptron(lr=0.001, n_iter=100)
    x_train = [[1000], [2000], [-1000], [-2000]]
    y_train = [1, 1, -1, -1]
    model.fit(x_train, y_train)
    predictions = model.predict([[1500], [-1500]])
    np.testing.assert_array_equal(predictions, np.array([1, -1]))

def test_predict_with_small_feature_values():
    model = p.Perceptron(lr=0.1, n_iter=100)
    x_train = [[0.01], [0.02], [-0.01], [-0.02]]
    y_train = [1, 1, -1, -1]
    model.fit(x_train, y_train)
    predictions = model.predict([[0.015], [-0.015]])
    np.testing.assert_array_equal(predictions, np.array([1, -1]))

def test_predict_non_linearly_separable():
    # Perceptron won't converge, but should not error.
    model = p.Perceptron(lr=0.1, n_iter=100)
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [-1, 1, 1, -1]
    model.fit(x_train, y_train)
    model.predict(x_train)
    assert True # Test passes if it runs without error

def test_fit_returns_self():
    model = p.Perceptron()
    x = [[1]]
    y = [1]
    fitted_model = model.fit(x, y)
    assert fitted_model is model

def test_predict_with_bias_shift():
    model = p.Perceptron(lr=0.1, n_iter=100)
    x_train = [[1], [2], [3], [4]]
    y_train = [-1, -1, 1, 1] # Decision boundary is between 2 and 3
    model.fit(x_train, y_train)
    predictions = model.predict([[1.5], [3.5]])
    np.testing.assert_array_equal(predictions, np.array([-1, 1]))

def test_predict_single_feature_as_scalar():
    model = p.Perceptron(n_iter=100)
    model.fit([1, 10], [1, -1])
    prediction = model.predict(0)
    np.testing.assert_array_equal(prediction, np.array([1]))