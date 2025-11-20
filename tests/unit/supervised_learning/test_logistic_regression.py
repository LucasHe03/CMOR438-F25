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

def test_predict_empty_input():
    model = logr.LogisticRegression()
    x_train = [[1], [2]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    predictions = model.predict([])
    assert predictions.shape == (0,)

def test_fit_numpy_input():
    model = logr.LogisticRegression()
    x = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model.fit(x, y)
    assert model.weights is not None
    assert model.bias is not None

def test_predict_numpy_input():
    model = logr.LogisticRegression()
    x_train = np.array([[1], [10]])
    y_train = np.array([0, 1])
    model.fit(x_train, y_train)
    x_test = np.array([0, 11])
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_refit_model():
    model = logr.LogisticRegression(lr=0.1, n_iter=100)
    x1, y1 = [[1], [2]], [0, 0]
    model.fit(x1, y1)
    pred1 = model.predict([1.5])
    np.testing.assert_array_equal(pred1, np.array([0]))

    x2, y2 = [[10], [11]], [1, 1]
    model.fit(x2, y2)
    pred2 = model.predict([10.5])
    np.testing.assert_array_equal(pred2, np.array([1]))

def test_predict_on_training_data():
    model = logr.LogisticRegression(lr=0.1, n_iter=1000)
    x_train = [[-2], [-1], [1], [2]]
    y_train = [0, 0, 1, 1]
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)
    np.testing.assert_array_equal(predictions, y_train)

def test_predict_2d_separable():
    model = logr.LogisticRegression(lr=0.1, n_iter=1000)
    x_train = [[-2, -1], [-1, -2], [1, 2], [2, 1]]
    y_train = [0, 0, 1, 1]
    model.fit(x_train, y_train)
    x_test = [[-1.5, -1.5], [1.5, 1.5]]
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_predict_single_1d_feature_as_list():
    model = logr.LogisticRegression(lr=0.1, n_iter=100)
    model.fit([1, 10], [0, 1])
    prediction = model.predict([11])
    np.testing.assert_array_equal(prediction, np.array([1]))

def test_predict_single_2d_feature_as_list():
    model = logr.LogisticRegression(lr=0.1, n_iter=100)
    model.fit([[1, 1], [10, 10]], [0, 1])
    prediction = model.predict([11, 11])
    np.testing.assert_array_equal(prediction, np.array([1]))

def test_predict_returns_numpy_array():
    model = logr.LogisticRegression()
    model.fit([1], [0])
    prediction = model.predict([1])
    assert isinstance(prediction, np.ndarray)

def test_predict_output_dtype_is_int():
    model = logr.LogisticRegression()
    model.fit([1], [0])
    prediction = model.predict([1])
    assert np.issubdtype(prediction.dtype, np.integer)

def test_fit_with_all_zeros_label():
    model = logr.LogisticRegression(lr=0.1, n_iter=100)
    x = [[1], [2], [3]]
    y = [0, 0, 0]
    model.fit(x, y)
    prediction = model.predict([[1.5], [2.5]])
    np.testing.assert_array_equal(prediction, np.array([0, 0]))

def test_fit_with_all_ones_label():
    model = logr.LogisticRegression(lr=0.1, n_iter=100)
    x = [[1], [2], [3]]
    y = [1, 1, 1]
    model.fit(x, y)
    prediction = model.predict([[1.5], [2.5]])
    np.testing.assert_array_equal(prediction, np.array([1, 1]))

def test_predict_far_point():
    model = logr.LogisticRegression(lr=0.1, n_iter=1000)
    model.fit([1, 2], [0, 1])
    prediction = model.predict([100])
    np.testing.assert_array_equal(prediction, np.array([1]))

def test_predict_point_between_classes():
    model = logr.LogisticRegression(lr=0.1, n_iter=1000)
    model.fit([-1, 1], [0, 1])
    prediction = model.predict([0])
    assert prediction[0] in [0, 1]

def test_predict_with_single_training_sample():
    model = logr.LogisticRegression()
    model.fit([[1]], [0])
    prediction = model.predict([[10]])
    np.testing.assert_array_equal(prediction, np.array([0]))

def test_predict_with_large_feature_values():
    model = logr.LogisticRegression(lr=0.001, n_iter=1000)
    x_train = [[1000], [2000], [-1000], [-2000]]
    y_train = [1, 1, 0, 0]
    model.fit(x_train, y_train)
    predictions = model.predict([[1500], [-1500]])
    np.testing.assert_array_equal(predictions, np.array([1, 0]))

def test_predict_with_small_feature_values():
    model = logr.LogisticRegression(lr=0.1, n_iter=1000)
    x_train = [[0.01], [0.02], [-0.01], [-0.02]]
    y_train = [1, 1, 0, 0]
    model.fit(x_train, y_train)
    predictions = model.predict([[0.015], [-0.015]])
    np.testing.assert_array_equal(predictions, np.array([1, 0]))

def test_predict_with_few_iterations():
    model = logr.LogisticRegression(n_iter=2)
    model.fit([[1], [10]], [0, 1])
    model.predict([5])
    assert True

def test_predict_non_linearly_separable():
    # XOR problem
    model = logr.LogisticRegression(lr=0.1, n_iter=1000)
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [0, 1, 1, 0]
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)
    accuracy = np.mean(predictions == y_train)
    assert accuracy in [0.5, 0.75]
