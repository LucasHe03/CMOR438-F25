import numpy as np
import pytest
import rice2025.supervised_learning.multilayer_perceptron as mlp

"""
Test __init__ function
"""
def test_init_default():
    model = mlp.MultilayerPerceptron()
    assert model.n_hidden == 100
    assert model.lr == 0.01
    assert model.n_iter == 1000
    assert model.w1 is None and model.b1 is None
    assert model.w2 is None and model.b2 is None

def test_init_custom():
    model = mlp.MultilayerPerceptron(n_hidden = 50, lr = 0.1, n_iter = 500)
    assert model.n_hidden == 50
    assert model.lr == 0.1
    assert model.n_iter == 500

"""
Test fit function
"""
def test_fit_success():
    model = mlp.MultilayerPerceptron()
    x = [[0, 0], [0, 1]]
    y = [0, 1]
    model.fit(x, y)
    assert model.w1 is not None and model.b1 is not None
    assert model.w2 is not None and model.b2 is not None

def test_fit_mismatched_lengths():
    model = mlp.MultilayerPerceptron()
    x = [[0, 0], [0, 1]]
    y = [0]
    with pytest.raises(ValueError, match = "x and y must have equal lengths"):
        model.fit(x, y)

def test_fit_empty():
    model = mlp.MultilayerPerceptron()
    x = []
    y = []
    with pytest.raises(ValueError, match = "x and y must be non-empty"):
        model.fit(x, y)

"""
Test predict function
"""
def test_predict_not_fitted():
    model = mlp.MultilayerPerceptron()
    with pytest.raises(ValueError, match = "model must be fit before predicting"):
        model.predict([[0, 0]])

def test_predict_single_point():
    model = mlp.MultilayerPerceptron(n_hidden = 10, lr = 0.1, n_iter = 5000)
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [0, 1, 1, 0]
    model.fit(x_train, y_train)
    
    prediction = model.predict([[1, 1]])
    np.testing.assert_array_equal(prediction, np.array([0]))

def test_predict_mismatched_features():
    model = mlp.MultilayerPerceptron()
    x_train = [[1, 1], [2, 2]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    
    x_test_mismatched = [[1, 1, 1]]
    with pytest.raises(ValueError):
        model.predict(x_test_mismatched)

def test_predict_1d_input():
    model = mlp.MultilayerPerceptron()
    x_train = [-3, -2, -1, 1, 2, 3]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)

    x_test = [-2.5, 2.5]
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_predict_empty_input():
    model = mlp.MultilayerPerceptron()
    x_train = [[0, 0], [1, 1]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    predictions = model.predict([])
    assert predictions.shape == (0,)

def test_fit_numpy_input():
    model = mlp.MultilayerPerceptron(n_iter=10)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    model.fit(x, y)
    assert model.w1 is not None
    assert model.w2 is not None

def test_predict_numpy_input():
    model = mlp.MultilayerPerceptron(n_iter=100)
    x_train = np.array([[-2], [2]])
    y_train = np.array([0, 1])
    model.fit(x_train, y_train)
    x_test = np.array([[-3], [3]])
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_refit_model():
    model = mlp.MultilayerPerceptron(n_iter=1000, lr=0.1)
    x1, y1 = [[-2], [-1]], [0, 0]
    model.fit(x1, y1)
    pred1 = model.predict([-1.5])
    np.testing.assert_array_equal(pred1, np.array([0]))

    x2, y2 = [[10], [11]], [1, 1]
    model.fit(x2, y2)
    pred2 = model.predict([10.5])
    np.testing.assert_array_equal(pred2, np.array([1]))

def test_predict_on_training_data_separable():
    model = mlp.MultilayerPerceptron(n_iter=2000, lr=0.1)
    x_train = [[-2], [-1], [1], [2]]
    y_train = [0, 0, 1, 1]
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)
    np.testing.assert_array_equal(predictions, y_train)

def test_predict_returns_numpy_array():
    model = mlp.MultilayerPerceptron(n_iter=10)
    model.fit([1], [0])
    prediction = model.predict([1])
    assert isinstance(prediction, np.ndarray)

def test_predict_output_dtype_is_int():
    model = mlp.MultilayerPerceptron(n_iter=10)
    model.fit([1], [0])
    prediction = model.predict([1])
    assert np.issubdtype(prediction.dtype, np.integer)

def test_fit_with_all_zeros_label():
    model = mlp.MultilayerPerceptron(n_iter=100)
    x = [[1], [2], [3]]
    y = [0, 0, 0]
    model.fit(x, y)
    prediction = model.predict([[1.5], [2.5]])
    np.testing.assert_array_equal(prediction, np.array([0, 0]))

def test_fit_with_all_ones_label():
    model = mlp.MultilayerPerceptron(n_iter=100)
    x = [[1], [2], [3]]
    y = [1, 1, 1]
    model.fit(x, y)
    prediction = model.predict([[1.5], [2.5]])
    np.testing.assert_array_equal(prediction, np.array([1, 1]))

def test_predict_with_single_training_sample():
    model = mlp.MultilayerPerceptron(n_iter=10)
    model.fit([[1]], [0])
    prediction = model.predict([[10]])
    np.testing.assert_array_equal(prediction, np.array([0]))

def test_predict_with_few_iterations():
    model = mlp.MultilayerPerceptron(n_iter=2)
    model.fit([[1], [10]], [0, 1])
    model.predict([5])
    assert True

def test_fit_returns_losses():
    model = mlp.MultilayerPerceptron(n_iter=10)
    losses = model.fit([[0]], [0])
    assert isinstance(losses, list)
    assert len(losses) == 10
    assert all(isinstance(loss, float) for loss in losses)

def test_weight_shapes():
    model = mlp.MultilayerPerceptron(n_hidden=50)
    x = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    model.fit(x, y)
    assert model.w1.shape == (5, 50)
    assert model.b1.shape == (1, 50)
    assert model.w2.shape == (50, 1)
    assert model.b2.shape == (1, 1)

def test_predict_linearly_separable_2d():
    model = mlp.MultilayerPerceptron(lr=0.1, n_iter=1000)
    x_train = [[-2, -1], [-1, -2], [1, 2], [2, 1]]
    y_train = [0, 0, 1, 1]
    model.fit(x_train, y_train)
    x_test = [[-1.5, -1.5], [1.5, 1.5]]
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_predict_single_1d_feature_as_scalar():
    model = mlp.MultilayerPerceptron(n_iter=100)
    model.fit([1, 10], [0, 1])
    prediction = model.predict(11)
    np.testing.assert_array_equal(prediction, np.array([1]))

def test_predict_single_2d_feature_as_list():
    model = mlp.MultilayerPerceptron(n_iter=100)
    model.fit([[1, 1], [10, 10]], [0, 1])
    prediction = model.predict([11, 11])
    np.testing.assert_array_equal(prediction, np.array([1]))