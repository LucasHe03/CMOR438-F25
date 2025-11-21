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