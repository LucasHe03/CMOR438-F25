import numpy as np
import pytest
import rice2025.knn as knn

"""
Test __init__ function
"""
def test_init_default():
    model = knn.KNN()
    assert model.k == 3

def test_init_custom():
    model = knn.KNN(5)
    assert model.k == 5

def test_init_zero():
    with pytest.raises(ValueError):
        model = knn.KNN(0)

def test_init_negative():
    with pytest.raises(ValueError):
        model = knn.KNN(-1)

"""
Test fit function
"""
def test_fit_success():
    model = knn.KNN()
    x = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = [0, 1, 0, 1]
    model.fit(x, y)
    np.testing.assert_array_equal(model.x_train, np.array(x, dtype=np.float64))
    np.testing.assert_array_equal(model.y_train, np.array(y))

def test_fit_success_classify():
    model = knn.KNN()
    x = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = ["cat", "dog", "cat", "dog"]
    model.fit(x, y)
    np.testing.assert_array_equal(model.x_train, np.array(x, dtype=np.float64))
    np.testing.assert_array_equal(model.y_train, np.array(y))

def test_fit_mismatched_lengths():
    model = knn.KNN()
    x = [[1, 2], [3, 4]]
    y = [0]
    with pytest.raises(ValueError):
        model.fit(x, y)

def test_fit_empty():
    model = knn.KNN()
    x = []
    y = []
    with pytest.raises(ValueError):
        model.fit(x, y)

def test_fit_k_too_large():
    model = knn.KNN(5)
    x = [[1, 2], [3, 4]]
    y = [0, 1]
    with pytest.raises(ValueError):
        model.fit(x, y)
