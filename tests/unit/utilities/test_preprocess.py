import numpy as np
import pytest
import rice2025.utilities.preprocess as prep

"""
Create NumPy arrays to be used across tests
"""
empty = np.array([])
one_d = np.array([0, 5, 10])
two_d = np.array([[0, 2, 3],
                     [5, 4, 3],
                     [10, 10, 9]])
negative = np.array([-5, 0, 5])
rep = np.array([5, 5, 5])
float_1d = np.array([0.5, 1.0, 2.5, 5.0])

"""
Testing scale function.
"""
def test_scale_empty():
    scaled = prep.scale(empty)
    exp = np.array([])
    np.testing.assert_equal(scaled, exp)

def test_scale_1d():
    scaled = prep.scale(one_d)
    exp = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(scaled, exp, 1e-8)

def test_scale_2d():
    scaled = prep.scale(two_d)
    exp = np.array([[0.0, 0.0, 0.0], 
                    [0.5, 0.25, 0.0], 
                    [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(scaled, exp, 1e-8)

def test_scale_negative():
    scaled = prep.scale(negative)
    exp = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(scaled, exp, 1e-8)

def test_scale_rep():
    scaled = prep.scale(rep)
    exp = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(scaled, exp, 1e-8)

def test_scale_float():
    scaled = prep.scale(float_1d)
    exp = np.array([0., 1/9, 4/9, 1.])
    np.testing.assert_allclose(scaled, exp, 1e-8)

def test_scale_single_element():
    scaled = prep.scale(np.array([42]))
    np.testing.assert_allclose(scaled, np.array([0.0]), 1e-8)

def test_scale_2d_rep_col():
    data = np.array([[1, 5, 8], [2, 5, 9]])
    scaled = prep.scale(data)
    exp = np.array([[0., 0., 0.], [1., 0., 1.]])
    np.testing.assert_allclose(scaled, exp, 1e-8)

"""
Test normalize function.
"""
def test_normalize_empty():
    normalized = prep.normalize(empty)
    exp = np.array([])
    np.testing.assert_equal(normalized, exp)

def test_normalize_1d():
    normalized = prep.normalize(one_d)
    exp = np.array([-1.22474487, 0.0, 1.22474487])
    np.testing.assert_allclose(normalized, exp, 1e-8)

def test_normalize_2d():
    normalized = prep.normalize(two_d)
    exp = np.array([[-1.22474487, -0.98058068, -0.70710678],
                    [ 0.        , -0.39223227, -0.70710678],
                    [ 1.22474487,  1.37281295,  1.41421356]])
    np.testing.assert_allclose(normalized, exp, 1e-8)

def test_normalize_negative():
    normalized = prep.normalize(negative)
    exp = np.array([-1.22474487, 0.0, 1.22474487])
    np.testing.assert_allclose(normalized, exp, 1e-8)

def test_normalize_rep():
    normalized = prep.normalize(rep)
    exp = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(normalized, exp, 1e-8)

def test_normalize_float():
    normalized = prep.normalize(float_1d)
    mean = np.mean(float_1d)
    std = np.std(float_1d)
    exp = (float_1d - mean) / std
    np.testing.assert_allclose(normalized, exp, 1e-8)

def test_normalize_single_element():
    normalized = prep.normalize(np.array([42]))
    np.testing.assert_allclose(normalized, np.array([0.0]), 1e-8)

def test_normalize_2d_rep_col():
    data = np.array([[1, 5, 8], [2, 5, 9]])
    normalized = prep.normalize(data)
    np.testing.assert_allclose(normalized[:, 1], np.array([0., 0.]), 1e-8)
"""
Test train_test_split function.
"""
empty_X = np.array([]).reshape(0, 1)
empty_y = np.array([])

def split_helper(test, train):
    assert len(np.unique(train)) == train.size
    assert len(np.unique(test)) == test.size
    assert len(np.intersect1d(train, test)) == 0

def split_2d_helper(original, train, test):
    assert train.shape[1] == original.shape[1]
    assert test.shape[1] == original.shape[1]
    assert train.shape[0] + test.shape[0] == original.shape[0]

# ---------- TESTS ----------

def test_train_test_split_empty():
    X_train, X_test, y_train, y_test = prep.train_test_split(empty_X, empty_y)
    assert X_train.size == 0 and X_test.size == 0
    assert y_train.size == 0 and y_test.size == 0

def test_train_test_split_1d():
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = prep.train_test_split(X, y, test_size=0.2)
    assert X_train.shape[0] == 8 and X_test.shape[0] == 2
    assert y_train.shape[0] == 8 and y_test.shape[0] == 2
    split_helper(y_test, y_train)

def test_train_test_split_2d():
    X = np.array([[0, 1, 2, 3],
                  [4, 5, 6, 7], 
                  [8, 9, 10, 11]])
    y = np.array([0, 1, 2])
    X_train, X_test, y_train, y_test = prep.train_test_split(X, y, test_size=0.33)
    split_2d_helper(X, X_train, X_test)
    split_helper(y_test, y_train)

def test_train_test_split_custom_size():
    X = np.arange(20).reshape(-1, 1)
    y = np.arange(20)
    X_train, X_test, y_train, y_test = prep.train_test_split(X, y, test_size=0.5)
    assert X_train.shape[0] == 10
    assert X_test.shape[0] == 10
    assert y_train.shape[0] == 10
    assert y_test.shape[0] == 10
    split_helper(y_test, y_train)

def test_train_test_split_size_one():
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = prep.train_test_split(X, y, test_size=0.1)
    assert X_train.shape[0] == 9
    assert X_test.shape[0] == 1
    assert y_train.shape[0] == 9
    assert y_test.shape[0] == 1
    split_helper(y_test, y_train)

def test_train_test_split_invalid_size():
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10)
    with pytest.raises(ValueError):
        prep.train_test_split(X, y, test_size=1.5)