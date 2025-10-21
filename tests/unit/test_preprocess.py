import numpy as np
import pytest
import rice2025.preprocess as prep

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
    np.testing.assert_equal(scaled, exp)

def test_scale_negative():
    scaled = prep.scale(negative)
    exp = np.array([0.0, 0.5, 1.0])
    np.testing.assert_equal(scaled, exp)

def test_scale_rep():
    scaled = prep.scale(rep)
    exp = np.array([0.0, 0.0, 0.0])
    np.testing.assert_equal(scaled, exp)

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

"""
Test train_test_split function.
"""
def test_train_test_split_empty():
    train, test = prep.train_test_split(empty)
    assert train.size == 0 and test.size == 0

def test_train_test_split_1d():
    data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    train, test = prep.train_test_split(data)
    assert train.size == 8 and test.size == 2
    split_helper(test, train)

def test_train_test_split_1d2():
    data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    train, test = prep.train_test_split(data)
    assert train.size == 9 and test.size == 3
    split_helper(test, train)

def test_train_test_split_2d():
    data = np.array([[0, 1, 2, 3],
                     [4, 5, 6, 7], 
                     [8, 9, 10, 11]])
    train, test = prep.train_test_split(data)
    assert train.size == 8 and test.size == 4
    split_helper(test, train)
    split_2d_helper(data, train, test)

def test_train_test_split_2d2():
    data = np.array([[0, 1], [2, 3],
                    [4, 5], [6, 7], 
                    [8, 9], [10, 11],
                    [12, 13], [14, 15]])
    train, test = prep.train_test_split(data)
    assert train.size == 12 and test.size == 4
    split_helper(test, train)
    split_2d_helper(data, train, test)

def split_helper(test, train):
    assert len(np.unique(train)) == train.size
    assert len(np.unique(test)) == test.size
    assert len(np.intersect1d(train, test)) == 0

def split_2d_helper(original, train, test):
    assert train.shape[1] == original.shape[1]
    assert test.shape[1] == original.shape[1]
    assert train.shape[0] + test.shape[0] == original.shape[0]