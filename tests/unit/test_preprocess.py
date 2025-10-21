import numpy as np
import pytest
import rice2025.preprocess as prep

"""
Create NumPy arrays to be used in all tests
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