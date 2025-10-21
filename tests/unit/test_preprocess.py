import numpy as np
import pytest
import rice2025.preprocess as prep

"""
Testing normalize function.
"""
def test_normalize_empty():
    data = np.array([])
    normalized = prep.normalize(data)
    exp = np.array([])
    np.testing.assert_equal(normalized, exp)

def test_normalize_1d():
    data = np.array([0, 5, 10])
    normalized = prep.normalize(data)
    exp = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(normalized, exp, 1e-8)

def test_normalize_2d():
    data = np.array([[0, 2, 3],
                     [5, 4, 3],
                     [10, 10, 9]])
    normalized = prep.normalize(data)
    exp = np.array([[0.0, 0.0, 0.0], 
                    [0.5, 0.25, 0.0], 
                    [1.0, 1.0, 1.0]])
    np.testing.assert_equal(normalized, exp)

def test_normalize_negative():
    data = np.array([-5, 0, 5])
    normalized = prep.normalize(data)
    exp = np.array([0.0, 0.5, 1.0])
    np.testing.assert_equal(normalized, exp)