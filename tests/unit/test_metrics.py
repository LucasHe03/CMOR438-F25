import numpy as np
import pytest
import rice2025.metrics as met

"""
Create NumPy arrays to be used across tests
"""
empty = np.array([])
one_d_a = np.array([1, 2, 3])
one_d_b = np.array([4, 5, 6])
two_d_a = np.array([[1, 2], [3, 4]])
two_d_b = np.array([[5, 6], [7, 8]])
neg_a = np.array([-1, -2])
neg_b = np.array([2, 2])

"""
Test euclidean_distance function
"""
def test_euclidean_empty():
    dist = met.euclidean_distance(empty, empty)
    assert dist == 0.0

def test_euclidean_identical():
    dist = met.euclidean_distance(one_d_a, one_d_a)
    assert dist == 0.0

def test_euclidean_1d():
    dist = met.euclidean_distance(one_d_a, one_d_b)
    np.testing.assert_allclose(dist, 5.19615242, 1e-8)

def test_euclidean_2d():
    dist = met.euclidean_distance(two_d_a, two_d_b)
    assert dist == 8.0

"""
Test manhattan_distance function
"""
def test_manhattan_empty():
    dist = met.manhattan_distance(empty, empty)
    assert dist == 0.0

def test_manhattan_identical():
    dist = met.manhattan_distance(one_d_a, one_d_a)
    assert dist == 0.0

def test_manhattan_1d():
    dist = met.manhattan_distance(one_d_a, one_d_b)
    assert dist == 9.0

def test_manhattan_2d():
    dist = met.manhattan_distance(two_d_a, two_d_b)
    assert dist == 16.0