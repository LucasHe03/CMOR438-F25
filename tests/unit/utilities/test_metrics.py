import numpy as np
import pytest
import rice2025.utilities.metrics as met

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
float_a = np.array([0.5, 1.5])
float_b = np.array([2.5, 3.0])

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

def test_euclidean_neg():
    dist = met.euclidean_distance(neg_a, neg_b)
    assert dist == 5.0

def test_euclidean_float():
    dist = met.euclidean_distance(float_a, float_b)
    np.testing.assert_allclose(dist, 2.5, 1e-8)

def test_euclidean_list_input():
    dist = met.euclidean_distance([1, 2, 3], [4, 5, 6])
    np.testing.assert_allclose(dist, 5.19615242, 1e-8)

def test_euclidean_commutative():
    dist1 = met.euclidean_distance(one_d_a, one_d_b)
    dist2 = met.euclidean_distance(one_d_b, one_d_a)
    assert dist1 == dist2

def test_euclidean_with_zero():
    dist = met.euclidean_distance(one_d_a, np.zeros_like(one_d_a))
    np.testing.assert_allclose(dist, np.sqrt(14), 1e-8)

def test_euclidean_mismatched_shape():
    with pytest.raises(ValueError):
        met.euclidean_distance(one_d_a, two_d_a)

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

def test_manhattan_neg():
    dist = met.manhattan_distance(neg_a, neg_b)
    assert dist == 7.0

def test_manhattan_float():
    dist = met.manhattan_distance(float_a, float_b)
    assert dist == 3.5

def test_manhattan_list_input():
    dist = met.manhattan_distance([1, 2, 3], [4, 5, 6])
    assert dist == 9.0

def test_manhattan_commutative():
    dist1 = met.manhattan_distance(one_d_a, one_d_b)
    dist2 = met.manhattan_distance(one_d_b, one_d_a)
    assert dist1 == dist2

def test_manhattan_with_zero():
    dist = met.manhattan_distance(one_d_a, np.zeros_like(one_d_a))
    assert dist == 6.0

def test_manhattan_mismatched_shape():
    with pytest.raises(ValueError):
        met.manhattan_distance(one_d_a, two_d_a)