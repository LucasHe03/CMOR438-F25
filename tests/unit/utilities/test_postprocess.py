import numpy as np
import pytest
import rice2025.utilities.postprocess as postp

"""
Create NumPy arrays to be used across tests
"""
empty = np.array([])
class1 = np.array(["cat", "dog", "mouse", "cat", "dog", "dog"])
class2 = np.array(["cat"])
class3 = np.array(["cat", "dog"])
reg1 = np.array([1, 2, 3, 4, 5])
reg2 = np.array([1, 1, 1, 1])
reg3 = np.array([[1, 2, 3], [4, 5, 6]])
reg_neg = np.array([-1, -2, -3, -4, -5])
reg_float = np.array([1.5, 2.5, 3.5])

"""
Test majority_label function
"""
def test_majority_empty():
    label = postp.majority_label(empty)
    assert label == None

def test_majority_1():
    label = postp.majority_label(class1)
    assert label == "dog"

def test_majority_2():
    label = postp.majority_label(class2)
    assert label == "cat"

def test_majority_3():
    label = postp.majority_label(class3)
    assert label == "cat" or label == "dog"

def test_majority_list_input():
    label = postp.majority_label(["a", "b", "a"])
    assert label == "a"

def test_majority_integer_labels():
    label = postp.majority_label([1, 2, 2, 3, 2, 1])
    assert label == 2

def test_majority_tie_break_alpha():
    label = postp.majority_label(['c', 'b', 'c', 'b'])
    assert label == 'b'

def test_majority_tie_break_numeric():
    label = postp.majority_label([2, 1, 2, 1])
    assert label == 1

def test_majority_all_unique():
    label = postp.majority_label(['cat', 'dog', 'mouse'])
    assert label == 'cat'
"""
Test average_label function
"""
def test_average_empty():
    label = postp.average_label(empty)
    assert np.isnan(label)

def test_average_1():
    label = postp.average_label(reg1)
    np.testing.assert_equal(label, 3.0)

def test_average_2():
    label = postp.average_label(reg2)
    np.testing.assert_equal(label, 1.0)

def test_average_3():
    label = postp.average_label(reg3)
    np.testing.assert_equal(label, 21/6)

def test_average_list_input():
    label = postp.average_label([1, 2, 3])
    assert label == 2.0

def test_average_single_element():
    label = postp.average_label([42])
    assert label == 42.0

def test_average_negative_numbers():
    label = postp.average_label(reg_neg)
    assert label == -3.0

def test_average_mixed_sign():
    label = postp.average_label([-1, 0, 1])
    assert label == 0.0

def test_average_float():
    label = postp.average_label(reg_float)
    assert label == 2.5

def test_average_float_precision():
    label = postp.average_label([0.1, 0.2])
    np.testing.assert_allclose(label, 0.15, 1e-8)