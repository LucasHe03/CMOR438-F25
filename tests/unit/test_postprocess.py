import numpy as np
import pytest
import rice2025.postprocess as postp

"""
Create NumPy arrays to be used across tests
"""
empty = np.array([])
class1 = np.array(["cat", "dog", "mouse", "cat" "dog", "dog"])
class2 = np.array(["cat"])
class3 = np.array(["cat", "dog"])
reg1 = np.array([1, 2, 3, 4, 5])
reg2 = np.array([1, 1, 1, 1])
reg3 = np.array([[1, 2, 3], [4, 5, 6]])

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