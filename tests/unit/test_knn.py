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