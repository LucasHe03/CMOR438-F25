import numpy as np
import pytest
import rice2025.unsupervised_learning.dbscan as db

"""
Test __init__ function
"""
def test_init_default():
    model = db.DBSCAN()
    assert model.eps == 0.5
    assert model.min_samples == 5

def test_init_custom():
    model = db.DBSCAN(eps=1.0, min_samples=10)
    assert model.eps == 1.0
    assert model.min_samples == 10

def test_init_invalid_eps():
    with pytest.raises(ValueError, match="eps must be a positive number"):
        db.DBSCAN(eps=0)
    with pytest.raises(ValueError, match="eps must be a positive number"):
        db.DBSCAN(eps=-1.0)

def test_init_invalid_min_samples():
    with pytest.raises(ValueError, match="min_samples must be a positive integer"):
        db.DBSCAN(min_samples=0)

"""
Test fit and fit_predict functions
"""
def test_fit_predict_simple_clusters():
    x = np.array([
        [1, 2], [1.1, 2.2], [0.9, 1.9], [1, 2.1],
        [8, 8], [8.1, 8.2], [7.9, 7.9], [8, 8.1],
        [20, 20]
    ])
    model = db.DBSCAN(eps=0.5, min_samples=3)
    labels = model.fit_predict(x)

    assert len(np.unique(labels[labels != -1])) == 2
    assert labels[8] == -1  # Noise point
    assert labels[0] == labels[1] == labels[2] == labels[3]
    assert labels[4] == labels[5] == labels[6] == labels[7]
    assert labels[0] != labels[4]

def test_fit_returns_self():
    model = db.DBSCAN()
    x = [[1, 1]]
    fitted_model = model.fit(x)
    assert fitted_model is model

def test_fit_empty_input():
    model = db.DBSCAN()
    labels = model.fit_predict([])
    assert labels.shape == (0,)

def test_all_noise():
    x = np.array([[1, 1], [5, 5], [10, 10]])
    model = db.DBSCAN(eps=1.0, min_samples=2)
    labels = model.fit_predict(x)
    np.testing.assert_array_equal(labels, np.array([-1, -1, -1]))

def test_one_big_cluster():
    x = np.array([[1, 1], [1.1, 1.1], [1.2, 1.2], [1.3, 1.3], [1.4, 1.4]])
    model = db.DBSCAN(eps=0.5, min_samples=3)
    labels = model.fit_predict(x)
    np.testing.assert_array_equal(labels, np.array([0, 0, 0, 0, 0]))

def test_border_point():
    x = np.array([[0,0], [0,1], [0,2], [0,3.5]])
    model = db.DBSCAN(eps=1.1, min_samples=3)
    labels = model.fit_predict(x)
    assert labels[0] == 0
    assert labels[1] == 0
    assert labels[2] == 0
    assert labels[3] == -1

def test_1d_data():
    x = np.array([1, 1.1, 1.2, 5, 5.2, 5.3, 10])
    model = db.DBSCAN(eps=0.5, min_samples=2)
    labels = model.fit_predict(x)
    assert labels[0] == 0
    assert labels[1] == 0
    assert labels[2] == 0
    assert labels[3] == 1
    assert labels[4] == 1
    assert labels[5] == 1
    assert labels[6] == -1

def test_large_eps_one_cluster():
    x = np.array([[1, 1], [10, 10], [100, 100]])
    model = db.DBSCAN(eps=200, min_samples=2)
    labels = model.fit_predict(x)
    np.testing.assert_array_equal(labels, np.array([0, 0, 0]))

def test_small_eps_all_noise():
    x = np.array([[1, 1], [1.1, 1.1], [1.2, 1.2]])
    model = db.DBSCAN(eps=0.01, min_samples=2)
    labels = model.fit_predict(x)
    np.testing.assert_array_equal(labels, np.array([-1, -1, -1]))

def test_min_samples_one():
    x = np.array([[1, 1], [5, 5]])
    model = db.DBSCAN(eps=1.0, min_samples=1)
    labels = model.fit_predict(x)
    assert labels[0] == 0
    assert labels[1] == 1

def test_duplicate_points():
    x = np.array([[1, 1], [1, 1], [1, 1], [5, 5]])
    model = db.DBSCAN(eps=0.5, min_samples=3)
    labels = model.fit_predict(x)
    assert labels[0] == 0
    assert labels[1] == 0
    assert labels[2] == 0
    assert labels[3] == -1

def test_3d_data():
    x = np.array([[1, 1, 1], [1.1, 1.1, 1.1], [5, 5, 5]])
    model = db.DBSCAN(eps=0.5, min_samples=2)
    labels = model.fit_predict(x)
    assert labels[0] == 0
    assert labels[1] == 0
    assert labels[2] == -1

def test_labels_are_numpy_array():
    model = db.DBSCAN()
    labels = model.fit_predict([[0,0]])
    assert isinstance(labels, np.ndarray)

def test_labels_are_integer_type():
    model = db.DBSCAN()
    labels = model.fit_predict([[0,0]])
    assert np.issubdtype(labels.dtype, np.integer)

def test_fit_predict_equals_fit_then_labels():
    model1 = db.DBSCAN(eps=1, min_samples=2)
    model2 = db.DBSCAN(eps=1, min_samples=2)
    x = [[0,0], [0.5, 0.5], [5,5]]
    labels1 = model1.fit_predict(x)
    model2.fit(x)
    labels2 = model2.labels_
    np.testing.assert_array_equal(labels1, labels2)

def test_refit_model():
    model = db.DBSCAN(eps=1, min_samples=2)
    labels1 = model.fit_predict([[0,0], [0.5,0.5]])
    labels2 = model.fit_predict([[100,100], [100.5, 100.5]])
    np.testing.assert_array_equal(labels1, np.array([0, 0]))
    np.testing.assert_array_equal(labels2, np.array([0, 0]))

def test_single_data_point():
    model = db.DBSCAN(min_samples=2)
    labels = model.fit_predict([[1,1]])
    np.testing.assert_array_equal(labels, np.array([-1]))

def test_just_enough_points_for_cluster():
    x = [[0,0], [0.1, 0.1], [0.2, 0.2]]
    model = db.DBSCAN(eps=0.5, min_samples=3)
    labels = model.fit_predict(x)
    np.testing.assert_array_equal(labels, np.array([0, 0, 0]))

def test_points_on_boundary():
    x = [[0,0], [1,0]]
    model = db.DBSCAN(eps=1.0, min_samples=2)
    labels = model.fit_predict(x)
    np.testing.assert_array_equal(labels, np.array([-1, -1]))

def test_points_just_inside_boundary():
    x = [[0,0], [0.99, 0]]
    model = db.DBSCAN(eps=1.0, min_samples=2)
    labels = model.fit_predict(x)
    np.testing.assert_array_equal(labels, np.array([0, 0]))

def test_noise_point_in_middle():
    x = [[0,0], [0.1,0.1], [5,5], [9.9,9.9], [10,10]]
    model = db.DBSCAN(eps=0.5, min_samples=2)
    labels = model.fit_predict(x)
    assert labels[0] == 0
    assert labels[1] == 0
    assert labels[2] == -1
    assert labels[3] == 1
    assert labels[4] == 1

def test_labels_start_from_zero():
    x = [[0,0], [10,10], [0.1,0.1], [10.1,10.1]]
    model = db.DBSCAN(eps=0.5, min_samples=2)
    labels = model.fit_predict(x)
    unique_labels = np.unique(labels[labels != -1])
    np.testing.assert_array_equal(unique_labels, np.array([0, 1]))

def test_no_core_points():
    x = [[0,0], [2,2], [4,4], [6,6]]
    model = db.DBSCAN(eps=1, min_samples=3)
    labels = model.fit_predict(x)
    np.testing.assert_array_equal(labels, np.array([-1, -1, -1, -1]))

def test_all_points_are_core():
    x = [[0,0], [0.1,0], [0.2,0], [0.3,0]]
    model = db.DBSCAN(eps=0.15, min_samples=3)
    labels = model.fit_predict(x)
    np.testing.assert_array_equal(labels, np.array([0, 0, 0, 0]))

def test_two_unconnected_clusters():
    x = [[0,0], [0.1,0], [10,0], [10.1,0]]
    model = db.DBSCAN(eps=0.5, min_samples=2)
    labels = model.fit_predict(x)
    assert labels[0] == 0
    assert labels[1] == 0
    assert labels[2] == 1
    assert labels[3] == 1