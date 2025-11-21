import numpy as np
import pytest
import rice2025.unsupervised_learning.k_means_clustering as kmc

"""
Test __init__ function
"""
def test_init_default():
    model = kmc.KMeansClustering()
    assert model.k == 3
    assert model.max_iter == 100
    assert model.centroids is None
    assert model.labels is None

def test_init_custom():
    model = kmc.KMeansClustering(k=5, max_iter=200)
    assert model.k == 5
    assert model.max_iter == 200

def test_init_invalid_k():
    with pytest.raises(ValueError, match="k must be a positive integer"):
        kmc.KMeansClustering(k=0)
    with pytest.raises(ValueError, match="k must be a positive integer"):
        kmc.KMeansClustering(k=-1)

def test_init_invalid_max_iter():
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        kmc.KMeansClustering(max_iter=0)

"""
Test fit function
"""
def test_fit_success():
    np.random.seed(42)
    model = kmc.KMeansClustering(k=2)
    x = np.array([[1, 1], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    model.fit(x)
    assert model.centroids is not None
    assert model.labels is not None
    assert model.centroids.shape == (2, 2)
    assert model.labels.shape == (6,)

def test_fit_k_too_large():
    model = kmc.KMeansClustering(k=5)
    x = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="Number of samples must be greater than or equal to k"):
        model.fit(x)

def test_fit_1d_data():
    np.random.seed(42)
    model = kmc.KMeansClustering(k=2)
    x = np.array([1, 2, 10, 12])
    model.fit(x)
    assert model.centroids.shape == (2, 1)
    assert model.labels.shape == (4,)

"""
Test predict function
"""
def test_predict_not_fitted():
    model = kmc.KMeansClustering()
    with pytest.raises(ValueError, match="Model must be fit before predicting"):
        model.predict([[1, 1]])

def test_predict_simple_clusters():
    np.random.seed(42)
    model = kmc.KMeansClustering(k=2)
    x_train = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
    model.fit(x_train)

    x_test = [[0.5, 0.5], [10.5, 10.5]]
    predictions = model.predict(x_test)

    pred_cluster_1 = model.predict([[0.5, 0.5]])
    pred_cluster_2 = model.predict([[10.5, 10.5]])
    assert pred_cluster_1 != pred_cluster_2

def test_predict_mismatched_features():
    model = kmc.KMeansClustering(k=2)
    model.fit([[1, 1], [10, 10]])
    with pytest.raises(ValueError):
        model.predict([1, 2, 3])

def test_predict_returns_correct_labels():
    np.random.seed(42)
    model = kmc.KMeansClustering(k=2, max_iter=10)
    x = np.array([[0], [1], [10], [11]])
    model.fit(x)
    predictions = model.predict([0.5, 10.5])
    assert len(np.unique(predictions)) == 2

    model.fit(x)
    predictions = model.predict([0.5, 10.5])
    assert len(np.unique(predictions)) == 2

def test_refit_model():
    np.random.seed(42)
    model = kmc.KMeansClustering(k=2)
    x1 = [[1, 1], [2, 2]]
    model.fit(x1)
    pred1 = model.predict([[1.5, 1.5]])

    x2 = [[100, 100], [101, 101]]
    model.fit(x2)
    pred2 = model.predict([[100.5, 100.5]])

    assert model.labels[0] != model.labels[1]

def test_predict_on_training_data():
    np.random.seed(42)
    model = kmc.KMeansClustering(k=2)
    x_train = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
    model.fit(x_train)
    predictions = model.predict(x_train)
    np.testing.assert_array_equal(predictions, model.labels)

def test_predict_returns_numpy_array():
    model = kmc.KMeansClustering(k=1)
    model.fit([[1, 1]])
    prediction = model.predict([[2, 2]])
    assert isinstance(prediction, np.ndarray)

def test_predict_output_dtype_is_int():
    model = kmc.KMeansClustering(k=1)
    model.fit([[1, 1]])
    prediction = model.predict([[2, 2]])
    assert np.issubdtype(prediction.dtype, np.integer)

def test_fit_returns_self():
    model = kmc.KMeansClustering(k=1)
    x = [[1, 1]]
    fitted_model = model.fit(x)
    assert fitted_model is model

def test_k_equals_one():
    np.random.seed(42)
    model = kmc.KMeansClustering(k=1)
    x = np.array([[1, 1], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    model.fit(x)
    predictions = model.predict([[0, 0], [100, 100]])
    np.testing.assert_array_equal(predictions, np.array([0, 0]))
    assert model.centroids.shape == (1, 2)

def test_k_equals_n_samples():
    np.random.seed(42)
    x = np.array([[1, 1], [5, 8], [9, 11]])
    model = kmc.KMeansClustering(k=len(x))
    model.fit(x)
    predictions = model.predict(x)
    assert len(np.unique(predictions)) == len(x)

def test_max_iter_one():
    np.random.seed(42)
    model = kmc.KMeansClustering(k=2, max_iter=1)
    x = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
    model.fit(x)
    assert model.centroids is not None
    assert model.labels is not None

def test_fit_all_identical_points():
    model = kmc.KMeansClustering(k=2)
    x = np.array([[5, 5], [5, 5], [5, 5], [5, 5]])
    model.fit(x)
    assert len(np.unique(model.labels)) == 1

def test_predict_single_point_1d():
    model = kmc.KMeansClustering(k=2)
    model.fit([0, 10])
    prediction = model.predict([1])
    assert prediction.shape == (1,)

def test_predict_single_point_2d():
    model = kmc.KMeansClustering(k=2)
    model.fit([[0, 0], [10, 10]])
    prediction = model.predict([1, 1])
    assert prediction.shape == (1,)

def test_predict_3d_data():
    np.random.seed(42)
    model = kmc.KMeansClustering(k=2)
    x_train = np.array([[0, 0, 0], [1, 1, 1], [10, 10, 10], [11, 11, 11]])
    model.fit(x_train)
    predictions = model.predict([[0.5, 0.5, 0.5], [10.5, 10.5, 10.5]])
    assert predictions[0] != predictions[1]

def test_convergence_check():
    np.random.seed(42)
    model = kmc.KMeansClustering(k=2, max_iter=100)
    x = np.array([[0], [1], [10], [11]])
    model.fit(x)
    predictions = model.predict([-1, 12])
    assert predictions[0] != predictions[1]

def test_predict_far_point():
    np.random.seed(42)
    model = kmc.KMeansClustering(k=2)
    model.fit([[0, 0], [1, 1]])
    pred_near = model.predict([[0.5, 0.5]])
    pred_far = model.predict([[100, 100]])
    assert pred_far[0] == pred_near[0]

def test_predict_point_equidistant():
    model = kmc.KMeansClustering(k=2)
    model.centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    prediction = model.predict([[5, 5]])
    assert prediction[0] == 0

def test_predict_with_large_feature_values():
    np.random.seed(42)
    model = kmc.KMeansClustering(k=2)
    x_train = np.array([[1e6, 1e6], [1e6+1, 1e6+1], [-1e6, -1e6], [-1e6-1, -1e6-1]])
    model.fit(x_train)
    predictions = model.predict([[1e6+0.5, 1e6+0.5], [-1e6-0.5, -1e6-0.5]])
    assert predictions[0] != predictions[1]

def test_predict_with_small_feature_values():
    np.random.seed(42)
    model = kmc.KMeansClustering(k=2)
    x_train = np.array([[1e-6, 1e-6], [2e-6, 2e-6], [-1e-6, -1e-6], [-2e-6, -2e-6]])
    model.fit(x_train)
    predictions = model.predict([[1.5e-6, 1.5e-6], [-1.5e-6, -1.5e-6]])
    assert predictions[0] != predictions[1]

def test_predict_single_scalar_for_1d_model():
    np.random.seed(42)
    model = kmc.KMeansClustering(k=2)
    model.fit([0, 10])
    prediction = model.predict(1)
    assert prediction.shape == (1,)
