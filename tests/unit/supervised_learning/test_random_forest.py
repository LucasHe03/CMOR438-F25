import numpy as np
import pytest
import rice2025.supervised_learning.random_forest as rf

"""
Test __init__ function
"""
def test_init_default():
    model = rf.RandomForest()
    assert model.n_estimators == 100
    assert model.min_samples_split == 2
    assert model.max_depth == 100
    assert model.trees == []

def test_init_custom():
    model = rf.RandomForest(n_estimators=50, min_samples_split=5, max_depth=10)
    assert model.n_estimators == 50
    assert model.min_samples_split == 5
    assert model.max_depth == 10

"""
Test fit function
"""
def test_fit_success():
    model = rf.RandomForest(n_estimators=5)
    x = [[1], [2]]
    y = [0, 1]
    model.fit(x, y)
    assert model.trees is not None
    assert len(model.trees) == 5

def test_fit_mismatched_lengths():
    model = rf.RandomForest()
    x = [[1], [2]]
    y = [0]
    with pytest.raises(ValueError, match="x and y must have equal lengths"):
        model.fit(x, y)

def test_fit_empty():
    model = rf.RandomForest()
    x = []
    y = []
    with pytest.raises(ValueError, match="x and y must be non-empty"):
        model.fit(x, y)

"""
Test predict function
"""
def test_predict_not_fitted():
    model = rf.RandomForest()
    with pytest.raises(ValueError, match="model must be fit before predicting"):
        model.predict([[1]])

def test_predict_classification():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=10, max_depth=5)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)
    
    x_test = [[1.5], [9.5]]
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_predict_regression():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=10, max_depth=5)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    model.fit(x_train, y_train)
    
    x_test = [[2.5], [8.5]]
    predictions = model.predict(x_test)
    np.testing.assert_array_almost_equal(predictions, np.array([10.0, 20.0]), decimal=1)

def test_predict_single_point_classification():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=10)
    x_train = [[1], [10]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    
    prediction = model.predict([11])
    np.testing.assert_array_equal(prediction, np.array([1]))

def test_predict_single_point_regression():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=10)
    x_train = [[1], [10]]
    y_train = [10.0, 20.0]
    model.fit(x_train, y_train)
    
    prediction = model.predict([11])
    assert prediction[0] > 15.0

def test_predict_empty_input():
    model = rf.RandomForest()
    x_train = [[0, 0], [1, 1]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    predictions = model.predict([])
    assert predictions.shape == (0,)

def test_fit_numpy_input():
    model = rf.RandomForest(n_estimators=5)
    x = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])
    model.fit(x, y)
    assert len(model.trees) == 5
    assert model._is_classifier is True

def test_predict_numpy_input():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=10)
    x_train = np.array([[1], [10]])
    y_train = np.array([0, 1])
    model.fit(x_train, y_train)
    x_test = np.array([[0], [11]])
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_refit_model():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=5)
    x1, y1 = [[1], [2]], [0, 0]
    model.fit(x1, y1)
    pred1 = model.predict([1.5])
    np.testing.assert_array_equal(pred1, np.array([0]))

    x2, y2 = [[10], [11]], [1, 1]
    model.fit(x2, y2)
    pred2 = model.predict([10.5])
    np.testing.assert_array_equal(pred2, np.array([1]))
    assert len(model.trees) == 5

def test_predict_on_training_data_classification():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=10, max_depth=5)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)
    accuracy = np.mean(predictions == y_train)
    assert accuracy >= 0.8

def test_predict_on_training_data_regression():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=10, max_depth=5)
    x_train = np.array([[1], [2], [3], [8], [9], [10]])
    y_train = np.array([10.0, 10.0, 10.0, 20.0, 20.0, 20.0])
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)
    mse = np.mean((predictions - y_train)**2)
    assert mse < 5.0

def test_predict_mismatched_features():
    model = rf.RandomForest()
    model.fit([[1, 2]], [1])
    with pytest.raises(ValueError):
        model.predict([[1, 2, 3]])

def test_classification_string_labels():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=10, max_depth=5)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = ['A', 'A', 'A', 'B', 'B', 'B']
    model.fit(x_train, y_train)
    predictions = model.predict([[1.5], [9.5]])
    np.testing.assert_array_equal(predictions, np.array(['A', 'B']))

def test_n_estimators_one():
    model = rf.RandomForest(n_estimators=1)
    model.fit([[1], [10]], [0, 1])
    assert len(model.trees) == 1
    prediction = model.predict([[5]])
    assert prediction[0] in [0, 1]

def test_max_depth_zero():
    model = rf.RandomForest(max_depth=0)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)
    predictions = model.predict([[1.5], [9.5]])
    assert predictions[0] == predictions[1]

def test_min_samples_split_large():
    model = rf.RandomForest(min_samples_split=10)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)
    predictions = model.predict([[1.5], [9.5]])
    assert predictions[0] == predictions[1]

def test_fit_with_all_same_labels():
    model = rf.RandomForest(n_estimators=5)
    x = [[1], [2], [3], [4]]
    y = [1, 1, 1, 1]
    model.fit(x, y)
    prediction = model.predict([[5]])
    np.testing.assert_array_equal(prediction, np.array([1]))

def test_predict_returns_numpy_array():
    model = rf.RandomForest()
    model.fit([[1]], [1])
    prediction = model.predict([[2]])
    assert isinstance(prediction, np.ndarray)

def test_predict_regression_output_dtype():
    model = rf.RandomForest()
    model.fit([[1], [2]], [1.0, 2.0])
    prediction = model.predict([[1.5]])
    assert np.issubdtype(prediction.dtype, np.floating)

def test_predict_multi_feature_classification():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=10, max_depth=5)
    x_train = [[1, 10], [2, 11], [8, 1], [9, 2]]
    y_train = [0, 0, 1, 1]
    model.fit(x_train, y_train)
    predictions = model.predict([[1.5, 20], [8.5, 0]])
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_predict_multi_feature_regression():
    np.random.seed(42)
    model = rf.RandomForest(n_estimators=10, max_depth=5)
    x_train = [[1, 10], [2, 11], [8, 1], [9, 2]]
    y_train = [100, 100, 200, 200]
    model.fit(x_train, y_train)
    predictions = model.predict([[1.5, 20], [8.5, 0]])
    np.testing.assert_array_almost_equal(predictions, np.array([100, 200]), decimal=1)

def test_is_classifier_flag_classification():
    model = rf.RandomForest()
    x = [[1], [2], [3], [4]]
    y = [0, 1, 0, 1]
    model.fit(x, y)
    assert model._is_classifier is True

def test_is_classifier_flag_string_labels():
    model = rf.RandomForest()
    x = [[1], [2], [3], [4]]
    y = ['A', 'B', 'A', 'B']
    model.fit(x, y)
    assert model._is_classifier is True