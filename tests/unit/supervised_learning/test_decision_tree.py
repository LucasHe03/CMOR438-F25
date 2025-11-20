import numpy as np
import pytest
import rice2025.supervised_learning.decision_tree as dt

"""
Test __init__ function
"""
def test_init_default():
    model = dt.DecisionTree()
    assert model.min_samples_split == 2
    assert model.max_depth == 100
    assert model.root is None

def test_init_custom():
    model = dt.DecisionTree(min_samples_split=5, max_depth=10)
    assert model.min_samples_split == 5
    assert model.max_depth == 10

def test_init_invalid_min_samples():
    with pytest.raises(ValueError, match="min_samples_split must be an integer >= 2"):
        dt.DecisionTree(min_samples_split=1)

def test_init_invalid_max_depth():
    with pytest.raises(ValueError, match="max_depth must be an integer >= 0"):
        dt.DecisionTree(max_depth=-1)

"""
Test fit function
"""
def test_fit_success():
    model = dt.DecisionTree()
    x = [[1], [2]]
    y = [0, 1]
    model.fit(x, y)
    assert model.root is not None

def test_fit_numpy_input():
    model = dt.DecisionTree()
    x = np.array([[1], [2]])
    y = np.array([0, 1])
    model.fit(x, y)
    assert model.root is not None
    assert model.root.num_samples == 2

def test_fit_mismatched_lengths():
    model = dt.DecisionTree()
    x = [[1], [2]]
    y = [0]
    with pytest.raises(ValueError, match="x and y must have equal lengths"):
        model.fit(x, y)

def test_fit_empty():
    model = dt.DecisionTree()
    x = []
    y = []
    with pytest.raises(ValueError, match="x and y must be non-empty"):
        model.fit(x, y)

def test_fit_refit():
    model = dt.DecisionTree(max_depth=1)
    x1, y1 = [[1], [10]], [0, 1]
    model.fit(x1, y1)
    pred1 = model.predict([[5]])
    np.testing.assert_array_equal(pred1, np.array([0]))

    x2, y2 = [[100], [200]], [10, 20]
    model.fit(x2, y2)
    pred2 = model.predict([[150]])
    np.testing.assert_array_equal(pred2, np.array([10]))
    assert model.root.feature_index == 0
    assert model.root.threshold == 150.0
    assert model.root.left.value == 10
    assert model.root.right.value == 20

"""
Test predict function
"""
def test_predict_not_fitted():
    model = dt.DecisionTree()
    with pytest.raises(ValueError, match="model must be fit before predicting"):
        model.predict([[1]])

def test_predict_classification():
    model = dt.DecisionTree(max_depth=5)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)
    
    x_test = [[1.5], [9.5]]
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_predict_classification_numpy():
    model = dt.DecisionTree(max_depth=5)
    x_train = np.array([[1], [2], [3], [8], [9], [10]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    model.fit(x_train, y_train)
    
    x_test = np.array([[1.5], [9.5]])
    predictions = model.predict(x_test)
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_predict_on_split_boundary():
    model = dt.DecisionTree(max_depth=1)
    model.fit([[1], [3]], [0, 1])
    predictions = model.predict([[2.0]])
    np.testing.assert_array_equal(predictions, np.array([0]))

def test_predict_regression():
    model = dt.DecisionTree(max_depth=5)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    model.fit(x_train, y_train)
    
    x_test = [[2.5], [8.5]]
    predictions = model.predict(x_test)
    np.testing.assert_array_almost_equal(predictions, np.array([10.0, 20.0]))

def test_predict_regression_unseen_range():
    model = dt.DecisionTree(max_depth=1)
    x_train = [[1], [10]]
    y_train = [100, 200]
    model.fit(x_train, y_train)
    predictions = model.predict([[-5], [100]])
    np.testing.assert_array_almost_equal(predictions, np.array([100, 200]))

def test_predict_single_point():
    model = dt.DecisionTree()
    x_train = [[1], [10]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    
    prediction = model.predict([11])
    np.testing.assert_array_equal(prediction, np.array([1]))

def test_predict_single_point_2d_feature():
    model = dt.DecisionTree()
    x_train = [[1, 5], [10, 5]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    
    prediction = model.predict([[11, 5]])
    np.testing.assert_array_equal(prediction, np.array([1]))

def test_predict_max_depth():
    model = dt.DecisionTree(max_depth=0)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 1, 1, 1, 1]
    model.fit(x_train, y_train)
    
    predictions = model.predict([[1], [9]])
    np.testing.assert_array_equal(predictions, np.array([1, 1]))

def test_predict_max_depth_1():
    model = dt.DecisionTree(max_depth=1)
    x_train = [[1], [2], [8], [9]]
    y_train = [0, 0, 1, 1]
    model.fit(x_train, y_train)
    assert not model.root.is_leaf_node()
    assert model.root.left.is_leaf_node()
    assert model.root.right.is_leaf_node()

def test_predict_min_samples_split():
    model = dt.DecisionTree(min_samples_split=10)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)

    assert model.root.is_leaf_node()

def test_predict_min_samples_split_prevents_split():
    model = dt.DecisionTree(min_samples_split=3)
    x_train = [[1], [2], [8], [9]]
    y_train = [0, 1, 0, 1]
    model.fit(x_train, y_train)
    assert not model.root.is_leaf_node()
    assert model.root.left.is_leaf_node()

def test_predict_min_samples_split_tie_classification():
    model = dt.DecisionTree(min_samples_split=10)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)

    predictions = model.predict([[5]])
    np.testing.assert_array_equal(predictions, np.array([0]))

def test_pure_node_becomes_leaf():
    model = dt.DecisionTree()
    x_train = [[1], [2], [3], [4]]
    y_train = [1, 1, 1, 1]
    model.fit(x_train, y_train)
    assert model.root.is_leaf_node()
    assert model.root.value == 1

def test_pure_node_regression():
    model = dt.DecisionTree()
    x_train = [[1], [2], [3], [4]]
    y_train = [5.5, 5.5, 5.5, 5.5]
    model.fit(x_train, y_train)
    assert model.root.is_leaf_node()
    assert model.root.value == 5.5

def test_no_information_gain():
    model = dt.DecisionTree()
    x_train = [[1], [1], [1], [1]]
    y_train = [0, 1, 0, 1]
    model.fit(x_train, y_train)
    assert model.root.is_leaf_node()
    assert model.root.value == 0

def test_constant_feature_multiple_values():
    model = dt.DecisionTree()
    x_train = [[1, 5], [2, 5], [3, 5]]
    y_train = [0, 1, 0]
    model.fit(x_train, y_train)
    assert model.root.feature_index == 0

def test_predict_multi_feature_classification():
    model = dt.DecisionTree(max_depth=5)
    x_train = [[1, 10], [2, 11], [8, 1], [9, 2]]
    y_train = [0, 0, 1, 1]
    model.fit(x_train, y_train)
    predictions = model.predict([[1.5, 20], [8.5, 0]])
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_predict_multi_feature_regression():
    model = dt.DecisionTree(max_depth=5)
    x_train = [[1, 10], [2, 11], [8, 1], [9, 2]]
    y_train = [100, 100, 200, 200]
    model.fit(x_train, y_train)
    predictions = model.predict([[1.5, 20], [8.5, 0]])
    np.testing.assert_array_almost_equal(predictions, np.array([100, 200]))

def test_predict_empty_input():
    model = dt.DecisionTree()
    model.fit([[1]], [1])
    predictions = model.predict([])
    assert predictions.shape == (0,)

def test_predict_mismatched_features():
    model = dt.DecisionTree()
    model.fit([[1, 2]], [1])
    with pytest.raises(ValueError, match="Number of features of the model must match the input"):
        model.predict([[1, 2, 3]])

def test_predict_mismatched_features_single():
    model = dt.DecisionTree()
    model.fit([[1, 2]], [1])
    with pytest.raises(ValueError, match="Number of features of the model must match the input"):
        model.predict([[1]])

def test_classification_string_labels():
    model = dt.DecisionTree(max_depth=1)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = ['A', 'A', 'A', 'B', 'B', 'B']
    model.fit(x_train, y_train)
    predictions = model.predict([[1.5], [9.5]])
    np.testing.assert_array_equal(predictions, np.array(['A', 'B']))

def test_classification_string_labels_tie_break():
    model = dt.DecisionTree(min_samples_split=10)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = ['B', 'B', 'B', 'A', 'A', 'A']
    model.fit(x_train, y_train)
    predictions = model.predict([[5]])
    np.testing.assert_array_equal(predictions, np.array(['A']))

def test_predict_on_training_data_classification():
    model = dt.DecisionTree(max_depth=5)
    x_train = [[1], [2], [3], [8], [9], [10]]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)
    np.testing.assert_array_equal(predictions, y_train)

def test_predict_with_single_sample_in_leaf():
    model = dt.DecisionTree(max_depth=2)
    x_train = [[1], [2], [10]]
    y_train = [0, 0, 1]
    model.fit(x_train, y_train)
    prediction = model.predict([[11]])
    np.testing.assert_array_equal(prediction, np.array([1]))