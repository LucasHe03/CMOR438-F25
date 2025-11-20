import numpy as np
import pytest
import rice2025.supervised_learning.knn as knn

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

"""
Test fit function
"""
def test_fit_success():
    model = knn.KNN()
    x = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = [0, 1, 0, 1]
    model.fit(x, y)
    np.testing.assert_array_equal(model.x_train, np.array(x, dtype=np.float64))
    np.testing.assert_array_equal(model.y_train, np.array(y))

def test_fit_success_classify():
    model = knn.KNN()
    x = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = ["cat", "dog", "cat", "dog"]
    model.fit(x, y)
    np.testing.assert_array_equal(model.x_train, np.array(x, dtype=np.float64))
    np.testing.assert_array_equal(model.y_train, np.array(y))

def test_fit_mismatched_lengths():
    model = knn.KNN()
    x = [[1, 2], [3, 4]]
    y = [0]
    with pytest.raises(ValueError):
        model.fit(x, y)

def test_fit_empty():
    model = knn.KNN()
    x = []
    y = []
    with pytest.raises(ValueError):
        model.fit(x, y)

def test_fit_k_too_large():
    model = knn.KNN(5)
    x = [[1, 2], [3, 4]]
    y = [0, 1]
    with pytest.raises(ValueError):
        model.fit(x, y)

"""
Test predict function
"""
def test_predict_not_fitted():
    model = knn.KNN()
    with pytest.raises(ValueError, match="model most be fit before predicting"):
        model.predict([[1, 1]])

def test_predict_regression():
    model = knn.KNN(3)
    x_train = [[1, 1], [1, 2], [2, 2], [8, 8], [8, 9], [9, 9]]
    y_train = [10, 10, 10, 20, 20, 20]
    model.fit(x_train, y_train)
    prediction = model.predict([[2, 1]])
    np.testing.assert_equal(prediction, np.array([10.0]))

def test_predict_regression2():
    model = knn.KNN(2)
    x_train = [1, 1, 2, 2]
    y_train = [5, 10, 15, 20]
    model.fit(x_train, y_train)
    prediction = model.predict([1])
    np.testing.assert_equal(prediction, np.array([5]))

def test_predict_classification():
    model = knn.KNN(3)
    x_train = [[1, 1], [1, 2], [2, 2], [8, 8], [8, 9], [9, 9]]
    y_train = ["cat", "cat", "cat", "dog", "dog", "dog"]
    model.fit(x_train, y_train)
    prediction = model.predict([[2, 1]])
    assert prediction[0] == "cat"

def test_predict_classification_tie():
    model = knn.KNN(4)
    x_train = [[1, 1], [2, 2], [8, 8], [9, 9]]
    y_train = ["cat", "dog", "cat", "dog"]
    model.fit(x_train, y_train)
    prediction = model.predict([[5, 5]])
    assert prediction[0] == "cat" or prediction[0] == "dog"

def test_predict_single_1d_input():
    model = knn.KNN(1)
    x_train = [[1, 1], [10, 10]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    prediction = model.predict([2, 2])
    np.testing.assert_equal(prediction, np.array([0.0]))

def test_predict_multiple_points():
    model = knn.KNN(1)
    x_train = [[0, 0], [10, 10]]
    y_train = ["cat", "dog"]
    model.fit(x_train, y_train)
    x_test = [[1, 1], [9, 9]]
    predictions = model.predict(x_test)
    np.testing.assert_equal(predictions, np.array(["cat", "dog"]))

def test_predict_mismatched_features():
    model = knn.KNN(1)
    x_train = [[1, 1], [2, 2]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    x_test_mismatched = [[1, 1, 1]]
    with pytest.raises(ValueError):
        model.predict(x_test_mismatched)

def test_fit_numpy_input():
    model = knn.KNN()
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model.fit(x, y)
    np.testing.assert_array_equal(model.x_train, x.astype(np.float64))
    np.testing.assert_array_equal(model.y_train, y)

def test_predict_numpy_input():
    model = knn.KNN(1)
    x_train = np.array([[0, 0], [10, 10]])
    y_train = np.array(["cat", "dog"])
    model.fit(x_train, y_train)
    x_test = np.array([[1, 1], [9, 9]])
    predictions = model.predict(x_test)
    np.testing.assert_equal(predictions, np.array(["cat", "dog"]))

def test_refit_model():
    model = knn.KNN(1)
    x1, y1 = [[1, 1]], [10]
    model.fit(x1, y1)
    pred1 = model.predict([[2, 2]])
    np.testing.assert_array_equal(pred1, np.array([10]))

    x2, y2 = [[100, 100]], [20]
    model.fit(x2, y2)
    pred2 = model.predict([[101, 101]])
    np.testing.assert_array_equal(pred2, np.array([20]))

def test_predict_on_training_data_classification():
    model = knn.KNN(1)
    x_train = [[1, 1], [2, 2], [10, 10], [11, 11]]
    y_train = [0, 0, 1, 1]
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)
    np.testing.assert_array_equal(predictions, y_train)

def test_predict_k_equals_n_samples_classification():
    model = knn.KNN(4)
    x_train = [[1, 1], [2, 2], [10, 10], [11, 11]]
    y_train = [0, 0, 1, 1]
    model.fit(x_train, y_train)
    prediction = model.predict([[5, 5]])
    np.testing.assert_array_equal(prediction, np.array([0])) # tie-break goes to first class

def test_predict_point_identical_to_training_point():
    model = knn.KNN(3)
    x_train = [[1, 1], [2, 2], [3, 3]]
    y_train = [0, 1, 0]
    model.fit(x_train, y_train)
    prediction = model.predict([[2, 2]])
    np.testing.assert_array_equal(prediction, np.array([0])) # neighbors are [2,2], [1,1], [3,3] -> labels [1,0,0] -> majority 0

def test_predict_all_training_points_identical():
    model = knn.KNN(3)
    x_train = [[5, 5], [5, 5], [5, 5], [5, 5]]
    y_train = [0, 1, 0, 1]
    model.fit(x_train, y_train)
    prediction = model.predict([[1, 1]])
    np.testing.assert_array_equal(prediction, np.array([0])) # tie-break

def test_predict_3d_features():
    model = knn.KNN(1)
    x_train = [[1, 1, 1], [10, 10, 10]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    prediction = model.predict([[2, 2, 2]])
    np.testing.assert_array_equal(prediction, np.array([0]))

def test_predict_negative_features():
    model = knn.KNN(1)
    x_train = [[-1, -1], [1, 1]]
    y_train = [0, 1]
    model.fit(x_train, y_train)
    prediction = model.predict([[0, 0]])
    # The prediction could be either 0 or 1 as [0,0] is equidistant.
    # We check if the prediction is one of the possible outcomes.
    assert prediction[0] in [0, 1]

def test_predict_far_point_classification():
    model = knn.KNN(3)
    x_train = [[1, 1], [2, 2], [3, 3]]
    y_train = [0, 0, 1]
    model.fit(x_train, y_train)
    prediction = model.predict([[100, 100]])
    np.testing.assert_array_equal(prediction, np.array([0]))
def test_fit_1d_data():
    model = knn.KNN(1)
    x = [1, 2, 3, 4]
    y = [0, 1, 0, 1]
    model.fit(x, y)
    np.testing.assert_array_equal(model.x_train, np.array([[1], [2], [3], [4]], dtype=np.float64))
    np.testing.assert_array_equal(model.y_train, np.array(y))

def test_predict_1d_classification():
    model = knn.KNN(3)
    x_train = [1, 2, 3, 8, 9, 10]
    y_train = [0, 0, 0, 1, 1, 1]
    model.fit(x_train, y_train)
    prediction = model.predict([4])
    np.testing.assert_array_equal(prediction, np.array([0]))

def test_init_float_k():
    with pytest.raises(TypeError):
        knn.KNN(3.5)

def test_predict_single_feature_mismatch():
    model = knn.KNN(1)
    model.fit([[1, 2]], [0])
    with pytest.raises(ValueError):
        model.predict([1])

def test_predict_with_single_training_sample_regression():
    model = knn.KNN(1)
    model.fit([[10, 10]], [100])
    prediction = model.predict([[1, 1]])
    np.testing.assert_array_equal(prediction, np.array([100]))

def test_predict_with_single_training_sample_classification():
    model = knn.KNN(1)
    model.fit([[10, 10]], ["A"])
    prediction = model.predict([[1, 1]])
    np.testing.assert_array_equal(prediction, np.array(["A"]))
