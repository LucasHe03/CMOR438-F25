import rice2025 as ml

def test_knn():
    knn = ml.KNN(3)
    x_train = [[1, 2], [2, 3], [3, 4], [5, 6], [7, 8]]
    y_train = [0, 0, 0, 1, 1]
    knn.fit(x_train, y_train)
    x_test = [[2, 2]]
    predictions = knn.predict(x_test)
    assert predictions == [0]