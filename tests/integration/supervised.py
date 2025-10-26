import rice2025 as ml

knn = ml.KNN(3)
x_train = [1, 1, 2, 2, 2, 3, 4]
y_train = [0, 1, 0, 1, 2, 0, 1]
knn.fit(x_train, y_train)
x_test = [2]
predictions = knn.predict(x_test)
print(predictions)