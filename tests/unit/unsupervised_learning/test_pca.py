import numpy as np
import pytest
import rice2025.unsupervised_learning.pca as pca

"""
Test __init__ function
"""
def test_init():
    model = pca.PCA(n_components=2)
    assert model.n_components == 2

def test_init_invalid_n_components():
    with pytest.raises(ValueError, match="n_components must be a positive integer"):
        pca.PCA(n_components=0)
    with pytest.raises(ValueError, match="n_components must be a positive integer"):
        pca.PCA(n_components=-1)

"""
Test fit, transform, and fit_transform functions
"""
def test_fit_success():
    model = pca.PCA(n_components=1)
    x = np.array([[1, 2], [3, 4], [5, 6]])
    model.fit(x)
    assert model.mean_ is not None
    assert model.components_ is not None
    assert model.components_.shape == (2, 1)

def test_fit_n_components_too_large():
    model = pca.PCA(n_components=3)
    x = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="n_components cannot be greater than the number of features"):
        model.fit(x)

def test_transform_not_fitted():
    model = pca.PCA(n_components=1)
    with pytest.raises(ValueError, match="PCA model must be fit before transforming data."):
        model.transform([[1, 1]])

def test_fit_transform_shape():
    model = pca.PCA(n_components=1)
    x = np.array([[1, 10], [2, 11], [3, 12]])
    x_transformed = model.fit_transform(x)
    assert x_transformed.shape == (3, 1)

def test_fit_transform_simple_case():
    x = np.array([[1, 1], [2, 2], [3, 3]])
    model = pca.PCA(n_components=1)
    x_transformed = model.fit_transform(x)

    component = model.components_[:, 0]
    expected_comp = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    assert np.allclose(np.abs(component), expected_comp)

    model_2d = pca.PCA(n_components=2).fit(x)
    x_transformed_2d = model_2d.transform(x)
    assert np.isclose(np.var(x_transformed_2d[:, 1]), 0)

def test_fit_returns_self():
    model = pca.PCA(n_components=1)
    x = [[1, 2], [3, 4]]
    fitted_model = model.fit(x)
    assert fitted_model is model

def test_1d_data():
    model = pca.PCA(n_components=1)
    x = [1, 2, 3, 4, 5]
    x_transformed = model.fit_transform(x)
    assert x_transformed.shape == (5, 1)

def test_data_centering():
    x = np.array([[1, 10], [2, 20], [3, 30]])
    model = pca.PCA(n_components=1)
    model.fit(x)
    np.testing.assert_array_almost_equal(model.mean_, np.array([2, 20]))

def test_transformed_data_is_centered():
    x = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
    model = pca.PCA(n_components=2)
    x_transformed = model.fit_transform(x)
    mean_transformed = np.mean(x_transformed, axis=0)
    np.testing.assert_array_almost_equal(mean_transformed, np.array([0, 0]))

def test_components_are_orthogonal():
    x = np.random.rand(10, 3)
    model = pca.PCA(n_components=3)
    model.fit(x)
    components = model.components_
    identity = np.dot(components.T, components)
    np.testing.assert_array_almost_equal(identity, np.eye(3))

def test_fit_transform_vs_fit_then_transform():
    x = np.array([[1, 10], [2, 11], [3, 12]])
    model1 = pca.PCA(n_components=1)
    xt1 = model1.fit_transform(x)

    model2 = pca.PCA(n_components=1)
    model2.fit(x)
    xt2 = model2.transform(x)
    np.testing.assert_array_almost_equal(xt1, xt2)

def test_data_with_zero_variance():
    x = np.array([[1, 1], [1, 1], [1, 1]])
    model = pca.PCA(n_components=1)
    x_transformed = model.fit_transform(x)
    # should be all zeros
    np.testing.assert_array_almost_equal(x_transformed, np.zeros((3, 1)))

def test_transform_different_data():
    x_train = np.array([[1, 1], [2, 2], [3, 3]])
    model = pca.PCA(n_components=1)
    model.fit(x_train)

    x_test = np.array([[4, 4], [5, 5]])
    x_test_transformed = model.transform(x_test)
    assert x_test_transformed.shape == (2, 1)

def test_empty_input_fit():
    model = pca.PCA(n_components=1)
    with pytest.raises(ValueError):
        model.fit([])

def test_refit_model():
    model = pca.PCA(n_components=1)
    x1 = np.array([[1, 1], [2, 2]])
    model.fit(x1)
    mean1 = model.mean_.copy()

    x2 = np.array([[10, 10], [20, 20]])
    model.fit(x2)
    mean2 = model.mean_.copy()

    assert not np.allclose(mean1, mean2)