import numpy as np
import pytest
import rice2025.unsupervised_learning.community_detection as cd

"""
Test __init__ function
"""
def test_init_default():
    model = cd.CommunityDetection()
    assert model.max_iter == 100

def test_init_custom():
    model = cd.CommunityDetection(max_iter=50)
    assert model.max_iter == 50

def test_init_invalid_max_iter():
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        cd.CommunityDetection(max_iter=0)

"""
Test fit and fit_predict functions
"""
def test_fit_simple_communities():
    adj_matrix = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ])
    model = cd.CommunityDetection()
    labels = model.fit_predict(adj_matrix)
    
    assert len(np.unique(labels)) == 2
    
    community1 = labels[0]
    community2 = labels[3]
    assert community1 != community2
    np.testing.assert_array_equal(labels[:3], [community1, community1, community1])
    np.testing.assert_array_equal(labels[3:], [community2, community2, community2])

def test_fit_returns_self():
    model = cd.CommunityDetection()
    adj_matrix = np.array([[0, 1], [1, 0]])
    fitted_model = model.fit(adj_matrix)
    assert fitted_model is model

def test_fit_empty_graph():
    model = cd.CommunityDetection()
    labels = model.fit_predict([])
    assert labels.shape == (0,)

def test_graph_with_no_edges():
    adj_matrix = np.zeros((5, 5))
    model = cd.CommunityDetection()
    labels = model.fit_predict(adj_matrix)
    assert len(np.unique(labels)) == 5
    np.testing.assert_array_equal(np.sort(labels), np.arange(5))

def test_single_large_component():
    adj_matrix = np.ones((5, 5)) - np.eye(5)
    model = cd.CommunityDetection()
    labels = model.fit_predict(adj_matrix)
    assert len(np.unique(labels)) == 1
    np.testing.assert_array_equal(labels, np.zeros(5))

def test_three_components():
    adj_matrix = np.zeros((7, 7))
    # Component 1: 0, 1
    adj_matrix[0, 1] = adj_matrix[1, 0] = 1
    # Component 2: 2, 3, 4
    adj_matrix[2, 3] = adj_matrix[3, 2] = 1
    adj_matrix[3, 4] = adj_matrix[4, 3] = 1
    # Component 3: 5, 6
    adj_matrix[5, 6] = adj_matrix[6, 5] = 1
    
    model = cd.CommunityDetection()
    labels = model.fit_predict(adj_matrix)
    assert len(np.unique(labels)) == 3
    assert labels[0] == labels[1]
    assert labels[2] == labels[3] == labels[4]
    assert labels[5] == labels[6]
    assert labels[0] != labels[2] and labels[0] != labels[5]

def test_graph_with_isolated_node():
    adj_matrix = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])
    model = cd.CommunityDetection()
    labels = model.fit_predict(adj_matrix)
    assert len(np.unique(labels)) == 2
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]

def test_non_square_matrix_input():
    adj_matrix = np.zeros((3, 4))
    model = cd.CommunityDetection()
    with pytest.raises(ValueError, match="Input must be a square adjacency matrix."):
        model.fit(adj_matrix)

def test_weighted_graph_is_treated_as_unweighted():
    adj_matrix = np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]])
    model = cd.CommunityDetection()
    labels = model.fit_predict(adj_matrix)
    assert len(np.unique(labels)) == 2
    assert labels[0] == labels[1]

def test_line_graph():
    adj_matrix = np.diag(np.ones(4), k=1) + np.diag(np.ones(4), k=-1)
    model = cd.CommunityDetection()
    labels = model.fit_predict(adj_matrix)
    assert len(np.unique(labels)) == 1

def test_single_node_graph():
    adj_matrix = np.array([[0]])
    model = cd.CommunityDetection()
    labels = model.fit_predict(adj_matrix)
    np.testing.assert_array_equal(labels, [0])

def test_refit_model():
    model = cd.CommunityDetection()
    adj1 = np.array([[0, 1], [1, 0]])
    labels1 = model.fit_predict(adj1)
    assert len(np.unique(labels1)) == 1

    adj2 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    labels2 = model.fit_predict(adj2)
    assert len(np.unique(labels2)) == 2