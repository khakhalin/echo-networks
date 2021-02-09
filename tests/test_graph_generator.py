import pytest
from esn import create_reservoir as creator
from esn import utils
import numpy as np


def test_make_graphs():
    n_nodes = 4
    n_edges = 4
    for net_type in ['erdos', 'ws']:
        for i in range(100): # Generate a bunch of random graphs
            g = creator.make_graph(n_nodes=n_nodes, n_edges=n_edges, network_type=net_type)
            assert isinstance(g, dict)
            assert len(g.keys()) <= 20 # Left-side nodes
            assert len(set([i for k,v in g.items() for i in v])) <= 20 # Right side nodes
            assert len(utils.edges(g)) == n_edges, f"graph: {g}"  # n_edges


def test_activation_functions():
    f = creator.activation('tanh')
    assert f(1) == np.tanh(1)


def test_graph_to_weights():
    w = creator.graph_to_weights({0: [0, 1], 1:[0]}, inhibition='none')
    assert (w == np.array([[1, 1], [1, 0]])).all()
    w = creator.graph_to_weights({0: [0, 1], 1: [0]}, inhibition='alternating')
    assert (w == np.array([[-1, 1], [1, 0]])).all()
    w = creator.graph_to_weights({0: [0, 1], 1: [0]}, inhibition='distributed', alpha=0.2)
    assert (w == np.array([[1, 1], [1, -0.2]])).all()


def test_weights_in():
    w = creator.weights_in(2, 'flat')
    assert (w == np.array([1, 1])).all()
    w = creator.weights_in(2, 'alternating')
    assert (w == np.array([-1, 1])).all()
