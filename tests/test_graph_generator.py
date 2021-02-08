import pytest
from esn import create_reservoir as creator
import numpy as np


def test_basic_graphs():
    for net_type in ['erdos', 'ws']:
        g = creator.make_graph(n_nodes=20, n_edges=10, network_type=net_type)
        assert isinstance(g, dict)
        assert len(g.keys()) <= 20 # Left-side nodes
        assert len(set([i for k,v in g.items() for i in v])) <= 20 # Right side nodes
        assert len(set([tuple(sorted([i,j])) for i,v in g.items() for j in v])) == 10 # n_edges


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
