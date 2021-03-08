import pytest
from esn import create_reservoir as creator
from esn.create_reservoir.graph_to_weights import _spectral_radius
from esn import utils
import numpy as np


def test_make_graphs():
    for net_type in ['erdos', 'ws']:
        for i in range(100): # Generate a bunch of random graphs
            n_nodes = np.random.randint(4, 20)
            n_edges = np.random.randint(1, max(n_nodes, n_nodes*(n_nodes-2))) # Not too dense
            g = creator.make_graph(n_nodes=n_nodes, n_edges=n_edges, network_type=net_type)
            assert isinstance(g, dict)
            assert len(g.keys()) <= 20 # Left-side nodes
            assert len(set([i for k,v in g.items() for i in v])) <= 20 # Right side nodes
            assert len(utils.edges(g)) == n_edges, f"graph: {g}"  # n_edges


def test_activation_functions():
    f = creator.activation('tanh')
    assert f(1) == np.tanh(1)


def test_spectral_radius():
    assert _spectral_radius(1) == 1
    assert _spectral_radius(np.array([[1,0],[0,1]])) == 1
    assert _spectral_radius(np.array([[2, 0], [0, 1]])) == 2
    assert _spectral_radius(np.array([[0,0],[0,0]])) == 0


def test_graph_to_weights():
    w,_ = creator.graph_to_weights({0: [1], 1:[0]}, inhibition='none')
    assert (w == np.array([[0, 1], [1, 0]])).all()
    w,_ = creator.graph_to_weights({0: [1], 1: [0]}, inhibition='none', rho=0.9) # rho should scale this one
    assert (w == 0.9*np.array([[0, 1], [1, 0]])).all()
    w,_ = creator.graph_to_weights({0: [1,0], 1: [0]}, inhibition='none') # Loops should be removed
    assert (w == np.array([[0, 1], [1, 0]])).all()
    w,_ = creator.graph_to_weights({0: [1,2], 1: [0]}, inhibition='alternating')
    assert (w == np.array([[0, 1, -1], [1, 0, 0], [0, 0, 0]])).all()
    w,_ = creator.graph_to_weights({0: [1], 1: []}, inhibition='distributed')
    assert (w == np.array([[0, 1], [-1, 0]])).all()
    w,_ = creator.graph_to_weights({0: [1, 2], 1: [0]}, inhibition='distributed')
    assert (w == np.array([[0, 1, 1], [1, 0, -1], [-1, -1, 0]])).all()


def test_weights_in():
    w = creator.weights_in(2, 'flat')
    assert (w == np.array([1, 1])).all()
    w = creator.weights_in(2, 'alternating')
    assert (w == np.array([-1, 1])).all()

