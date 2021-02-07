import pytest
from esn import create_reservoir as creator
import numpy as np


def test_basic_():
    for net_type in ['erdos', 'ws']:
        g = creator.make_graph(n_nodes=20, n_edges=10, network_type=net_type)
        assert isinstance(g, dict)
        assert len(g.keys()) <= 20 # Left-side nodes
        assert len(set([i for k,v in g.items() for i in v])) <= 20 # Right side nodes
        assert len(set([tuple(sorted([i,j])) for i,v in g.items() for j in v])) == 10 # n_edges