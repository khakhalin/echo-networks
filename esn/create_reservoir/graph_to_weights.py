import numpy as np
from esn.utils import utils


def graph_to_weights(graph_dict, n_nodes=None, inhibition='alternating'):
    """Creates a numpy weights matrix from a graph.

    parameters:
    graph_dict: a dictionary defining the graph
    inhibition (string): how to introduce inhibition. Options include:
        'none' - no inhibition
        'alternating' - checkerboard pattern, with even edges excitatory
        'distributed' - all edges are excitatory, but all missing edges are weakly inhibitory
    """

    if n_nodes is None: # Try go guess n_nodes from the graph_dict itself
        keys,values = zip(*graph_dict.items())
        n_nodes = max(list(keys) + [i for edges in values for i in edges]) + 1
    weights = np.zeros((n_nodes, n_nodes))
    edges = [(i,j) for i,j in utils.edges(graph_dict) if i != j]  # Remove loops
    for (i,j) in edges:
        weights[i, j] = 1   # The matrix is not flipped!

    if inhibition == 'alternating':
        for i in range(n_nodes):
            for j in range(n_nodes):
                weights[i,j] *= ((i+j) % 2)*2 - 1 # Checkerboard

    elif inhibition == 'distributed':
        strength = len(edges)/(n_nodes*(n_nodes-1)-len(edges))
        weights = weights*(1 + strength) - strength
        i = range(n_nodes)

    weights[i,i] = 0  # No self-inhibition (we already have leak)
    return weights
