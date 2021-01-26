import numpy as np

# # self.weights = creator.graph_to_weights(self.graph, inhibition='distributed')

@staticmethod
def graph_to_weights(graph_dict, n_nodes=None, inhibition='alternating', params=None):
    """Creates a numpy weights matrix from a graph.

    parameters:
    graph_dict: a dictionary defining the graph
    inhibition (string): how to introduce inhibition. Options include:
        'none' - no inhibition
        'alternating' - checkerboard pattern, with even edges excitatory
        'distributed' - all edges are excitatory, but all missing edges are weakly inhibitory
    """

    if not n_nodes: # Try go guess n_nodes from the graph_dict itself
        keys,values = zip(*graph_dict.items())
        n_nodes = max(list(keys) + [i for edges in values for i in edges]) + 1
    weights = np.zeros((n_nodes, n_nodes))
    for i, edges in graph_dict.items():
        for j in edges:
            weights[i, j] = 1   # The matrix is not flipped!

    if inhibition == 'alternating':
        for i in range(n_nodes):
            for j in range(n_nodes):
                weights[i,j] *= ((i+j) % 2)*2 - 1
    elif inhibition == 'distributed':
        if params:
            alpha = params
        else:
            alpha = 0.1
        weights = weights*(1 + alpha) - alpha

    return weights