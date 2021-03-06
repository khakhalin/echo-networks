import numpy as np
import scipy.linalg as spla
from esn.utils import utils


def graph_to_weights(graph_dict, n_nodes=None, rho=None, inhibition='alternating'):
    """Creates a numpy weights matrix from a graph.

    parameters:
    graph_dict: a dictionary defining the graph
    rho: target spectral radius. Default=None (no adjustment)
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
    for i in range(weights.shape[0]):
        weights[i,i] = 0  # No self-inhibition (we already have leak)

    if inhibition == 'alternating':
        for i in range(n_nodes):
            for j in range(n_nodes):
                weights[i,j] *= ((i+j) % 2)*2 - 1 # Checkerboard

    elif inhibition == 'distributed':
        # Sum across the entire matrix ==0
        strength = len(edges)/(n_nodes*(n_nodes-1)-len(edges))
        weights = weights*(1 + strength) - strength
        i = range(n_nodes)

    elif inhibition == 'balanced_in':
        # Sum converging on each neuron ==0, unless all edges are either 1 or 0
        total_input = np.sum(weights, axis=0)
        total_input = total_input / np.maximum((weights.shape[0] - 1 - total_input), 1)
        weights = weights * (1 + total_input[np.newaxis, :]) - total_input[np.newaxis, :]

    elif inhibition == 'balanced_out':
        # Sum of outputs of each neuron ==0 (unless out_edges are either all 1 or all 0)
        total_input = np.sum(weights, axis=1)
        total_input = total_input / np.maximum((weights.shape[0] - 1 - total_input), 1)
        weights = weights * (1 + total_input[:, np.newaxis]) - total_input[:,np.newaxis]

    elif inhibition is None:
        pass # Explicitly do nothing
    else:
        raise(ValueError(f'Unrecognized inhibition type: {inhibition}'))

    for i in range(weights.shape[0]):
        weights[i,i] = 0  # Cleanup the diagonal once again (some methods above spoil it)

    sr = _spectral_radius(weights)
    if rho is not None:  # Need to correct spectral radius
        if sr != 0: # This matrix is hopeless as a weight matrix, but at least let's not divide by 0
            weights = weights/sr*rho
    return weights, sr


def _spectral_radius(mat):
    """Calculates spectral radius of a matrix."""
    if isinstance(mat, float) or isinstance(mat, int):
        mat = np.array([[mat]])
    n = mat.shape[0]
    # r = spla.eigh(mat, eigvals_only=True, subset_by_index=(n-1)) # Once scipy is updated to 1.6.1
    # r = spla.eigh(mat, eigvals_only=True, eigvals=(n-1, n-1))
    r = max(np.linalg.eigvals(mat))
    return np.real(r)  # I'm not sure it's in the definition, but feels right in this context?