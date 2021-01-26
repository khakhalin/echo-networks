import numpy as np


@staticmethod
def weights_in(n_nodes, mode='alternating'):
    """Creaes a vector of input weights.

    Parameters:
        n_nodes (int): number of nodes
        mode (string): what values input weights should take. Options:
            'alternating': alternating between 1 and -1
            'flat': all equal to +1
    """

    if mode == 'flat':
        return np.ones(n_nodes)
    if mode == 'alternating':
        return np.array([(i % 2)*2.0 - 1 for i in range(n_nodes)])
    raise ValueError('Unrecognized input weights type.')