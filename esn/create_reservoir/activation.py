import numpy as np

def activation(name='tanh'):
    """Returns an activation function."""
    if name == 'tanh':
        fun = np.tanh
    else:
        raise ValueError('Unknown activation function.')
    return fun