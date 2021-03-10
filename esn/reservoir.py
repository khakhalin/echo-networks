import numpy as np
import sklearn.linear_model as lm
from . import create_reservoir as creator


class Reservoir(object):
    """ Create a echo-networks model with the predefined tuning.

    Args:
        n_nodes (int): Number of processing neurons in internal reservoir
        n_edges (int): Number of edges. Default = n_nodes*2
        network_type (str): ws, erdos
        leak (float):  leak (aka alpha) for reservoir state update. Default=0.05
        rho (float): target spectral radius. Default=0.8. Set to None for no spectral rescaling.
        inhibition (str): alternating (default), distributed, none
    """

    def __init__(self, n_nodes=20, n_edges=None, network_type='ws',
                 leak=0.05, rho=0.9, l2=0.0,
                 inhibition='alternating', weights_in='alternating'):
        self.n_nodes = n_nodes
        self.network_type = network_type
        self.leak = leak
        self.l2 = l2             # Ridge regression l2 regularization

        if n_edges is None:
            n_edges = n_nodes*2 if n_nodes>3 else 2  # Heuristic that doesn't break for very small n_edges

        # Creator is a stateless all-static-methods utility class
        self.meta = {}
        self.graph = creator.make_graph(n_nodes, n_edges=n_edges, network_type=network_type)
        self.weights, spectral_radius = creator.graph_to_weights(self.graph, n_nodes, inhibition=inhibition, rho=rho)
        self.meta['original_rho'] = spectral_radius
        self.weights_in = creator.weights_in(n_nodes, weights_in)
        self.norm_input = None       # Inputs normalization: [mean std]
        self.weights_out = None      # Originally the model is not fit
        self.norm_out = None         # Output normalization: [intercept std]
        self.activation = creator.activation('tanh')

        self.state = np.zeros(n_nodes)

    def __str__(self):
        return f"Reservoir of {self.n_nodes} nodes, `{self.network_type}` type."


    def _forward(self, drive=None):
        """Make 1 step forward, update reservoir state.
        If input is not provided, perform self-generation."""
        if not drive:
            drive = self.state @ self.weights_out * self.norm_out[1] + self.norm_out[0]  # Try to self-drive
        self.state = (self.state * (1-self.leak) +
                      self.leak * self.activation((self.weights.T @ self.state) +
                                                   self.weights_in * drive))


    def run(self, input, n_steps=None):
        """Run the model several steps forward, driving it with input signal.

        Arguments:
            input (1D numpy): input signal
            n_steps (int): how many steps to run
            """
        if n_steps is None:
            n_steps = len(input)
        history = np.zeros((n_steps, self.n_nodes))
        self.state = np.zeros(self.n_nodes)
        for i_step in range(n_steps):
            if i_step < input.shape[0]:
                self._forward(input[i_step])
            else:
                self._forward()
            history[i_step, :] = self.state
        return history


    def fit(self, x, y, skip=None):
        """
        Fit model with input X and y
        Args:
            x (1d numpy array):  input series
            y (1d numpy array):  output series
            skip (int): how many first points to ignore (default: min(n_time / 4, n_nodes*4)

        Returns: a vector of output weights (that is also saved as weights_out in the self object).
        """
        if len(y.shape) == 1: # Simple vectors needs to be turned to column-vectors
            y = y[:, np.newaxis]
        if skip is None:
            skip = min(self.n_nodes*4 , len(y) // 4)
        self.norm_input = [np.mean(x), np.std(x)]
        self.norm_out = [np.mean(y), np.std(y)]
        history = self.run((x - self.norm_input[0]) / self.norm_input[1])
        if self.l2 is None:  # Simple regression
            self.weights_out = (np.linalg.pinv(history[skip:, :].T @ history[skip:, :]) @
                                (history[skip:, :].T @ (y[skip:] - self.norm_out[0])/self.norm_out[1])
                                )
        else:  # Ridge regression
            y_norm = (y - self.norm_out[0])/self.norm_out[1]
            clf = lm.Ridge(alpha=self.l2, fit_intercept=False)
            clf.fit(history, y_norm)  # <-        HERE THIS IS WRONG FOR NOW
            self.weights_out = clf.coef_.T
        return self      # In scikit-learn style, fit is supposed to return self


    def predict(self, x, length=None):
        """
        Args:
            x (numpy array): input signal
            n_steps (int): for how long to run

        Returns:
            y (numpy array): output
        """
        if length is None:
            length = len(x)
        if self.weights_out is None:
            raise Exception('The model needs to be fit first.')
        history = self. run((x - self.norm_input[0]) / self.norm_input[1], length)
        return (history @ self.weights_out * self.norm_out[1] + self.norm_out[0]).squeeze()
