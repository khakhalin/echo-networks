import numpy as np
from . import create_reservoir as creator


class Reservoir(object):
    """ Create a echo-networks model with the predefined tuning.

    Args:
        n_nodes (int): Number of processing neurons in internal reservoir
        network_type (str): ws, erdos
        leak (float):  leakage for reservoir state update. Default=0.95
        alpha (float): integration step. Default=0.05
        inhibition (str): alternating (default), distributed, none
    """

    def __init__(self, n_nodes=20, network_type='ws', leak=0.95, alpha=0.05, inhibition='alternating'):
        self.n_nodes = n_nodes
        self.network_type = network_type
        self.leak = leak
        self.alpha = alpha

        # Creator is a stateless all-static-methods utility class
        self.graph = creator.make_graph(n_nodes, network_type=network_type)
        self.weights = creator.graph_to_weights(self.graph, n_nodes, inhibition=inhibition)
        self.weights_in = creator.weights_in(n_nodes)
        self.input_norm = None       # Inputs should be normalized
        self.weights_out = None      # Originally the model is not fit
        self.bias_out = None         # Intercept.
        self.activation = creator.activation('tanh')

        self.state = np.zeros(n_nodes)

    def __str__(self):
        return f"Reservoir of {self.n_nodes} nodes, `{self.network_type}` type."


    def _forward(self, drive=None):
        """Make 1 step forward, update reservoir state.
        If input is not provided, perform self-generation."""
        if not drive:
            drive = self.state @ self.weights_out.T + self.bias_out  # Try to self-drive
        self.state = (self.state * self.leak +
                      self.alpha * self.activation((self.weights.T @ self.state) +
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
        self.input_norm = [np.mean(x), np.std(x)]
        self.bias_out = np.mean(y)
        history = self.run((x - self.input_norm[0])/self.input_norm[1])
        self.weights_out = (((y[skip:].T - self.bias_out) @ history[skip:, :]) @
                            np.linalg.pinv(history[skip:, :].T @ history[skip:, :]))
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
        history = self. run((x - self.input_norm[0])/self.input_norm[1], length)
        return (history @ self.weights_out.T + self.bias_out).squeeze()
