import numpy as np
from . import create_reservoir as creator


class Reservoir(object):
    """ Create a echo-networks model with the predefined tuning.

    Args:
        n_nodes (int): Number of processing neurons in internal reservoir
        leak (float):  leakage for reservoir state update
    """

    def __init__(self, n_nodes=20, network_type='ws', leak=0.95, alpha=0.05):
        self.n_nodes = n_nodes
        self.network_type = network_type
        self.leak = leak
        self.alpha = alpha

        # Creator is a stateless all-static-methods utility class
        self.graph = creator.make_graph(n_nodes, network_type=network_type)
        self.weights = creator.graph_to_weights(self.graph, n_nodes, inhibition='alternating')
        self.weights_in = creator.weights_in(n_nodes)
        self.weights_out = np.zeros(n_nodes)  # We could put it in a function, but why?
        self.activation = creator.activation('tanh')

        self.state = np.zeros(n_nodes)

    def __str__(self):
        return f"Reservoir of {self.n_nodes} nodes, `{self.network_type}` type."

    def fit(self, x, y):
        """
        Fit model with input X and y
        Args:
            x (1d numpy array):  input series
            y (1d numpy array):  output series

        Returns: a vector of output weights (that is also saved as weights_out in the self object).
        """
        # x and y are supposed to be column-vectors
        x = x[:, np.newaxis]
        x = x[:, np.newaxis]
        self.weights_out = (y.T @ x) @ np.linalg.pinv(x.T @ x)
        return self.weights_out


    def _forward(self, drive=None):
        """Make 1 step forward, update reservoir state.
        If input is not provided, perform self-generation."""
        if not drive:
            drive = self.state.T @ self.weights_out
        self.state = (self.state * self.leak +
                      self.alpha * self.activation((self.weights.T @ self.state) +
                                                   self.weights_in * drive))


    def run(self, input, n_steps=None):
        """Run the model several steps forward, driving it with input signal.

        Arguments:
            n_steps (int): how many steps to run
            input (1D numpy): input to drive it with
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

    def predict(self, x):
        """
        Args:
            x (numpy array): input series

        Returns:
            y (numpy array): output

        """
        # Error if model is not fitted, else
        self.forward()
        # y =
        return y
