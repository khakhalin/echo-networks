import numpy
from ._create_reservoir import creator

class Reservoir(object):
    """ Create a echo-networks model with the predefined tuning.

    Args:
        n_nodes (int): Number of processing neurons in internal reservoir
        leak (float):  leakage for reservoir state update
    """

    def __init__(self, n_nodes=20, network_type=None, leak=None):
        self.n_nodes = n_nodes
        self.network_type = network_type
        self.leak = leak

        self.graph = creator.make_graph(n_nodes, network_type)
        # self.weights = creator.graph_to_weights(self.graph, inhibition='distributed')
        # self.weights_in = creator.weights_in(self.n_nodes)
        # self.weights_out = creator.weights_out(self.n_nodes
        # self.activation = creator.activation('tanh')

        # I'm actually not sure here. It feels like creator would benefit from knowing everything
        # about SELF, like network type and what not. Maybe make creator a mix-in, and inherit to it?
        # Or is there a benefit in keeping these

    def fit(self, x, y):
        """
        Fit model with input X and y
        Args:
            x (1d numpy array):  input series
            y (1d numpy array):  output series

        Returns:

        TODO:
        fit
        """

        pass

    def run(self, n_steps):
        # Run n_steps forward
        pass

    def forward(self):
        # Make 1 step forward, update reservoir state
        pass


    def predict(self, x):
        """

        Args:
            x (numpy array): input series

        Returns:
            y (numpy array): output

        """
        #Error of model is not fitted, else
        self.forward()
        #y =
        return y
