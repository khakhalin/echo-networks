class Reservoir:
    """ Create basic echo-networks model with the predefined internal state

    Args:
        n_internal (int): Number of processing neurons in internal reservoir
        leak (float):  leakage for reservoir state update


    """

    def __init__(self, n_nodes=20, leak=None, network_type=None):
        self.n_nodes = n_nodes
        # probability of edge existence
        self.leak = leak
        self._weights = None
        self.weights_in = None
        self.weights_out = None
        self.network_type = network_type
        self.graph_dict = None


    @property
    def weights(self):
        return self._weights

    @weights.setter
    def internal_weights(self):
        pass
    def set_weights_in(self):
        #we need init weights
        pass


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

    def forward(self):
        #return reservoir state
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
