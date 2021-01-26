
class creator(object):
    """A class of static methods for creating and tuning reservoirs."""

    from ._make_graph import make_graph
    from ._graph_to_weights import graph_to_weights
    from ._activation import activation

    # Usage:
    # self.graph = creator.make_graph(n_nodes, network_type)
    # self.weights = creator.graph_to_weights(self.graph, inhibition='distributed')
    # self.weights_in = creator.weights_in(self.n_nodes)
    # self.weights_out = creator.weights_out(self.n_nodes)
    # self.activation = creator.activation('tanh')