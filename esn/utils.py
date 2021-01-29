import networkx as nx
import matplotlib.pyplot as plt


def plot_data(x, y, title='Lorenz attractor'):
    if not x.ndim == 1:
       raise ValueError("Argument x_array should be 1 dimensional. "
                         "It actually is {0} dimensional".format(x.ndim))
    if not y.ndim == 1:
        raise ValueError("Argument y_array should be 1 dimensional. "
                         "It actually is {0} dimensional".format(y.ndim))
    #if len(x) != len(y):
    #    raise RuntimeError("Arguments x_array and y_array should have same length. "
    #                       "But x_array has length {0} and y_array has length {1}".format(len(x),
    #                                                                                      len(y)
    #                                                                                      )
    #                       )
    fig, ax = plt.subplots()
    ax.plot(x, y, ".")
    ax.set(xlabel="X-comp", ylabel="Z-comp", title=title)
    return fig
# Static class, never initialized

class utils():
    """A collection of relevant utilities."""

    @staticmethod
    def plot_graph(graph_dictionary):
        """Utility: plots a graph from a gdict."""
        G = nx.Graph()
        for node, edges in graph_dictionary.items():
            for other in edges:
                G.add_edge(node, other)  # Undirected graph, so the order doesn't matter
        nx.draw_kamada_kawai(G, node_size=30)
        return G

