import networkx as nx
import matplotlib.pyplot as plt


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

    @staticmethod
    def plot_data(x, y, title='Lorenz attractor'):
        """AKH NOTE: I have no idea why we would need this. But keeping for now."""
        if not x.ndim == 1:
            raise ValueError("Argument x_array should be 1 dimensional. "
                             "It actually is {0} dimensional".format(x.ndim))
        if not y.ndim == 1:
            raise ValueError("Argument y_array should be 1 dimensional. "
                             "It actually is {0} dimensional".format(y.ndim))
        fig, ax = plt.subplots()
        ax.plot(x, y, ".")
        ax.set(xlabel="X-comp", ylabel="Z-comp", title=title)
        return fig