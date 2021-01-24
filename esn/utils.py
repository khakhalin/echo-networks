
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
        return