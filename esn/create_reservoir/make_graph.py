import numpy as np

def make_graph(n_nodes=20, n_edges=None, network_type='erdos'):
    """Creates a graph dictionary.

    Parameters:
    n_nodes: number of nodes
    n_edges: number of edges (optinal)
    network_tpe: 'erdos', 'ws' (for Watts-Strogatz; default)
    """
    if n_edges is None:
        n_edges = n_nodes*2
    if network_type == 'erdos':
        graph_dict = _make_erdos(n_nodes, n_edges)
    elif network_type == 'ws':
        graph_dict = _make_ws(n_nodes, n_edges) # Actually there's also a beta parameter, but we park it for now
    else:
        raise ValueError('Unrecognized graph type.')
    return graph_dict

def _make_erdos(n, e):
    """Create Erdos graph with N nodes and E edges."""
    edges = [(i,j) for j in range(n) for i in range(n) if i != j]
    edges = [edges[i] for i in np.random.choice(len(edges), e, replace=False)]
    g = {i:list(set([j for k,j in edges if k==i])) for i in range(n)}
    return g

def _make_ws(n, e, beta=0.5):
    """Makes an oriented Watts-Strogatz network with N nodes, E edges, and beta rewiring."""
    if e > n*(n - 1):
        e = n*(n - 1)             # Max possible number of edges for a graph of N nodes
    degree = (e // n)                  # Average out-degree, rounded down
    n_with_extra_edge = e - (degree * n)    # Number of elements with k+2 degree
    g = {i: [] for i in range(n)} # Empty graph for now

    for i in range(n): # First, create a ring
        edges_left  = degree // 2
        edges_right = degree - edges_left
        if i < n_with_extra_edge:
            jlist = range(i - edges_left, i + edges_right + 2)
        else:
            jlist = range(i - edges_left, i + edges_right + 1)
        for j in jlist:
            if j == i:    # Don't connect to itself
                continue
            jp = j % n # Loop around the ring
            if jp not in g[i]: g[i].append(jp)

    # Now rewire edges:
    for i in range(n):                      # For every node in the graph
        js = [j for j in g[i] if (j-i) % n < (n // 2)]     # Only Those to the right are to be rewired
        for j in js:                        # For every edge on the right
            if np.random.uniform() < beta:  # Toss a weighted coin if this edge needs to be rewired
                k = i           # New edge destination; set to a deliberately bad choice (self)
                while k == i or (k in g[i]):  # Draw while unhappy (self, or existing)
                    k = np.random.randint(n)
                    # Note that with high enough e, we'll get an infinite loop here,
                    # as rewiring will be impossible.
                g[i].remove(j)  # Unwire
                g[i].append(k)
    return g