import numpy as np

@staticmethod
def make_graph(n_nodes=20, n_edges=None, network_type='erdos'):
    """Creates a graph dictionary.

    Parameters:
    n_nodes: number of nodes
    network_tpe: erdos, ws (for Watts-Strogatz)
    """
    if not n_edges:
        n_edges = n_nodes*2
    if network_type == 'erdos':
        gdict = _make_random(n_nodes, n_edges)
    elif network_type == 'ws':
        gdict = _make_ws(n_nodes, n_edges) # Actually there's also a beta parameter, but we park it for now
    else:
        raise ValueError('Unrecognized graph type.')
    return gdict

def _make_random(n, e=None):
    """Create Erdos graph."""
    edges = [(i,j) for j in range(n) for i in range(j)]
    edges = [edges[i] for i in np.random.choice(n*(n-1)//2, e, replace=False)]
    g = {i:list(set([j for k,j in edges if k==i] + [k for k,j in edges if j==i])) for i in range(n)}
    return g

def _make_ws(n, e=None, beta=0.3):
    """Makes a Watts-Strogatz network."""
    if e > n * (n - 1) / 2:
        e = n * (n - 1) // 2
    k = (e // n) * 2  # Average degree, rounded down, and forced to be even
    n_group1 = e - (k * n // 2)  # Number of elements with k+2 degree
    g = {i: [] for i in range(n)}
    for i in range(n):
        k1 = k // 2
        if i < n_group1:
            jlist = range(i - k1 - 1, i + k1 + 1)
        else:
            jlist = range(i - k1, i + k1 + 1)
        for j in jlist:
            jp = j % n
            if (jp) == i:
                continue
            if jp not in g[i]: g[i].append(jp)
            if i not in g[jp]: g[jp].append(i)

    for i in range(n):  # Rewire something for each node
        js = [j for j in g[i] if j > i]  # Those to the right are to be rewired
        for j in js:
            if np.random.uniform() < beta:  # Coin toss
                g[i].remove(j)  # Unwire
                k = i  # Set to a deliberately bad choice (self)
                while k == i or k == j or (k in g[i]):  # Draw while unhappy (self, old, or existing)
                    k = np.random.randint(n)
                g[i].append(k)
                g[k].append(i)
    return g