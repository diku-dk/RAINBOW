import numpy as np
import numba as nb
import networkx as nx
from collections import defaultdict


@nb.njit(parallel=True, nogil=True, cache=True)
def _get_edges(J):
    """ Get the edges of the contact graph.

    Args:
        J (ArrayLike): The contact jacobi matrix.

    Returns:
        (ArrayLike, ArrayLike): The source and target of the edges.
    """
    K = J.shape[0] // 4 # The number of contact points, here the each contact point has 4 rows of J.
    step_size = 24 # Here the step_size is 24 , because each contact point has 8 non-zero blocks and each block has 3 columns.
    cols = J.shape[1] # The number of columns of J.
    
    # Preallocate arrays for source and target of edges
    max_possible_edges = (cols // step_size) * K * K
    source = np.full(max_possible_edges, -1, dtype=np.int64)
    target = np.full(max_possible_edges, -1, dtype=np.int64)
    edge_count = 0

    for idx in nb.prange(0, cols//step_size):
        col_idx = idx * step_size
        contacts = np.where(J[:, col_idx: col_idx+step_size])[0]
        contacts = np.unique(contacts // 4)
        for i, contact_i in enumerate(contacts):
            for j, contact_j in enumerate(contacts):
                if i < j:
                    source[edge_count] = contact_i
                    target[edge_count] = contact_j
                    edge_count += 1

    # Trim the arrays to the actual size
    valid_indices = source != -1
    filtered_source = source[valid_indices]
    filtered_source = target[valid_indices]

    return filtered_source, filtered_source


def build_contact_graph(J):
    """ Create the contact graph based on the contact jacobi matrix.
        The vertices of the graph are the contact points, and the edges of the graph are the value of the contact jacobi matrix is not zero between two contact points.

    Args:
        J (ArrayLike): The contact jacobi matrix.

    Returns:
        graph: The contact graph.
    """
    G = nx.Graph()
    K = J.shape[0] // 4  # The number of contact points, here the each contact point has 4 rows of J.
    G.add_nodes_from(np.arange(K)) # Add the contact points as the vertices of the graph.
    sources, targets = _get_edges(J)
    for s, t in zip(sources, targets):
        G.add_edge(s, t)
    
    return G


def greedy_graph_coloring(G):
    """ Greedy graph coloring algorithm

    Args:
        G (Graph): The contact grpah is created from the contact jacobi matrix.

    Returns:
        ArrayLike: The color dictionary, the key is the color, the value is a array of the block location.
    """
    C = nx.coloring.greedy_color(G)
    color_groups = defaultdict(list)
    sorted_C = dict(sorted(C.items()))

    for k, color in sorted_C.items():
        block_start = 4 * k # The start index of a block
        block_end = block_start + 4 # The end index of a block
        color_groups[color].append((block_start, block_end))
    return color_groups


def random_graph_coloring(G, max_iter = 200):
    """ Random graph coloring algorithm, which is posted as this paper: "Vivace: a Practical Gauss-Seidel Method for Stable Soft Body Dynamics"
        
    Note: According to this paper, random graph coloring algorithm is faster than greedy graph coloring algorithm, and it can get a better result 
            than greedy graph coloring algorithm; but I did not got the same result as the paper. In my test, greedy graph coloring algorithm is better than random graph coloring algorithm.
            I think the reason is that the random graph coloring algorithm is not stable, it can get different result in different test.
            So I did not use this algorithm in the final version of the code, however we can save this algorithm for future use (maybe can achieve better performance in different scene).
        
    Args:
        G (Graph): The contact grpah is created from the contact jacobi matrix.
        max_iter (int, optional): The maximum iterations . Defaults to 200.

    Returns:
        dict: A color dictionary, the key is the contact index, the value is the color index.  
    """
    degrees = np.array([degree for _, degree in G.degree()])
    colors = (degrees // 1).astype(int)
    palettes = [set(np.arange(c)) for c in colors]
    palette_dict = {node: palette for node, palette in zip(G.nodes(), palettes)}

    U = set(G.nodes())
    C = defaultdict(int)
    iter_count = 0
    while len(U) > 0 and iter_count < max_iter:
        for v in U:
            if palette_dict[v]:
                C[v] = np.random.choice(list(palette_dict[v]))

        I = set()
        for v in U:
            neighbors_colors = {C[u] for u in G.neighbors(v) if u in U}
            if C[v] not in neighbors_colors:
                I.add(v)
                for u in G.neighbors(v):
                    if u in U:
                        palette_dict[u].discard(C[v])
        U -= I
        max_used_color = max(C.values(), default=-1)
        for v in U:
            if not palette_dict[v]:
                max_used_color += 1
                palette_dict[v].add(max_used_color)
        
        iter_count += 1
        
    if U:
        max_used_color = max(C.values(), default=-1)
        for v in U:
            max_used_color += 1
            C[v] = max_used_color  

    color_groups = defaultdict(list)
    sorted_C = dict(sorted(C.items()))
    for key, value in sorted_C.items():
        color_groups[value].append(key)

    return color_groups
