import numpy as np
import networkx as nx
from collections import defaultdict


def build_contact_graph(contact_points, type="rigid_body"):
    """ Create the contact graph based on the contact points for rigid body or soft body.
        The vertices of the graph are the contact points, and the edges of the graph are :
        1. If the type is rigid body, the edges are the contact points which belong to the same rigid body.
        2. If the type is soft body, the edges are the contact points which belong to the same tetrahedron.

    :param contact_points: The contact points array of the engine.
    :param type: The type of the body, 'rigid body' or 'soft body', defaults to "rigid_body"
    :raises ValueError: If the type of body is not supported.
    :return: The contact graph.
    """
    G = nx.Graph()
    K = len(contact_points) # The number of contact points.
    G.add_nodes_from(np.arange(K)) # Add the contact points as the vertices of the graph.

    for k in range(K):
        cp1 = contact_points[k]
        if type == "rigid_body":
            for i in range(k+1, K):
                cp2 = contact_points[i]
                if cp1.bodyA.idx == cp2.bodyA.idx or cp1.bodyA.idx == cp2.bodyB.idx or cp1.bodyB.idx == cp2.bodyA.idx or cp1.bodyB.idx == cp2.bodyB.idx:
                    G.add_edge(k, i)
        elif type == "soft_body":
            cp1_iA, cp1_jA, cp1_kA, cp1_mA = cp1.bodyA.T[cp1.idx_tetA] + cp1.bodyA.offset
            cp1_iB, cp1_jB, cp1_kB, cp1_mB = cp1.bodyB.T[cp1.idx_tetB] + cp1.bodyB.offset
            set1 = set([cp1_iA, cp1_jA, cp1_kA, cp1_mA, cp1_iB, cp1_jB, cp1_kB, cp1_mB])
            for i in range(k+1, K):
                cp2 = contact_points[i]
                cp2_iA, cp2_jA, cp2_kA, cp2_mA = cp2.bodyA.T[cp2.idx_tetA] + cp2.bodyA.offset
                cp2_iB, cp2_jB, cp2_kB, cp2_mB = cp2.bodyB.T[cp2.idx_tetB] + cp2.bodyB.offset
                set2 = set([cp2_iA, cp2_jA, cp2_kA, cp2_mA, cp2_iB, cp2_jB, cp2_kB, cp2_mB])
                if len(set1.intersection(set2)) > 0:
                    G.add_edge(k, i)
        else:
            raise ValueError("The type of body is not supported.")
    
    return G


def greedy_graph_coloring(G):
    """ Greedy graph coloring algorithm

    :param G: The contact grpah.
    :return: The color dictionary, the key is the color, the value is a array of the block location.
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
    """ Random graph coloring algorithm, which is posted as this paper: "Vivace: a Practical Gauss-Seidel Method for Stable Soft Body Dynamics". According to this paper, random graph coloring algorithm is faster than greedy graph coloring algorithm, and it can get a better result than greedy graph coloring algorithm; but I did not got the same result as the paper. In my test, greedy graph coloring algorithm is better than random graph coloring algorithm. I think the reason is that the random graph coloring algorithm is not stable, it can get different result in different test. So I did not use this algorithm in the final version of the code, however we can save this algorithm for future use (maybe can achieve better performance in different scene).

    :param G: The contact grpah.
    :param max_iter: The maximum iterations, defaults to 200.
    :return:  A color dictionary, the key is the contact index, the value is the color index.  

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
