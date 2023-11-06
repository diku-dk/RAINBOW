import numpy as np
import numba as nb
import networkx as nx
from collections import defaultdict
from rainbow.util.timer import Timer


def _build_contact_graph(J):
    """ Create the contact graph based on the contact jacobi matrix.
        The vertices of the graph are the contact points, and the edges of the graph are the value of the contact jacobi matrix is not zero between two contact points.

    Args:
        J (ArrayLike): The contact jacobi matrix.

    Returns:
        graph: The contact graph.
    """
    G = nx.Graph()
    K = J.shape[0] // 4
    G.add_nodes_from(np.arange(K))
    sources, targets = _get_edges(J)
    for s, t in zip(sources, targets):
        G.add_edge(s, t) 
    
    return G


@nb.njit(parallel=True, nogil=True, cache=True)
def _get_edges(J):
    """ Get the edges of the contact graph.

    Args:
        J (ArrayLike): The contact jacobi matrix.

    Returns:
        (ArrayLike, ArrayLike): The source and target of the edges.
    """
    K = J.shape[0] // 4
    step_size = 24
    cols = J.shape[1]
    
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


def _greedy_graph_coloring(G):
    """ Greedy graph coloring algorithm

    Args:
        G (Graph): The contact grpah is created by on the contact jacobi matrix.

    Returns:
        ArrayLike: The color dictionary, the key is the color, the value is a array of the block location.
    """
    C = nx.coloring.greedy_color(G)
    color_groups = defaultdict(list)
    sorted_C = dict(sorted(C.items()))

    for k, color in sorted_C.items():
        block_start = 4 * k
        block_end = block_start + 4
        color_groups[color].append((block_start, block_end))
    return color_groups


def _random_graph_coloring(G, max_iter = 200):
    """ Random graph coloring algorithm, which is posted as this paper: "Vivace: a Practical Gauss-Seidel Method for Stable Soft Body Dynamics"
        
    Note: According to this paper, random graph coloring algorithm is faster than greedy graph coloring algorithm, and it can get a better result 
            than greedy graph coloring algorithm; but I did not got the same result as the paper. In my test, greedy graph coloring algorithm is better than random graph coloring algorithm.
            I think the reason is that the random graph coloring algorithm is not stable, it can get different result in different test.
            So I did not use this algorithm in the final version of the code, however we can save this algorithm for future use (maybe can achieve better performance in different scene).
        
    Args:
        G (Graph): The contact grpah is created by on the contact jacobi matrix.
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

    return C


@nb.njit(parallel=True, nogil=True, cache=True)
def sweep_worker(color_group, J, WJT, b, mu, r, x, w, delta_ws, friction_solver):
    """ The worker function of the parallel gauss seidel algorithm.

    Args:
        color_group (ArrayLike): The color group, the value is  the block location, containing the start and end index.
        J (ArrayLike): The contact jacobi matrix.
        WJT (ArrayLike): The WJ^T matrix, here W = M^{-1}, and the M is the mass matrix.
        b (ArrayLike): b = Ju^t + ∆t JM^{-1}h + EJu^t
        mu (float): The coefficient of friction.
        r (float): r-factor value.
        x (ArrayLike): The current contact force.
        w (ArrayLike): The WJT.dot(x).
        delta_ws (ArrayLike): A array of the delta_w, each delta_w = WJT.dot(delta_x), here delta_x is the change of the contact force.
        friction_solver (callable): The friction cone function.

    Returns:
        (ArrayLike, ArrayLike): The new contact force and the new delta_ws.
    """
    for i in nb.prange(len(color_group)):
        block_start, block_end = color_group[i]
        block = np.arange(block_start, block_end)
        x_b = x[block]
        r_b = r[block]
        b_b = b[block]

        delta = x_b.copy()
        z_b = x_b - np.multiply(r_b, (J.dot(w)[block] + b_b))

        x_b[0] = np.max(np.array([0.0, z_b[0]]))

        mu_k = mu[block_start // 4]
        x_b[1], x_b[2], x_b[3] = friction_solver(z_b[1], z_b[2], z_b[3], mu_k, x_b[0])

        np.subtract(x_b, delta, delta)
        delta_w = WJT[:, block].dot(delta)

        x[block] = x_b
        delta_ws[block_start//4] = delta_w

    return x, delta_ws


def sweep_parallel(K, J, WJT, b, mu, r, x, friction_solver):
    """ Parallel Gauss-Seidel algorithm:
        This method aims to parallelize the Gauss-Seidel algorithm by dividing contact points into different groups based on graph coloring. Each group, representing a unique color, is processed in parallel within itself, but serially across groups. Here are the detailed steps:

        1. Construct the contact graph: Vertices of this graph represent contact points, and edges indicate non-zero values in the contact Jacobian matrix between two such points. For more information, refer to: https://dspace5.zcu.cz/bitstream/11025/11058/1/Lazarevych.pdf.

        2. Graph Coloring: Utilize the graph coloring algorithm to segregate contact points into distinct groups (colors). We experimented with both the Random Graph Coloring and Greedy Graph Coloring algorithms. However, the Greedy Graph Coloring proved superior, hence our choice.

        3. Parallel Gauss Seidel computation: Each color group's contact force can be computed in parallel. However, computations for each color group are handled serially.


        Why we can parallel the gauss seidel algorithm?
        The origin idea is easy to understand why we can parallel the gauss seidel algorithm, if the two rows of the contact jacobi matrix are independent, then we can compute the new contact force of these two rows in parallel, the indepandent rows defined the value of the contact jacobi matrix is zero between these two contact points. We can move to a easy case of linear system as follows:
        Ax = b
        [[1, 1, 0, 0],     [x_1,         [1,
         [2, 2, 2, 0],      x_2,    =     2,
         [0, 1, 1, 1],      x_3,          3,
         [0, 0, 2, 2]]      x_4]          4]
         We can find A_{0,3} = 0 and A_{2,0} = 0, so we can compute x_1 and x_3 in parallel.
         

    Args:
        K (int): Contact points number.
        J (ArrayLike): The contact jacobi matrix.
        WJT (ArrayLike): The WJ^T matrix, here W = M^{-1}.
        b (ArrayLike): b = Ju^t + ∆t JM^{-1}h + EJu^t
        mu (float): The coefficient of friction.
        r (float): r-factor value.
        x (ArrayLike): The current contact force.
        friction_solver (callable): The friction cone function.

    Returns:
        ArrayLike: The new contact force.
    """
    color_groups = defaultdict(list)
    G = _build_contact_graph(J.toarray())
    color_groups = _greedy_graph_coloring(G)
    nb.set_num_threads(4 if (nb.config.NUMBA_NUM_THREADS // 2) >= 4 else nb.config.NUMBA_NUM_THREADS // 2)

    w0 = WJT.dot(x)
    delta_ws = np.zeros((WJT.shape[1]//4, w0.shape[0]), dtype=np.float64)
    for color_group in color_groups.values():
        w = w0.copy()
        block_start, _ = color_group[0]
        for i in range(block_start // 4):
            w += delta_ws[i]
        x, delta_ws = sweep_worker(np.array(color_group), J.toarray(), WJT.toarray(), b, mu, r, x, w, delta_ws, friction_solver)

    return x


def sweep(K, J, WJT, b, mu, r, x, friction_solver):
    """ Serial Gauss Seidel algorithm.

    Args:
        K (int): Contact points number.
        J (ArrayLike): The contact jacobi matrix.
        WJT (ArrayLike): The WJ^T matrix, here W = M^{-1}.
        b (ArrayLike): b = Ju^t + ∆t JM^{-1}h + EJu^t
        mu (float): The coefficient of friction.
        r (float): r-factor value.
        x (ArrayLike): The current contact force.
        friction_solver (callable): The friction cone function.

    Returns:
        ArrayLike: The new contact force.
    """
    w = WJT.dot(x)
    for k in range(K):
        block = range(4 * k, 4 * k + 4)
        mu_k = mu[k]  # Only isotropic Coulomb friction
        x_b = x[block]
        delta = (
            x_b.copy()
        )  # Used to keep the old values and compute the change in values
        r_b = r[block]
        b_b = b[block]

        # By definition
        #       z = x - r (J WJ^T x  + b)
        #         = x - r ( A x  + b)
        # We use
        #        w =  WJ^T x
        # so
        #       z  = x - r ( J w  + b)
        z_b = x_b - np.multiply(r_b, (J.dot(w)[block] + b_b))

        # Solve:         x_n = prox_{R^+}( x_n - r (A x_n + b) )
        x_b[0] = np.max([0.0, z_b[0]])

        # Solve:         x_f = prox_C( x_f - r (A x_f + b))
        x_b[1], x_b[2], x_b[3] = friction_solver(z_b[1], z_b[2], z_b[3], mu_k, x_b[0])
        # Put updated contact forces back into solution vector
        x[block] = x_b
        # Get the change in the x_block
        np.subtract(x_b, delta, delta)
        # Update w
        w += WJT.tocsr()[:, block].dot(delta)
    return x


def solve(J, WJT, b, mu, friction_solver, engine, stats, debug_on, prefix):
    """ Gauss Seidel Proximal Solver, which supports parallel computing and serial computing.

    Args:
        J (ArrayLike): The contact jacobi matrix.
        WJT (ArrayLike): The WJ^T matrix, here W = M^{-1}.
        b (ArrayLike): b = Ju^t + ∆t JM^{-1}h + EJu^t
        mu (float): The coefficient of friction.
        friction_solver (callable): The friction cone function.
        engine (Object): The engine object.
        stats (dict): The statistics information.
        debug_on (boolean): Whether to debug.
        prefix (string): The prefix of the statistics information.

    Returns:
        (ArrayLike, dict): The new contact force and the statistics information.
    """
    timer = None
    if debug_on:
        timer = Timer("Gauss Seidel")
        stats[prefix + "residuals"] = (
            np.ones(engine.params.max_iterations, dtype=np.float64) * np.inf
        )
        stats[prefix + "lambda"] = np.zeros(
            [engine.params.max_iterations] + list(b.shape), dtype=np.float64
        )
        stats[prefix + "reject"] = np.zeros(engine.params.max_iterations, dtype=bool)
        stats[prefix + "exitcode"] = 0
        stats[prefix + "iterations"] = engine.params.max_iterations
        timer.start()

    K = len(engine.contact_points)
    x = np.zeros(b.shape, dtype=np.float64)  # The current iterate
    sol = np.zeros(
        b.shape, dtype=np.float64
    )  # The last best known solution, used for restarting if divergence
    error = np.zeros(b.shape, dtype=np.float64)  # The residual vector

    # Compute initial r-factor value
    delassus_diag = np.sum(J.multiply(WJT.T), axis=1).A1
    delassus_diag[delassus_diag == 0] = 1
    r = 0.1 / delassus_diag

    # Extract parameter values for controlling the adaptive r-factor strategy
    nu_reduce = engine.params.nu_reduce
    nu_increase = engine.params.nu_increase
    too_small_merit_change = engine.params.too_small_merit_change

    last_merit = np.Inf    

    for iteration in range(engine.params.max_iterations):
        if engine.params.proximal_solver['parallel']:
            x = sweep_parallel(K, J, WJT, b, mu, r, x, friction_solver)
                # w = delta_w
        else:
            x = sweep(K, J, WJT, b, mu, r, x, friction_solver)

        np.subtract(x, sol, error)
        merit = np.linalg.norm(error, np.inf)
        if debug_on:
            stats[prefix + "lambda"][iteration] = x
            stats[prefix + "residuals"][iteration] = merit
        # Test stopping criteria
        if merit < engine.params.absolute_tolerance:
            if debug_on:
                stats[prefix + "iterations"] = iteration
                stats[prefix + "exitcode"] = 1
                timer.end()
                stats[prefix + "solver_time"] = timer.elapsed
            return x, stats
        if np.abs(merit - last_merit) < engine.params.relative_tolerance * last_merit:
            if debug_on:
                stats[prefix + "iterations"] = iteration
                stats[prefix + "exitcode"] = 2
                timer.end()
                stats[prefix + "solver_time"] = timer.elapsed
            return x, stats

        # Update r-factors
        if merit > last_merit:
            # Divergence detected: reduce R-factor and roll-back solution to last known good iterate!
            np.multiply(nu_reduce, r, r)
            np.copyto(x, sol)
            if debug_on:
                stats[prefix + "reject"][iteration] = True
        else:
            if last_merit - merit < too_small_merit_change:
                # Convergence is slow: increase r-factor
                np.multiply(nu_increase, r, r)
            # Convergence detected: accept x as better solution
            last_merit = merit
            np.copyto(sol, x)

    # If this point of the code is reached then it means the method did not converge within the given iterations.
    if debug_on:
        timer.end()
        stats[prefix + "solver_time"] = timer.elapsed
    return sol, stats
