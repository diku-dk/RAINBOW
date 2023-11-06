import numpy as np
import numba as nb
from rainbow.util.timer import Timer


@nb.njit(parallel=True, nogil=True, cache=True)
def sweep(K, w, z, b, mu, x, friction_solver, hybrid = True):
    """ Parallel Jacobi Proximal Algorithm.

    Args:
        K (int): Contact points number.
        w (ArrayLike): The WJT.dot(x).
        z (ArrayLike): The current contact force.
        b (ArrayLike): b = Ju^t + ∆t JM^{-1}h + EJu^t
        mu (float): The coefficient of friction.
        x (ArrayLike): The current contact force.
        friction_solver (callable): The friction cone function.
        hybrid (bool): If True, use the hybrid parallel Jacobi algorithm.

    Returns:
        ArrayLike: The new contact force.
    """
    if not hybrid:
        x_old = x.copy()
    for k in nb.prange(K):
        block = np.arange(4 * k, 4 * k + 4)
        mu_k = mu[k]  # Only isotropic Coulomb friction
        x_b = x[block]
        b_b = b[block]
        z_b = z[block]
        
        # Solve:         x_n = prox_{R^+}( x_n - r (A x_n + b) )
        x_b[0] = np.max(np.array([0.0, z_b[0]]))

        # Solve:         x_f = prox_C( x_f - r (A x_f + b))
        x_b_0 = x_b[0] if hybrid else x_old[block][0]
        x_b[1], x_b[2], x_b[3] = friction_solver(z_b[1], z_b[2], z_b[3], mu_k, x_b_0)
        # Put updated contact forces back into solution vector
        x[block] = x_b
    return x


def solve(J, WJT, b, mu, friction_solver, engine, stats, debug_on, prefix):
    """ Jacobi Proximal Solver, which just supports parallel computing.

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

    # Set number of threads for Numba
    nb.set_num_threads(4 if (nb.config.NUMBA_NUM_THREADS // 2) >= 4 else nb.config.NUMBA_NUM_THREADS // 2)
    
    for iteration in range(engine.params.max_iterations):
        w = WJT.dot(x)
        z = x - np.multiply(r, (J.dot(w) + b))
        hybrid = True if engine.params.proximal_solver['scheme'] == 'jacboi_hybrid' else False
        x = sweep(K, w, z, b, mu, x, friction_solver, hybrid)

        np.subtract(x, sol, error)
        merit = np.linalg.norm(error, np.inf)
        if debug_on:
            stats[prefix + "lambda"][iteration] = x
            stats[prefix + "residuals"][iteration] = merit
        # Test stopping criteria
        if merit < engine.params.absolute_tolerance:
            if debug_on:
                stats[prefix + "iterations"] = iteration+1
                stats[prefix + "exitcode"] = 1
                timer.end()
                stats[prefix + "solver_time"] = timer.elapsed
                # print("convergence_iteration1 = ", iteration+1)
            return x, stats
        if np.abs(merit - last_merit) < engine.params.relative_tolerance * last_merit:
            if debug_on:
                stats[prefix + "iterations"] = iteration+1
                stats[prefix + "exitcode"] = 2
                timer.end()
                stats[prefix + "solver_time"] = timer.elapsed
                # print("convergence_iteration2 = ", iteration+1)
            return x, stats

        # Update r-factors
        if merit > last_merit:
            # Divergence detected: reduce R-factor and roll-back solution to last known good iterate!
            np.multiply(nu_reduce, r,  r)
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