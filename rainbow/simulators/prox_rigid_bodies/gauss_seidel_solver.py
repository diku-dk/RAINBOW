import numpy as np
from rainbow.util.timer import Timer


def prox_sphere(z_s, z_t, z_tau, mu, x_n):
    """
    Proximal point of z to a sphere.

    :param z_s:       s-component of current z-point.
    :param z_t:       t-component of current z-point.
    :param z_tau:     tau-component of current z-point.
    :param mu:        The coefficient of friction.
    :param x_n:       The current normal force magnitude.
    :return:          The new z-point which will be the closest point to the sphere with radius my*x_n.
    """
    if x_n <= 0.0:
        return 0.0, 0.0, 0.0
    radius = mu * x_n
    sqr_z_norm = z_s * z_s + z_t * z_t + z_tau * z_tau
    if sqr_z_norm <= radius * radius:
        return z_s, z_t, z_tau
    scale = radius / np.sqrt(sqr_z_norm)
    return z_s * scale, z_t * scale, z_tau * scale


def prox_origin(z_s, z_t, z_tau, mu, x_n):
    """
    Proximal point of z to the origin.

    :param z_s:       s-component of current z-point.
    :param z_t:       t-component of current z-point.
    :param z_tau:     tau-component of current z-point.
    :param mu:        The coefficient of friction.
    :param x_n:       The current normal force magnitude.
    :return:          The new z-point which will be the closest point to the sphere with radius my*x_n.
    """
    return 0.0, 0.0, 0.0


def prox_ellipsoid(engine, z_s, z_t, z_tau, mu_s, mu_t, mu_tau, x_n):
    # 2022-05-09 Kenny TODO: This interface is currently not compatible with
    #                   the friction_solver interface that we assume in the
    #                   sweep function.
    """

    :param engine:
    :param z_s:
    :param z_t:
    :param z_tau:
    :param mu_s:
    :param mu_t:
    :param mu_tau:
    :param x_n:
    :return:
    """
    if x_n <= 0.0:
        return 0.0, 0.0, 0.0

    # If surface friction is frictionless, just return prox origin
    if mu_s == 0 and mu_t == 0 and mu_tau == 0:
        return prox_origin(z_s, z_t, z_tau, mu_s, x_n)

    if mu_s == mu_t and mu_s == mu_tau:
        return prox_sphere(z_s, z_t, z_tau, mu_s, x_n)

    a = mu_s * x_n
    b = mu_t * x_n
    c = mu_tau * x_n

    scale = 1 / max(1, a, b, c, abs(z_s), abs(z_t), abs(z_tau))
    sa, sb, sc, sx, sy, sz = (
        scale * a,
        scale * b,
        scale * c,
        scale * z_s,
        scale * z_t,
        scale * z_tau,
    )

    # Precompute squared values
    saa, sbb, scc, sxx, syy, szz = sa * sa, sb * sb, sc * sc, sx * sx, sy * sy, sz * sz

    # Check if point is already on or inside ellipsoid
    f0 = (sxx / saa) + (syy / sbb) + (szz / scc) - 1
    if f0 < engine.params.ellipsoid_tolerance:
        return z_s, z_t, z_tau

    t0 = 0
    t1 = max(sa, sb, sc) * np.linalg.norm([sx, sy, sz])
    g0 = (
        (saa * sxx) / ((saa + t0) ** 2)
        + (sbb * syy) / ((sbb + t0) ** 2)
        + (scc * szz) / ((scc + t0) ** 2)
        - 1
    )
    g1 = (
        (saa * sxx) / ((saa + t1) ** 2)
        + (sbb * syy) / ((sbb + t1) ** 2)
        + (scc * szz) / ((scc + t1) ** 2)
        - 1
    )
    while g1 > 0:
        t1 *= engine.params.ellipsoid_expansion
        g1 = (
            (saa * sxx) / ((saa + t1) ** 2)
            + (sbb * syy) / ((sbb + t1) ** 2)
            + (scc * szz) / ((scc + t1) ** 2)
            - 1
        )

    tk = (t0 + t1) * 0.5
    for iteration in range(engine.params.ellipsoid_max_iterations):
        # Stagnation test
        if abs(t1 - t0) < engine.params.ellipsoid_tolerance:
            break
        gk = (
            (saa * sxx) / ((saa + tk) ** 2)
            + (sbb * syy) / ((sbb + tk) ** 2)
            + (scc * szz) / ((scc + tk) ** 2)
            - 1
        )
        # Absolute convergence test
        if abs(gk) < engine.params.ellipsoid_tolerance:
            break
        if gk > 0:
            t0 = tk
        else:
            t1 = tk
        tk = (t0 + t1) * 0.5

    tk /= scale * scale

    x_s = (a * a * z_s) / (a * a + tk)
    x_t = (b * b * z_t) / (b * b + tk)
    x_tau = (c * c * z_tau) / (c * c + tk)

    return x_s, x_t, x_tau


def sweep(K, J, WJT, b, mu, r, x, friction_solver, engine):
    """

    :param K:
    :param J:
    :param WJT:
    :param b:
    :param mu:
    :param r:
    :param x:
    :param friction_solver:
    :param engine:
    :return:
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
        # Updating w so it reflect the change in x, remember w = WJT delta
        # TODO 2020-08-17 Kristian: WJT is in bsr matrix format, which does not support indexing and we can therefore
        #  not access the block sub-matrix. Currently we circumvent this by converting it to a csr matrix instead,
        #  however another solution might be better.
        w += WJT.tocsr()[:, block].dot(delta)
    return x


def solve(J, WJT, b, mu, friction_solver, engine, stats, debug_on, prefix):
    """

    :param J:
    :param WJT:
    :param b:
    :param mu:
    :param friction_solver:
    :param engine:
    :param stats:
    :param debug_on:
    :param prefix:
    :return:
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
    r = 1.0 / delassus_diag

    # Extract parameter values for controlling the adaptive r-factor strategy
    nu_reduce = engine.params.nu_reduce
    nu_increase = engine.params.nu_increase
    too_small_merit_change = engine.params.too_small_merit_change

    last_merit = np.Inf

    for iteration in range(engine.params.max_iterations):
        x = sweep(K, J, WJT, b, mu, r, x, friction_solver, engine)

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
