import numpy as np
import numba as nb


@nb.njit(cache=True)
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


@nb.njit(cache=True)
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


@nb.njit(cache=True)
def prox_ellipsoid(engine, z_s, z_t, z_tau, mu_s, mu_t, mu_tau, x_n):
    # 2022-05-09 Kenny TODO: This interface is currently not compatible with
    #                   the friction_solver interface that we assume in the
    #                   sweep function.
    """

    :param engine:
    :param z_s: s-component of current z-point.
    :param z_t: t-component of current z-point.
    :param z_tau: tau-component of current z-point.
    :param mu_s: The coefficient of friction.
    :param mu_t: The coefficient of friction.
    :param mu_tau: The coefficient of friction.
    :param x_n:  The current normal force magnitude.
    :return: The x_s, x_t, x_tau which will be the closest point to the ellipsoid with radius my*x_n.
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