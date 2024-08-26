"""
This module contains proximal operator implementations.

A proximal operator is basically a projection to some convex set.

 Different operators can be specialized for performance when knowing the kind of convex set they work on.
"""

from typing import Tuple

import numpy as np


def prox_sphere(z_s: float, z_t: float, z_tau: float,
                mu: float, x_n: float
                ) -> Tuple[float, float, float]:
    """
    Proximal point of z to a sphere.

    :param z_s:       The s-component of the current z-point.
    :param z_t:       The t-component of the current z-point.
    :param z_tau:     The tau-component of current z-point.
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
