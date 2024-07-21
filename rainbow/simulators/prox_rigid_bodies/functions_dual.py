from typing import Tuple

import scipy.sparse as sparse
import rainbow.simulators.prox_rigid_bodies.mass as MASS
import rainbow.simulators.proximal_contact.prox_solvers as CONTACT_SOLVERS
from rainbow.simulators.prox_rigid_bodies.types import *
import numpy as np


def compute_contact_jacobian_matrix(engine):
    """
    This function creates the contact Jacobian matrix by iterating over all contact points in the
    engine and assembling the Jacobian.

    :param engine:   A reference to the engine holding the contact point information.
    :return:         The Jacobian matrix.
    """
    N = len(engine.bodies)
    K = len(engine.contact_points)

    cols = N * 6
    rows = K * 4

    data = np.zeros(K * 48, dtype=np.float64)
    row = np.zeros(K * 48, dtype=np.float64)
    col = np.zeros(K * 48, dtype=np.float64)
    next_entry = 0
    for k in range(K):
        cp = engine.contact_points[k]
        # Compute the vector arms from point of contact to center of mass of the rigid bodies.
        rA = cp.p - cp.bodyA.r
        rB = cp.p - cp.bodyB.r
        # Compute a contact frame where two first vectors span the contact plane, and the last
        # vector is orthogonal and defines the contact normal direction of the local contact frame system.
        vs, vt, vn = V3.make_orthonormal_vectors(cp.n)
        # Now compute the Jacobian matrix blocks, the essentially maps the body velocities into
        # a relative contact space velocity.
        JA_v = -np.array([vn, vs, vt, V3.zero()], dtype=np.float64)
        JA_w = -np.array(
            [V3.cross(rA, vn), V3.cross(rA, vs), V3.cross(rA, vt), vn], dtype=np.float64
        )
        JB_v = np.array([vn, vs, vt, V3.zero()], dtype=np.float64)
        JB_w = np.array(
            [V3.cross(rB, vn), V3.cross(rB, vs), V3.cross(rB, vt), vn], dtype=np.float64
        )
        # Now we have all the values we need. Next step is to fill the values into the global Jacobian
        # matrix in the right "entries". The for loops below is doing all the index bookkeeping for making
        # this mapping.
        idx_A = cp.bodyA.idx
        idx_B = cp.bodyB.idx
        for i in range(4):
            row_idx = k * 4 + i
            col_idx = idx_A * 6
            for j in range(3):
                data[next_entry] = JA_v[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx += 3
            for j in range(3):
                data[next_entry] = JA_w[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx = idx_B * 6
            for j in range(3):
                data[next_entry] = JB_v[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx += 3
            for j in range(3):
                data[next_entry] = JB_w[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
    # Finally, we can create the specific sparse matrix format that is suitable for our
    # simulator. Here, we go with CSR, because we in most places just need efficient
    # matrix-vector operations.
    J = sparse.csr_matrix((data, (row, col)), shape=(rows, cols))
    return J


def get_friction_coefficient_vector(engine):
    """
    This function looks up the friction coefficients for each material pairs of
    a contact pair and collects all those values into a global vector. The vector
    is used to pass along the friction parameters to the underlying numerical
    solvers used when computing contact forces.

    :param engine: A reference to the engine that holds all contact points and surface interaction information.
    :return: A global vector that has one friction parameter value for each contact point.
    """
    K = len(engine.contact_points)
    mu = np.zeros(K, dtype=np.float64)
    for k in range(K):
        cp = engine.contact_points[k]
        behavior = engine.surfaces_interactions.get_interaction(
            cp.bodyA.material, cp.bodyB.material
        )
        mu[k] = behavior.mu[0]
    return mu


def get_restitution_vector(engine):
    """
    This function extract information about restitution coefficient for each contact point and
    assembles this into one global vector. Restitution is used to modeling bounce when rigid bodies are
    colliding with each other. The assembled restitution vector has a different dimension that the
    friction coefficient vector because the restitution vector is multiplied with the block-wise
    pattern for a contact.

    :param engine:  The engine that holds all contact points and surface interaction information.
    :return:        The global restitution vector.
    """
    K = len(engine.contact_points)
    e = np.zeros(4 * K, dtype=np.float64)
    for k in range(K):
        cp = engine.contact_points[k]
        behavior = engine.surfaces_interactions.get_interaction(
            cp.bodyA.material, cp.bodyB.material
        )
        e[4 * k + 0] = behavior.epsilon
        e[4 * k + 1] = 0.0
        e[4 * k + 2] = 0.0
        e[4 * k + 3] = 0.0
    return e


def get_pre_stabilization_vector(dt, v, engine):
    """
    Compute and return the pre-stabilization vector.

    Pre-stabilization works like a small spring that is created at each contact point. The spring coefficient are
    carefully chosen to remove a specified fraction of drift error. That is if the gap-value is different from
    zero, we create springs in normal direction to remove a specified fraction of the non-zero gap-value.

    The "spring" view of stabilization is a pure mental picture, the springs do not really exists as objects, rather
    we just model this as some "excess" displacement/velocity that the contact forces must bring to zero as well
    as dealing with the non-penetration constraints.

    :param dt:
    :param v:
    :param engine:
    :return:
    """
    K = len(engine.contact_points)
    g = np.zeros(4 * K, dtype=np.float64)
    rate = engine.params.gap_reduction / dt
    upper = -engine.params.max_gap_value / dt
    lower = -engine.params.min_gap_value
    for k in range(K):
        cp = engine.contact_points[k]
        if cp.g < lower and v[4 * k] <= 0.0:
            g[4 * k + 0] = max(upper, rate * cp.g)
    return g


def get_largest_gap_error(engine) -> float:
    """
    This function iterates over all contacts currently in the engine and determines the
    largest gap-value error. That is the most negative gap-value, and returns the absolute
    value of that.

    :param engine:   The engine holding all the contact points.
    :return:         A float value that is the absolute value of the most negative gap-value.
    """
    gap = 0.0
    for cp in engine.contact_points:
        gap = min(gap, cp.g)
    return -gap






def compute_joint_jacobian_matrix(engine):
    """
    This function creates the contact Jacobian matrix by iterating over all contact points in the
    engine and assembling the Jacobian.

    :param engine:   A reference to the engine holding the contact point information.
    :return:         The Jacobian matrix.
    """
    N = len(engine.bodies)
    K = len(engine.hin)

    cols = N * 6
    rows = K * 4

    data = np.zeros(K * 48, dtype=np.float64)
    row = np.zeros(K * 48, dtype=np.float64)
    col = np.zeros(K * 48, dtype=np.float64)
    next_entry = 0
    for k in range(K):
        cp = engine.contact_points[k]
        # Compute the vector arms from point of contact to center of mass of the rigid bodies.
        rA = cp.p - cp.bodyA.r
        rB = cp.p - cp.bodyB.r
        # Compute a contact frame where two first vectors span the contact plane, and the last
        # vector is orthogonal and defines the contact normal direction of the local contact frame system.
        vs, vt, vn = V3.make_orthonormal_vectors(cp.n)
        # Now compute the Jacobian matrix blocks, the essentially maps the body velocities into
        # a relative contact space velocity.
        JA_v = -np.array([vn, vs, vt, V3.zero()], dtype=np.float64)
        JA_w = -np.array(
            [V3.cross(rA, vn), V3.cross(rA, vs), V3.cross(rA, vt), vn], dtype=np.float64
        )
        JB_v = np.array([vn, vs, vt, V3.zero()], dtype=np.float64)
        JB_w = np.array(
            [V3.cross(rB, vn), V3.cross(rB, vs), V3.cross(rB, vt), vn], dtype=np.float64
        )
        # Now we have all the values we need. Next step is to fill the values into the global Jacobian
        # matrix in the right "entries". The for loops below is doing all the index bookkeeping for making
        # this mapping.
        idx_A = cp.bodyA.idx
        idx_B = cp.bodyB.idx
        for i in range(4):
            row_idx = k * 4 + i
            col_idx = idx_A * 6
            for j in range(3):
                data[next_entry] = JA_v[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx += 3
            for j in range(3):
                data[next_entry] = JA_w[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx = idx_B * 6
            for j in range(3):
                data[next_entry] = JB_v[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx += 3
            for j in range(3):
                data[next_entry] = JB_w[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
    # Finally, we can create the specific sparse matrix format that is suitable for our
    # simulator. Here, we go with CSR, because we in most places just need efficient
    # matrix-vector operations.
    J = sparse.csr_matrix((data, (row, col)), shape=(rows, cols))
    return J

