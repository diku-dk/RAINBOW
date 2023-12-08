from typing import Tuple

import scipy.sparse as sparse
import rainbow.math.matrix3 as M3
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.mass as MASS
import rainbow.simulators.prox_rigid_bodies.collision_detection as CD
import rainbow.simulators.proximal_contact.prox_solvers as CONTACT_SOLVERS
from rainbow.simulators.prox_rigid_bodies.types import *
from rainbow.util.timer import Timer
import numpy as np


def transform_shape_into_body_frame(shape) -> None:
    """
    This function computes the mass properties of a given shape assuming unit-mass-density. It
    then assigns the body frame mass properties to the shape instance and adjust the surface
    mesh of the shape to be given wrt the computed body frame.

    The translation and rotation used to transform the mesh into the body frame is kept with
    the shape. The reason for this is to be able to get back the "modelling" frame that was
    used when defining the shape. This is needed for making it easy for end-users to set up
    and rig their simulations. However, internally inside the simulator the modeling frame
    is never seen, here only the body frame information is used.

    Warning if invoked multiple times on the same shape reference then the modeling
    frame information will be lost. The reason is that in the second invocation the
    mesh now lives in the body frame, so when computing the mass properties one will
    find a zero-translation and identity transform for bringing the mesh into its
    body frame.

    :param shape:  The reference to the shape that should be transformed.
    :return:       Nothing.
    """
    prop = MASS.compute_mass_properties(shape.mesh.V, shape.mesh.T, 1.0)

    (shape.r, shape.q, shape.mass, shape.inertia) = MASS.xform_model_2_body_space(prop)
    #
    # shape.r and shape.q gives the rigid body transform from body to model space
    # We need to do the inverse transform here
    #
    MESH.translate(shape.mesh, -shape.r)
    MESH.rotate(shape.mesh, Q.conjugate(shape.q))


def get_position_vector(engine):
    """
    This function extract the position of center of mass and the orientation
    of the body frames as a quaternion for all rigid bodies in the engine
    and stack these into one global generalized position vector.

    :param engine:  The engine from which to extract position and orientation of rigid bodies.
    :return:        The generalized position vector.
    """
    x = np.zeros(len(engine.bodies) * 7, dtype=np.float64)
    k = 0
    for body in engine.bodies.values():
        offset = 7 * k
        x[offset : offset + 3] = body.r
        x[offset + 3 : offset + 7] = body.q
        k += 1
    return x


def set_position_vector(x, engine) -> None:
    """
    This function works opposite of the get_position_vector function. It updates
    all the positions and orientations of rigid bodies in the engine to be equal
    to the corresponding values from a generalized position vector.

    :param x:          The generalized position vector.
    :param engine:     A reference to an engine holding the rigid bodies that will be updated.
    :return:           Nothing.
    """
    k = 0
    for body in engine.bodies.values():
        offset = 7 * k
        body.r = x[offset : offset + 3]
        body.q = x[offset + 3 : offset + 7]
        k += 1


def get_velocity_vector(engine):
    """
    This function loops over all rigid bodies and extract linear and angular
    velocities and stack these into a generalized velocity vector. Observe that
    the dimensionality of the generalized position and velocity vectors are
    different. If one has N bodies then the generalized position vector will be 7N
    long whereas the generalized velocity vector will be 6N long. The reason is due
    to the quaternion representation of the rigid body orientation.

    :param engine: A reference to the engine holding the rigid bodies to extract velocities from.
    :return:       The generalized velocity vector.
    """
    u = np.zeros((len(engine.bodies) * 6,), dtype=np.float64)
    k = 0
    for body in engine.bodies.values():
        offset = 6 * k
        u[offset : offset + 3] = body.v
        u[offset + 3 : offset + 6] = body.w
        k += 1
    return u


def set_velocity_vector(u, engine) -> None:
    """
    This function works opposite of the get_velocity_vector function. It
    maps a generalized velocity vector to the rigid bodies inside
    an engine instance.

    :param u:       The generalized velocity vector.
    :param engine:  A reference to the engine holding the rigid bodies.
    :return:        Nothing.
    """
    k = 0
    for body in engine.bodies.values():
        offset = 6 * k
        body.v = u[offset : offset + 3]
        body.w = u[offset + 3 : offset + 6]
        k += 1


def position_update(x, u, dt, engine) -> None:
    """
    This function performs an explicit Euler update of the kinematic relation between
    positions and velocities. The update is slightly more complicated due to using quaternions
    as one needs to solve

    .. math::

        q^{t+1} \leftarrow q^{t} + \Delta t \, {\Omega} \otimes q^t

    where :math:`q` is the quaternion,:math:`\Delta t` the time-step size,
    :math:`\Omega` is the angular velocity written as a quaternion,
    and :math:`\otimes` is the quaternion product. This update rule is highly
    nonlinear, and it also requires renormalization of :math:`q^{t+1}`.

    This function encapsulates all the ugliness of this time-integration.

    :param x:       The generalized position vector.
    :param u:       The generalized velocity vector.
    :param dt:      The time-step size.
    :param engine:  A reference to the engine holding all the rigid bodies.
    :return:        Nothing.
    """
    k = 0
    for body in engine.bodies.values():
        x_offset = 7 * k
        u_offset = 6 * k

        r = x[x_offset : x_offset + 3]
        q = x[x_offset + 3 : x_offset + 7]
        v = u[u_offset : u_offset + 3]
        w = u[u_offset + 3 : u_offset + 6]

        if not body.is_fixed:
            r += v * dt
            q += Q.prod(Q.from_vector3(w), q) * dt * 0.5
        x[x_offset : x_offset + 3] = r
        x[x_offset + 3 : x_offset + 7] = Q.unit(q)
        k += 1


def compute_total_external_forces(x, u, engine):
    """
    This function iterates over all rigid bodies and evaluates
    all force types acting on them and accumulates the result. Finally,
    it stacks all total force and torque terms into a generalized force
    vector.

    Observe that we collect gyroscopic force terms into the external generalized force vector. That is

    .. math::

        f \equiv \sum_i f_i

        T \equiv \omega x I \, \omega + \sum_i T_i

    where :math:`f,T` are total force and torque on a single body, :math:`f_i,T_i` are the
    contributions from the forces types, :math:`\omega`
    is angular velocity, :math:`I` is world
    space inertia tensor, and :math:`x` is the vector cross product.

    :param x:        The generalized position vector used for evaluating the external forces.
    :param u:        The generalized velocity vector used for evaluating the external forces.
    :param engine:   A reference to the engine holding all the rigid bodies.
    :return:         The generalized force vector.
    """
    f_ext = np.zeros((len(engine.bodies) * 6,), dtype=np.float64)
    k = 0
    for body in engine.bodies.values():
        x_offset = 7 * k
        u_offset = 6 * k

        F = V3.zero()
        T = V3.zero()

        if not body.is_fixed:
            r = x[x_offset : x_offset + 3]
            q = x[x_offset + 3 : x_offset + 7]
            v = u[u_offset : u_offset + 3]
            w = u[u_offset + 3 : u_offset + 6]
            for force_type in body.forces:
                (Fi, Ti) = force_type.compute(body, r, q, v, w)
                F += Fi
                T += Ti
            R = Q.to_matrix(q)
            I_wcs = MASS.update_inertia_tensor(R, body.inertia)
            T -= np.cross(w, np.dot(I_wcs, w), axis=0)

        f_ext[u_offset : u_offset + 3] = F
        f_ext[u_offset + 3 : u_offset + 6] = T

        k += 1
    return f_ext


def compute_inverse_mass_matrix(x, engine):
    """
    This function computes the inverse mass matrix of all the rigid bodies in the engine.

    :param x:        The generalized position vector, this is used to proper update inertia tensors before inverting.
    :param engine:   The engine that holds all the rigid bodies.
    :return:         A sparse matrix that holds the value of the inverse mass matrix.
    """
    N = len(engine.bodies)
    blocks = [np.zeros((3, 3), dtype=np.float64) for _ in range(N * 2)]
    k = 0
    for body in engine.bodies.values():
        x_offset = 7 * k  # Position offset into x-array
        if not body.is_fixed:
            q = x[x_offset + 3 : x_offset + 7]  # Extract rotation part
            R = Q.to_matrix(q)
            I_wcs = MASS.update_inertia_tensor(R, 1.0 / body.inertia)
            m = 1.0 / body.mass
            blocks[2 * k] = M3.diag(m, m, m)
            blocks[2 * k + 1] = I_wcs
        k += 1
    # TODO 2022-01-05 Kenny: These sparse matrix formats may bot be the most efficient ones. The inverse mass matrix
    #  is multiplied onto the Jacobian matrix or a force vector. Hence, we just need a sparse format that is
    #  efficient for that purpose. Current implementation just build a block diagonal matrix and then converts
    #  it in the end. This was done to simplify the programming effort and not with performance in mind.
    D = sparse.block_diag(blocks)
    W = D.tobsr(blocksize=(3, 3))
    return W


def compute_jacobian_matrix(engine):
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


def apply_post_stabilization(J, WJT, x, engine, stats: dict, debug_on) -> dict:
    """
    Apply post-stabilization to remove gap errors in the simulation.
    This function solves for actual displacements that will make all gap-values become non-negative. This is
    different from pre-stabilization. Pre-stabilization just adds a bit more to the contact problem to make contact
    forces stronger, so the forces can remove more of the error.
    In post-stabilization we measure the actual error and solve for instantaneous displacement that will resolve all
    the errors, taking coupling and non-linearity and everything into account.

    :param J:         The contact Jacobian matrix.
    :param WJT:       The transpose contact Jacobian matrix pre-multiplied by the inverse mass matrix.
    :param x:         The generalized position vector.
    :param engine:    The engine that holds all the rigid bodies.
    :param stats:     A dictionary that collects performance measurements and statistics when debugging mode is on.
    :param debug_on:  A boolean flag to toggle debug mode.
    :return:          A dictionary with collected statistics and performance measurements.
    """
    K = len(engine.contact_points)
    g = np.zeros(4 * K, dtype=np.float64)
    for k in range(K):
        cp = engine.contact_points[k]
        if cp.g < -engine.params.min_gap_value:
            g[4 * k + 0] = cp.g
    # If the gap to correct is all zeros, then just return
    if not g.any():
        return stats
    mu = np.zeros(K, dtype=np.float64)
    sol, stats = CONTACT_SOLVERS.solve(
        J, WJT, g, mu, CONTACT_SOLVERS.prox_origin, engine, stats, debug_on,
        prefix="post_stabilization_", 
        scheme=engine.params.proximal_solver
    )
    vector_positional_update = WJT.dot(sol)
    position_update(x, vector_positional_update, 1, engine)
    return stats


def get_total_energy(engine) -> Tuple[float, float]:
    """
    Compute the total kinetic and potential energy of the whole system in the engine. This function is
    used to monitor energies during simulation which is a very useful debug and analysis tool. The energies are
    computed and collected in each time step if the simulator is running in debug mode. One may invoke the function
    directly if the energies is needed, as the function does not update or change any information in the engine.

    :param engine:  A reference to the engine that is holding all the rigid bodies.
    :return:        A pair of floats that represent the total kinetic energy and potential energy (in that order).
    """
    kinetic = 0.0
    potential = 0.0
    for body in engine.bodies.values():
        if body.is_fixed:
            continue
        m = body.mass
        h = 0

        G = V3.make(0, 0, 0)
        isGravity = False
        for force in engine.forces:
            if isinstance(engine.forces[force], Gravity):
                gravity_force = engine.forces[force]
                G += gravity_force.up * gravity_force.g
                isGravity = True

        v = np.linalg.norm(body.v)
        w = body.w
        I_bf = body.inertia
        R = Q.to_matrix(body.q)
        I_wcs = MASS.update_inertia_tensor(R, I_bf)
        wIw = np.dot(w, np.dot(I_wcs, w))
        kinetic += 0.5 * (m * (v ** 2) + wIw)
        if isGravity:
            up = V3.unit(G)
            g = V3.norm(G)
            h = up.dot(body.r)
            potential += m * g * h

    return kinetic, potential


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


class SemiImplicitStepper:
    """
    This class implements a semi-implicit first order Euler time-stepper.
    """

    def __init__(self, engine: Engine) -> None:
        self.log = []

    def step(self, dt: float, engine: Engine, debug_on: bool) -> None:
        """
        This is the main simulation method that is responsible for stepping time
        forward to the next time-step.

        :param dt:        The time-step to step time forward in the engine.
        :param engine:    The engine that holds all the rigid bodies.
        :param debug_on:  A boolean flag to toggle debug mode.
        :return:          None
        """
        timer = None
        if debug_on:
            timer = Timer("Stepper")
            timer.start()
        stats = {}

        x = get_position_vector(engine)
        u = get_velocity_vector(engine)
        f_ext = compute_total_external_forces(x, u, engine)
        du_contact = np.zeros(u.shape, dtype=np.float64)
        W = compute_inverse_mass_matrix(x, engine)
        du_ext = W.dot(dt * f_ext)

        stats = CD.run_collision_detection(engine, stats, debug_on)

        J = None
        WJT = None
        if len(engine.contact_points) > 0:
            J = compute_jacobian_matrix(engine)
            WJT = W.dot(J.T)
            v = J.dot(u)
            if engine.params.use_pre_stabilization:
                g = get_pre_stabilization_vector(dt, v, engine)
            else:
                g = np.zeros(v.shape, dtype=np.float64)

            if engine.params.use_bounce:
                e = get_restitution_vector(engine)
            else:
                e = np.zeros(v.shape, dtype=np.float64)

            mu = get_friction_coefficient_vector(engine)
            b = np.multiply(1 + e, v) + J.dot(du_ext) + g
            sol, stats = CONTACT_SOLVERS.solve(
                J, WJT, b, mu, CONTACT_SOLVERS.prox_sphere, engine, stats, debug_on, 
                prefix="",
                scheme=engine.params.proximal_solver
            )
            du_contact = WJT.dot(sol)

        du_total = du_ext + du_contact
        u += du_total
        position_update(x, u, dt, engine)
        set_position_vector(x, engine)
        set_velocity_vector(u, engine)

        if engine.params.use_post_stabilization:
            if len(engine.contact_points) > 0:
                stats = apply_post_stabilization(J, WJT, x, engine, stats, debug_on)

        if debug_on:
            timer.end()
            stats["stepper_time"] = timer.elapsed
            stats["dt"] = dt
            stats["contact_points"] = len(engine.contact_points)
            kinetic_energy, potential_energy = get_total_energy(engine)
            stats["kinetic_energy"] = kinetic_energy
            stats["potential_energy"] = potential_energy
            stats["max_penetration"] = get_largest_gap_error(engine)
            self.log.append(stats)
