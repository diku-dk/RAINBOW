"""
This module implements a container for all the body state information in a rigid body simulator.
"""

from typing import Tuple, Optional

import numpy as np
import scipy.sparse as sparse

import rainbow.math.matrix3 as M3
import rainbow.simulators.prox_rigid_bodies.mass as MASS
from rainbow.simulators.prox_rigid_bodies.types import *


class StateStorage:
    """
    The StateStorage class holds information about all the bodies in the simulation.

    Here all body information is agglomerated into vectors rather than keep it as
    local variables in the body instances themselves. Essentially the per body instance
    information is stacked into larger vectors. This has the advantage of making it easier
    to apply linear algebra routines on the whole system information in one go.

    ToDo 2024-07-22 Kenny: We should refactor this one step further, such that the body
     class is only a view into these larger state vectors. Hence, no local data other
     than geometry/collision detection should be stored in the rigid body class. Such
     a redesign would remove the need for retrieving and storing state information
     back-and-forth from the state-info class to the body instances. The redesign should
     likely be replicated such that all constraints (contacts and joints) are similar shallow
     views into larger vectors/matrices.
    """

    def __init__(self):
        self.x: Optional[np.ndarray] = None  # Generalized positions of all rigid bodies, 7 N dimensions.
        self.u: Optional[np.ndarray] = None  # Generalized velocities of all rigid bodies, 6 N dimensions.
        self.f_ext: Optional[np.ndarray] = None  # Generalized forces of all rigid bodies, 6 N dimensions.
        self.W: Optional[sparse.bsr_matrix] = None  # Inverse mass matrix of all rigid bodies, 6N by 6N dimensions.

        # Acceleration change due to external forces, should be multiplied by a time-stp
        # prior to any velocity update.
        self.delta_u_ext: Optional[np.ndarray] = None

    def copy_from_engine(self, engine: Engine) -> None:
        """
        Initialize the state storage with information taken from the given engine instance.

        :param engine:  The engine to extract information from.
        """
        self.x = StateStorage._get_position_vector(engine)
        self.u = StateStorage._get_velocity_vector(engine)
        self.f_ext = self._compute_total_external_forces(engine)
        self.W = self._compute_inverse_mass_matrix(engine)
        self.delta_u_ext = self.W.dot(self.f_ext)

    def copy_to_engine(self, engine: Engine) -> None:
        """
        Stores information from the state storage back into a given engine.

        :param engine:   The engine that should contain the state information.
        """
        for (k, body) in enumerate(engine.bodies.values()):
            offset = 6 * k
            body.v = self.u[offset: offset + 3]
            body.w = self.u[offset + 3: offset + 6]

            offset = 7 * k
            body.r = self.x[offset: offset + 3]
            body.q = self.x[offset + 3: offset + 7]

    @staticmethod
    def _get_position_vector(engine: Engine) -> np.ndarray:
        """
        This function extracts the position value for the center of mass and the orientation
        of the body frames as a quaternion for all rigid bodies in the engine
        and stacks these into one global generalized position vector.

        :param engine:  The engine from which to extract the position and orientation of rigid bodies.
        :return:        The generalized position vector.
        """
        x = np.zeros(len(engine.bodies) * 7, dtype=np.float64)
        for (k, body) in enumerate(engine.bodies.values()):
            offset = 7 * k
            x[offset: offset + 3] = body.r
            x[offset + 3: offset + 7] = body.q
        return x

    @staticmethod
    def _get_velocity_vector(engine: Engine) -> np.ndarray:
        """
        This function loops over all rigid bodies and extracts linear and angular
        velocities and stack these into a generalized velocity vector. Observe that
        the dimensionality of the generalized position and velocity vectors are
        different. If one has N bodies, then the generalized position vector will be 7N
         long, whereas the generalized velocity vector will be 6N long. The reason is due
        to the quaternion representation of the rigid body orientation.

        :param engine: A reference to the engine holding the rigid bodies to extract velocities from.
        :return:       The generalized velocity vector.
        """
        u = np.zeros((len(engine.bodies) * 6,), dtype=np.float64)
        for (k, body) in enumerate(engine.bodies.values()):
            offset = 6 * k
            u[offset: offset + 3] = body.v
            u[offset + 3: offset + 6] = body.w
        return u

    # TODO 2024-07-28 Kenny: Can we refactor code to drop Engine argument.
    def _compute_total_external_forces(self, engine: Engine) -> np.ndarray:
        """
        Evaluate and accumulate all force types acting on all rigid bodies, then stack these into a generalized force
        vector.

        Gyroscopic force terms are collected into the external generalized force vector.

        .. math::

            F \equiv \sum_i f_i

            T \equiv \omega x I \, \omega + \sum_i T_i

        where:
            - :math:`f,T` are total force and torque on a single body,
            - :math:`f_i,T_i` are the contributions from the forces types,
            - :math:`\omega` is angular velocity,
            - :math:`I` is world space inertia tensor,
            - :math:`x` is the vector cross product.

        :param engine:   A reference to the engine holding all the rigid bodies.
        :return:         The generalized force vector.
        """
        f_ext = np.zeros((len(engine.bodies) * 6,), dtype=np.float64)
        for (k, body) in enumerate(engine.bodies.values()):
            # TODO 2024-07-28 Kenny: The body fixed state is one reason why we need the engine instance as an
            #  argument.
            if body.is_fixed:
                continue
            x_offset = 7 * k
            u_offset = 6 * k
            r = self.x[x_offset: x_offset + 3]
            q = self.x[x_offset + 3: x_offset + 7]
            v = self.u[u_offset: u_offset + 3]
            w = self.u[u_offset + 3: u_offset + 6]
            R = Q.to_matrix(q)
            # TODO 2024-07-28 Kenny: The body inertia state is one reason why we need the engine instance as an
            #  argument.
            I_wcs = MASS.update_inertia_tensor(R, body.inertia)
            F = V3.zero()
            T = -V3.cross(w, np.dot(I_wcs, w))
            # TODO 2024-07-28 Kenny: The body class is the only one that knows about associated forces on the body.
            for force_type in body.forces:
                (Fi, Ti) = force_type.compute(body, r, q, v, w)
                F += Fi
                T += Ti
            f_ext[u_offset: u_offset + 3] = F
            f_ext[u_offset + 3: u_offset + 6] = T
        return f_ext

    def _compute_inverse_mass_matrix(self, engine: Engine) -> sparse.bsr_matrix:
        """
        This function computes the inverse mass matrix of all the rigid bodies in the engine.

        :param engine:   The engine that holds all the rigid bodies.
        :return:         A sparse matrix that holds the value of the inverse mass matrix.
        """
        N = len(engine.bodies)
        blocks = [np.zeros((3, 3), dtype=np.float64) for _ in range(N * 2)]
        for (k, body) in enumerate(engine.bodies.values()):
            x_offset = 7 * k  # Position offset into x-array
            if not body.is_fixed:
                q = self.x[x_offset + 3: x_offset + 7]  # Extract rotation part
                R = Q.to_matrix(q)
                I_wcs = MASS.update_inertia_tensor(R, 1.0 / body.inertia)
                m = 1.0 / body.mass
                blocks[2 * k] = M3.diag(m, m, m)
                blocks[2 * k + 1] = I_wcs
        # TODO 2022-01-05 Kenny: These sparse matrix formats may bot be the most efficient ones. The inverse mass matrix
        #  is multiplied onto the Jacobian matrix or a force vector. Hence, we just need a sparse format that is
        #  efficient for that purpose. Current implementation just build a block diagonal matrix and then converts
        #  it in the end. This was done to simplify the programming effort and not with performance in mind.
        D = sparse.block_diag(blocks)
        W = D.tobsr(blocksize=(3, 3))
        return W

    # TODO 2024-07-28 Kenny: Can we refactor code to drop Engine argument.
    def compute_total_energy(self, engine) -> Tuple[float, float]:
        """
        Compute the total kinetic and potential energy of the whole system in the engine.
        This function is used to monitor energies during simulation, which is a beneficial debug and analysis tool.

        :param engine:  A reference to the engine that is holding all the rigid bodies.
        :return:        A pair of floats that represent the total kinetic energy and potential energy (in that order).
        """
        kinetic = 0.0
        potential = 0.0
        for (k, body) in enumerate(engine.bodies.values()):

            # TODO 2024-07-28 Kenny: The body fixed  below is one reason why we need the engine instance as an
            #  argument.
            if body.is_fixed:
                continue

            x_offset = 7 * k
            u_offset = 6 * k

            r = self.x[x_offset: x_offset + 3]
            q = self.x[x_offset + 3: x_offset + 7]
            R = Q.to_matrix(q)
            v = self.u[u_offset: u_offset + 3]
            w = self.u[u_offset + 3: u_offset + 6]

            # TODO 2024-07-28 Kenny: The body mass and inertia access is another reason why we need the engine instance.
            m = body.mass
            I_bf = body.inertia
            I_wcs = MASS.update_inertia_tensor(R, I_bf)

            G = V3.make(0, 0, 0)
            has_gravity = False
            # TODO 2024-07-28 Kenny: The engine is the only place that can tells us about forces attached to a rigid
            #  body.
            for force in engine.forces:
                if isinstance(engine.forces[force], Gravity):
                    gravity_force = engine.forces[force]
                    G += gravity_force.up * gravity_force.g
                    has_gravity = True
            if has_gravity:
                up = V3.unit(G)
                g = V3.norm(G)
                h = up.dot(r)
                potential += m * g * h

            wIw = np.dot(w, np.dot(I_wcs, w))
            kinetic += 0.5 * (m * (v.dot(v)) + wIw)

        return kinetic, potential

    def velocity_update(self, delta_u: np.ndarray) -> None:
        """
        Update the generalized velocity vector by adding the change in velocity.

        :param delta_u:  The change in the generalized velocity vector.
        """
        if self.u is not None:
            self.u += delta_u
        else:
            raise ValueError("Generalized velocity vector 'u' is not initialized.")

    # TODO 2024-07-28 Kenny: Can we refactor code to drop Engine argument.
    def position_update(self, dt: float, engine: Engine, u: np.ndarray = None) -> None:
        """
         Performs an explicit Euler update of the kinematic relation between positions and velocities.

        This function updates the position of the object using the explicit Euler method, taking into account the
        complexities introduced by quaternion representation.

        The update process involves solving the following equation:

        .. math::

            Q^{t+1} \leftarrow q^{t} + \Delta t \, {\Omega} \otimes q^t

        Where

            - :Math:`q` is the quaternion,
            - :math:`\Delta t` the time-step size,
            - :math:`\Omega` is the angular velocity written as a quaternion,
            - :math:`\otimes` is the quaternion product.

        Due to the nonlinearity of this update rule, :math:`q^{t+1}` must be renormalized after the update.

        This function encapsulates all the ugliness of this time-integration.

        :param dt:      The time-step size.
        :param u:       The generalized velocity vector. If set to None, then the internal velocity of state is used.
        :param engine:  A reference to the engine holding all the rigid bodies.
        :return:        Nothing.
        """
        if self.x is None or self.u is None:
            raise ValueError("Generalized position vector 'x' or velocity vector 'u' is not initialized.")

        u = self.u if u is None else u
        for (k, body) in enumerate(engine.bodies.values()):
            x_offset = 7 * k
            u_offset = 6 * k

            r = self.x[x_offset: x_offset + 3]
            q = self.x[x_offset + 3: x_offset + 7]
            v = u[u_offset: u_offset + 3]
            w = u[u_offset + 3: u_offset + 6]

            # TODO 2024-07-28 Kenny: The body instance below is the only reason why we need the engine instance as an
            #  argument.
            if not body.is_fixed:
                r += v * dt
                q += Q.prod(Q.from_vector3(w), q) * dt * 0.5

            self.x[x_offset: x_offset + 3] = r
            self.x[x_offset + 3: x_offset + 7] = Q.unit(q)
