"""
This module contains contact problem definition.
"""
from .problem import Problem

from typing import Optional

import numpy as np
import scipy.sparse as sparse


from rainbow.simulators.prox_rigid_bodies.types import *
import rainbow.simulators.prox_rigid_bodies.state_storage as STORAGE
import rainbow.simulators.prox_rigid_bodies.prox_operators as OP


class Contacts(Problem):

    def __init__(self):
        super().__init__("contact_problem")
        self.K: int = 0  # The number of contact points in the contact problem
        self.J: Optional[sparse.csr_matrix] = None  # The contact Jacobian.
        self.WJT: Optional[sparse.csr_matrix] = None  # The inverse mass matrix multiplied by Jacobian transpose,
        self.g: Optional[np.ndarray] = None  # The gap vector.
        self.e: Optional[np.ndarray] = None   # Restitution coefficient vector.
        self.mu: Optional[np.ndarray] = None  # Friction coefficient vector.
        self.b: Optional[np.ndarray] = None   # The right-hand side vector in v = A x + b
        self.diag: Optional[np.ndarray] = None  # Diagonal of Delassus operator, A-matrix.
        self.delta_u: Optional[np.ndarray] = None  # Velocity change vector given by the Lagrange multiplier solution.

    def initialize(self, dt: float, state: STORAGE.StateStorage, engine: Engine) -> None:
        """
        See also:

        - Problem.initialize: for the base method documentation.
        """
        self.K = len(engine.contact_points)
        self.J = Contacts._compute_jacobian_matrix(engine)
        self.WJT = state.W.dot(self.J.T)
        v = self.J.dot(state.u)
        self.g = Contacts._compute_error_vector(dt, v, engine)
        self.e = Contacts._get_restitution_vector(engine)
        self.mu = Contacts._get_friction_coefficient_vector(engine)

        # Compute right-hand side vector, w = A x + b
        self.b = np.multiply(1 + self.e, v) + self.J.dot(dt * state.delta_u_ext) + self.g

        # Compute diagonal values of Delassus operator (aka system matrix A).
        self.diag = np.sum(self.J.multiply(self.WJT.T), axis=1).A1
        self.diag[self.diag == 0] = 1

        self.r = 1.0 / self.diag  # Initial r-factor values for prox solver.
        self.x = np.zeros(self.b.shape, dtype=np.float64)  # The current solution
        self.sol = np.zeros(self.b.shape, dtype=np.float64)  # The last best known convergent solution
        self.error = np.zeros(self.b.shape, dtype=np.float64)  # The residual vector

    def finalize_solution(self) -> np.ndarray:
        """
        See also:

        - Problem.finalize_solution: for the base method documentation.
        """
        self.delta_u = self.WJT.dot(self.sol)
        return self.delta_u

    def sweep(self) -> None:
        """
        See also:

        - Problem.sweep: for the base method documentation.
        """
        w = self.WJT.dot(self.x)
        for k in range(self.K):
            block = range(4 * k, 4 * k + 4)
            mu_k = self.mu[k]  # Only isotropic Coulomb friction
            x_b = self.x[block]
            delta = (
                x_b.copy()
            )  # Used to keep the old values and compute the change in values
            r_b = self.r[block]
            b_b = self.b[block]
            # By definition
            #       z = x - r (J WJ^T x  + b)
            #         = x - r ( A x  + b)
            # We use
            #        w =  WJ^T x
            # so
            #       z  = x - r ( J w  + b)
            z_b = x_b - np.multiply(r_b, (self.J.dot(w)[block] + b_b))

            # Solve:         x_n = prox_{R^+}( x_n - r (A x_n + b) )
            x_b[0] = np.max([0.0, z_b[0]])

            # Solve: x_f = prox_C( x_f - r (A x_f + b))
            x_b[1], x_b[2], x_b[3] = OP.prox_sphere(z_b[1], z_b[2], z_b[3], mu_k, x_b[0])
            # Put updated contact forces back into solution vector
            self.x[block] = x_b
            # Get the change in the x_block
            np.subtract(x_b, delta, delta)
            # Updating w, so it reflects the change in x, remember w = WJT delta

            # TODO 2020-08-17 Kristian: WJT is in bsr matrix format, which does not support indexing and we can
            #  therefore not access the block sub-matrix.
            #  Currently we circumvent this by converting it to a csr matrix instead, however another solution might
            #  be better.

            w += self.WJT.tocsr()[:, block].dot(delta)

    @staticmethod
    def _compute_jacobian_matrix(engine: Engine) -> sparse.csr_matrix:
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

        # Each contact has 4 rows and two rigid bodies each mapping to 6 columns. Hence, we
        # have 4*2*6 =48 nonzero values per contact point.
        data = np.zeros(K * 48, dtype=np.float64)
        row = np.zeros(K * 48, dtype=np.float64)
        col = np.zeros(K * 48, dtype=np.float64)
        next_entry = 0
        for k in range(K):
            cp = engine.contact_points[k]
            # Compute the vector arms from the point of contact to the rigid body center of mass.
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
            # Now we have all the values we need.
            #
            # The next step is to fill the values into the global Jacobian matrix in the right "entries".
            #
            # The loops below are doing all the index bookkeeping for making this mapping.
            #
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

    @staticmethod
    def _compute_error_vector(dt: float, v: np.ndarray, engine: Engine) -> np.ndarray:
        """
        Compute and return the pre-stabilization vector.

        Pre-stabilization works like a small spring created at each contact point.

        The spring coefficient is carefully chosen to remove a specified fraction of drift error.

        That is, if the gap-value is different from zero, we create springs in normal direction to remove a
        specified fraction of the non-zero gap-value.

        The "spring" view of stabilization is a pure conceptual view.

        The springs do not really exist as objects, rather we just model this as some "excess" displacement/velocity
        that the contact forces must bring to zero as well as dealing with the non-penetration constraints.

        :param dt:             The time-step used by the current time-stepping method.
        :param v:              The relative contact velocities.
        :param engine:         The engine that contains all the contact points.
        :return:               The resulting error vector to use when solving the contact problem.
        """
        K = len(engine.contact_points)
        g = np.zeros(4 * K, dtype=np.float64)
        if not engine.params.use_pre_stabilization:
            return g
        rate = engine.params.gap_reduction / dt
        upper = -engine.params.max_gap_value / dt
        lower = -engine.params.min_gap_value
        for (k, cp) in enumerate(engine.contact_points):
            if cp.g < lower and v[4 * k] <= 0.0:
                g[4 * k + 0] = max(upper, rate * cp.g)
        return g

    @staticmethod
    def get_largest_penetration(engine: Engine) -> float:
        """
        This function iterates over all contacts currently in the engine and determines the
        largest gap-value error. That is the most negative gap-value, and returns the absolute
        value of that.

        :param engine:   The engine holds all the contact points.
        :return:         A float value that is the absolute value of the most negative gap-value.
        """
        gap = 0.0
        for cp in engine.contact_points:
            gap = min(gap, cp.g)
        return -gap

    @staticmethod
    def _get_friction_coefficient_vector(engine: Engine) -> np.ndarray:
        """
        This function looks up the friction coefficients for each material pair of
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

    @staticmethod
    def _get_restitution_vector(engine: Engine) -> np.ndarray:
        """
        This function extract information about restitution coefficient for each contact point and
        assemble this into one global vector.

        Restitution is used to model bounce when rigid bodies are colliding with each other.

        The restitution vector is multiplied with the block-wise pattern for a contact.

        This means that the assembled restitution vector has a different dimension than the friction coefficient
        vector.

        :param engine:  The engine that holds all contact points and surface interaction information.
        :return:        The global restitution vector.
        """
        K = len(engine.contact_points)
        e = np.zeros(4 * K, dtype=np.float64)
        if not engine.params.use_bounce:
            return e
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
