"""
This module contains the post-stabilization problem definitions.
"""

from .problem import Problem
from .contacts import Contacts
from typing import Optional

import numpy as np
import scipy.sparse as sparse

from rainbow.simulators.prox_rigid_bodies.types import *
import rainbow.simulators.prox_rigid_bodies.state_storage as STORAGE


class PostStabilization(Problem):

    def __init__(self, contact_problem: Contacts):
        super().__init__("post_stabilization_problem")
        self.K: int = contact_problem.K  # The number of contact points in the contact problem
        self.J: Optional[sparse.csr_matrix] = contact_problem.J  # The contact Jacobian.
        self.WJT: Optional[sparse.csr_matrix] = contact_problem.WJT  # The inverse mass matrix multiplied by Jacobian transpose,
        self.g: Optional[np.ndarray] = None  # The gap vector.
        self.diag: Optional[np.ndarray] = contact_problem.diag.copy()  # Diagonal of Delassus operator, A-matrix.
        self.delta_r: Optional[np.ndarray] = None  # Positional change vector given by the Lagrange multiplier solution.

    def initialize(self, dt: float, state: STORAGE.StateStorage, engine: Engine) -> None:
        """
        See also:

        - Problem.initialize: for the base method documentation.
        """
        self.g = PostStabilization._compute_error_vector(engine)
        self.r = 1.0 / self.diag  # Initial r-factor values for prox solver.
        self.x = np.zeros(self.g.shape, dtype=np.float64)  # The current solution.
        self.sol = np.zeros(self.g.shape, dtype=np.float64)  # The last best known convergent solution.
        self.error = np.zeros(self.g.shape, dtype=np.float64)  # The residual vector.

    def finalize_solution(self) -> np.ndarray:
        """
        See also:

        - Problem.finalize_solution: for the base method documentation.
        """
        self.delta_r = self.WJT.dot(self.sol)
        return self.delta_r

    def sweep(self) -> None:
        """
        See also:

        - Problem.sweep: for the base method documentation.
        """
        w = self.WJT.dot(self.x)
        for k in range(self.K):
            block = range(4 * k, 4 * k + 4)
            x_b = self.x[block]
            delta = (
                x_b.copy()
            )  # Used to keep the old values and compute the change in values
            r_b = self.r[block]
            g_b = self.g[block]
            # By definition
            #       z = x - r (J WJ^T x  + g)
            #         = x - r ( A x  + g)
            # We use
            #        w =  WJ^T x
            # so
            #       z  = x - r ( J w  + g)
            z_b = x_b - np.multiply(r_b, (self.J.dot(w)[block] + g_b))
            # Solve:         x_n = prox_{R^+}( x_n - r (A x_n + b) )
            x_b[0] = np.max([0.0, z_b[0]])
            x_b[1], x_b[2], x_b[3] = 0, 0, 0
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
    def _compute_error_vector(engine: Engine) -> np.ndarray:
        """
        Compute and return the post-stabilization vector.

        The post-stabilization vector is like a desired displacement at the contacts.

        Hence, the error vector computes the distance that should be "pushed" at each contact point.

        :param engine:  The engine that holds the contact point information.
        :return:        The error vector with the "distances" that should be fixed by the post-stabilization.
        """
        K = len(engine.contact_points)
        g = np.zeros(4 * K, dtype=np.float64)
        for (k, cp) in enumerate(engine.contact_points):
            if cp.g < -engine.params.min_gap_value:
                g[4 * k + 0] = cp.g
        return g
