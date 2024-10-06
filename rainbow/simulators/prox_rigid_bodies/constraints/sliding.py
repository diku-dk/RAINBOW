from typing import Optional

from .problem import Problem

import numpy as np
import scipy.sparse as sparse

import rainbow.math.matrix3 as M3
from rainbow.simulators.prox_rigid_bodies.types import *
import rainbow.simulators.prox_rigid_bodies.state_storage as STORAGE


class SlidingJoints(Problem):
    """
    The sliding joints problem class
    """

    def __init__(self):
        super().__init__("sliding_joint_problem")
        self.K: int = 0 # number of sliding joints
        self.J: Optional[sparse.csr_matrix] = None # The Jacobian
        self.WJT: Optional[sparse.csr_matrix] = None  # The inverse mass matrix multiplied by Jacobian transpose,
        self.g: Optional[np.ndarray] = None  # The gap vector.
        self.b: Optional[np.ndarray] = None   # The right-hand side vector in v = A x + b
        self.diag: Optional[np.ndarray] = None  # Diagonal of Delassus operator, A-matrix.
        self.delta_u: Optional[np.ndarray] = None  # Velocity change vector given by the Lagrange multiplier solution.
    
    def initialize(self, dt: float, state: STORAGE.StateStorage, engine: Engine) -> None:
        self.K = len(engine.sliding_joints)
        self.J = SlidingJoints._compute_jacobian_matrix(engine)
        self.WJT = state.W.dot(self.J.T)
        self.g = SlidingJoints._compute_error_vector(dt, engine)
        self.b = self.J.dot(state.u) + self.J.dot(dt * state.delta_u_ext) + self.g
        
        # Compute diagonal values of Delassus operator (aka system matrix A).
        self.diag = np.sum(self.J.multiply(self.WJT.T), axis=1).A1
        self.diag[self.diag == 0] = 1

        self.r = 1.0 / self.diag  # Initial r-factor values for prox solver.
        self.x = np.zeros(self.b.shape, dtype=np.float64)  # The current solution
        self.sol = np.zeros(self.b.shape,
                            dtype=np.float64)  # The last best known solution, used for restarting if divergence
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
        raise NotImplementedError()
    
    @staticmethod
    def _compute_jacobian_matrix(engine: Engine) -> sparse.csr_matrix:
        """
        This function creates the sliding joint Jacobian matrix by iterating over 
        all sliding joints in the engine and assembling the Jacobian.

        :param engine:   A reference to the engine holding the sliding joint information.
        :return:         The Jacobian matrix.
        """
        N = len(engine.bodies)
        K = len(engine.sliding_joints)
        
        # Preallocate matrices for temporaries
        JA_v = np.zeros((5,3), dtype=np.float64)
        JA_w = np.zeros((5,3), dtype=np.float64)
        JB_v = np.zeros((5,3), dtype=np.float64)
        JB_w = np.zeros((5,3), dtype=np.float64)
        
        cols = N * 6
        rows = K * 5
        
        # A hinge joint has 5 constraint rows and two rigid bodies each of 6 DOFs.
        # This gives a maximum of 12 nonzero values per row. A total of 60 non-zeroes
        # per hinge.
        data = np.zeros(K * 60, dtype=np.float64)
        row = np.zeros(K * 60, dtype=np.float64)
        col = np.zeros(K * 60, dtype=np.float64)
        next_entry = 0

        for slider in engine.sliding_joints.values():
            k = slider.idx
            
            n_p = Q.rotate(slider.parent.q, slider.axis_p)
            t1_p, t2_p, n_p = V3.make_orthonormal_vectors(n_p)
            
            half_c = 0.5 * slider.offset_init
            half_c_t1_p_T = np.cross(half_c, t1_p).T
            half_c_t2_p_T = np.cross(half_c, t2_p).T
            
            # Compute Jacobian matrix blocks
            JA_v[0:3, :] = M3.zero()
            JA_v[3, :] = t1_p.T
            JA_v[4, :] = t2_p.T
            JA_w[0:3, :] = M3.identity()
            JA_w[3, :] = half_c_t1_p_T
            JA_w[4, :] = half_c_t2_p_T
            
            JB_v[0:3, :] = M3.zero()
            JB_v[3, :] = - t1_p.T
            JB_v[4, :] = - t2_p.T
            JB_w[0:3, :] = - M3.identity()
            JB_w[3, :] = half_c_t1_p_T
            JB_w[4, :] = half_c_t2_p_T
            
            idx_A = slider.parent.idx
            idx_B = slider.child.idx
            for i in range(5):
                row_idx = k * 5 + i
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
    def _compute_error_vector(dt: float, engine: Engine) -> np.ndarray:
        """
        Compute and return the error vector.

        :param dt:
        :param engine:
        :return:
        """
        K = len(engine.sliding_joints)
        g = np.zeros(5 * K, dtype=np.float64)
        if not engine.params.use_pre_stabilization:
            return g
        for sliding_joint in engine.sliding_joints.values():
            offset = sliding_joint.idx * 5

            r_p = Q.rotate(sliding_joint.parent.q, sliding_joint.arm_p) + sliding_joint.parent.r
            r_c = Q.rotate(sliding_joint.child.q, sliding_joint.arm_c) + sliding_joint.child.r

            n_p = Q.rotate(sliding_joint.parent.q, sliding_joint.axis_p)
            t_p, b_p, n_p = V3.make_orthonormal_vectors(n_p)

            n_c = Q.rotate(sliding_joint.child.q, sliding_joint.axis_c)

            u = V3.cross(n_p, n_c)

            g[offset:offset + 3] = (r_c - r_p)
            g[offset + 3] = t_p.dot(u)
            g[offset + 4] = b_p.dot(u)
        rate = engine.params.gap_reduction / dt
        g *= rate
        return g
