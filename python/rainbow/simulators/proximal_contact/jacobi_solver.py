import numpy as np
import numba as nb
from rainbow.simulators.proximal_contact.solver_interface import SolverInterface


@nb.njit(parallel=True, nogil=True, cache=True)
def sweep_worker(K, w, z, b, mu, x, friction_solver, hybrid):
    """ Sweep worker for parallel Jacobi solver.

    :param K: Contact points number.
    :param w: The WJT.dot(x).
    :param z: The current contact force.
    :param b:  b = Ju^t + ∆t JM^{-1}h + EJu^t.
    :param mu: The coefficient of friction.
    :param x: The current contact force.
    :param friction_solver: The proximal operator of friction cone function.
    :param hybrid: If True, use the hybrid parallel Jacobi algorithm.
    :return: The new contact force.
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


class ParallelJacobiSolver(SolverInterface):
    """ Parallel Jacobi Proximal Algorithm.
    """

    def sweep(self):
        w = self.WJT.dot(self.x)
        z = self.x - np.multiply(self.r, (self.J.dot(w) + self.b))
        self.x = sweep_worker(
            self.K,
            w,
            z,
            self.b,
            self.mu,
            self.x,
            self.friction_solver,
            False,
        )


class ParallelJacobiHybridSolver(SolverInterface):
    """ Parallel Jacobi Hybrid Proximal Algorithm.
        Here hybrid means that : 
            1. The normal force is update by the old value, which is Jacobi-scheme.
            2. The friction force is update by the lastest value, which is Gauss-Seidel-scheme.
        Hence, we call it hybrid compared with the ParallelJacobiSolver.
    """

    def sweep(self):
        w = self.WJT.dot(self.x)
        z = self.x - np.multiply(self.r, (self.J.dot(w) + self.b))
        self.x = sweep_worker(
            self.K,
            w,
            z,
            self.b,
            self.mu,
            self.x,
            self.friction_solver,
            True,
        )