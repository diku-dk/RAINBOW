import numpy as np
import numba as nb
from collections import defaultdict
import rainbow.simulators.proximal_contact.blocking as BLOCKING
from rainbow.simulators.proximal_contact.solver_interface import SolverInterface


class GaussSeidelSolver(SolverInterface):
    """ Serial Gauss Seidel Solver.
    """

    def sweep(self):
        """ Sweep the contact force.
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

            # Solve:         x_f = prox_C( x_f - r (A x_f + b))
            x_b[1], x_b[2], x_b[3] = self.friction_solver(z_b[1], z_b[2], z_b[3], mu_k, x_b[0])
            # Put updated contact forces back into solution vector
            self.x[block] = x_b
            # Get the change in the x_block
            np.subtract(x_b, delta, delta)
            # Update w
            w += self.WJT.tocsr()[:, block].dot(delta)


@nb.njit(parallel=True, nogil=True, cache=True)
def sweep_worker(color_group, J, WJT, b, mu, r, x, w, delta_ws, friction_solver):
    """ Sweep worker for parallel Gauss Seidel solver.

    :param color_group: The color group, the value is  the block location, containing the start and end index.
    :param J: The contact jacobi matrix.
    :param WJT: The WJ^T matrix, here W = M^{-1}, and the M is the mass matrix.
    :param b: b = Ju^t + ∆t JM^{-1}h + EJu^t
    :param mu: The coefficient of friction.
    :param r: r-factor value.
    :param x: The current contact force.
    :param w: The WJT.dot(x).
    :param delta_ws: A array of the delta_w, each delta_w = WJT.dot(delta_x), here delta_x is the change of the contact force.
    :param friction_solver: The proximal operator of friction cone function.
    :return: The new contact force and the new delta_ws.
    """

    for i in nb.prange(len(color_group)):
        block_start, block_end = color_group[i]
        block = np.arange(block_start, block_end)
        x_b = x[block]
        r_b = r[block]
        b_b = b[block]

        delta = x_b.copy()
        z_b = x_b - np.multiply(r_b, (J.dot(w)[block] + b_b))

        x_b[0] = np.max(np.array([0.0, z_b[0]]))

        mu_k = mu[block_start // 4]
        x_b[1], x_b[2], x_b[3] = friction_solver(z_b[1], z_b[2], z_b[3], mu_k, x_b[0])

        np.subtract(x_b, delta, delta)
        delta_w = WJT[:, block].dot(delta)

        x[block] = x_b
        delta_ws[block_start//4] = delta_w

    return x, delta_ws


class ParallelGaussSeidelSolver(SolverInterface):
    """ Parallel Gauss Seidel Solver.
    """
    
    def sweep(self):
        """ Sweep the contact force.
        """
        # Compute the color group
        color_groups = defaultdict(list)
        G = BLOCKING.build_contact_graph(self.engine.contact_points, self.engine.simulator_type)
        color_groups = BLOCKING.greedy_graph_coloring(G)

        # Set the number of threads
        nb.set_num_threads(4 if (nb.config.NUMBA_NUM_THREADS // 2) >= 4 else nb.config.NUMBA_NUM_THREADS // 2)

        # Compute the new contact force
        w0 = self.WJT.dot(self.x)
        delta_ws = np.zeros((self.WJT.shape[1]//4, w0.shape[0]), dtype=np.float64)
        for color_group in color_groups.values():
            w = w0.copy()
            block_start, _ = color_group[0]
            for i in range(block_start // 4):
                w += delta_ws[i]

            self.x, delta_ws = sweep_worker(np.array(color_group), 
                                            self.J.toarray(), 
                                            self.WJT.toarray(), 
                                            self.b, 
                                            self.mu, 
                                            self.r, 
                                            self.x, 
                                            w,
                                            delta_ws, 
                                            self.friction_solver)
