import numpy as np
from abc import ABC, abstractmethod
from rainbow.util.timer import Timer


class SolverInterface(ABC):
    """ A interface class of proximal solver. This class is used to define the Jacobi or Gauss-Seidel scheme of proximal solver.
    """

    def __init__(self, J, WJT, b, mu, friction_solver, engine, stats, debug_on, prefix) -> None:
        """ Initialize the solver.

        :param J: contact jacobi matrix
        :param WJT: WJT matrix, here W = M^{-1}.
        :param b: b = Ju^t + ∆t JM^{-1}h + EJu^t
        :param mu: The coefficient of friction.
        :param friction_solver: The proximal operator of the friction cone.
        :param engine: The engine object.
        :param stats: The statistics information.
        :param debug_on: Whether to debug (Ture or False).
        :param prefix: The prefix of the statistics information.
        """
       
        self.J = J # The contact jacobi matrix
        self.WJT = WJT # The WJ^T matrix, here W = M^{-1}.
        self.b = b # b = Ju^t + ∆t JM^{-1}h + EJu^t
        self.mu = mu # The coefficient of friction.
        self.friction_solver = friction_solver # The proximal operator of the friction cone.
        self.engine = engine # The engine object.
        self.stats = stats # The statistics information.
        self.debug_on = debug_on # Whether to debug (Ture or False).
        self.prefix = prefix # The prefix of the statistics information.
        self.timer = None # The timer used to record the time
        self.K = len(engine.contact_points) # Number of contact points.
        self.iteration = 0 # The current iteration.
        self.x = np.zeros(self.b.shape, dtype=np.float64)  # The current iterate
        self.sol = np.zeros(
            self.b.shape, dtype=np.float64
        )  # The last best known solution, used for restarting if divergence
        self.error = np.zeros(self.b.shape, dtype=np.float64)  # The residual vector
        self.last_merit = np.Inf  # The last merit value
        
    def compute_initial_r_factor(self):
        """ Compute the initial r-factor value.
        """
        delassus_diag = np.sum(self.J.multiply(self.WJT.T), axis=1).A1
        delassus_diag[delassus_diag == 0] = 1
        self.r = 0.1 / delassus_diag
    
    def update_r_factor_and_handle_sol_state(self):
        """ Update the r-factor value.
        """
        nu_reduce = self.engine.params.nu_reduce
        nu_increase = self.engine.params.nu_increase
        too_small_merit_change = self.engine.params.too_small_merit_change

        if self.merit > self.last_merit:
            # Divergence detected: reduce R-factor and roll-back solution to last known good iterate!
            np.multiply(nu_reduce, self.r, self.r)
            np.copyto(self.x, self.sol)
            if self.debug_on:
                self.stats[self.prefix + "reject"][self.iteration] = True
        else:
            if self.last_merit - self.merit < too_small_merit_change:
                # Convergence is slow: increase r-factor
                np.multiply(nu_increase, self.r, self.r)
            # Convergence detected: accept x as better solution
            self.last_merit = self.merit
            np.copyto(self.sol, self.x)

    def initialize_stats(self, timer):
        """ Initialize the statistics information, when debug_on is True.

        :param timer: The timer used to record the time.
        """
        self.timer = timer
        self.stats[self.prefix + "residuals"] = (
                np.ones(self.engine.params.max_iterations, dtype=np.float64) * np.inf
            )
        self.stats[self.prefix + "lambda"] = np.zeros(
                [self.engine.params.max_iterations] + list(self.b.shape), dtype=np.float64
            )
        self.stats[self.prefix + "reject"] = np.zeros(self.engine.params.max_iterations, dtype=bool)
        self.stats[self.prefix + "exitcode"] = 0
        self.stats[self.prefix + "iterations"] = self.engine.params.max_iterations
        self.timer.start()

    def check_convergence(self):
        """ Check the convergence.

        :return: if True, the solver converges, otherwise(False), the solver does not converge.
        """
        np.subtract(self.x, self.sol, self.error)
        self.merit = np.linalg.norm(self.error, np.inf)

        if self.debug_on:
            self.stats[self.prefix + "lambda"][self.iteration] = self.x
            self.stats[self.prefix + "residuals"][self.iteration] = self.merit

        if self.merit < self.engine.params.absolute_tolerance:
            if self.debug_on:
                self.stats[self.prefix + "iterations"] = self.iteration
                self.stats[self.prefix + "exitcode"] = 1
                self.timer.end()
                self.stats[self.prefix + "solver_time"] = self.timer.elapsed
            return True
        if np.abs(self.merit - self.last_merit) < self.engine.params.relative_tolerance * self.last_merit:
            if self.debug_on:
                self.stats[self.prefix + "iterations"] = self.iteration
                self.stats[self.prefix + "exitcode"] = 2
                self.timer.end()
                self.stats[self.prefix + "solver_time"] = self.timer.elapsed
            return True
        return False
    
    @abstractmethod
    def sweep(self):
        """ Sweep the contact points to compute the new contact force.
            This method should be implemented in the derived class.
        """
        pass
    
    def solve(self):
        """ Solve the contact problem.

        :return: The new contact force and the statistics information.
        """
        if self.debug_on:
            self.initialize_stats(Timer("CONTACT_SOLVER"))

        # Compute initial r-factor value
        self.compute_initial_r_factor()

        for iteration in range(self.engine.params.max_iterations):
            self.iteration = iteration
            self.sweep()
            # Check convergence
            if self.check_convergence():
                return self.sol, self.stats
            # Update r-factors, and handle the state of the solution based on the convergence and divergence criteria.:
            self.update_r_factor_and_handle_sol_state()

        # If this point of the code is reached then it means the method did not converge within the given iterations.
        if self.debug_on:
            self.timer.end()
            self.stats[self.prefix + "solver_time"] = self.timer.elapsed
        return self.sol, self.stats