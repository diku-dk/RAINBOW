"""
This module contains the abstract base class for constraint problem definitions.

TODO 2024-07-27 Kenny: The current design is great for supporting a generic way to extend simulator with new
 constraint types.
  The drawback is that the there are some redundancy in the individual problem implementations.
  Like in the methods for computing the Jacobian matrices and the sweep method.
  Here 80% or more of the code is generic boiler plate and duplicate.
  This is by choice due to to Python performance worries.
  Having too many nested function calls could be awfully.
  Instead we are suffering redundant code by writing out the code in full.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

import rainbow.simulators.prox_rigid_bodies.state_storage as STORAGE
from rainbow.simulators.prox_rigid_bodies.types import *


class Problem(ABC):
    """
    This is the base class for all constraint problems.

    It defines all the common functionality to support the Gauss-Seidel Proximal Operator Solver.

    To add a new type of constraint to the rigid body simulator, one simply derives from this class.
    """

    def __init__(self, name: str):
        self.name: str = name  # The name of the problem type that is solved.
        self.error: Optional[np.ndarray] = None  # The residual vector, basically error = x - sol.
        self.sol: Optional[np.ndarray] = None  # Last known good solution.
        self.x: Optional[np.ndarray] = None  # The current solution.
        self.r: Optional[np.ndarray] = None  # The R-factor values.

    def rollback_iterate(self) -> None:
        """
        The solver involves this function when divergence has been detected.

        This is to roll back the current solution to a previous best known solution.
        """
        np.copyto(self.x, self.sol)  # x = sol

    def accept_iterate(self) -> None:
        """
        The solver calls this method when the current solution is convergent
        and improves the merit value of the constraint problem.
        """
        np.copyto(self.sol, self.x)  # sol = x

    def compute_merit_value(self) -> float:
        """
        This method is called when the solver wants to know the current metrit value for the problem.

        :return:  The computed metrit value.
        """
        np.subtract(self.x, self.sol, self.error)  # error = x - sol
        value = np.linalg.norm(self.error, np.inf)
        return value

    def change_r_factors(self, rate) -> None:
        """
        The solver invokes this method when it tries to adjust the R-factor values to obtain better
        convergence behavior.

        :param rate:     The rate is a positive number multiplied with the current R-values.
        """
        np.multiply(rate, self.r, self.r)

    @abstractmethod
    def initialize(self, dt: float, state: STORAGE.StateStorage, engine: Engine) -> None:
        """
        Initialize the problem.

        This method sets up internal variables that are needed when running a solver.

        This often involves computing a Jacobian matrix and/or a constraint vector.

        :param dt:             The time step used by the time stepping method.
        :param state:          The current body states information that should be used to initialize the problem.
        :param engine:         The engine instance that holds all other information about the simulation.
        """
        pass

    @abstractmethod
    def finalize_solution(self) -> np.ndarray:
        """
        This method is called after the solver is completed to finalize_solution the computation of the solution.

        This essential means computing the "velocity" change caused by the constraint forces.

        @return: The velocity change vector.
        """
        pass

    @abstractmethod
    def sweep(self) -> None:
        """
        Solver invokes this to compute the proximal operator update for all constraints.
        """
        pass
