"""
This module implements the Gauss-Seidel solver.

The Gauss-Seidel solver is used to solve contact and joint constraints in the rigid body simulator.

One uses the problem classes to pass information about the constraints to the solver.
"""

import numpy as np

from rainbow.util.timer import Timer


def solve(engine, Problems: list, performance_data: dict, profiling_on: bool, prefix: str) -> None:
    """

    :param engine:
    :param Problems:
    :param performance_data:
    :param profiling_on:
    :param prefix:
    :return:
    """
    timer = None
    if profiling_on:
        performance_data[prefix + "residuals"] = (
                np.ones(engine.params.max_iterations, dtype=np.float64) * np.inf
        )
        timer = Timer("Gauss Seidel")
        timer.start()

    # Extract parameter values for controlling the adaptive r-factor strategy; this is for readability of the code.
    nu_reduce = engine.params.nu_reduce
    nu_increase = engine.params.nu_increase
    too_small_merit_change = engine.params.too_small_merit_change

    last_merit_values = [np.Inf for _ in Problems]
    global_last_merit = np.inf

    for iteration in range(engine.params.max_iterations):

        # Keep track of the overall merit across all problems we are solving
        global_merit = 0

        for index, problem in enumerate(Problems):
            last_merit = last_merit_values[index]

            problem.sweep()  # This will update the current iterate x

            merit = problem.compute_merit_value()

            # Update r-factors to ensure we have a convergent scheme
            if merit > last_merit:
                # Divergence was detected: we will reduce R-factor
                problem.change_r_factors(nu_reduce)
                #  Next, we roll back the current iterate to the last known good solution
                problem.rollback_iterate()
                # Update the global merit, so we later can test if all problems have converged
                global_merit = np.maximum(global_merit, last_merit)
            else:
                # The Scheme is convergent, so we accept the iterate as the current best solution
                problem.accept_iterate()
                # Test if convergence was too slow and increase R-factor if this is the case
                if last_merit - merit < too_small_merit_change:
                    problem.change_r_factors(nu_increase)
                # Update merit value for next iteration
                last_merit_values[index] = merit
                # Update the global merit, so we later can test if all problems have converged
                global_merit = np.maximum(global_merit, merit)

        if profiling_on:
            performance_data[prefix + "residuals"][iteration] = global_merit
        # Test for absolute convergence
        if global_merit < engine.params.absolute_tolerance:
            if profiling_on:
                timer.end()
                performance_data[prefix + "iterations"] = iteration
                performance_data[prefix + "exitcode"] = 1
                performance_data[prefix + "solver_time"] = timer.elapsed
            return
        # Test for relative convergence
        if np.abs(global_merit - global_last_merit) < engine.params.relative_tolerance * global_last_merit:
            if profiling_on:
                timer.end()
                performance_data[prefix + "iterations"] = iteration
                performance_data[prefix + "exitcode"] = 2
                performance_data[prefix + "solver_time"] = timer.elapsed
            return
        # Update the global last merit value, so we are ready for the next iteration
        global_last_merit = global_merit

    # If this code is reached, then it means the method did not converge within the given number of maximum iterations.
    if profiling_on:
        timer.end()
        performance_data[prefix + "iterations"] = engine.params.max_iterations
        performance_data[prefix + "exitcode"] = 0
        performance_data[prefix + "solver_time"] = timer.elapsed
