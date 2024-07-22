import numpy as np

import rainbow.simulators.prox_rigid_bodies.problems as PROBLEMS
from rainbow.util.timer import Timer


def solve(engine, Problems: list, stats: dict, debug_on: bool, prefix: str) -> None:

    last_merit = np.Inf

    for problem in Problems:
        problem.init(engine)  # sets up x, sol, error, merit and r

    for iteration in range(engine.params.max_iterations):
        for problem in Problems:
            problem.sweep()

            merit = problem.compute_merit_value()



        merits = np.array([ problem.get_merit() for problem in Problems ])
        merit = np.max(merits)

        if debug_on:
            stats[prefix + "lambda"][iteration] = x
            stats[prefix + "residuals"][iteration] = merit
        # Test stopping criteria
        if merit < engine.params.absolute_tolerance:
            if debug_on:
                stats[prefix + "iterations"] = iteration
                stats[prefix + "exitcode"] = 1
                timer.end()
                stats[prefix + "solver_time"] = timer.elapsed
            return x, stats
        if np.abs(merit - last_merit) < engine.params.relative_tolerance * last_merit:
            if debug_on:
                stats[prefix + "iterations"] = iteration
                stats[prefix + "exitcode"] = 2
                timer.end()
                stats[prefix + "solver_time"] = timer.elapsed
            return x, stats

        # Update r-factors
        if merit > last_merit:
            # Divergence detected: reduce R-factor and roll-back solution to last known good iterate!
            problem.change_r_factors(nu_reduce)
            problem.rollback_iterate()
            if debug_on:
                stats[prefix + "reject"][iteration] = True
        else:
            if last_merit - merit < too_small_merit_change:
                problem.change_r_factors(nu_reduce)
            last_merit = merit
            problem.accept_iterate()
