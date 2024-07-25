import rainbow.simulators.prox_rigid_bodies.collision_detection as CD
import rainbow.simulators.prox_rigid_bodies.gauss_seidel_solver as SOLVER
import rainbow.simulators.prox_rigid_bodies.state_storage as STORAGE
import rainbow.simulators.prox_rigid_bodies.problems as PROBLEMS
from rainbow.simulators.prox_rigid_bodies.types import *
from rainbow.util.timer import Timer
import numpy as np


def apply_post_stabilization(J, WJT, x, engine, profile_data: dict, profiling_on) -> dict:
    """
    Apply post-stabilization to remove gap errors in the simulation.
    This function solves for actual displacements that will make all gap-values become non-negative. This is
    different from pre-stabilization. Pre-stabilization just adds a bit more to the contact problem to make contact
    forces stronger, so the forces can remove more of the error.
    In post-stabilization we measure the actual error and solve for instantaneous displacement that will resolve all
    the errors, taking coupling and non-linearity and everything into account.

    :param J:         The contact Jacobian matrix.
    :param WJT:       The transpose contact Jacobian matrix pre-multiplied by the inverse mass matrix.
    :param x:         The generalized position vector.
    :param engine:    The engine that holds all the rigid bodies.
    :param stats:     A dictionary that collects performance measurements and statistics when debugging mode is on.
    :param debug_on:  A boolean flag to toggle debug mode.
    :return:          A dictionary with collected statistics and performance measurements.
    """
#    K = len(engine.contact_points)
#    g = np.zeros(4 * K, dtype=np.float64)
#    for k in range(K):
#        cp = engine.contact_points[k]
#        if cp.g < -engine.params.min_gap_value:
#            g[4 * k + 0] = cp.g
#    # If the gap to correct is all zeros, then just return
#    if not g.any():
#        return stats
#    mu = np.zeros(K, dtype=np.float64)
#    CONTACT_SOLVER.solve(
#        J, WJT, g, mu, CONTACT_SOLVER.prox_origin, engine, profile_data, profiling_on,
#        prefix="post_stabilization_"
#    )
#    vector_positional_update = WJT.dot(sol)
#    FUNC.Bodies.position_update(x, vector_positional_update, 1, engine)
#    return stats
pass


class SemiImplicitStepper:
    """
    This class implements a semi-implicit first order Euler time-stepper.
    """

    def __init__(self, engine: Engine) -> None:
        self.log = []

    def step(self, dt: float, engine: Engine, profiling_on: bool) -> None:
        """
        This is the main simulation method that is responsible for stepping time
        forward to the next time-step.

        :param dt:        The time-step to step time forward in the engine.
        :param engine:    The engine that holds all the rigid bodies.
        :param debug_on:  A boolean flag to toggle debug mode.
        :return:          None
        """
        timer = None
        if profiling_on:
            timer = Timer("SemiImplicitStepper")
            timer.start()
        profile_data = {}

        state = STORAGE.StateStorage()
        state.initialize(engine)

        CD.run_collision_detection(engine, profile_data, profiling_on)

        problems = []
        if len(engine.contact_points) > 0:
            problems.append(PROBLEMS.Contacts())
        if len(engine.hinges) > 0:
            problems.append(PROBLEMS.Hinges())

        for prb in problems:
            prb.initialize(dt, state, engine)

        SOLVER.solve(engine, problems, profile_data, profiling_on, prefix="")

        delta_u = dt * state.delta_u_ext
        for prb in problems:
            delta_u += prb.finalize()

        state.velocity_update(delta_u)
        state.position_update(dt, engine)
        state.finalize(engine)

        if engine.params.use_post_stabilization and len(engine.contact_points) > 0:
            projection = PROBLEMS.PostStabilization()
            projection.initialize(dt, state, engine)

            if not projection.g.any():
                SOLVER.solve(engine, [projection], profile_data, profiling_on, prefix="post_stabilization")
                delta_r = projection.finalize()
                state.position_update(dt=dt, engine=engine, u=delta_r)

        if profiling_on:
            timer.end()
            profile_data["stepper_time"] = timer.elapsed
            profile_data["dt"] = dt
            profile_data["contact_points"] = len(engine.contact_points)
            kinetic_energy, potential_energy = state.compute_total_energy(engine)
            profile_data["kinetic_energy"] = kinetic_energy
            profile_data["potential_energy"] = potential_energy
            profile_data["max_penetration"] = PROBLEMS.Contacts.get_largest_penetration(engine)
            self.log.append(profile_data)
