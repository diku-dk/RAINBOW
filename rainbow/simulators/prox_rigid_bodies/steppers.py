"""
This module contains time steppers for the rigid body simulator.

Currently, we only support semi-implicit first order time-stepping.
"""

import rainbow.simulators.prox_rigid_bodies.collision_detection as CD
import rainbow.simulators.prox_rigid_bodies.gauss_seidel_solver as SOLVER
import rainbow.simulators.prox_rigid_bodies.state_storage as STORAGE
import rainbow.simulators.prox_rigid_bodies.constraints as PROBLEMS
from rainbow.simulators.prox_rigid_bodies.types import *
from rainbow.util.timer import Timer


def _apply_post_stabilization(engine: Engine,
                              contact_problem: PROBLEMS.Contacts,
                              state: STORAGE.StateStorage,
                              profile_data: dict[str, any],
                              profiling_on: bool
                              ) -> None:
    """
    This function is used to apply post-stabilization (back projection) to a rigid body simulation.

    :param engine:            The engine containing the simulation.
    :param contact_problem:   The current contact problem of the simulation.
    :param state:             The current body state of the simulation.
    :param profile_data:      Dictionary to store profiling data if profiling is enabled.
    :param profiling_on:      Boolean flag to enable profiling.
    """
    projection = PROBLEMS.PostStabilization(contact_problem=contact_problem)
    projection.initialize(dt=1.0, state=state, engine=engine)
    if not projection.g.any():
        return
    SOLVER.solve(engine, [projection], profile_data, profiling_on, prefix="post_stabilization_")
    delta_r = projection.finalize_solution()
    state.position_update(dt=1.0, engine=engine, u=delta_r)


class SemiImplicitStepper:
    """
    This class implements a semi-implicit first order Euler time-stepper.
    """

    def __init__(self) -> None:
        self.log: list[dict[str, any]] = []

    def step(self, dt: float, engine: Engine, profiling_on: bool) -> None:
        """
        Step the simulation forward by a given time-step.

        :param dt:            The time-step size.
        :param engine:        The engine holding all the rigid bodies.
        :param profiling_on:  Boolean flag to enable profiling.
        """
        timer = Timer("SemiImplicitStepper") if profiling_on else None
        profile_data = {}

        if profiling_on:
            timer.start()

        state = STORAGE.StateStorage()
        state.copy_from_engine(engine)

        delta_u = dt * state.delta_u_ext

        CD.run_collision_detection(engine, profile_data, profiling_on)

        problems: list[PROBLEMS.Problem] = []
        contact_problem = PROBLEMS.Contacts()
        hinges_problem = PROBLEMS.Hinges()
        sliding_joints_problem = PROBLEMS.SlidingJoints()

        if engine.contact_points:
            problems.append(contact_problem)
        if engine.hinges:
            problems.append(hinges_problem)
        if engine.sliding_joints:
            problems.append(sliding_joints_problem)

        if problems:

            for prb in problems:
                prb.initialize(dt, state, engine)

            SOLVER.solve(engine, problems, profile_data, profiling_on, prefix="")

            for prb in problems:
                delta_u += prb.finalize_solution()

        state.velocity_update(delta_u)
        state.position_update(dt, engine)

        if engine.contact_points and engine.params.use_post_stabilization:
            _apply_post_stabilization(engine, contact_problem, state, profile_data, profiling_on)

        state.copy_to_engine(engine)

        if profiling_on:
            timer.end()

            kinetic_energy, potential_energy = state.compute_total_energy(engine)

            profile_data["stepper_time"] = timer.elapsed
            profile_data["dt"] = dt
            profile_data["contact_points"] = len(engine.contact_points)
            profile_data["kinetic_energy"] = kinetic_energy
            profile_data["potential_energy"] = potential_energy
            profile_data["max_penetration"] = PROBLEMS.Contacts.get_largest_penetration(engine)
            self.log.append(profile_data)
