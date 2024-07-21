import rainbow.simulators.prox_rigid_bodies.collision_detection as CD
import rainbow.simulators.proximal_contact.prox_solvers as CONTACT_SOLVERS
from rainbow.simulators.prox_rigid_bodies.types import *
from rainbow.util.timer import Timer
import numpy as np



class SemiImplicitStepper:
    """
    This class implements a semi-implicit first order Euler time-stepper.
    """

    def __init__(self, engine: Engine) -> None:
        self.log = []

    def step(self, dt: float, engine: Engine, debug_on: bool) -> None:
        """
        This is the main simulation method that is responsible for stepping time
        forward to the next time-step.

        :param dt:        The time-step to step time forward in the engine.
        :param engine:    The engine that holds all the rigid bodies.
        :param debug_on:  A boolean flag to toggle debug mode.
        :return:          None
        """
        timer = None
        if debug_on:
            timer = Timer("Stepper")
            timer.start()
        stats = {}

        x = get_position_vector(engine)
        u = get_velocity_vector(engine)
        f_ext = compute_total_external_forces(x, u, engine)
        du_contact = np.zeros(u.shape, dtype=np.float64)
        W = compute_inverse_mass_matrix(x, engine)
        du_ext = W.dot(dt * f_ext)

        stats = CD.run_collision_detection(engine, stats, debug_on)

        J = None
        WJT = None
        if len(engine.contact_points) > 0:
            J = compute_contact_jacobian_matrix(engine)
            WJT = W.dot(J.T)
            v = J.dot(u)
            if engine.params.use_pre_stabilization:
                g = get_pre_stabilization_vector(dt, v, engine)
            else:
                g = np.zeros(v.shape, dtype=np.float64)

            if engine.params.use_bounce:
                e = get_restitution_vector(engine)
            else:
                e = np.zeros(v.shape, dtype=np.float64)

            mu = get_friction_coefficient_vector(engine)
            b = np.multiply(1 + e, v) + J.dot(du_ext) + g
            sol, stats = CONTACT_SOLVERS.solve(
                J, WJT, b, mu, CONTACT_SOLVERS.prox_sphere, engine, stats, debug_on,
                prefix="",
                scheme=engine.params.proximal_solver
            )
            du_contact = WJT.dot(sol)

        du_total = du_ext + du_contact
        u += du_total
        position_update(x, u, dt, engine)
        set_position_vector(x, engine)
        set_velocity_vector(u, engine)

        if engine.params.use_post_stabilization:
            if len(engine.contact_points) > 0:
                stats = apply_post_stabilization(J, WJT, x, engine, stats, debug_on)

        if debug_on:
            timer.end()
            stats["stepper_time"] = timer.elapsed
            stats["dt"] = dt
            stats["contact_points"] = len(engine.contact_points)
            kinetic_energy, potential_energy = get_total_energy(engine)
            stats["kinetic_energy"] = kinetic_energy
            stats["potential_energy"] = potential_energy
            stats["max_penetration"] = get_largest_gap_error(engine)
            self.log.append(stats)
