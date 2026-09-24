"""Time-integration algorithms for the tetrahedral soft-body solver.

The time steppers own the state-update and nonlinear-solve orchestration.  A
``SoftBody`` supplies mesh, material, loads, and force operations; it does not
implement an integration algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .forces import compute_element_jacobians
from .nonlinear import solve_lbfgs

if TYPE_CHECKING:
    from .solver import SoftBody


def _validate_step_inputs(dt: float, gravity: np.ndarray) -> np.ndarray:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    gravity = np.asarray(gravity, dtype=np.float64)
    if gravity.shape != (3,):
        raise ValueError("gravity must have shape (3,)")
    if not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must be finite")
    return gravity


def step_semi_implicit(body: "SoftBody", dt: float, gravity: np.ndarray, sync: bool = True):
    """Advance ``body`` with semi-implicit Euler."""
    gravity = _validate_step_inputs(dt, gravity)
    if body._jax_enabled:
        from .solver import _jax_step, jnp

        if body._jax_x is None:
            body._jax_x = jnp.asarray(body.x)
            body._jax_v = jnp.asarray(body.v)
        body._jax_x, body._jax_v = _jax_step(
            body._jax_x,
            body._jax_v,
            *body._static,
            body._jax_x0,
            *body._pressure_static,
            body._external_forces_device,
            jnp.asarray(gravity),
            *body.material.compute_lame_parameters(),
            body.material.model_code,
            dt,
        )
        if sync:
            body.synchronize()
        else:
            body.x, body.v = body._jax_x, body._jax_v
        return body.x, body.v

    forces = body._noninertial_forces(np.asarray(body.x))
    active = ~body.fixed
    body.v[active] += dt * (forces[active] * body.mesh.inverse_lumped_mass[active, None] + gravity)
    body.x[active] += dt * body.v[active]
    body.v[body.fixed] = 0.0
    body.x[body.fixed] = body.mesh.x0[body.fixed]
    return body.x, body.v


def step_implicit(body: "SoftBody", dt: float, gravity: np.ndarray, settings: dict | None = None):
    """Advance ``body`` with backward Euler and matrix-free L-BFGS."""
    gravity = _validate_step_inputs(dt, gravity)
    cfg = {
        "max_iterations": 25,
        "tolerance": 1.0e-6,
        "history_size": 10,
        "line_search": True,
        "max_line_search_iterations": 12,
        "line_search_reduction": 0.5,
        "line_search_c1": 1.0e-4,
        "curvature_tolerance": 1.0e-10,
        "directional_epsilon": 1.0e-6,
        "directional_residual_strategy": "tangent_action",
        "prevent_inversion": None,
        "minimum_jacobian": 0.0,
        "raise_on_failure": False,
    }
    if settings is not None:
        unknown = set(settings) - set(cfg)
        if unknown:
            raise ValueError(f"unknown implicit solver settings: {sorted(unknown)}")
        cfg.update(settings)
    if cfg["max_iterations"] < 1 or cfg["history_size"] < 0:
        raise ValueError("max_iterations must be positive and history_size cannot be negative")
    if cfg["max_line_search_iterations"] < 1:
        raise ValueError("max_line_search_iterations must be positive")
    if not (0.0 < cfg["line_search_reduction"] < 1.0):
        raise ValueError("line_search_reduction must be in (0, 1)")
    if not (0.0 <= cfg["line_search_c1"] < 1.0):
        raise ValueError("line_search_c1 must be in [0, 1)")
    if cfg["tolerance"] <= 0.0 or cfg["directional_epsilon"] <= 0.0 or cfg["curvature_tolerance"] < 0.0:
        raise ValueError("solver tolerances and directional_epsilon must be valid positive values")
    if cfg["prevent_inversion"] is not None and not isinstance(cfg["prevent_inversion"], (bool, np.bool_)):
        raise ValueError("prevent_inversion must be True, False, or None")
    if not np.isfinite(cfg["minimum_jacobian"]) or cfg["minimum_jacobian"] < 0.0:
        raise ValueError("minimum_jacobian must be finite and non-negative")
    strategy_codes = {"tangent_action": 0, "closed_form": 1, "finite_difference": 2}
    strategy = cfg["directional_residual_strategy"]
    if strategy not in strategy_codes:
        raise ValueError("directional_residual_strategy must be 'tangent_action', 'closed_form', or 'finite_difference'")

    prevent_inversion = body.material.model_code == 0 if cfg["prevent_inversion"] is None else bool(cfg["prevent_inversion"])
    if prevent_inversion and np.any(compute_element_jacobians(body.x, body.mesh) <= cfg["minimum_jacobian"]):
        raise ValueError("current state contains a collapsed or inverted element")

    x_n = np.asarray(body.x, dtype=np.float64).copy()
    v_n = np.asarray(body.v, dtype=np.float64).copy()
    free = np.flatnonzero(~body.fixed)
    dofs = np.arange(body.mesh.node_count * 3).reshape((-1, 3))[free].reshape(-1)

    if body._jax_enabled:
        from .solver import _jax_implicit_step, jnp

        x_device, v_device, info = _jax_implicit_step(
            jnp.asarray(x_n), jnp.asarray(v_n), jnp.asarray(body.mesh.x0), *body._static,
            jnp.asarray(body.mesh.lumped_mass), *body._pressure_static,
            body._external_forces_device, jnp.asarray(gravity),
            *body.material.compute_lame_parameters(), body.material.model_code, dt,
            jnp.asarray(dofs, dtype=jnp.int32),
            jnp.asarray(np.repeat(body.mesh.lumped_mass[free], 3) / (dt * dt)),
            jnp.asarray(np.repeat(body.mesh.inverse_lumped_mass[free], 3)),
            cfg["tolerance"], int(cfg["max_iterations"]), int(cfg["history_size"]),
            bool(cfg["line_search"]), int(cfg["max_line_search_iterations"]),
            cfg["line_search_reduction"], cfg["line_search_c1"], cfg["curvature_tolerance"],
            cfg["directional_epsilon"], strategy_codes[strategy],
            cfg["minimum_jacobian"], prevent_inversion,
        )
        x_device.block_until_ready()
        body._jax_x, body._jax_v = x_device, v_device
        body.x, body.v = np.asarray(x_device), np.asarray(v_device)
        body.last_implicit_info = {
            "converged": bool(info[0]), "iterations": int(info[1]),
            "final_residual_norm": float(info[2]), "initial_residual_norm": float(info[3]),
            "residual_reduction_factor": float(info[6]), "line_search_steps": int(info[4]),
            "history_length": int(info[5]),
        }
        if not body.last_implicit_info["converged"] and cfg["raise_on_failure"]:
            raise RuntimeError(f"implicit BFGS solve did not converge: {body.last_implicit_info}")
        return body.x, body.v

    x = x_n + dt * v_n
    x[body.fixed] = body.mesh.x0[body.fixed]
    mass_dof = np.repeat(body.mesh.lumped_mass[free], 3)
    inv_mass_dof = np.repeat(body.mesh.inverse_lumped_mass[free], 3)
    scale = mass_dof / (dt * dt)

    def residual(position: np.ndarray) -> np.ndarray:
        position_array = np.asarray(position).reshape(body.x.shape)
        force = body._total_forces(position_array, gravity)
        result = np.zeros(position.size, dtype=np.float64)
        result[dofs] = scale * (position_array.reshape(-1)[dofs] - x_n.reshape(-1)[dofs] - dt * v_n.reshape(-1)[dofs]) - force[free].reshape(-1)
        return result

    def directional_residual(position: np.ndarray, direction: np.ndarray) -> np.ndarray:
        position_array = np.asarray(position).reshape(body.x.shape)
        direction_array = np.asarray(direction).reshape(body.x.shape)
        df = body._directional_force(position_array, direction_array, gravity, cfg["directional_epsilon"], strategy)
        result = np.zeros(position.size, dtype=np.float64)
        result[dofs] = scale * direction_array.reshape(-1)[dofs] - df[free].reshape(-1)
        return result

    feasible = None
    if prevent_inversion:
        feasible = lambda position: bool(np.all(compute_element_jacobians(np.asarray(position).reshape(body.x.shape), body.mesh) > cfg["minimum_jacobian"]))

    x = x.reshape(-1)
    diagonal = np.zeros(x.size, dtype=np.float64)
    diagonal[dofs] = dt * dt * inv_mass_dof
    x, _, body.last_implicit_info = solve_lbfgs(x, residual, directional_residual, diagonal, cfg, feasible=feasible)
    if not body.last_implicit_info["converged"] and cfg["raise_on_failure"]:
        raise RuntimeError(f"implicit BFGS solve did not converge: {body.last_implicit_info}")
    body.x = x.reshape(x_n.shape)
    body.v = (body.x - x_n) / dt
    body.x[body.fixed] = body.mesh.x0[body.fixed]
    body.v[body.fixed] = 0.0
    return body.x, body.v


__all__ = ["step_implicit", "step_semi_implicit"]
