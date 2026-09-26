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
            body._jax_x = jnp.asarray(body._x)
            body._jax_v = jnp.asarray(body._v)
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
        body._jax_host_synced = False
        if sync:
            body.synchronize()
        else:
            # Keep the host snapshot unchanged. Rendering can explicitly call
            # get_x()/get_v(), while the solver continues from device state.
            return body._jax_x, body._jax_v
        return body.get_x(), body.get_v()

    current_x = np.asarray(body._x)
    current_v = np.asarray(body._v)
    next_x = current_x.copy()
    next_v = current_v.copy()
    forces = body._noninertial_forces(current_x)
    active = ~body.fixed
    next_v[active] += dt * (forces[active] * body.mesh.inverse_lumped_mass[active, None] + gravity)
    next_x[active] += dt * next_v[active]
    next_v[body.fixed] = 0.0
    next_x[body.fixed] = body.mesh.x0[body.fixed]
    body._v = next_v
    body._x = next_x
    return body.get_x(), body.get_v()


def step_implicit(body: "SoftBody", dt: float, gravity: np.ndarray, settings: dict | None = None):
    """Advance ``body`` with backward Euler and matrix-free L-BFGS.

    NumPy uses residual-norm backtracking with a trial-state Jacobian guard and
    can additionally use a preconditioned gradient fallback or bounded
    ``globalization="watchdog"`` mode. These are controlled by
    ``enable_gradient_fallback``, ``max_watchdog_steps``, and
    ``watchdog_growth_factor``. Diagnostics are stored in
    ``body.last_implicit_info``. The JAX implementation supports the
    backtracking contract and reports the same diagnostic keys; watchdog mode
    is intentionally NumPy-only because it requires a dynamic restoration
    policy.
    """
    gravity = _validate_step_inputs(dt, gravity)
    if body._jax_enabled and body._jax_x is not None:
        body.synchronize()
    cfg = {
        "max_iterations": 25,
        "absolute_tolerance": 1.0e-6,
        "relative_tolerance": 1.0e-6,
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
        "globalization": "backtracking",
        "enable_gradient_fallback": True,
        "max_watchdog_steps": 3,
        "watchdog_growth_factor": 1.1,
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
    if cfg["absolute_tolerance"] < 0.0 or cfg["relative_tolerance"] < 0.0 or (cfg["absolute_tolerance"] == 0.0 and cfg["relative_tolerance"] == 0.0) or cfg["directional_epsilon"] <= 0.0 or cfg["curvature_tolerance"] < 0.0:
        raise ValueError("solver tolerances and directional_epsilon must be valid positive values")
    if cfg["prevent_inversion"] is not None and not isinstance(cfg["prevent_inversion"], (bool, np.bool_)):
        raise ValueError("prevent_inversion must be True, False, or None")
    if not np.isfinite(cfg["minimum_jacobian"]) or cfg["minimum_jacobian"] < 0.0:
        raise ValueError("minimum_jacobian must be finite and non-negative")
    if cfg["globalization"] not in ("backtracking", "watchdog"):
        raise ValueError("globalization must be 'backtracking' or 'watchdog'")
    if not isinstance(cfg["enable_gradient_fallback"], (bool, np.bool_)):
        raise ValueError("enable_gradient_fallback must be boolean")
    if cfg["max_watchdog_steps"] < 1:
        raise ValueError("max_watchdog_steps must be positive")
    if not np.isfinite(cfg["watchdog_growth_factor"]) or cfg["watchdog_growth_factor"] < 1.0:
        raise ValueError("watchdog_growth_factor must be finite and at least one")
    strategy_codes = {"tangent_action": 0, "closed_form": 1, "finite_difference": 2}
    strategy = cfg["directional_residual_strategy"]
    if strategy not in strategy_codes:
        raise ValueError("directional_residual_strategy must be 'tangent_action', 'closed_form', or 'finite_difference'")

    prevent_inversion = body.material.model_code == 0 if cfg["prevent_inversion"] is None else bool(cfg["prevent_inversion"])
    if prevent_inversion and np.any(compute_element_jacobians(body._x, body.mesh) <= cfg["minimum_jacobian"]):
        raise ValueError("current state contains a collapsed or inverted element")

    x_n = np.asarray(body._x, dtype=np.float64).copy()
    v_n = np.asarray(body._v, dtype=np.float64).copy()
    free = np.flatnonzero(~body.fixed)
    dofs = np.arange(body.mesh.node_count * 3).reshape((-1, 3))[free].reshape(-1)

    if body._jax_enabled:
        from .solver import _jax_implicit_step, jnp

        if cfg["globalization"] != "backtracking":
            raise ValueError("JAX implicit stepping supports globalization='backtracking' only")
        if not cfg["enable_gradient_fallback"]:
            raise ValueError("JAX implicit stepping requires enable_gradient_fallback=True")
        if cfg["max_watchdog_steps"] != 3 or cfg["watchdog_growth_factor"] != 1.1:
            raise ValueError("watchdog settings are unsupported by the JAX implicit kernel")

        x_device, v_device, info = _jax_implicit_step(
            jnp.asarray(x_n), jnp.asarray(v_n), jnp.asarray(body.mesh.x0), *body._static,
            jnp.asarray(body.mesh.lumped_mass), *body._pressure_static,
            body._external_forces_device, jnp.asarray(gravity),
            *body.material.compute_lame_parameters(), body.material.model_code, dt,
            jnp.asarray(dofs, dtype=jnp.int32),
            jnp.asarray(np.repeat(body.mesh.lumped_mass[free], 3) / (dt * dt)),
            jnp.asarray(np.repeat(body.mesh.inverse_lumped_mass[free], 3)),
            cfg["absolute_tolerance"], cfg["relative_tolerance"], int(cfg["max_iterations"]), int(cfg["history_size"]),
            bool(cfg["line_search"]), int(cfg["max_line_search_iterations"]),
            cfg["line_search_reduction"], cfg["line_search_c1"], cfg["curvature_tolerance"],
            cfg["directional_epsilon"], strategy_codes[strategy],
            cfg["minimum_jacobian"], prevent_inversion,
        )
        x_device.block_until_ready()
        body._jax_x, body._jax_v = x_device, v_device
        object.__setattr__(body, "_x", np.asarray(x_device))
        object.__setattr__(body, "_v", np.asarray(v_device))
        body._jax_host_synced = True
        history_count = int(info[1]) + (0 if bool(info[0]) else 1)
        body.last_implicit_info = {
            "converged": bool(info[0]), "iterations": int(info[1]),
            "final_residual_norm": float(info[2]), "initial_residual_norm": float(info[3]),
            "convergence_threshold": float(info[4]), "residual_reduction_factor": float(info[7]), "line_search_steps": int(info[5]),
            "history_length": int(info[6]),
            "gradient_fallback_steps": int(info[8]),
            "direction_fallback_steps": int(info[9]),
            "watchdog_steps": 0,
            "watchdog_acceptances": 0,
            "rescue_steps": int(info[10]),
            "residual_norm_history": np.asarray(info[11], dtype=np.float64)[:history_count].tolist(),
        }
        if not body.last_implicit_info["converged"] and cfg["raise_on_failure"]:
            raise RuntimeError(f"implicit BFGS solve did not converge: {body.last_implicit_info}")
        return body.get_x(), body.get_v()

    x = x_n + dt * v_n
    x[body.fixed] = body.mesh.x0[body.fixed]
    mass_dof = np.repeat(body.mesh.lumped_mass[free], 3)
    inv_mass_dof = np.repeat(body.mesh.inverse_lumped_mass[free], 3)
    scale = mass_dof / (dt * dt)

    def residual(position: np.ndarray) -> np.ndarray:
        position_array = np.asarray(position).reshape(body._x.shape)
        force = body._total_forces(position_array, gravity)
        result = np.zeros(position.size, dtype=np.float64)
        result[dofs] = scale * (position_array.reshape(-1)[dofs] - x_n.reshape(-1)[dofs] - dt * v_n.reshape(-1)[dofs]) - force[free].reshape(-1)
        return result

    def directional_residual(position: np.ndarray, direction: np.ndarray) -> np.ndarray:
        position_array = np.asarray(position).reshape(body._x.shape)
        direction_array = np.asarray(direction).reshape(body._x.shape)
        df = body._directional_force(position_array, direction_array, gravity, cfg["directional_epsilon"], strategy)
        result = np.zeros(position.size, dtype=np.float64)
        result[dofs] = scale * direction_array.reshape(-1)[dofs] - df[free].reshape(-1)
        return result

    feasible = None
    if prevent_inversion:
        feasible = lambda position: bool(np.all(compute_element_jacobians(np.asarray(position).reshape(body._x.shape), body.mesh) > cfg["minimum_jacobian"]))

    x = x.reshape(-1)
    diagonal = np.zeros(x.size, dtype=np.float64)
    diagonal[dofs] = dt * dt * inv_mass_dof
    x, _, body.last_implicit_info = solve_lbfgs(x, residual, directional_residual, diagonal, cfg, feasible=feasible)
    if not body.last_implicit_info["converged"] and cfg["raise_on_failure"]:
        raise RuntimeError(f"implicit BFGS solve did not converge: {body.last_implicit_info}")
    body._x = x.reshape(x_n.shape)
    body._v = (body._x - x_n) / dt
    body._x[body.fixed] = body.mesh.x0[body.fixed]
    body._v[body.fixed] = 0.0
    return body.get_x(), body.get_v()


__all__ = ["step_implicit", "step_semi_implicit"]
