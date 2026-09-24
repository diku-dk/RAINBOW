"""Reusable NumPy nonlinear-solver building blocks for soft-body stepping."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .types import Array


Residual = Callable[[Array], Array]
Feasibility = Callable[[Array], bool]


def compute_lbfgs_direction(
    gradient: Array,
    history_s: list[Array],
    history_y: list[Array],
    history_rho: list[float],
    diagonal: Array,
) -> Array:
    """Apply the limited-memory BFGS inverse-Hessian approximation."""
    direction = -np.asarray(gradient, dtype=np.float64).copy()
    alphas: list[float] = []
    for s, y, rho in reversed(list(zip(history_s, history_y, history_rho))):
        alpha = rho * float(np.dot(s, direction))
        alphas.append(alpha)
        direction -= alpha * y
    direction *= diagonal
    for (s, y, rho), alpha in zip(
        zip(history_s, history_y, history_rho), reversed(alphas)
    ):
        direction += s * (alpha - rho * float(np.dot(y, direction)))
    return direction


def update_lbfgs_history(
    history_s: list[Array],
    history_y: list[Array],
    history_rho: list[float],
    s: Array,
    y: Array,
    history_size: int,
    curvature_tolerance: float,
) -> bool:
    """Append a curvature pair when it is numerically safe to use."""
    curvature = float(np.dot(s, y))
    threshold = curvature_tolerance * np.linalg.norm(s) * max(np.linalg.norm(y), 1.0e-30)
    if not np.isfinite(curvature) or curvature <= threshold or history_size == 0:
        return False
    history_s.append(np.asarray(s, dtype=np.float64).copy())
    history_y.append(np.asarray(y, dtype=np.float64).copy())
    history_rho.append(1.0 / curvature)
    while len(history_s) > history_size:
        history_s.pop(0)
        history_y.pop(0)
        history_rho.pop(0)
    return True


def compute_backtracking_line_search(
    position: Array,
    gradient: Array,
    direction: Array,
    residual: Residual,
    enabled: bool,
    max_iterations: int,
    reduction: float,
    c1: float,
    feasible: Feasibility | None = None,
) -> dict:
    """Perform residual backtracking with an optional trial feasibility test."""
    phi = 0.5 * float(np.dot(gradient, gradient))
    slope = float(np.dot(gradient, direction))
    step_length = 1.0
    for iteration in range(max_iterations):
        trial = position + step_length * direction
        if feasible is not None and not feasible(trial):
            step_length *= reduction
            continue
        trial_gradient = residual(trial)
        trial_phi = 0.5 * float(np.dot(trial_gradient, trial_gradient))
        if not enabled or trial_phi <= phi + c1 * step_length * slope:
            return {
                "accepted": True,
                "position": trial,
                "gradient": trial_gradient,
                "step_length": step_length,
                "iterations": iteration + 1,
            }
        step_length *= reduction
    return {
        "accepted": False,
        "position": position,
        "gradient": gradient,
        "step_length": 0.0,
        "iterations": max_iterations,
    }


def solve_lbfgs(
    position: Array,
    residual: Residual,
    directional_residual: Residual,
    diagonal: Array,
    settings: dict,
    feasible: Feasibility | None = None,
) -> tuple[Array, Array, dict]:
    """Solve a residual equation with matrix-free limited-memory BFGS.

    ``directional_residual(position, direction)`` must return the residual
    directional derivative. The diagonal is the initial inverse-Hessian
    approximation. The returned info dictionary is suitable for
    ``SoftBody.last_implicit_info``.
    """
    x = np.asarray(position, dtype=np.float64).copy()
    diagonal = np.asarray(diagonal, dtype=np.float64)
    g = np.asarray(residual(x), dtype=np.float64)
    initial_norm = max(float(np.linalg.norm(g)), 1.0)
    history_s: list[Array] = []
    history_y: list[Array] = []
    history_rho: list[float] = []
    converged = False
    line_search_steps = 0
    iterations = 0

    for iteration in range(int(settings["max_iterations"])):
        iterations = iteration + 1
        if np.linalg.norm(g) <= float(settings["tolerance"]) * initial_norm:
            converged = True
            break

        direction = compute_lbfgs_direction(g, history_s, history_y, history_rho, diagonal)
        if float(np.dot(direction, g)) >= 0.0 or not np.all(np.isfinite(direction)):
            direction = -diagonal * g

        line_search = compute_backtracking_line_search(
            x,
            g,
            direction,
            residual,
            bool(settings["line_search"]),
            int(settings["max_line_search_iterations"]),
            float(settings["line_search_reduction"]),
            float(settings["line_search_c1"]),
            feasible=feasible,
        )
        line_search_steps += line_search["iterations"]
        if not line_search["accepted"]:
            break

        trial = line_search["position"]
        trial_g = line_search["gradient"]
        s = trial - x
        y = np.asarray(directional_residual(trial, s), dtype=np.float64)
        update_lbfgs_history(
            history_s,
            history_y,
            history_rho,
            s,
            y,
            int(settings["history_size"]),
            float(settings["curvature_tolerance"]),
        )
        x, g = trial, trial_g

    info = {
        "converged": converged,
        "iterations": iterations,
        "final_residual_norm": float(np.linalg.norm(g)),
        "initial_residual_norm": initial_norm,
        "residual_reduction_factor": float(np.linalg.norm(g) / initial_norm),
        "line_search_steps": line_search_steps,
        "history_length": len(history_s),
    }
    return x, g, info
