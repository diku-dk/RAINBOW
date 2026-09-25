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
    merit_limit: float | None = None,
    accept_best: bool = False,
) -> dict:
    """Perform residual backtracking with feasibility and rescue support."""
    phi = 0.5 * float(np.dot(gradient, gradient))
    slope = float(np.dot(gradient, direction))
    step_length = 1.0
    best = None
    for iteration in range(max_iterations):
        trial = position + step_length * direction
        if feasible is not None and not feasible(trial):
            step_length *= reduction
            continue
        trial_gradient = residual(trial)
        trial_phi = 0.5 * float(np.dot(trial_gradient, trial_gradient))
        if np.isfinite(trial_phi) and (best is None or trial_phi < best["merit"]):
            best = {
                "position": trial.copy(),
                "gradient": trial_gradient.copy(),
                "step_length": step_length,
                "merit": trial_phi,
            }
        limit = phi if merit_limit is None else merit_limit
        if np.isfinite(trial_phi) and (not enabled or trial_phi <= limit + c1 * step_length * slope):
            return {
                "accepted": True,
                "position": trial,
                "gradient": trial_gradient,
                "step_length": step_length,
                "iterations": iteration + 1,
                "rescued": False,
            }
        step_length *= reduction
    if accept_best and best is not None and best["merit"] < phi:
        return {
            "accepted": True,
            "position": best["position"],
            "gradient": best["gradient"],
            "step_length": best["step_length"],
            "iterations": max_iterations,
            "rescued": True,
        }
    return {
        "accepted": False,
        "position": position,
        "gradient": gradient,
        "step_length": 0.0,
        "iterations": max_iterations,
        "rescued": False,
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
    residual_norm_history = [float(np.linalg.norm(g))]
    history_s: list[Array] = []
    history_y: list[Array] = []
    history_rho: list[float] = []
    converged = False
    line_search_steps = 0
    iterations = 0
    fallback_steps = 0
    direction_fallback_steps = 0
    watchdog_steps = 0
    watchdog_x = None
    watchdog_g = None
    watchdog_merit = None
    watchdog_acceptances = 0
    rescue_steps = 0
    globalization = settings.get("globalization", "backtracking")
    enable_gradient_fallback = bool(settings.get("enable_gradient_fallback", True))
    max_watchdog_steps = int(settings.get("max_watchdog_steps", 3))
    watchdog_growth = float(settings.get("watchdog_growth_factor", 1.1))

    for iteration in range(int(settings["max_iterations"])):
        iterations = iteration + 1
        if np.linalg.norm(g) <= float(settings["tolerance"]) * initial_norm:
            converged = True
            break

        direction = compute_lbfgs_direction(g, history_s, history_y, history_rho, diagonal)
        if float(np.dot(direction, g)) >= 0.0 or not np.all(np.isfinite(direction)):
            direction_fallback_steps += 1
            direction = -diagonal * g

        merit_limit = None
        current_merit = 0.5 * float(np.dot(g, g))
        position_before_trial = x.copy()
        gradient_before_trial = g.copy()
        if globalization == "watchdog" and watchdog_steps > 0:
            merit_limit = watchdog_merit * watchdog_growth
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
            merit_limit=merit_limit,
        )
        line_search_steps += line_search["iterations"]
        if not line_search["accepted"]:
            if globalization == "watchdog":
                if watchdog_steps == 0:
                    watchdog_x = x.copy()
                    watchdog_g = g.copy()
                    watchdog_merit = 0.5 * float(np.dot(g, g))
                line_search = compute_backtracking_line_search(
                    x, g, direction, residual,
                    True,
                    int(settings["max_line_search_iterations"]),
                    float(settings["line_search_reduction"]),
                    float(settings["line_search_c1"]),
                    feasible=feasible,
                    merit_limit=watchdog_merit * watchdog_growth,
                )
                line_search_steps += line_search["iterations"]
            if not line_search["accepted"] and enable_gradient_fallback:
                fallback_steps += 1
                gradient_direction = -diagonal * g
                line_search = compute_backtracking_line_search(
                    x, g, gradient_direction, residual,
                    True,
                    int(settings["max_line_search_iterations"]),
                    float(settings["line_search_reduction"]),
                    float(settings["line_search_c1"]),
                    feasible=feasible,
                    accept_best=True,
                )
                line_search_steps += line_search["iterations"]
            if not line_search["accepted"]:
                break

        trial = line_search["position"]
        trial_g = line_search["gradient"]
        rescue_steps += int(line_search.get("rescued", False))
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
        residual_norm_history.append(float(np.linalg.norm(g)))

        trial_merit = 0.5 * float(np.dot(g, g))
        if globalization == "watchdog":
            if trial_merit > current_merit:
                if watchdog_steps == 0:
                    # Keep the last strict iterate as the restoration point.
                    watchdog_x = position_before_trial
                    watchdog_g = gradient_before_trial
                    watchdog_merit = current_merit
                watchdog_steps += 1
                watchdog_acceptances += 1
            elif watchdog_steps > 0 and trial_merit < watchdog_merit:
                watchdog_steps = 0
            if watchdog_steps > max_watchdog_steps:
                x, g = watchdog_x, watchdog_g
                history_s.clear()
                history_y.clear()
                history_rho.clear()
                watchdog_steps = 0

    info = {
        "converged": converged,
        "iterations": iterations,
        "final_residual_norm": float(np.linalg.norm(g)),
        "initial_residual_norm": initial_norm,
        "residual_reduction_factor": float(np.linalg.norm(g) / initial_norm),
        "line_search_steps": line_search_steps,
        "history_length": len(history_s),
        "gradient_fallback_steps": fallback_steps,
        "direction_fallback_steps": direction_fallback_steps,
        "watchdog_steps": watchdog_steps,
        "watchdog_acceptances": watchdog_acceptances,
        "rescue_steps": rescue_steps,
        "residual_norm_history": residual_norm_history,
    }
    return x, g, info
