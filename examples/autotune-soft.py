"""Auto-tune all soft-body timestepper/backend combinations.

The reference is a small-timestep semi-implicit cantilever bending under
gravity. Candidate semi-implicit and implicit backward-Euler runs sweep
timestep multipliers. Implicit runs also sweep solver settings, and the
fastest valid candidate is selected independently for each backend.

The fine semi-implicit NumPy trajectory is used as a reference. Candidate
semi-implicit and implicit backward-Euler runs sweep timestep multipliers.
Implicit runs also sweep solver settings, and the fastest valid candidate is
selected independently for each backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darerl.simulators.soft import (
    SoftBody,
    create_bending_baseline,
    create_compress_baseline,
    create_stretch_baseline,
    create_twist_baseline,
)
from darerl.simulators.soft.mesh import compute_boundary_faces

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_FACTORIES = {
    "bending": create_bending_baseline,
    "twist": create_twist_baseline,
    "compress": create_compress_baseline,
    "stretch": create_stretch_baseline,
}


def make_case_body(case: str, i: int, j: int, k: int, use_jax: bool) -> tuple[SoftBody, np.ndarray]:
    baseline = CASE_FACTORIES[case](i, j, k)
    return baseline.create_body(use_jax=use_jax), baseline.fixed


def compute_energy_magnitude(body: SoftBody, gravity: np.ndarray) -> float:
    """Return a positive scale for monitoring total stored mechanical energy."""
    gravity = np.asarray(gravity, dtype=np.float64)
    x = body.get_x()
    v = body.get_v()
    kinetic = 0.5 * np.sum(body.mesh.lumped_mass[:, None] * v * v)
    potential = -np.sum(body.mesh.lumped_mass[:, None] * x * gravity[None, :])
    elastic = body.compute_elastic_energy(x)
    return float(abs(kinetic) + abs(potential) + abs(elastic))


def simulate_reference(case: str, i: int, j: int, k: int, dt: float, duration: float) -> tuple[SoftBody, np.ndarray, float, np.ndarray]:
    body, fixed = make_case_body(case, i, j, k, use_jax=False)
    gravity = CASE_FACTORIES[case](i, j, k).gravity
    steps = int(round(duration / dt))
    if not np.isclose(steps * dt, duration, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("duration must be an integer multiple of reference_dt")
    initial_x = body.get_x()
    states = np.empty((steps + 1,) + initial_x.shape, dtype=np.float64)
    states[0] = initial_x
    energy_magnitudes = np.empty(steps + 1, dtype=np.float64)
    energy_magnitudes[0] = compute_energy_magnitude(body, gravity)
    start = time.perf_counter()
    for step in range(1, steps + 1):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            body.step(dt, gravity=gravity)
        x = body.get_x()
        v = body.get_v()
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)):
            raise RuntimeError(
                f"reference trajectory became non-finite at step {step}; "
                "reduce --baseline-dt or use a less aggressive baseline load"
            )
        states[step] = x
        energy_magnitudes[step] = compute_energy_magnitude(body, gravity)
        if not np.isfinite(energy_magnitudes[step]):
            raise RuntimeError(f"reference energy became non-finite at step {step}")
    elapsed = time.perf_counter() - start
    return body, states, elapsed, energy_magnitudes


def simulate_candidate(
    case: str,
    i: int,
    j: int,
    k: int,
    dt: float,
    duration: float,
    method: str,
    use_jax: bool,
    settings: dict,
    energy_limit: float,
) -> dict:
    steps = int(round(duration / dt))
    if not np.isclose(steps * dt, duration, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("duration must be an integer multiple of every candidate dt")
    jit_seconds = 0.0
    gravity = CASE_FACTORIES[case](i, j, k).gravity
    start = time.perf_counter()
    try:
        if use_jax:
            warmup, _ = make_case_body(case, i, j, k, use_jax)
            jit_start = time.perf_counter()
            if method == "implicit_bfgs":
                warmup.step_implicit(dt, gravity=gravity, settings=settings)
            else:
                warmup.step(dt, gravity=gravity)
            warmup.synchronize()
            jit_seconds = time.perf_counter() - jit_start

        body, _ = make_case_body(case, i, j, k, use_jax)
        initial_x = body.get_x()
        states = np.empty((steps + 1,) + initial_x.shape, dtype=np.float64)
        states[0] = initial_x
        iterations = []
        energy_magnitudes = np.empty(steps + 1, dtype=np.float64)
        energy_magnitudes[0] = compute_energy_magnitude(body, gravity)
        start = time.perf_counter()
        for step in range(1, steps + 1):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if method == "implicit_bfgs":
                    body.step_implicit(dt, gravity=gravity, settings=settings)
                else:
                    body.step(dt, gravity=gravity)
            x = body.get_x()
            v = body.get_v()
            states[step] = x
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)):
                return {
                    "valid": False,
                    "error": f"non-finite state at step {step}",
                    "elapsed": time.perf_counter() - start,
                    "jit_seconds": jit_seconds,
                }
            energy_magnitudes[step] = compute_energy_magnitude(body, gravity)
            if not np.isfinite(energy_magnitudes[step]):
                return {
                    "valid": False,
                    "error": f"non-finite energy at step {step}",
                    "elapsed": time.perf_counter() - start,
                    "jit_seconds": jit_seconds,
                }
            if energy_magnitudes[step] > energy_limit:
                return {
                    "valid": False,
                    "error": f"energy growth exceeded {energy_limit:.3e} J at step {step}",
                    "elapsed": time.perf_counter() - start,
                    "jit_seconds": jit_seconds,
                }
            if method == "implicit_bfgs":
                iterations.append(body.last_implicit_info["iterations"])
    except Exception as error:
        return {
            "valid": False,
            "error": str(error),
            "elapsed": max(0.0, time.perf_counter() - start),
            "jit_seconds": jit_seconds,
        }
    elapsed = time.perf_counter() - start
    return {
        "valid": True,
        "body": body,
        "states": states,
        "elapsed": elapsed,
        "jit_seconds": jit_seconds,
        "iterations": iterations,
    }


def compute_trajectory_error(candidate: np.ndarray, reference: np.ndarray, reference_dt: float, candidate_dt: float) -> float:
    reference_indices = np.rint(np.arange(len(candidate)) * candidate_dt / reference_dt).astype(np.int64)
    differences = candidate - reference[reference_indices]
    scale = max(float(np.ptp(reference[:, :, 0])), 1.0)
    error = float(np.max(np.linalg.norm(differences, axis=2)) / scale)
    return error if np.isfinite(error) else float("nan")


def parse_list(value: str, converter):
    return [converter(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path: Path, results: list[dict]) -> None:
    fields = ["method", "backend", "directional_residual_strategy", "dt", "max_iterations", "history_size", "absolute_tolerance", "relative_tolerance", "line_search", "valid", "stability_valid", "accuracy_valid", "performance_valid", "error", "elapsed", "jit_seconds", "mean_iterations", "trajectory_error", "trajectory_error_percent"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fields})


def write_settings(path: Path, best_by_combination: dict, baseline_dt: float, max_error_percent: float) -> None:
    """Write clean per-combination runtime settings."""
    document = {
        "combinations": best_by_combination,
        "metadata": {
            "baseline_dt": float(baseline_dt),
            "max_error_percent": float(max_error_percent),
        "preferred_combination": next(
            (key for key in best_by_combination if key.startswith("implicit_bfgs/") and key.endswith("/JAX")),
            next((key for key in best_by_combination if key.startswith("implicit_bfgs/") and key.endswith("/NumPy")), ""),
        ),
        },
    }
    with path.open("w") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")


def plot_results(
    path: Path,
    results: list[dict],
    tolerance: float,
    baseline_states: np.ndarray,
    baseline_elements: np.ndarray,
    baseline_dt: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        return
    valid = [result for result in results if result["valid"]]
    with PdfPages(path) as pdf:
        faces = compute_boundary_faces(baseline_elements)
        frame_indices = np.linspace(0, len(baseline_states) - 1, 4, dtype=int)
        figure, axes = plt.subplots(
            1, 4, figsize=(14, 4), subplot_kw={"projection": "3d"}, constrained_layout=True
        )
        bounds_min = np.min(baseline_states, axis=(0, 1))
        bounds_max = np.max(baseline_states, axis=(0, 1))
        for axis, index in zip(axes, frame_indices):
            state = baseline_states[index]
            axis.plot_trisurf(
                state[:, 0], state[:, 1], state[:, 2], triangles=faces,
                linewidth=0.05, alpha=0.85, color="tab:blue"
            )
            axis.set_title(f"t = {index * baseline_dt:.4f} s")
            axis.set_xlim(bounds_min[0], bounds_max[0])
            axis.set_ylim(bounds_min[1], bounds_max[1])
            axis.set_zlim(bounds_min[2], bounds_max[2])
            axis.set_box_aspect(bounds_max - bounds_min)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_zlabel("z")
        figure.suptitle("Semi-implicit baseline bending-beam deformation")
        pdf.savefig(figure)
        plt.close(figure)

        if valid:
            figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
            labels = sorted({f"{result['method']}/{result['backend']}" for result in valid})
            for label in labels:
                group = [result for result in valid if f"{result['method']}/{result['backend']}" == label]
                axis.loglog(
                    [result["trajectory_error"] for result in group],
                    [result["elapsed"] for result in group],
                    "o",
                    label=label,
                )
            axis.axvline(tolerance, color="black", linestyle="--", label="error tolerance")
            axis.set_xlabel("maximum trajectory position error")
            axis.set_ylabel("runtime [s]")
            axis.set_title("Soft-body auto-tuning: accuracy versus runtime")
            axis.grid(True, which="both", alpha=0.25)
            axis.legend()
            pdf.savefig(figure)
            plt.close(figure)

            # Pages 3+: one-parameter error slices through each sweep.
            for label in labels:
                group = [result for result in valid if f"{result['method']}/{result['backend']}" == label]
                best = min(
                    (result for result in group if result["trajectory_error"] <= tolerance),
                    key=lambda result: result["elapsed"],
                )
                parameters = ["dt"]
                if best["method"] == "implicit_bfgs":
                    parameters.extend(("max_iterations", "history_size", "relative_tolerance", "line_search"))
                figure, axes = plt.subplots(
                    1, len(parameters), figsize=(4.2 * len(parameters), 3.8),
                    squeeze=False, constrained_layout=True,
                )
                for axis, parameter in zip(axes[0], parameters):
                    other_parameters = [item for item in parameters if item != parameter]

                    def equal_value(left, right):
                        if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
                            return bool(np.isclose(left, right, rtol=1.0e-12, atol=1.0e-15))
                        return left == right

                    slice_results = [
                        result for result in group
                        if all(equal_value(result[item], best[item]) for item in other_parameters)
                    ]
                    slice_results.sort(key=lambda result: result[parameter])
                    x_values = [result[parameter] for result in slice_results]
                    y_values = [100.0 * result["trajectory_error"] for result in slice_results]
                    if parameter == "line_search":
                        x_values = [int(value) for value in x_values]
                        axis.set_xticks((0, 1), ("False", "True"))
                    axis.plot(x_values, y_values, "o-", color="tab:blue")
                    axis.scatter(
                        [best[parameter]], [100.0 * best["trajectory_error"]],
                        color="tab:red", marker="*", s=110, zorder=3, label="best",
                    )
                    axis.axhline(100.0 * tolerance, color="black", linestyle="--", linewidth=0.8, label="limit")
                    axis.set_xlabel(parameter)
                    axis.set_ylabel("trajectory error [%]")
                    axis.set_title(f"best={best[parameter]}")
                    axis.grid(True, alpha=0.25)
                    axis.legend(fontsize="small")
                figure.suptitle(f"Sensitivity around best setting: {label}")
                pdf.savefig(figure)
                plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASE_FACTORIES), default="bending")
    parser.add_argument("--i", type=int, default=12, help="beam nodes along its length")
    parser.add_argument("--j", type=int, default=4, help="beam nodes across y")
    parser.add_argument("--k", type=int, default=4, help="beam nodes across z")
    parser.add_argument("--baseline-dt", type=float, default=1.0e-4, help="semi-implicit baseline timestep")
    parser.add_argument("--baseline-steps", type=int, default=1000, help="number of semi-implicit baseline steps when --validation-duration is omitted")
    parser.add_argument(
        "--validation-duration",
        type=float,
        default=None,
        help="candidate/reference validation horizon in seconds; defaults to at least 1 s to expose accumulated instability",
    )
    parser.add_argument(
        "--max-energy-growth",
        type=float,
        default=10.0,
        help="maximum candidate energy envelope relative to the reference envelope",
    )
    parser.add_argument("--max-dt", type=float, default=0.1, help="largest candidate timestep")
    parser.add_argument(
        "--max-iterations", default="10,20,30",
        help="candidate L-BFGS iteration caps; swept one parameter at a time",
    )
    parser.add_argument(
        "--history-sizes", default="4,8,12",
        help="candidate L-BFGS history sizes; swept one parameter at a time",
    )
    parser.add_argument(
        "--relative-tolerances", default="1e-4,1e-6",
        help="candidate relative residual tolerances; swept one parameter at a time",
    )
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-6, help="absolute L-BFGS residual tolerance used for every candidate")
    parser.add_argument("--line-search", default="true,false")
    parser.add_argument("--max-error-percent", type=float, default=5.0, help="maximum trajectory error relative to baseline, in percent")
    parser.add_argument("--backend", choices=("both", "numpy", "jax"), default="both")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--settings-output", type=Path, default=Path("output/auto-tuned-settings.json"))
    args = parser.parse_args()

    if args.baseline_dt <= 0.0 or args.baseline_steps < 1 or args.max_error_percent < 0.0:
        parser.error("baseline-dt must be positive, baseline-steps must be positive, and max-error-percent cannot be negative")
    if args.validation_duration <= 0.0 or args.max_energy_growth <= 0.0:
        parser.error("validation-duration and max-energy-growth must be positive")
    duration = max(args.baseline_dt * args.baseline_steps, 1.0) if args.validation_duration is None else args.validation_duration
    reference_steps = int(round(duration / args.baseline_dt))
    if not np.isclose(reference_steps * args.baseline_dt, duration, rtol=1.0e-10, atol=1.0e-14):
        parser.error("validation-duration must be an integer multiple of baseline-dt")
    max_error = args.max_error_percent / 100.0

    try:
        import jax  # noqa: F401
    except ImportError:
        has_jax = False
        print("JAX is unavailable; only NumPy will be tuned")
    else:
        has_jax = True
    backends = [("NumPy", False)]
    if has_jax and args.backend != "numpy":
        backends.append(("JAX", True))
    if args.backend == "jax":
        backends = [("JAX", True)] if has_jax else []
    if not backends:
        parser.error("JAX was requested but is unavailable")

    max_iterations = parse_list(args.max_iterations, int)
    history_sizes = parse_list(args.history_sizes, int)
    relative_tolerances = parse_list(args.relative_tolerances, float)
    line_search_values = [value.lower() in {"true", "1", "yes", "on"} for value in parse_list(args.line_search, str)]
    if not max_iterations or not history_sizes or not relative_tolerances or not line_search_values:
        parser.error("all solver sweep lists must contain at least one value")
    if args.max_dt < args.baseline_dt:
        parser.error("max-dt must be at least baseline-dt")
    candidate_dts = []
    power = 0
    while args.baseline_dt * (10**power) <= args.max_dt * (1.0 + 1.0e-12):
        candidate_dts.append(args.baseline_dt * (10**power))
        power += 1
    for dt in candidate_dts:
        steps = round(duration / dt)
        if not np.isclose(steps * dt, duration, rtol=1.0e-10, atol=1.0e-14):
            parser.error(f"baseline duration {duration} is not an integer multiple of candidate dt {dt}")

    print("Computing fine semi-implicit reference...")
    reference_body, reference_states, reference_runtime, reference_energy = simulate_reference(args.case, args.i, args.j, args.k, args.baseline_dt, duration)
    print(f"reference: {len(reference_states) - 1} steps, {reference_runtime:.3f} s")
    reference_energy_limit = max(float(np.max(reference_energy)) * args.max_energy_growth, np.finfo(float).tiny)
    print(f"reference energy envelope: {np.max(reference_energy):.3e} J; candidate limit: {reference_energy_limit:.3e} J")
    max_displacement = float(np.max(np.linalg.norm(reference_states - reference_states[0], axis=2)))
    beam_length = float(np.ptp(reference_states[0, :, 0]))
    displacement_percent = 100.0 * max_displacement / max(beam_length, np.finfo(float).eps)
    print(
        f"reference max displacement: {max_displacement:.4e} m "
        f"({displacement_percent:.2f}% of beam length)"
    )

    results = []
    best_by_combination = {}
    solver_combinations = (
        ("semi_implicit", None),
        ("implicit_bfgs", "tangent_action"),
        ("implicit_bfgs", "closed_form"),
        ("implicit_bfgs", "finite_difference"),
    )
    for backend, use_jax in backends:
        for method, strategy in solver_combinations:
            label = f"{method}/{backend}" if strategy is None else f"{method}/{strategy}/{backend}"
            print(f"\n=== {label} ===")
            combination_results = []
            seen = set()

            def evaluate(dt, iterations=0, history=0, relative_tolerance=0.0, line_search=False):
                key = (float(dt), int(iterations), int(history), float(relative_tolerance), bool(line_search))
                if key in seen:
                    return
                seen.add(key)
                settings = {} if strategy is None else {
                    "max_iterations": iterations,
                    "history_size": history,
                    "absolute_tolerance": args.absolute_tolerance,
                    "relative_tolerance": relative_tolerance,
                    "line_search": line_search,
                    "raise_on_failure": True,
                    "directional_residual_strategy": strategy,
                }
                candidate = simulate_candidate(args.case, args.i, args.j, args.k, dt, duration, method, use_jax, settings, reference_energy_limit)
                result = {
                    "method": method,
                    "backend": backend,
                    "directional_residual_strategy": strategy,
                    "dt": dt,
                    "max_iterations": iterations,
                    "history_size": history,
                    "absolute_tolerance": args.absolute_tolerance,
                    "relative_tolerance": relative_tolerance,
                    "line_search": line_search,
                    "valid": candidate["valid"],
                    "elapsed": candidate["elapsed"],
                    "jit_seconds": candidate.get("jit_seconds", 0.0),
                }
                if candidate["valid"]:
                    result["trajectory_error"] = compute_trajectory_error(candidate["states"], reference_states, args.baseline_dt, dt)
                    result["trajectory_error_percent"] = 100.0 * result["trajectory_error"]
                    result["mean_iterations"] = float(np.mean(candidate["iterations"])) if candidate["iterations"] else 0.0
                    result["error"] = ""
                    result["stability_valid"] = True
                    result["accuracy_valid"] = bool(result["trajectory_error"] <= max_error)
                    result["performance_valid"] = result["accuracy_valid"]
                else:
                    result["trajectory_error"] = np.nan
                    result["trajectory_error_percent"] = np.nan
                    result["mean_iterations"] = np.nan
                    result["error"] = candidate["error"]
                    result["stability_valid"] = False
                    result["accuracy_valid"] = False
                    result["performance_valid"] = False
                results.append(result)
                combination_results.append(result)
                if not result["stability_valid"]:
                    status = f" INVALID: {result['error']}"
                elif not result["accuracy_valid"]:
                    status = f" REJECTED: error exceeds {args.max_error_percent:g}%"
                else:
                    status = ""
                if strategy is not None:
                    print(
                        f"dt={dt:.3e} max_it={iterations:2d} history={history:2d} "
                        f"abs_tol={args.absolute_tolerance:.1e} rel_tol={relative_tolerance:.1e} line_search={line_search!s:5s} "
                        f"compute={result['elapsed']:.3f}s jit={result['jit_seconds']:.3f}s "
                        f"error={result['trajectory_error_percent']:.4f}% "
                        f"mean_it={result['mean_iterations']:.2f}{status}"
                    )
                else:
                    print(
                        f"dt={dt:.3e} compute={result['elapsed']:.3f}s jit={result['jit_seconds']:.3f}s "
                        f"error={result['trajectory_error_percent']:.4f}%{status}"
                    )

            if method == "semi_implicit":
                for dt in candidate_dts:
                    evaluate(dt)
            else:
                # First establish a robust reference solver profile at every
                # timestep. The remaining parameters are then varied one at a
                # time, avoiding the combinatorial Cartesian product.
                robust = {
                    "iterations": max(max_iterations),
                    "history": max(history_sizes),
                    "relative_tolerance": min(relative_tolerances),
                    "line_search": True if True in line_search_values else line_search_values[0],
                }
                for dt in candidate_dts:
                    evaluate(dt, **robust)

                acceptable_reference = [
                    result for result in combination_results
                    if result["valid"] and result["trajectory_error"] <= max_error
                ]
                valid_reference = [result for result in combination_results if result["valid"]]
                reference_pool = acceptable_reference or valid_reference
                reference_pool.sort(key=lambda result: result["dt"], reverse=True)
                if reference_pool:
                    boundary_dt = reference_pool[0]["dt"]
                    boundary_index = candidate_dts.index(boundary_dt)
                    # Include the largest robust timestep and the next larger
                    # candidate. This is where solver settings can recover a
                    # timestep rejected by the robust profile.
                    exploration_indices = {boundary_index}
                    if boundary_index + 1 < len(candidate_dts):
                        exploration_indices.add(boundary_index + 1)
                    exploration_dts = [candidate_dts[index] for index in sorted(exploration_indices)]
                else:
                    exploration_dts = candidate_dts[:2]

                # These values cover the meaningful regimes: low/medium/high
                # iteration and history budgets, two useful convergence
                # tolerances, and the line-search on/off choice. Each sweep is
                # local to the robust profile, so interactions do not multiply
                # into a full Cartesian product.
                parameter_values = {
                    "iterations": max_iterations,
                    "history": history_sizes,
                    "relative_tolerance": relative_tolerances,
                    "line_search": line_search_values,
                }
                for dt in exploration_dts:
                    for parameter, values in parameter_values.items():
                        for value in values:
                            trial = dict(robust)
                            trial[parameter] = value
                            evaluate(dt, **trial)

                setting_candidates = [
                    result for result in combination_results
                    if result["valid"] and result["trajectory_error"] <= max_error
                ]
                setting_candidates = setting_candidates or [
                    result for result in combination_results if result["valid"]
                ]
                seed = min(setting_candidates, key=lambda result: result["elapsed"]) if setting_candidates else None
                if seed is not None:
                    seed_settings = {
                        "iterations": int(seed["max_iterations"]),
                        "history": int(seed["history_size"]),
                        "relative_tolerance": float(seed["relative_tolerance"]),
                        "line_search": bool(seed["line_search"]),
                    }
                    # Re-test the selected solver profile over every timestep;
                    # this makes the final dt comparison fair and complete.
                    for dt in candidate_dts:
                        evaluate(dt, **seed_settings)

            stable = [result for result in combination_results if result["stability_valid"]]
            accurate = [result for result in stable if result["accuracy_valid"]]
            if stable:
                stability_best = max(stable, key=lambda result: result["dt"])
                print(
                    f"Stability {label}: largest stable dt={stability_best['dt']:.3e} "
                    f"(energy-bounded and finite)"
                )
            else:
                print(f"Stability {label}: no stable candidate")
            if accurate:
                accuracy_best = max(accurate, key=lambda result: result["dt"])
                performance_best = min(accurate, key=lambda result: result["elapsed"])
                print(
                    f"Accuracy {label}: largest acceptable dt={accuracy_best['dt']:.3e} "
                    f"(error={accuracy_best['trajectory_error_percent']:.4f}%)"
                )
                print(
                    f"Performance {label}: fastest acceptable dt={performance_best['dt']:.3e} "
                    f"(compute={performance_best['elapsed']:.3f}s)"
                )
            else:
                print(f"Accuracy {label}: no candidate within {args.max_error_percent:g}%")
                print(f"Performance {label}: unavailable because no candidate met the accuracy criterion")

            acceptable = accurate
            if not acceptable:
                failed_settings = {} if strategy is None else {
                    "max_iterations": float("nan"),
                    "history_size": float("nan"),
                    "absolute_tolerance": float("nan"),
                    "relative_tolerance": float("nan"),
                    "line_search": float("nan"),
                    "raise_on_failure": False,
                    "directional_residual_strategy": strategy,
                }
                best_by_combination[label] = {
                    "method": method,
                    "backend": backend,
                    "directional_residual_strategy": strategy,
                    "valid": False,
                    "dt": float("nan"),
                    "solver_settings": failed_settings,
                    "trajectory_error": float("nan"),
                    "trajectory_error_percent": float("nan"),
                    "elapsed": float("nan"),
                    "jit_seconds": float("nan"),
                    "mean_iterations": float("nan"),
                    "failure": f"no candidate met max error of {args.max_error_percent:g}%",
                }
                print(f"No valid candidate met --max-error for {label}; writing NaN settings")
                continue
            best = min(acceptable, key=lambda result: result["elapsed"])
            best_by_combination[label] = {
                "method": method,
                "backend": backend,
                "directional_residual_strategy": strategy,
                "valid": True,
                "dt": float(best["dt"]),
                "solver_settings": {} if strategy is None else {
                    "max_iterations": int(best["max_iterations"]),
                    "history_size": int(best["history_size"]),
                    "absolute_tolerance": float(best["absolute_tolerance"]),
                    "relative_tolerance": float(best["relative_tolerance"]),
                    "line_search": bool(best["line_search"]),
                    "raise_on_failure": True,
                    "directional_residual_strategy": strategy,
                },
                "trajectory_error": float(best["trajectory_error"]),
                "trajectory_error_percent": float(100.0 * best["trajectory_error"]),
                "elapsed": float(best["elapsed"]),
                "jit_seconds": float(best["jit_seconds"]),
                "mean_iterations": float(best["mean_iterations"]),
            }
            print(f"Best {label}: {best_by_combination[label]}")
    if not best_by_combination:
        raise RuntimeError("no timestepper/backend combination met --max-error")
    args.output = args.output or Path(f"output/soft_{args.case}_autotune.csv")
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, results)
    plot_results(
        output.with_suffix(".pdf"),
        results,
        max_error,
        reference_states,
        reference_body.mesh.elements,
        args.baseline_dt,
    )
    settings_output = args.settings_output if args.settings_output.is_absolute() else PROJECT_ROOT / args.settings_output
    settings_output.parent.mkdir(parents=True, exist_ok=True)
    write_settings(settings_output, best_by_combination, args.baseline_dt, args.max_error_percent)
    for label, best in best_by_combination.items():
        safe_label = label.replace("/", "-")
        individual = settings_output.with_name(f"{settings_output.stem}-{safe_label}{settings_output.suffix}")
        write_settings(individual, {label: best}, args.baseline_dt, args.max_error_percent)
    print(f"Wrote results to {output}")
    print(f"Wrote tuning plot to {output.with_suffix('.pdf')}")
    print(f"Wrote runtime settings to {settings_output}")
    print(f"Wrote {len(best_by_combination)} per-combination settings files")


if __name__ == "__main__":
    main()
