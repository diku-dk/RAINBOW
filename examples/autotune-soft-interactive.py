"""Interactive graphics-oriented auto-tuning for soft-body timestepper/backend combinations.

The reference is a fine-step semi-implicit cantilever simulation. Candidates
are classified by finite-state, bounded-motion, and runtime criteria. The
interactive profile permits inexact implicit solves and numerical damping;
it is not an accuracy or energy-conservation study.
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


def compute_energy_state(body: SoftBody, gravity: np.ndarray, previous_x: np.ndarray, previous_applied_force: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Return energy magnitude, mechanical energy, applied work, and load."""
    gravity = np.asarray(gravity, dtype=np.float64)
    x = body.get_x()
    v = body.get_v()
    kinetic = 0.5 * np.sum(body.mesh.lumped_mass[:, None] * v * v)
    potential = -np.sum(body.mesh.lumped_mass[:, None] * x * gravity[None, :])
    elastic = body.compute_elastic_energy(x)
    mechanical = float(kinetic + potential + elastic)
    applied_force = body.compute_neumann_forces(x) + body.external_forces
    work_increment = 0.5 * np.sum((previous_applied_force + applied_force) * (x - previous_x))
    return float(abs(kinetic) + abs(potential) + abs(elastic)), mechanical, float(work_increment), applied_force


def simulate_reference(case: str, i: int, j: int, k: int, dt: float, duration: float) -> tuple[SoftBody, np.ndarray, float, dict[str, np.ndarray]]:
    body, fixed = make_case_body(case, i, j, k, use_jax=False)
    gravity = CASE_FACTORIES[case](i, j, k).gravity
    steps = int(round(duration / dt))
    if not np.isclose(steps * dt, duration, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("duration must be an integer multiple of reference_dt")
    initial_x = body.get_x()
    states = np.empty((steps + 1,) + initial_x.shape, dtype=np.float64)
    states[0] = initial_x
    previous_x = initial_x.copy()
    previous_force = body.compute_neumann_forces(previous_x) + body.external_forces
    energy_magnitudes = np.empty(steps + 1, dtype=np.float64)
    mechanical_energy = np.empty(steps + 1, dtype=np.float64)
    applied_work = np.zeros(steps + 1, dtype=np.float64)
    energy_magnitudes[0], mechanical_energy[0], _, previous_force = compute_energy_state(body, gravity, previous_x, previous_force)
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
        energy_magnitudes[step], mechanical_energy[step], work_increment, previous_force = compute_energy_state(body, gravity, previous_x, previous_force)
        applied_work[step] = applied_work[step - 1] + work_increment
        previous_x = x.copy()
        if not np.isfinite(energy_magnitudes[step]):
            raise RuntimeError(f"reference energy became non-finite at step {step}")
    elapsed = time.perf_counter() - start
    return body, states, elapsed, {
        "magnitude": energy_magnitudes,
        "mechanical": mechanical_energy,
        "work": applied_work,
    }


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
    interactive_limits: dict[str, float] | None = None,
) -> dict:
    steps = int(round(duration / dt))
    if not np.isclose(steps * dt, duration, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("duration must be an integer multiple of every candidate dt")
    jit_seconds = 0.0
    gravity = CASE_FACTORIES[case](i, j, k).gravity
    start = time.perf_counter()
    diagnostic_names = (
        "direction_fallback_steps",
        "gradient_fallback_steps",
        "watchdog_steps",
        "watchdog_acceptances",
        "rescue_steps",
        "line_search_steps",
    )
    diagnostic_totals = {name: 0 for name in diagnostic_names}
    nonconverged_steps = 0
    max_residual_reduction_factor = 0.0
    max_displacement = 0.0
    max_speed = 0.0
    max_frame_displacement = 0.0

    def diagnostic_payload() -> dict[str, int]:
        return {name: int(value) for name, value in diagnostic_totals.items()}

    def interactive_payload() -> dict[str, float | int]:
        return {
            "nonconverged_steps": int(nonconverged_steps),
            "max_residual_reduction_factor": float(max_residual_reduction_factor),
            "max_displacement": float(max_displacement),
            "max_speed": float(max_speed),
            "max_frame_displacement": float(max_frame_displacement),
        }

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
        previous_x = initial_x.copy()
        previous_force = body.compute_neumann_forces(previous_x) + body.external_forces
        energy_magnitudes = np.empty(steps + 1, dtype=np.float64)
        mechanical_energy = np.empty(steps + 1, dtype=np.float64)
        applied_work = np.zeros(steps + 1, dtype=np.float64)
        energy_magnitudes[0], mechanical_energy[0], _, previous_force = compute_energy_state(body, gravity, previous_x, previous_force)
        step_times_ms = []
        start = time.perf_counter()
        for step in range(1, steps + 1):
            step_start = time.perf_counter()
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if method == "implicit_bfgs":
                    body.step_implicit(dt, gravity=gravity, settings=settings)
                    info = body.last_implicit_info
                    for name in diagnostic_names:
                        diagnostic_totals[name] += int(info.get(name, 0))
                    if not info.get("converged", True):
                        nonconverged_steps += 1
                    max_residual_reduction_factor = max(
                        max_residual_reduction_factor,
                        float(info.get("residual_reduction_factor", 0.0)),
                    )
                else:
                    body.step(dt, gravity=gravity)
                # JAX dispatch is asynchronous. Include device completion in
                # the measured step; otherwise host-side profiling reports
                # dispatch latency rather than usable real-time latency.
                if use_jax:
                    body.synchronize()
            step_times_ms.append(1000.0 * (time.perf_counter() - step_start))
            x = body.get_x()
            v = body.get_v()
            states[step] = x
            max_displacement = max(max_displacement, float(np.max(np.linalg.norm(x - initial_x, axis=1))))
            max_speed = max(max_speed, float(np.max(np.linalg.norm(v, axis=1))))
            max_frame_displacement = max(max_frame_displacement, float(np.max(np.linalg.norm(x - previous_x, axis=1))))
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)):
                return {
                    "valid": False,
                    "error": f"non-finite state at step {step}",
                    "elapsed": time.perf_counter() - start,
                    "jit_seconds": jit_seconds,
                    "step_times_ms": np.asarray(step_times_ms),
                    **diagnostic_payload(),
                    **interactive_payload(),
                }
            if interactive_limits is not None:
                if max_displacement > interactive_limits["max_displacement"]:
                    return {
                        "valid": False,
                        "error": f"bounded-motion displacement exceeded {interactive_limits['max_displacement']:.3e} m at step {step}",
                        "elapsed": time.perf_counter() - start,
                        "jit_seconds": jit_seconds,
                        "step_times_ms": np.asarray(step_times_ms),
                        **diagnostic_payload(), **interactive_payload(),
                    }
                if max_speed > interactive_limits["max_speed"]:
                    return {
                        "valid": False,
                        "error": f"bounded-motion speed exceeded {interactive_limits['max_speed']:.3e} m/s at step {step}",
                        "elapsed": time.perf_counter() - start,
                        "jit_seconds": jit_seconds,
                        "step_times_ms": np.asarray(step_times_ms),
                        **diagnostic_payload(), **interactive_payload(),
                    }
                if max_frame_displacement > interactive_limits["max_frame_displacement"]:
                    return {
                        "valid": False,
                        "error": f"frame displacement exceeded {interactive_limits['max_frame_displacement']:.3e} m at step {step}",
                        "elapsed": time.perf_counter() - start,
                        "jit_seconds": jit_seconds,
                        "step_times_ms": np.asarray(step_times_ms),
                        **diagnostic_payload(), **interactive_payload(),
                    }
            energy_magnitudes[step], mechanical_energy[step], work_increment, previous_force = compute_energy_state(body, gravity, previous_x, previous_force)
            applied_work[step] = applied_work[step - 1] + work_increment
            previous_x = x.copy()
            if not np.isfinite(energy_magnitudes[step]):
                return {
                    "valid": False,
                    "error": f"non-finite energy at step {step}",
                    "elapsed": time.perf_counter() - start,
                    "jit_seconds": jit_seconds,
                    "step_times_ms": np.asarray(step_times_ms),
                    **diagnostic_payload(),
                    **interactive_payload(),
                }
            if energy_magnitudes[step] > energy_limit:
                return {
                    "valid": False,
                    "error": f"energy growth exceeded {energy_limit:.3e} J at step {step}",
                    "elapsed": time.perf_counter() - start,
                    "jit_seconds": jit_seconds,
                    "step_times_ms": np.asarray(step_times_ms),
                    **diagnostic_payload(),
                    **interactive_payload(),
                }
            if method == "implicit_bfgs":
                iterations.append(body.last_implicit_info["iterations"])
    except Exception as error:
        return {
            "valid": False,
            "error": str(error),
            "elapsed": max(0.0, time.perf_counter() - start),
            "jit_seconds": jit_seconds,
            "step_times_ms": np.asarray(locals().get("step_times_ms", [])),
            **diagnostic_payload(),
            **interactive_payload(),
        }
    elapsed = float(np.sum(step_times_ms)) / 1000.0
    energy_scale = max(float(np.max(energy_magnitudes)), np.finfo(float).tiny)
    balance_error = (mechanical_energy - mechanical_energy[0]) - applied_work
    return {
        "valid": True,
        "body": body,
        "states": states,
        "elapsed": elapsed,
        "jit_seconds": jit_seconds,
        "iterations": iterations,
        "step_times_ms": np.asarray(step_times_ms),
        "energy_magnitudes": energy_magnitudes,
        "mechanical_energy": mechanical_energy,
        "applied_work": applied_work,
        "energy_scale": energy_scale,
        "max_energy_balance_error": float(np.max(np.abs(balance_error))),
        **diagnostic_payload(),
        **interactive_payload(),
    }


def compute_trajectory_error(candidate: np.ndarray, reference: np.ndarray, reference_dt: float, candidate_dt: float) -> float:
    reference_indices = np.rint(np.arange(len(candidate)) * candidate_dt / reference_dt).astype(np.int64)
    reference_indices = np.clip(reference_indices, 0, len(reference) - 1)
    differences = candidate - reference[reference_indices]
    reference_extent = np.ptp(reference[0], axis=0)
    scale = max(float(np.linalg.norm(reference_extent)), np.finfo(float).eps)
    error = float(np.max(np.linalg.norm(differences, axis=2)) / scale)
    return error if np.isfinite(error) else float("nan")


def parse_list(value: str, converter):
    return [converter(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path: Path, results: list[dict]) -> None:
    fields = ["method", "backend", "directional_residual_strategy", "dt", "max_iterations", "history_size", "absolute_tolerance", "relative_tolerance", "line_search", "max_line_search_iterations", "globalization", "valid", "stability_valid", "accuracy_valid", "performance_valid", "error", "elapsed", "step_mean_ms", "step_p25_ms", "step_p75_ms", "jit_seconds", "mean_iterations", "nonconverged_steps", "max_residual_reduction_factor", "max_displacement", "max_speed", "max_frame_displacement", "direction_fallback_steps", "gradient_fallback_steps", "watchdog_steps", "watchdog_acceptances", "rescue_steps", "line_search_steps", "max_energy_ratio", "max_energy_balance_error", "trajectory_error", "trajectory_error_percent"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fields})


def write_settings(
    path: Path,
    best_by_combination: dict,
    baseline_dt: float,
    max_error_percent: float,
    validation_duration: float,
    max_energy_growth: float,
) -> None:
    """Write clean per-combination runtime settings."""
    document = {
        "combinations": best_by_combination,
        "metadata": {
            "baseline_dt": float(baseline_dt),
            "max_error_percent": float(max_error_percent),
            "validation_duration": float(validation_duration),
            "max_energy_growth": float(max_energy_growth),
        },
        "preferred_combination": next(
            (key for key in best_by_combination if key.startswith("implicit_bfgs/") and key.endswith("/JAX")),
            next((key for key in best_by_combination if key.startswith("implicit_bfgs/") and key.endswith("/NumPy")), ""),
        ),
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
    valid = [result for result in results if result.get("accuracy_valid", False)]
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
                    parameters.extend(("max_iterations", "history_size", "absolute_tolerance", "relative_tolerance", "line_search", "max_line_search_iterations", "globalization"))
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
        default=25.0,
        help="maximum candidate energy envelope; used only as an explosion guard",
    )
    parser.add_argument(
        "--max-displacement-factor", type=float, default=10.0,
        help="maximum displacement relative to the reference maximum displacement",
    )
    parser.add_argument(
        "--max-velocity-factor", type=float, default=10.0,
        help="maximum speed relative to the reference maximum speed, with a beam-scale floor",
    )
    parser.add_argument(
        "--max-frame-displacement", type=float, default=0.25,
        help="maximum displacement per candidate step as a fraction of beam length",
    )
    parser.add_argument("--max-dt", type=float, default=0.1, help="largest candidate timestep")
    parser.add_argument(
        "--max-iterations", default="5,10,20,30,50,75,100,150",
        help="candidate L-BFGS iteration caps; permissive pre-screening removes caps that cannot converge",
    )
    parser.add_argument(
        "--history-sizes", default="4,8,12",
        help="candidate L-BFGS history sizes; swept one parameter at a time",
    )
    parser.add_argument(
        "--relative-tolerances", default="5e-1,2e-1,1e-1,5e-2,1e-2,1e-3,1e-4,1e-6",
        help="candidate relative residual tolerances; permissive pre-screening removes values that cannot converge",
    )
    parser.add_argument(
        "--absolute-tolerances", default="1e-1,1e-2,1e-3,1e-4,1e-6,1e-8",
        help="candidate absolute residual tolerances; permissive pre-screening removes values that cannot converge",
    )
    parser.add_argument("--line-search", default="true,false")
    parser.add_argument(
        "--globalization", default="backtracking,watchdog",
        help="implicit globalization strategies to sweep; JAX supports backtracking only",
    )
    parser.add_argument(
        "--max-line-search-iterations", default="12,24",
        help="maximum backtracking trials for implicit candidates; swept one parameter at a time",
    )
    parser.add_argument("--max-error-percent", type=float, default=100.0, help="reported trajectory error; not an interactive acceptance criterion")
    parser.add_argument("--backend", choices=("both", "numpy", "jax"), default="both")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--settings-output", type=Path, default=Path("output/autotune/interactive/auto-tuned-settings.json"))
    args = parser.parse_args()

    if args.baseline_dt <= 0.0 or args.baseline_steps < 1 or args.max_error_percent < 0.0:
        parser.error("baseline-dt must be positive, baseline-steps must be positive, and max-error-percent cannot be negative")
    if (
        (args.validation_duration is not None and args.validation_duration <= 0.0)
        or args.max_energy_growth <= 0.0
        or args.max_displacement_factor <= 0.0
        or args.max_velocity_factor <= 0.0
        or args.max_frame_displacement <= 0.0
    ):
        parser.error("interactive bounds must be positive")
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
    absolute_tolerances = parse_list(args.absolute_tolerances, float)
    max_line_search_iterations = parse_list(args.max_line_search_iterations, int)
    line_search_values = [value.lower() in {"true", "1", "yes", "on"} for value in parse_list(args.line_search, str)]
    globalization_values = parse_list(args.globalization, str)
    if not max_iterations or not history_sizes or not relative_tolerances or not absolute_tolerances or not line_search_values or not max_line_search_iterations or not globalization_values:
        parser.error("all solver sweep lists must contain at least one value")
    if any(value not in {"backtracking", "watchdog"} for value in globalization_values):
        parser.error("globalization must contain only backtracking or watchdog")
    if (
        any(not np.isfinite(value) or value < 0.0 for value in absolute_tolerances)
        or any(not np.isfinite(value) or value < 0.0 for value in relative_tolerances)
        or any(value < 1 for value in max_iterations)
        or any(value < 0 for value in history_sizes)
        or any(value < 1 for value in max_line_search_iterations)
    ):
        parser.error("tolerances must be finite and non-negative; max-iterations must be positive; history sizes cannot be negative")
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
    try:
        reference_body, reference_states, reference_runtime, reference_energy = simulate_reference(
            args.case, args.i, args.j, args.k, args.baseline_dt, duration
        )
    except (RuntimeError, ValueError) as error:
        parser.error(f"unable to construct a finite reference trajectory: {error}")
    print(f"reference: {len(reference_states) - 1} steps, {reference_runtime:.3f} s")
    reference_energy_scale = max(float(np.max(reference_energy["magnitude"])), 1.0e-12)
    reference_energy_limit = reference_energy_scale * args.max_energy_growth
    print(f"reference energy envelope: {np.max(reference_energy['magnitude']):.3e} J; interactive explosion limit: {reference_energy_limit:.3e} J")
    max_displacement = float(np.max(np.linalg.norm(reference_states - reference_states[0], axis=2)))
    beam_length = float(np.ptp(reference_states[0, :, 0]))
    reference_velocities = np.diff(reference_states, axis=0) / args.baseline_dt
    reference_max_speed = float(np.max(np.linalg.norm(reference_velocities, axis=2)))
    interactive_limits = {
        "max_displacement": max_displacement * args.max_displacement_factor,
        "max_speed": max(reference_max_speed * args.max_velocity_factor, beam_length / max(duration, args.baseline_dt)),
        "max_frame_displacement": beam_length * args.max_frame_displacement,
    }
    displacement_percent = 100.0 * max_displacement / max(beam_length, np.finfo(float).eps)
    print(
        f"reference max displacement: {max_displacement:.4e} m "
        f"({displacement_percent:.2f}% of beam length)"
    )
    print(
        f"interactive bounds: displacement={interactive_limits['max_displacement']:.3e} m, "
        f"speed={interactive_limits['max_speed']:.3e} m/s, "
        f"per-step displacement={interactive_limits['max_frame_displacement']:.3e} m"
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
            active_globalization_values = ["backtracking"] if use_jax else globalization_values

            def evaluate(dt, iterations=0, history=0, absolute_tolerance=0.0, relative_tolerance=0.0, line_search=False, max_line_search_iterations=0, globalization="backtracking"):
                key = (float(dt), int(iterations), int(history), float(absolute_tolerance), float(relative_tolerance), bool(line_search), int(max_line_search_iterations), globalization)
                if key in seen:
                    return
                seen.add(key)
                settings = {} if strategy is None else {
                    "max_iterations": iterations,
                    "history_size": history,
                    "absolute_tolerance": absolute_tolerance,
                    "relative_tolerance": relative_tolerance,
                    "line_search": line_search,
                    "max_line_search_iterations": max_line_search_iterations,
                    "globalization": globalization,
                    "raise_on_failure": False,
                    "directional_residual_strategy": strategy,
                }
                candidate = simulate_candidate(
                    args.case, args.i, args.j, args.k, dt, duration, method, use_jax,
                    settings, reference_energy_limit, interactive_limits,
                )
                result = {
                    "method": method,
                    "backend": backend,
                    "directional_residual_strategy": strategy,
                    "dt": dt,
                    "max_iterations": iterations,
                    "history_size": history,
                    "absolute_tolerance": absolute_tolerance,
                    "relative_tolerance": relative_tolerance,
                    "line_search": line_search,
                    "max_line_search_iterations": max_line_search_iterations,
                    "globalization": globalization,
                    "valid": candidate["valid"],
                    "elapsed": candidate["elapsed"],
                    "jit_seconds": candidate.get("jit_seconds", 0.0),
                }
                for metric_name in (
                    "nonconverged_steps", "max_residual_reduction_factor",
                    "max_displacement", "max_speed", "max_frame_displacement",
                ):
                    result[metric_name] = candidate.get(metric_name, np.nan)
                for diagnostic_name in (
                    "direction_fallback_steps", "gradient_fallback_steps",
                    "watchdog_steps", "watchdog_acceptances", "rescue_steps",
                    "line_search_steps",
                ):
                    result[diagnostic_name] = candidate.get(diagnostic_name, 0)
                if candidate["valid"]:
                    result["trajectory_error"] = compute_trajectory_error(candidate["states"], reference_states, args.baseline_dt, dt)
                    result["trajectory_error_percent"] = 100.0 * result["trajectory_error"]
                    result["mean_iterations"] = float(np.mean(candidate["iterations"])) if candidate["iterations"] else 0.0
                    result["step_mean_ms"] = float(np.mean(candidate["step_times_ms"]))
                    result["step_p25_ms"] = float(np.quantile(candidate["step_times_ms"], 0.25))
                    result["step_p75_ms"] = float(np.quantile(candidate["step_times_ms"], 0.75))
                    result["max_energy_ratio"] = float(np.max(candidate["energy_magnitudes"]) / max(np.max(reference_energy["magnitude"]), np.finfo(float).tiny))
                    result["max_energy_balance_error"] = candidate["max_energy_balance_error"]
                    result["error"] = ""
                    result["stability_valid"] = True
                    # Interactive tuning intentionally does not reject
                    # bounded, damped, inexact trajectories for position error.
                    result["accuracy_valid"] = True
                    result["performance_valid"] = True
                else:
                    result["trajectory_error"] = np.nan
                    result["trajectory_error_percent"] = np.nan
                    result["mean_iterations"] = np.nan
                    result["step_mean_ms"] = np.nan
                    result["step_p25_ms"] = np.nan
                    result["step_p75_ms"] = np.nan
                    result["max_energy_ratio"] = np.nan
                    result["max_energy_balance_error"] = np.nan
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
                        f"abs_tol={absolute_tolerance:.1e} rel_tol={relative_tolerance:.1e} "
                        f"line_search={line_search!s:5s} ls_it={max_line_search_iterations:2d} "
                        f"globalization={globalization} "
                        f"compute={result['elapsed']:.3f}s jit={result['jit_seconds']:.3f}s "
                        f"step={result['step_mean_ms']:.3f}ms "
                        f"fallback={result['gradient_fallback_steps']} "
                        f"rescue={result['rescue_steps']} "
                        f"watchdog={result['watchdog_acceptances']} "
                        f"ls={result['line_search_steps']} "
                        f"error={result['trajectory_error_percent']:.4f}% "
                        f"mean_it={result['mean_iterations']:.2f}{status}"
                    )
                else:
                    print(
                        f"dt={dt:.3e} compute={result['elapsed']:.3f}s jit={result['jit_seconds']:.3f}s "
                        f"step={result['step_mean_ms']:.3f}ms "
                        f"ls={result['line_search_steps']} "
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
                    "absolute_tolerance": min(absolute_tolerances),
                    "relative_tolerance": min(relative_tolerances),
                    "line_search": True if True in line_search_values else line_search_values[0],
                    "max_line_search_iterations": max(max_line_search_iterations),
                    "globalization": active_globalization_values[0],
                }
                for dt in candidate_dts:
                    evaluate(dt, **robust)

                # Sweep solver settings at every candidate timestep. This
                # avoids missing a setting that recovers a larger stable
                # timestep than the robust profile can handle.
                exploration_dts = candidate_dts

                # First pre-screen each nonlinear parameter at the smallest
                # timestep with a deliberately permissive solve. A value that
                # cannot complete even this easiest candidate is not useful in
                # the one-at-a-time sweep: tightening tolerances or increasing
                # the timestep cannot make that same setting more viable.
                # This keeps the expanded ranges useful without repeating
                # obviously doomed experiments at every timestep.
                probe_dt = candidate_dts[0]
                probe_globalization = active_globalization_values[-1]
                probe_base = {
                    "iterations": max(max_iterations),
                    "history": max(history_sizes),
                    "absolute_tolerance": max(absolute_tolerances),
                    "relative_tolerance": max(relative_tolerances),
                    "line_search": True if True in line_search_values else line_search_values[0],
                    "max_line_search_iterations": max(max_line_search_iterations),
                    "globalization": probe_globalization,
                }

                def viable_probe(parameter, value):
                    trial = dict(probe_base)
                    trial[parameter] = value
                    evaluate(probe_dt, **trial)
                    matching = [result for result in combination_results if result["dt"] == probe_dt and (
                        result["max_iterations"], result["history_size"],
                        result["absolute_tolerance"], result["relative_tolerance"],
                        result["line_search"], result["max_line_search_iterations"],
                        result["globalization"],
                    ) == (
                        int(trial["iterations"]), int(trial["history"]),
                        float(trial["absolute_tolerance"]), float(trial["relative_tolerance"]),
                        bool(trial["line_search"]), int(trial["max_line_search_iterations"]),
                        trial["globalization"],
                    )]
                    return bool(matching and matching[-1]["valid"])

                viable_iterations = [value for value in max_iterations if viable_probe("iterations", value)]
                viable_history = [value for value in history_sizes if viable_probe("history", value)]
                viable_absolute = [value for value in absolute_tolerances if viable_probe("absolute_tolerance", value)]
                viable_relative = [value for value in relative_tolerances if viable_probe("relative_tolerance", value)]

                # Never leave a sweep empty: the permissive probe itself may
                # fail for a physically difficult case, but the subsequent
                # robust evaluations should still be recorded.
                viable_iterations = viable_iterations or [max(max_iterations)]
                viable_history = viable_history or [max(history_sizes)]
                viable_absolute = viable_absolute or [max(absolute_tolerances)]
                viable_relative = viable_relative or [max(relative_tolerances)]
                print(
                    "Prescreened nonlinear sweep: "
                    f"iterations={viable_iterations}, history={viable_history}, "
                    f"abs_tol={viable_absolute}, rel_tol={viable_relative}"
                )

                # These values cover the regimes that survived the permissive
                # prescreen. Each sweep is local to the robust profile, so
                # interactions do not multiply into a full Cartesian product.
                parameter_values = {
                    "iterations": viable_iterations,
                    "history": viable_history,
                    "absolute_tolerance": viable_absolute,
                    "relative_tolerance": viable_relative,
                    "line_search": line_search_values,
                    "max_line_search_iterations": max_line_search_iterations,
                    "globalization": active_globalization_values,
                }
                for dt in exploration_dts:
                    for parameter, values in parameter_values.items():
                        for value in values:
                            trial = dict(robust)
                            trial[parameter] = value
                            evaluate(dt, **trial)

                setting_candidates = [result for result in combination_results if result["valid"]]
                seed = min(setting_candidates, key=lambda result: result["elapsed"]) if setting_candidates else None
                if seed is not None:
                    seed_settings = {
                        "iterations": int(seed["max_iterations"]),
                        "history": int(seed["history_size"]),
                        "absolute_tolerance": float(seed["absolute_tolerance"]),
                        "relative_tolerance": float(seed["relative_tolerance"]),
                        "line_search": bool(seed["line_search"]),
                        "max_line_search_iterations": int(seed["max_line_search_iterations"]),
                        "globalization": seed["globalization"],
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
                    f"(finite and within interactive bounds)"
                )
            else:
                print(f"Stability {label}: no stable candidate")
            if accurate:
                accuracy_best = max(accurate, key=lambda result: result["dt"])
                performance_best = min(accurate, key=lambda result: result["elapsed"])
                print(
                    f"Interactive {label}: largest acceptable dt={accuracy_best['dt']:.3e} "
                    f"(trajectory error={accuracy_best['trajectory_error_percent']:.4f}%, not an acceptance gate)"
                )
                print(
                    f"Performance {label}: fastest acceptable dt={performance_best['dt']:.3e} "
                    f"(compute={performance_best['elapsed']:.3f}s)"
                )
            else:
                print(f"Interactive {label}: no bounded candidate")
                print(f"Performance {label}: unavailable because no candidate met the interactive bounds")

            acceptable = accurate
            if not acceptable:
                failed_settings = {} if strategy is None else {
                    "max_iterations": float("nan"),
                    "history_size": float("nan"),
                    "absolute_tolerance": float("nan"),
                    "relative_tolerance": float("nan"),
                    "line_search": float("nan"),
                    "max_line_search_iterations": float("nan"),
                    "globalization": "",
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
                    "diagnostics": {
                        name: 0 for name in (
                            "direction_fallback_steps", "gradient_fallback_steps",
                            "watchdog_steps", "watchdog_acceptances", "rescue_steps",
                            "line_search_steps",
                        )
                    },
                    "failure": "no candidate met the interactive boundedness criteria",
                    "stability_dt": float(max((result["dt"] for result in stable), default=np.nan)),
                    "accuracy_dt": float("nan"),
                    "performance_dt": float("nan"),
                }
                print(f"No bounded candidate met the interactive criteria for {label}; writing NaN settings")
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
                    "max_line_search_iterations": int(best["max_line_search_iterations"]),
                    "globalization": best["globalization"],
                    "raise_on_failure": False,
                    "directional_residual_strategy": strategy,
                },
                "trajectory_error": float(best["trajectory_error"]),
                "trajectory_error_percent": float(100.0 * best["trajectory_error"]),
                "elapsed": float(best["elapsed"]),
                "jit_seconds": float(best["jit_seconds"]),
                "mean_iterations": float(best["mean_iterations"]),
                "diagnostics": {
                    name: int(best[name]) for name in (
                        "direction_fallback_steps", "gradient_fallback_steps",
                        "watchdog_steps", "watchdog_acceptances", "rescue_steps",
                        "line_search_steps",
                    )
                },
                "stability_dt": float(max(result["dt"] for result in stable)),
                "accuracy_dt": float(max(result["dt"] for result in accurate)),
                "performance_dt": float(best["dt"]),
            }
            print(f"Best {label}: {best_by_combination[label]}")
    if not best_by_combination:
        raise RuntimeError("no timestepper/backend combination met --max-error")
    args.output = args.output or Path(f"output/autotune/interactive/soft_{args.case}_autotune.csv")
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
    write_settings(
        settings_output, best_by_combination, args.baseline_dt, args.max_error_percent,
        duration, args.max_energy_growth,
    )
    for label, best in best_by_combination.items():
        safe_label = label.replace("/", "-")
        individual = settings_output.with_name(f"{settings_output.stem}-{safe_label}{settings_output.suffix}")
        write_settings(
            individual, {label: best}, args.baseline_dt, args.max_error_percent,
            duration, args.max_energy_growth,
        )
    print(f"Wrote results to {output}")
    print(f"Wrote tuning plot to {output.with_suffix('.pdf')}")
    print(f"Wrote runtime settings to {settings_output}")
    print(f"Wrote {len(best_by_combination)} per-combination settings files")


if __name__ == "__main__":
    main()
