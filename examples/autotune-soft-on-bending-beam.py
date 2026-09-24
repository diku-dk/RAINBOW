"""Auto-tune all soft-body timestepper/backend combinations.

The reference is a small-timestep semi-implicit cantilever bending under
gravity. Candidate semi-implicit and implicit backward-Euler runs sweep
timestep multipliers. Implicit runs also sweep solver settings, and the
fastest valid candidate is selected independently for each backend.

Example::

The fine semi-implicit NumPy trajectory is used as a reference. Candidate
semi-implicit and implicit backward-Euler runs sweep timestep multipliers.
Implicit runs also sweep solver settings, and the fastest valid candidate is
selected independently for each backend.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darerl.simulators.soft import SoftBody, TetMesh, create_bending_baseline
from darerl.simulators.soft.mesh import boundary_faces

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENDING_GRAVITY = create_bending_baseline().gravity


def make_bending_body(i: int, j: int, k: int, use_jax: bool) -> tuple[SoftBody, np.ndarray]:
    baseline = create_bending_baseline(i, j, k)
    return baseline.create_body(use_jax=use_jax), baseline.fixed


def simulate_reference(i: int, j: int, k: int, dt: float, duration: float) -> tuple[SoftBody, np.ndarray, float]:
    body, fixed = make_bending_body(i, j, k, use_jax=False)
    steps = int(round(duration / dt))
    if not np.isclose(steps * dt, duration, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("duration must be an integer multiple of reference_dt")
    states = np.empty((steps + 1,) + body.x.shape, dtype=np.float64)
    states[0] = body.x
    start = time.perf_counter()
    for step in range(1, steps + 1):
        body.step(dt, gravity=BENDING_GRAVITY)
        states[step] = body.x
    elapsed = time.perf_counter() - start
    return body, states, elapsed


def simulate_candidate(
    i: int,
    j: int,
    k: int,
    dt: float,
    duration: float,
    method: str,
    use_jax: bool,
    settings: dict,
) -> dict:
    steps = int(round(duration / dt))
    if not np.isclose(steps * dt, duration, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("duration must be an integer multiple of every candidate dt")
    jit_seconds = 0.0
    try:
        if use_jax:
            warmup, _ = make_bending_body(i, j, k, use_jax)
            jit_start = time.perf_counter()
            if method == "implicit_bfgs":
                warmup.step_implicit(dt, gravity=BENDING_GRAVITY, settings=settings)
            else:
                warmup.step(dt, gravity=BENDING_GRAVITY)
            warmup.synchronize()
            jit_seconds = time.perf_counter() - jit_start

        body, _ = make_bending_body(i, j, k, use_jax)
        states = np.empty((steps + 1,) + body.x.shape, dtype=np.float64)
        states[0] = body.x
        iterations = []
        start = time.perf_counter()
        for step in range(1, steps + 1):
            if method == "implicit_bfgs":
                body.step_implicit(dt, gravity=BENDING_GRAVITY, settings=settings)
            else:
                body.step(dt, gravity=BENDING_GRAVITY)
            states[step] = body.x
            if not np.all(np.isfinite(body.x)) or not np.all(np.isfinite(body.v)):
                return {
                    "valid": False,
                    "error": f"non-finite state at step {step}",
                    "elapsed": time.perf_counter() - start,
                    "jit_seconds": jit_seconds,
                }
            if method == "implicit_bfgs":
                iterations.append(body.last_implicit_info["iterations"])
    except (OverflowError, RuntimeError, FloatingPointError, ValueError) as error:
        return {
            "valid": False,
            "error": str(error),
            "elapsed": time.perf_counter() - locals().get("start", time.perf_counter()),
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


def trajectory_error(candidate: np.ndarray, reference: np.ndarray, reference_dt: float, candidate_dt: float) -> float:
    reference_indices = np.rint(np.arange(len(candidate)) * candidate_dt / reference_dt).astype(np.int64)
    differences = candidate - reference[reference_indices]
    scale = max(float(np.ptp(reference[:, :, 0])), 1.0)
    error = float(np.max(np.linalg.norm(differences, axis=2)) / scale)
    return error if np.isfinite(error) else float("nan")


def parse_list(value: str, converter):
    return [converter(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path: Path, results: list[dict]) -> None:
    fields = ["method", "backend", "dt", "max_iterations", "history_size", "tolerance", "line_search", "valid", "error", "elapsed", "jit_seconds", "mean_iterations", "trajectory_error", "trajectory_error_percent"]
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
            "preferred_combination": "implicit_bfgs/JAX" if "implicit_bfgs/JAX" in best_by_combination else "implicit_bfgs/NumPy",
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
        faces = boundary_faces(baseline_elements)
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
                    parameters.extend(("max_iterations", "history_size", "tolerance", "line_search"))
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
    parser.add_argument("--i", type=int, default=12, help="beam nodes along its length")
    parser.add_argument("--j", type=int, default=4, help="beam nodes across y")
    parser.add_argument("--k", type=int, default=4, help="beam nodes across z")
    parser.add_argument("--baseline-dt", type=float, default=1.0e-4, help="semi-implicit baseline timestep")
    parser.add_argument("--baseline-steps", type=int, default=1000, help="number of semi-implicit baseline steps")
    parser.add_argument("--max-dt", type=float, default=0.1, help="largest candidate timestep")
    parser.add_argument("--max-iterations", default="5,10,15,20,25,30")
    parser.add_argument("--history-sizes", default=",".join(str(value) for value in range(1, 16)))
    parser.add_argument("--tolerances", default="1e-4,1e-6")
    parser.add_argument("--line-search", default="true,false")
    parser.add_argument("--max-error-percent", type=float, default=5.0, help="maximum trajectory error relative to baseline, in percent")
    parser.add_argument("--backend", choices=("both", "numpy", "jax"), default="both")
    parser.add_argument("--output", type=Path, default=Path("output/implicit_bending_autotune.csv"))
    parser.add_argument("--settings-output", type=Path, default=Path("output/auto-tuned-settings.json"))
    args = parser.parse_args()

    if args.baseline_dt <= 0.0 or args.baseline_steps < 1 or args.max_error_percent < 0.0:
        parser.error("baseline-dt must be positive, baseline-steps must be positive, and max-error-percent cannot be negative")
    duration = args.baseline_dt * args.baseline_steps
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
    tolerances = parse_list(args.tolerances, float)
    line_search_values = [value.lower() in {"true", "1", "yes", "on"} for value in parse_list(args.line_search, str)]
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
    reference_body, reference_states, reference_runtime = simulate_reference(args.i, args.j, args.k, args.baseline_dt, duration)
    print(f"reference: {len(reference_states) - 1} steps, {reference_runtime:.3f} s")
    max_displacement = float(np.max(np.linalg.norm(reference_states - reference_states[0], axis=2)))
    beam_length = float(np.ptp(reference_states[0, :, 0]))
    displacement_percent = 100.0 * max_displacement / max(beam_length, np.finfo(float).eps)
    print(
        f"reference max displacement: {max_displacement:.4e} m "
        f"({displacement_percent:.2f}% of beam length)"
    )

    results = []
    best_by_combination = {}
    for backend, use_jax in backends:
        for method in ("semi_implicit", "implicit_bfgs"):
            print(f"\n=== {method}/{backend} ===")
            if method == "semi_implicit":
                combinations = ((dt, 0, 0, 0.0, False) for dt in candidate_dts)
            else:
                combinations = itertools.product(candidate_dts, max_iterations, history_sizes, tolerances, line_search_values)
            combination_results = []
            for dt, iterations, history, tolerance, line_search in combinations:
                settings = {} if method == "semi_implicit" else {
                    "max_iterations": iterations,
                    "history_size": history,
                    "tolerance": tolerance,
                    "line_search": line_search,
                    "raise_on_failure": True,
                }
                candidate = simulate_candidate(args.i, args.j, args.k, dt, duration, method, use_jax, settings)
                result = {
                    "method": method,
                    "backend": backend,
                    "dt": dt,
                    "max_iterations": iterations,
                    "history_size": history,
                    "tolerance": tolerance,
                    "line_search": line_search,
                    "valid": candidate["valid"],
                    "elapsed": candidate["elapsed"],
                    "jit_seconds": candidate.get("jit_seconds", 0.0),
                }
                if candidate["valid"]:
                    result["trajectory_error"] = trajectory_error(candidate["states"], reference_states, args.baseline_dt, dt)
                    result["trajectory_error_percent"] = 100.0 * result["trajectory_error"]
                    result["mean_iterations"] = float(np.mean(candidate["iterations"])) if candidate["iterations"] else 0.0
                    result["error"] = ""
                else:
                    result["trajectory_error"] = np.nan
                    result["trajectory_error_percent"] = np.nan
                    result["mean_iterations"] = np.nan
                    result["error"] = candidate["error"]
                results.append(result)
                combination_results.append(result)
                if method == "implicit_bfgs":
                    print(
                        f"dt={dt:.3e} max_it={iterations:2d} history={history:2d} "
                        f"tol={tolerance:.1e} line_search={line_search!s:5s} "
                        f"compute={result['elapsed']:.3f}s jit={result['jit_seconds']:.3f}s "
                        f"error={result['trajectory_error_percent']:.4f}% "
                        f"mean_it={result['mean_iterations']:.2f}"
                    )
                else:
                    print(
                        f"dt={dt:.3e} compute={result['elapsed']:.3f}s jit={result['jit_seconds']:.3f}s "
                        f"error={result['trajectory_error_percent']:.4f}%"
                    )
            acceptable = [result for result in combination_results if result["valid"] and result["trajectory_error"] <= max_error]
            if not acceptable:
                failed_settings = {} if method == "semi_implicit" else {
                    "max_iterations": float("nan"),
                    "history_size": float("nan"),
                    "tolerance": float("nan"),
                    "line_search": float("nan"),
                    "raise_on_failure": False,
                }
                best_by_combination[f"{method}/{backend}"] = {
                    "method": method,
                    "backend": backend,
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
                print(f"No valid candidate met --max-error for {method}/{backend}; writing NaN settings")
                continue
            best = min(acceptable, key=lambda result: result["elapsed"])
            best_by_combination[f"{method}/{backend}"] = {
                "method": method,
                "backend": backend,
                "valid": True,
                "dt": float(best["dt"]),
                "solver_settings": {} if method == "semi_implicit" else {
                    "max_iterations": int(best["max_iterations"]),
                    "history_size": int(best["history_size"]),
                    "tolerance": float(best["tolerance"]),
                    "line_search": bool(best["line_search"]),
                    "raise_on_failure": True,
                },
                "trajectory_error": float(best["trajectory_error"]),
                "trajectory_error_percent": float(100.0 * best["trajectory_error"]),
                "elapsed": float(best["elapsed"]),
                "jit_seconds": float(best["jit_seconds"]),
                "mean_iterations": float(best["mean_iterations"]),
            }
            print(f"Best {method}/{backend}: {best_by_combination[f'{method}/{backend}']}")
    if not best_by_combination:
        raise RuntimeError("no timestepper/backend combination met --max-error")
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
