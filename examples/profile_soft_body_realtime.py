"""Profile soft-body solvers as a 30 FPS real-time frame producer.

Each rendered frame advances exactly ``1 / fps`` seconds.  A solver may use
multiple internal substeps within that frame.  For every mesh size and
solver/backend combination, the script first selects the fastest candidate
substep count that stays below the trajectory-error tolerance relative to a
fine semi-implicit reference.  It then measures the wall time needed to
produce frames and reports the number of timestep invocations per frame.

The implicit BFGS control settings are loaded from the auto-tuner JSON file::

    uv run python -m examples.profile_soft_body_realtime

Outputs are written to ``output/soft_body_realtime.pdf``, CSV, and JSON files.
The default material is a soft, nearly incompressible Ecoflex-like silicone.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from darerl.simulators.soft import SoftBody, SoftBaseline, create_bending_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAVITY = create_bending_baseline().gravity
METHODS = (
    ("semi_implicit", "NumPy", False),
    ("semi_implicit", "JAX", True),
    ("implicit_bfgs", "NumPy", False),
    ("implicit_bfgs", "JAX", True),
)


def make_beam(i: int, j: int, k: int) -> SoftBaseline:
    return create_bending_baseline(i, j, k)


def make_body(baseline: SoftBaseline, use_jax: bool) -> SoftBody:
    return baseline.create_body(use_jax=use_jax)


def advance_frame(body: SoftBody, method: str, frame_dt: float, substeps: int, settings: dict) -> None:
    dt = frame_dt / substeps
    for _ in range(substeps):
        if method == "implicit_bfgs":
            body.step_implicit(dt, gravity=GRAVITY, settings=settings)
        else:
            body.step(dt, gravity=GRAVITY, sync=False)
    body.synchronize()


def trajectory(
    baseline: SoftBaseline,
    method: str,
    use_jax: bool,
    duration: float,
    fps: float,
    substeps: int,
    settings: dict,
) -> tuple[np.ndarray, float]:
    frame_count = int(round(duration * fps))
    body = make_body(baseline, use_jax)
    states = np.empty((frame_count + 1,) + body.x.shape, dtype=np.float64)
    states[0] = body.x
    start = time.perf_counter()
    for frame in range(frame_count):
        advance_frame(body, method, 1.0 / fps, substeps, settings)
        states[frame + 1] = np.asarray(body.x)
    return states, time.perf_counter() - start


def error_against_reference(candidate: np.ndarray, reference: np.ndarray) -> float:
    scale = max(float(np.ptp(reference[:, :, 0])), 1.0)
    error = float(np.max(np.linalg.norm(candidate - reference, axis=2)) / scale)
    return error if np.isfinite(error) else float("inf")


def backend_label(use_jax: bool) -> str:
    return "JAX" if use_jax else "NumPy"


def tune_combination(
    baseline: SoftBaseline,
    method: str,
    use_jax: bool,
    fps: float,
    tuning_duration: float,
    reference_substeps: int,
    candidate_substeps: list[int],
    settings: dict,
    max_error: float,
) -> dict:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reference, _ = trajectory(
            baseline, "semi_implicit", False, tuning_duration, fps, reference_substeps, {}
        )
    candidates = []
    for substeps in candidate_substeps:
        try:
            # Compile JAX before the measured candidate run.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if use_jax:
                    trajectory(baseline, method, use_jax, 1.0 / fps, fps, substeps, settings)
                states, elapsed = trajectory(
                    baseline, method, use_jax, tuning_duration, fps, substeps, settings
                )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            candidates.append(
                {
                    "substeps": substeps,
                    "dt": 1.0 / (fps * substeps),
                    "error": float("inf"),
                    "tuning_seconds": float("inf"),
                    "failure": str(error),
                }
            )
            continue
        error = error_against_reference(states, reference)
        candidates.append(
            {
                "substeps": substeps,
                "dt": 1.0 / (fps * substeps),
                "error": error,
                "tuning_seconds": elapsed,
                "failure": "",
            }
        )
    acceptable = [candidate for candidate in candidates if candidate["error"] <= max_error]
    if not acceptable:
        return {
            "method": method,
            "backend": backend_label(use_jax),
            "valid": False,
            "substeps": None,
            "dt": None,
            "trajectory_error": float("inf"),
            "candidates": candidates,
            "solver_settings": settings,
            "failure": f"no candidate met error tolerance {max_error:g}",
        }
    # The shortest calibration run is the best proxy for frame cost.
    best = min(acceptable, key=lambda candidate: candidate["tuning_seconds"])
    return {
        "method": method,
        "backend": backend_label(use_jax),
        "valid": True,
        "substeps": int(best["substeps"]),
        "dt": float(best["dt"]),
        "trajectory_error": float(best["error"]),
        "candidates": candidates,
        "solver_settings": settings,
        "failure": "",
    }


def measure_frames(
    baseline: SoftBaseline,
    selection: dict,
    duration: float,
    fps: float,
) -> dict:
    body = make_body(baseline, selection["backend"] == "JAX")
    frame_count = int(round(duration * fps))
    samples = np.empty(frame_count, dtype=float)
    for frame in range(frame_count):
        start = time.perf_counter()
        advance_frame(body, selection["method"], 1.0 / fps, selection["substeps"], selection["solver_settings"])
        samples[frame] = 1000.0 * (time.perf_counter() - start)
    return {
        "mean_frame_ms": float(np.mean(samples)),
        "p95_frame_ms": float(np.quantile(samples, 0.95)),
        "max_frame_ms": float(np.max(samples)),
        "frame_budget_ms": 1000.0 / fps,
        "invocations_per_frame": int(selection["substeps"]),
        "dt": selection["dt"],
        "trajectory_error": selection["trajectory_error"],
    }


def mesh_sizes(min_elements: int, max_elements: int, count: int, j: int, k: int) -> list[int]:
    cells_per_x = 5 * (j - 1) * (k - 1)
    return [max(2, int(round(target / cells_per_x)) + 1) for target in np.geomspace(min_elements, max_elements, count)]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "elements", "nodes", "method", "backend", "valid", "failure", "substeps", "dt",
        "trajectory_error", "mean_frame_ms", "p95_frame_ms", "max_frame_ms",
        "frame_budget_ms", "invocations_per_frame",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def plot_results(path: Path, rows: list[dict], fps: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    labels = sorted({f"{row['method']}/{row['backend']}" for row in rows})
    styles = {
        "semi_implicit/NumPy": "o-",
        "semi_implicit/JAX": "s-",
        "implicit_bfgs/NumPy": "^-",
        "implicit_bfgs/JAX": "D-",
    }
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for label in labels:
            group = [
                row for row in rows
                if row["valid"] and f"{row['method']}/{row['backend']}" == label
            ]
            if not group:
                continue
            group.sort(key=lambda row: row["elements"])
            elements = np.array([row["elements"] for row in group])
            style = styles.get(label, "o-")
            axes[0].loglog(elements, [row["mean_frame_ms"] for row in group], style, label=label)
            axes[1].plot(elements, [row["invocations_per_frame"] for row in group], style, label=label)
        axes[0].axhline(1000.0 / fps, color="black", linestyle="--", label="frame budget")
        axes[0].set_ylabel("mean frame time [ms]")
        axes[0].set_title(f"30 FPS frame production ({fps:g} FPS)")
        axes[1].set_ylabel("timestep invocations per frame")
        axes[1].set_title("Selected stability substeps")
        for axis in axes:
            axis.set_xlabel("tetrahedral elements")
            axis.grid(True, which="both", alpha=0.25)
            axis.legend(fontsize="small")
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-elements", type=int, default=10_000)
    parser.add_argument("--max-elements", type=int, default=100_000)
    parser.add_argument("--num-sizes", type=int, default=10)
    parser.add_argument("--j", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--tuning-duration", type=float, default=None, help="calibration duration; defaults to the full simulation duration")
    parser.add_argument("--reference-substeps", type=int, default=64)
    parser.add_argument("--candidate-substeps", default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--max-error", type=float, default=1.0e-3)
    parser.add_argument("--settings", type=Path, default=Path("output/auto-tuned-settings.json"))
    parser.add_argument("--backend", choices=("both", "numpy", "jax"), default="both")
    parser.add_argument("--output", type=Path, default=Path("output/soft_body_realtime.pdf"))
    parser.add_argument("--csv", type=Path, default=Path("output/soft_body_realtime.csv"))
    parser.add_argument("--settings-output", type=Path, default=Path("output/soft_body_realtime_settings.json"))
    args = parser.parse_args()
    if args.fps <= 0.0 or args.duration <= 0.0:
        parser.error("fps and duration must be positive")
    tuning_duration = args.duration if args.tuning_duration is None else args.tuning_duration
    if tuning_duration <= 0.0 or tuning_duration > args.duration:
        parser.error("tuning-duration must be positive and no greater than duration")
    candidate_substeps = [int(value) for value in args.candidate_substeps.split(",") if value.strip()]
    if not candidate_substeps or any(value < 1 for value in candidate_substeps):
        parser.error("candidate-substeps must contain positive integers")
    settings_path = args.settings if args.settings.is_absolute() else PROJECT_ROOT / args.settings
    with settings_path.open() as stream:
        tuned = json.load(stream)
    tuned_combinations = tuned.get("combinations", {})

    def settings_for(method: str, backend: str) -> dict:
        entry = tuned_combinations.get(f"{method}/{backend}")
        if entry is not None:
            return dict(entry.get("solver_settings", {}))
        return {}
    try:
        import jax  # noqa: F401
    except ImportError:
        has_jax = False
    else:
        has_jax = True
    backends = [backend for backend in METHODS if backend[2] is False or (has_jax and args.backend != "numpy")]
    if args.backend == "jax":
        backends = [backend for backend in backends if backend[2]]
    elif args.backend == "numpy":
        backends = [backend for backend in backends if not backend[2]]

    rows = []
    selections = {}
    for i in mesh_sizes(args.min_elements, args.max_elements, args.num_sizes, args.j, args.k):
        baseline = make_beam(i, args.j, args.k)
        mesh = baseline.mesh
        print(f"\n{mesh.tet_count:,} elements / {mesh.node_count:,} nodes")
        for method, backend, use_jax in backends:
            settings = settings_for(method, backend)
            selection = tune_combination(
                baseline,
                method,
                use_jax,
                args.fps,
                tuning_duration,
                args.reference_substeps,
                candidate_substeps,
                settings,
                args.max_error,
            )
            measured = (
                measure_frames(baseline, selection, args.duration, args.fps)
                if selection["valid"]
                else {
                    "mean_frame_ms": float("nan"),
                    "p95_frame_ms": float("nan"),
                    "max_frame_ms": float("nan"),
                    "frame_budget_ms": 1000.0 / args.fps,
                    "invocations_per_frame": None,
                }
            )
            row = {
                "elements": mesh.tet_count,
                "nodes": mesh.node_count,
                **{key: value for key, value in selection.items() if key not in {"candidates", "solver_settings"}},
                **measured,
            }
            rows.append(row)
            selections[f"{mesh.tet_count}/{method}/{backend}"] = selection
            if selection["valid"]:
                print(
                    f"  {method:15s}/{backend:5s} dt={selection['dt']:.3e} "
                    f"invocations/frame={selection['substeps']:3d} "
                    f"mean={measured['mean_frame_ms']:.2f} ms "
                    f"p95={measured['p95_frame_ms']:.2f} ms "
                    f"error={selection['trajectory_error']:.2e}"
                )
            else:
                print(f"  {method:15s}/{backend:5s} unavailable: {selection['failure']}")

    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    csv_path = args.csv if args.csv.is_absolute() else PROJECT_ROOT / args.csv
    settings_output = args.settings_output if args.settings_output.is_absolute() else PROJECT_ROOT / args.settings_output
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    settings_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, rows)
    plot_results(output, rows, args.fps)
    with settings_output.open("w") as stream:
        json.dump(selections, stream, indent=2)
        stream.write("\n")
    print(f"Wrote frame-time report to {output}")
    print(f"Wrote measurements to {csv_path}")
    print(f"Wrote selected settings to {settings_output}")


if __name__ == "__main__":
    main()
