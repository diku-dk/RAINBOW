"""Compare semi-implicit and fully implicit soft-body stepping.

Two comparisons are performed on the bending-beam reference problem:

1. Both methods use ``dt=0.001``.  This intentionally favors the
   semi-implicit method because it does not solve a nonlinear system.
2. The implicit method uses the tuned ``dt`` and solver settings from the
   auto-tuner JSON file.  Semi-implicit Euler advances the same macro step
   using configurable sub-stepping.  This exposes the cost of obtaining a
   comparable time resolution with an explicit method.

Run from the project root with::

    uv run python -m examples.compare_soft_steppers

The result is written to ``output/soft_stepper_comparison.pdf``.
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


def make_body(i: int, j: int, k: int, use_jax: bool) -> SoftBody:
    return create_bending_baseline(i, j, k).create_body(use_jax=use_jax)


def advance(
    body: SoftBody,
    method: str,
    macro_dt: float,
    macro_steps: int,
    implicit_settings: dict,
    semi_substeps: int = 1,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Advance one fresh body and return elapsed time, final x and history."""
    history = np.empty((macro_steps + 1,) + body.x.shape, dtype=np.float64)
    history[0] = body.x
    start = time.perf_counter()
    for step in range(macro_steps):
        if method == "implicit_bfgs":
            body.step_implicit(macro_dt, gravity=GRAVITY, settings=implicit_settings)
        else:
            sub_dt = macro_dt / semi_substeps
            for _ in range(semi_substeps):
                body.step(sub_dt, gravity=GRAVITY, sync=False)
            # This also synchronizes pending JAX work before timing stops.
            body.synchronize()
        history[step + 1] = body.x
    body.synchronize()
    return time.perf_counter() - start, np.array(body.x, copy=True), history


def timed_case(
    name: str,
    i: int,
    j: int,
    k: int,
    use_jax: bool,
    duration: float,
    implicit_dt: float,
    implicit_settings: dict,
    semi_substeps: int,
    runs: int,
) -> list[dict]:
    """Measure both methods for one comparison case."""
    configurations = (
        ("semi_implicit", duration / round(duration / implicit_dt), 1),
        ("implicit_bfgs", duration / round(duration / implicit_dt), 1),
        ("semi_implicit_substepped", implicit_dt, semi_substeps),
        ("implicit_bfgs_tuned", implicit_dt, 1),
    )
    results = []
    for method, macro_dt, substeps in configurations:
        steps = round(duration / macro_dt)
        if steps < 1 or not np.isclose(steps * macro_dt, duration, rtol=1.0e-10, atol=1.0e-14):
            raise ValueError(f"duration {duration} must be divisible by dt {macro_dt}")
        # Compile/warm up outside the reported measurements for JAX.
        warmup_body = make_body(i, j, k, use_jax)
        advance(warmup_body, method.removesuffix("_tuned").removesuffix("_substepped"), macro_dt, 1, implicit_settings, substeps)
        elapsed = []
        final = None
        history = None
        actual_method = method.removesuffix("_tuned").removesuffix("_substepped")
        for _ in range(runs):
            body = make_body(i, j, k, use_jax)
            measured, final, history = advance(body, actual_method, macro_dt, steps, implicit_settings, substeps)
            elapsed.append(measured)
        results.append(
            {
                "case": name,
                "method": method,
                "dt": macro_dt,
                "substeps": substeps,
                "steps": steps,
                "mean_seconds": float(np.mean(elapsed)),
                "std_seconds": float(np.std(elapsed, ddof=1)) if runs > 1 else 0.0,
                "final_displacement": float(np.linalg.norm(final[-1] - make_body(i, j, k, False).mesh.x0[-1])),
                "history": history,
            }
        )
    return results


def write_csv(path: Path, results: list[dict]) -> None:
    fields = ["case", "method", "dt", "substeps", "steps", "mean_seconds", "std_seconds", "final_displacement"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})


def plot_results(path: Path, results: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = (
        ("semi\n(dt=.001)", "implicit\n(dt=.001)"),
        ("semi\nsubstepped", "implicit\ntuned"),
    )
    colors = ["tab:blue", "tab:orange", "tab:blue", "tab:orange"]
    figures, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for panel, case, case_labels in zip(axes, ("equal_dt", "tuned_dt"), labels):
        group = [result for result in results if result["case"] == case]
        x = np.arange(len(group))
        if panel is axes[0]:
            values = [result["mean_seconds"] * 1000.0 for result in group]
        else:
            values = [result["final_displacement"] for result in group]
        panel.bar(x, values, color=colors)
        panel.set_xticks(x, case_labels, rotation=20)
        panel.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("mean runtime [ms]")
    axes[0].set_title("Equal dt = 0.001")
    axes[1].set_ylabel("tip displacement [m]")
    axes[1].set_title("Tuned implicit dt with matched substeps")
    figure = axes[0].figure
    figure.suptitle("Semi-implicit versus fully implicit bending beam")
    figure.savefig(path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i", type=int, default=12)
    parser.add_argument("--j", type=int, default=4)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--duration", type=float, default=0.02)
    parser.add_argument("--settings", type=Path, default=Path("output/auto-tuned-settings.json"))
    parser.add_argument("--semi-substeps", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="jax")
    parser.add_argument("--output", type=Path, default=Path("output/soft_stepper_comparison.pdf"))
    parser.add_argument("--csv", type=Path, default=Path("output/soft_stepper_comparison.csv"))
    args = parser.parse_args()
    if args.semi_substeps < 1 or args.runs < 1:
        parser.error("--semi-substeps and --runs must be positive")

    settings_path = args.settings if args.settings.is_absolute() else PROJECT_ROOT / args.settings
    with settings_path.open() as stream:
        tuned = json.load(stream)
    tuned_combinations = tuned.get("combinations", {})

    def settings_for(backend: str) -> dict:
        entry = tuned_combinations.get(f"implicit_bfgs/{backend}")
        if entry is not None:
            return dict(entry.get("solver_settings", {}))
        return dict(tuned.get("solver_settings", {}))

    preferred_backend = "JAX" if args.backend == "jax" else "NumPy"
    implicit_entry = tuned_combinations.get(f"implicit_bfgs/{preferred_backend}")
    if implicit_entry is None:
        implicit_entry = tuned_combinations.get("implicit_bfgs/NumPy") or tuned_combinations.get("implicit_bfgs/JAX")
    if implicit_entry is None:
        raise ValueError("settings file does not contain an implicit_bfgs combination")
    implicit_dt = float(implicit_entry["dt"])
    implicit_settings = settings_for(preferred_backend)
    use_jax = args.backend == "jax"
    if use_jax:
        try:
            import jax  # noqa: F401
        except ImportError:
            print("JAX is unavailable; falling back to NumPy")
            use_jax = False

    equal_dt = 1.0e-3
    equal_steps = round(args.duration / equal_dt)
    if not np.isclose(equal_steps * equal_dt, args.duration, rtol=1.0e-10, atol=1.0e-14):
        parser.error("duration must be an integer multiple of 0.001")
    results = []
    results.extend(timed_case("equal_dt", args.i, args.j, args.k, use_jax, args.duration, equal_dt, implicit_settings, 1, args.runs)[:2])
    results.extend(timed_case("tuned_dt", args.i, args.j, args.k, use_jax, args.duration, implicit_dt, implicit_settings, args.semi_substeps, args.runs)[2:])
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    csv_path = args.csv if args.csv.is_absolute() else PROJECT_ROOT / args.csv
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, results)
    plot_results(output, results)
    print(f"Loaded implicit dt={implicit_dt:.6g} from {settings_path}")
    for result in results:
        print(
            f"{result['case']:9s} {result['method']:25s} "
            f"mean={1000.0 * result['mean_seconds']:.3f} ms "
            f"std={1000.0 * result['std_seconds']:.3f} ms "
            f"dt={result['dt']:.6g} substeps={result['substeps']}"
        )
    print(f"Wrote plot to {output}")
    print(f"Wrote measurements to {csv_path}")


if __name__ == "__main__":
    main()
