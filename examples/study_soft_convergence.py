"""Study implicit L-BFGS convergence over a soft-body trajectory.

The study uses the best strategy/backend-specific settings written by
``autotune-soft.py``. For every backward-Euler invocation it records the
residual reduction factor

```text
final_residual_norm / initial_residual_norm
```

and reports the equivalent logarithmic convergence rate
``-log10(reduction_factor)``. The PDF contains per-invocation curves and a
mean-with-quartiles summary for all tested solver combinations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darerl.simulators.soft import (
    create_bending_baseline,
    create_compress_baseline,
    create_stretch_baseline,
    create_twist_baseline,
)


CASE_FACTORIES = {
    "bending": create_bending_baseline,
    "twist": create_twist_baseline,
    "compress": create_compress_baseline,
    "stretch": create_stretch_baseline,
}
STRATEGIES = ("tangent_action", "closed_form", "finite_difference")


def load_settings(path: Path, backend: str, strategy: str) -> tuple[float, dict]:
    """Load tuned timestep and solver settings for one combination."""
    with path.open() as stream:
        document = json.load(stream)
    combinations = document.get("combinations", {})
    entry = combinations.get(f"implicit_bfgs/{strategy}/{backend}")
    if entry is None:
        entry = combinations.get(f"implicit_bfgs/{backend}")
    if entry is None or not entry.get("valid", True):
        raise KeyError(f"missing valid settings for implicit_bfgs/{strategy}/{backend}")
    dt = float(entry["dt"])
    settings = dict(entry.get("solver_settings", {}))
    settings["directional_residual_strategy"] = strategy
    settings["raise_on_failure"] = True
    return dt, settings


def run_combination(baseline, backend: str, strategy: str, duration: float, dt: float, settings: dict) -> dict:
    """Run one tuned implicit combination and collect invocation diagnostics."""
    body = baseline.create_body(use_jax=backend == "JAX")
    steps = max(1, int(round(duration / dt)))
    iterations = np.empty(steps, dtype=np.int32)
    residual_histories = []
    gravity = np.asarray(baseline.gravity, dtype=np.float64)
    for index in range(steps):
        body.step_implicit(dt, gravity=gravity, settings=settings)
        info = body.last_implicit_info
        iterations[index] = int(info["iterations"])
        residual_histories.append(np.asarray(info["residual_norm_history"], dtype=np.float64))
    return {
        "backend": backend,
        "strategy": strategy,
        "dt": dt,
        "steps": steps,
        "actual_duration": steps * dt,
        "iterations": iterations,
        "residual_history": residual_histories,
    }


def write_csv(path: Path, results: list[dict]) -> None:
    fields = ["backend", "strategy", "invocation", "iteration", "dt", "residual_norm"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for index in range(result["steps"]):
                for iteration, residual_norm in enumerate(result["residual_history"][index]):
                    writer.writerow({
                        "backend": result["backend"],
                        "strategy": result["strategy"],
                        "invocation": index + 1,
                        "iteration": iteration,
                        "dt": result["dt"],
                        "residual_norm": residual_norm,
                    })


def plot_report(path: Path, results: list[dict], case: str, duration: float) -> None:
    """Write two residual-history plots for every backend/strategy pair."""
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        for result in results:
            label = f"{result['backend']} / {result['strategy']}"
            histories = result["residual_history"]
            width = max(len(history) for history in histories)
            values = np.full((len(histories), width), np.nan, dtype=np.float64)
            for index, history in enumerate(histories):
                values[index, : len(history)] = history
            iterations = np.arange(width)

            figure, axes = plt.subplots(1, 2, figsize=(13, 5))
            for history in histories:
                axes[0].plot(np.arange(len(history)), history, color="tab:blue", alpha=0.18, linewidth=0.8)
            axes[0].set_yscale("log")
            axes[0].set_xlabel("L-BFGS iteration")
            axes[0].set_ylabel("residual norm")
            axes[0].set_title("All residual histories")
            axes[0].grid(True, which="both", alpha=0.25)

            mean = np.nanmean(values, axis=0)
            lower = np.nanquantile(values, 0.25, axis=0)
            upper = np.nanquantile(values, 0.75, axis=0)
            axes[1].fill_between(iterations, lower, upper, color="tab:blue", alpha=0.25, label="25th–75th percentile")
            axes[1].plot(iterations, mean, color="tab:blue", linewidth=1.5, label="mean")
            axes[1].set_yscale("log")
            axes[1].set_xlabel("L-BFGS iteration")
            axes[1].set_ylabel("residual norm")
            axes[1].set_title("Mean residual history with quartiles")
            axes[1].grid(True, which="both", alpha=0.25)
            axes[1].legend(fontsize=8)
            figure.suptitle(f"{label}: {case}, requested duration={duration:g} s")
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASE_FACTORIES), default="bending")
    parser.add_argument("--duration", type=float, default=3.0, help="requested simulated duration in seconds")
    parser.add_argument("--settings", type=Path, default=Path("output/auto-tuned-settings.json"))
    parser.add_argument("--backend", choices=("both", "numpy", "jax"), default="both")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0:
        raise ValueError("duration must be positive")
    project_root = Path(__file__).resolve().parents[1]
    settings_path = args.settings if args.settings.is_absolute() else project_root / args.settings
    if not settings_path.is_file():
        raise FileNotFoundError(f"settings file not found: {settings_path}; run examples/autotune-soft.py first")
    try:
        import jax  # noqa: F401
        has_jax = True
    except ImportError:
        has_jax = False
    backends = ["NumPy"]
    if args.backend in {"both", "jax"} and has_jax:
        backends.append("JAX")
    if args.backend == "jax" and not has_jax:
        raise RuntimeError("JAX was requested but is unavailable")
    baseline = CASE_FACTORIES[args.case](6, 3, 3)
    results = []
    for backend in backends:
        for strategy in STRATEGIES:
            dt, settings = load_settings(settings_path, backend, strategy)
            result = run_combination(baseline, backend, strategy, args.duration, dt, settings)
            results.append(result)
            final_factors = np.array([history[-1] / max(history[0], np.finfo(float).tiny) for history in result["residual_history"]])
            rates = -np.log10(np.maximum(final_factors, np.finfo(float).tiny))
            print(
                f"{backend:5s} {strategy:18s} dt={dt:.3e} invocations={result['steps']:5d} "
                f"mean_rate={np.mean(rates):.3f} "
                f"quartiles=({np.quantile(rates, .25):.3f}, {np.quantile(rates, .75):.3f})"
            )
    output = args.output or project_root / f"output/soft_{args.case}_convergence.pdf"
    csv_path = args.csv or project_root / f"output/soft_{args.case}_convergence.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, results)
    plot_report(output, results, args.case, args.duration)
    print(f"case={args.case}")
    print(f"report={output}")
    print(f"data={csv_path}")


if __name__ == "__main__":
    main()
