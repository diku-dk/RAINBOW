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
    factors = np.empty(steps, dtype=np.float64)
    rates = np.empty(steps, dtype=np.float64)
    iterations = np.empty(steps, dtype=np.int32)
    initial_norms = np.empty(steps, dtype=np.float64)
    final_norms = np.empty(steps, dtype=np.float64)
    gravity = np.asarray(baseline.gravity, dtype=np.float64)
    for index in range(steps):
        body.step_implicit(dt, gravity=gravity, settings=settings)
        info = body.last_implicit_info
        factor = float(info["residual_reduction_factor"])
        factors[index] = factor
        rates[index] = -math.log10(max(factor, np.finfo(float).tiny))
        iterations[index] = int(info["iterations"])
        initial_norms[index] = float(info["initial_residual_norm"])
        final_norms[index] = float(info["final_residual_norm"])
    return {
        "backend": backend,
        "strategy": strategy,
        "dt": dt,
        "steps": steps,
        "actual_duration": steps * dt,
        "factor": factors,
        "rate": rates,
        "iterations": iterations,
        "initial_norm": initial_norms,
        "final_norm": final_norms,
    }


def write_csv(path: Path, results: list[dict]) -> None:
    fields = ["backend", "strategy", "invocation", "dt", "iterations", "reduction_factor", "convergence_rate", "initial_residual_norm", "final_residual_norm"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for index in range(result["steps"]):
                writer.writerow({
                    "backend": result["backend"],
                    "strategy": result["strategy"],
                    "invocation": index + 1,
                    "dt": result["dt"],
                    "iterations": result["iterations"][index],
                    "reduction_factor": result["factor"][index],
                    "convergence_rate": result["rate"][index],
                    "initial_residual_norm": result["initial_norm"][index],
                    "final_residual_norm": result["final_norm"][index],
                })


def plot_report(path: Path, results: list[dict], case: str, duration: float) -> None:
    """Write per-invocation and quartile convergence plots."""
    from matplotlib.backends.backend_pdf import PdfPages

    labels = [f"{result['backend']} / {result['strategy']}" for result in results]
    with PdfPages(path) as pdf:
        columns = 2
        rows = max(1, math.ceil(len(results) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(12, max(7, 3.2 * rows)), squeeze=False)
        for axis, result, label in zip(axes.flat, results, labels):
            axis.plot(np.arange(1, result["steps"] + 1), result["rate"], linewidth=1.0)
            axis.axhline(np.mean(result["rate"]), color="tab:red", linestyle="--", linewidth=1.0, label="mean")
            axis.set_title(label, fontsize=9)
            axis.set_xlabel("implicit solver invocation")
            axis.set_ylabel("-log10 residual reduction")
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8)
        for axis in axes.flat[len(results):]:
            axis.axis("off")
        figure.suptitle(f"L-BFGS convergence per invocation: {case}, requested duration={duration:g} s")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(12, 5))
        x = np.arange(len(results))
        means = np.array([np.mean(result["rate"]) for result in results])
        lower = np.array([np.quantile(result["rate"], 0.25) for result in results])
        upper = np.array([np.quantile(result["rate"], 0.75) for result in results])
        axis.fill_between(x, lower, upper, color="tab:blue", alpha=0.25, label="25th–75th percentile")
        axis.plot(x, means, "o-", color="tab:blue", label="mean")
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.set_ylabel("-log10 residual reduction")
        axis.set_title("Mean L-BFGS convergence rate with quartiles")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
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
    settings_path = args.settings if args.settings.is_absolute() else Path(__file__).resolve().parents[1] / args.settings
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
            print(
                f"{backend:5s} {strategy:18s} dt={dt:.3e} invocations={result['steps']:5d} "
                f"mean_rate={np.mean(result['rate']):.3f} "
                f"quartiles=({np.quantile(result['rate'], .25):.3f}, {np.quantile(result['rate'], .75):.3f})"
            )
    output = args.output or Path(f"output/soft_{args.case}_convergence.pdf")
    csv_path = args.csv or Path(f"output/soft_{args.case}_convergence.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, results)
    plot_report(output, results, args.case, args.duration)
    print(f"case={args.case}")
    print(f"report={output}")
    print(f"data={csv_path}")


if __name__ == "__main__":
    main()
