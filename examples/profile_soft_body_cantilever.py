"""Profile and visualize stretch, bend, and twist modes of a soft beam.

The reference portfolio contains hanging, extension, compression, and twist
load cases. Both semi-implicit Euler and implicit BFGS are benchmarked. The
largest mesh also produces initial/final mesh figures and energy histories.

Run with::

    uv run python -m examples.profile_soft_body_cantilever
    uv run python -m examples.profile_soft_body_cantilever --scenario twist
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from darerl.simulators.soft import (
    BENDING_GRAVITY,
    SoftBaseline,
    SoftBody,
    create_bending_baseline,
    create_compress_baseline,
    create_stretch_baseline,
    create_twist_baseline,
)
from darerl.simulators.soft.mesh import boundary_faces

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("hanging", "extension", "compression", "twist")
METHODS = ("semi_implicit", "implicit_bfgs")
BACKENDS = (("NumPy", False), ("JAX", True))

# Approximate Ecoflex-like soft silicone properties.  The exact values depend
# on formulation and strain rate; these are intended as a soft, nearly
# incompressible profiling material rather than a product-data calibration.


@dataclass(frozen=True)
class LoadCase:
    name: str
    gravity: tuple[float, float, float]


def make_cantilever(i: int, j: int, k: int, scenario: str) -> SoftBaseline:
    factories = {
        "hanging": create_bending_baseline,
        "extension": create_stretch_baseline,
        "compression": create_compress_baseline,
        "twist": create_twist_baseline,
    }
    return factories[scenario](i, j, k)


def load_case(name: str) -> LoadCase:
    if name == "hanging":
        return LoadCase(name, BENDING_GRAVITY)
    if name in {"extension", "compression", "twist"}:
        return LoadCase(name, (0.0, 0.0, 0.0))
    raise ValueError(f"unknown load case: {name}")


def make_body(
    baseline: SoftBaseline,
    case: LoadCase,
    use_jax: bool,
) -> SoftBody:
    return baseline.create_body(use_jax=use_jax)


def time_force(body: SoftBody, repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        body.elastic_forces()
    return 1000.0 * (time.perf_counter() - start) / repeats


def time_steps(
    body: SoftBody,
    case: LoadCase,
    method: str,
    steps: int,
    dt: float,
    implicit_settings: dict,
) -> np.ndarray:
    samples = np.empty(steps, dtype=float)
    for step in range(steps):
        start = time.perf_counter()
        if method == "implicit_bfgs":
            body.step_implicit(dt, gravity=case.gravity, settings=implicit_settings)
        else:
            body.step(dt, gravity=case.gravity, sync=False)
            body.synchronize()
        samples[step] = 1000.0 * (time.perf_counter() - start)
    return samples


def benchmark_backend(
    baseline: SoftBaseline,
    case: LoadCase,
    method: str,
    steps: int,
    force_repeats: int,
    dt: float,
    use_jax: bool,
    implicit_settings: dict,
) -> dict:
    warmup = make_body(baseline, case, use_jax)
    compile_start = time.perf_counter()
    if method == "implicit_bfgs":
        warmup.step_implicit(dt, gravity=case.gravity, settings=implicit_settings)
    else:
        warmup.elastic_forces()
        warmup.step(dt, gravity=case.gravity)
    compile_seconds = time.perf_counter() - compile_start if use_jax else 0.0

    force_ms = time_force(make_body(baseline, case, use_jax), force_repeats)
    step_samples_ms = time_steps(
        make_body(baseline, case, use_jax),
        case,
        method,
        steps,
        dt,
        implicit_settings,
    )
    return {
        "compile_seconds": compile_seconds,
        "force_ms": force_ms,
        "step_samples_ms": step_samples_ms,
        "step_ms": float(np.mean(step_samples_ms)),
    }


def mesh_sizes(min_elements: int, max_elements: int, count: int, j: int, k: int) -> list[int]:
    cells_per_x = 5 * (j - 1) * (k - 1)
    requested = np.geomspace(min_elements, max_elements, count)
    return [max(2, int(round(target / cells_per_x)) + 1) for target in requested]


def summary(row: dict, backend: str, metric: str) -> tuple[float, float, float]:
    values = np.array([run[metric] for run in row["timings"][backend]], dtype=float)
    return float(np.mean(values)), float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))


def step_summary(row: dict, backend: str) -> tuple[float, float, float]:
    values = np.concatenate([run["step_samples_ms"] for run in row["timings"][backend]])
    return float(np.mean(values)), float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))


def trajectory(
    baseline: SoftBaseline,
    case: LoadCase,
    method: str,
    steps: int,
    dt: float,
    use_jax: bool,
    implicit_settings: dict,
) -> dict[str, np.ndarray | SoftBody]:
    mesh = baseline.mesh
    body = make_body(baseline, case, use_jax)
    x_previous = body.x.copy()
    pressure_previous = body.neumann_forces(x_previous)
    initial_potential = -np.sum(mesh.lumped_mass[:, None] * np.asarray(case.gravity) * mesh.x0)
    times = np.arange(steps + 1, dtype=float) * dt
    kinetic = np.zeros(steps + 1)
    potential = np.zeros(steps + 1)
    elastic = np.zeros(steps + 1)
    pressure_work = np.zeros(steps + 1)
    states = [body.x.copy()]
    for step in range(1, steps + 1):
        if method == "implicit_bfgs":
            body.step_implicit(dt, gravity=case.gravity, settings=implicit_settings)
        else:
            body.step(dt, gravity=case.gravity, sync=False)
            body.synchronize()
        x_current = np.asarray(body.x).copy()
        pressure_current = body.neumann_forces(x_current)
        displacement = x_current - x_previous
        pressure_work[step] = pressure_work[step - 1] + 0.5 * np.sum((pressure_previous + pressure_current) * displacement)
        kinetic[step] = 0.5 * np.sum(mesh.lumped_mass[:, None] * body.v * body.v)
        potential[step] = -np.sum(mesh.lumped_mass[:, None] * np.asarray(case.gravity) * x_current) - initial_potential
        elastic[step] = body.elastic_energy(x_current)
        states.append(x_current)
        x_previous = x_current
        pressure_previous = pressure_current
    return {
        "times": times,
        "kinetic": kinetic,
        "potential": potential,
        "elastic": elastic,
        "mechanical": kinetic + potential + elastic,
        "pressure_work": pressure_work,
        "initial": states[0],
        "final": states[-1],
    }


def plot_scaling(all_results: dict, output: Path, runs: int, steps: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    styles = {"NumPy": "o-", "JAX": "s-"}
    with PdfPages(output) as pdf:
        for case_name, method_results in all_results.items():
            for method, results in method_results.items():
                elements = np.array([row["elements"] for row in results])
                backends = sorted({backend for row in results for backend in row["timings"]})
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
                for backend in backends:
                    force = np.array([summary(row, backend, "force_ms") for row in results])
                    step = np.array([step_summary(row, backend) for row in results])
                    line = styles.get(backend, "o-")
                    axes[0].loglog(elements, force[:, 0], line, label=backend)
                    axes[0].fill_between(elements, force[:, 1], force[:, 2], alpha=0.2)
                    axes[1].loglog(elements, step[:, 0], line, label=backend)
                    axes[1].fill_between(elements, step[:, 1], step[:, 2], alpha=0.2)
                axes[0].set_title(f"{case_name}/{method}: force ({runs} runs)")
                axes[1].set_title(f"{case_name}/{method}: one step ({steps} samples/run)")
                axes[0].set_ylabel("force evaluation [ms]")
                axes[1].set_ylabel("completed timestep [ms]")
                for axis in axes:
                    axis.set_xlabel("tetrahedral elements")
                    axis.grid(True, which="both", alpha=0.25)
                    axis.legend()
                pdf.savefig(fig)
                plt.close(fig)


def plot_trajectory_pages(trajectories: dict, surface: np.ndarray, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output) as pdf:
        for case_name, methods in trajectories.items():
            fig, axes = plt.subplots(1, len(methods) + 1, figsize=(5 * (len(methods) + 1), 5), subplot_kw={"projection": "3d"})
            for axis, (label, data) in zip(axes, methods.items()):
                axis.plot_trisurf(data["final"][:, 0], data["final"][:, 1], data["final"][:, 2], triangles=surface, linewidth=0.05, alpha=0.8)
                axis.set_title(f"{case_name}: final {label}")
            first = next(iter(methods.values()))
            axes[-1].plot_trisurf(first["initial"][:, 0], first["initial"][:, 1], first["initial"][:, 2], triangles=surface, linewidth=0.05, alpha=0.8)
            axes[-1].set_title(f"{case_name}: initial")
            for axis in axes:
                axis.set_box_aspect((6, 1, 1))
            pdf.savefig(fig)
            plt.close(fig)

            fig, axes = plt.subplots(len(methods), 1, figsize=(9, 3.5 * len(methods)), squeeze=False, constrained_layout=True)
            for axis, (label, data) in zip(axes[:, 0], methods.items()):
                axis.plot(data["times"], data["kinetic"], label="kinetic")
                axis.plot(data["times"], data["potential"], label="gravity potential")
                axis.plot(data["times"], data["elastic"], label="elastic")
                axis.plot(data["times"], data["mechanical"], label="mechanical total", linewidth=2)
                axis.plot(data["times"], data["pressure_work"], label="pressure work", linestyle="--")
                axis.set_title(f"{case_name}: {label}")
                axis.set_xlabel("time [s]")
                axis.set_ylabel("energy [J]")
                axis.grid(True, alpha=0.25)
                axis.legend()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default=None)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--min-elements", type=int, default=10_000)
    parser.add_argument("--max-elements", type=int, default=100_000)
    parser.add_argument("--num-sizes", type=int, default=10)
    parser.add_argument("--j", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--duration", type=float, default=3.0, help="simulated motion time in seconds")
    parser.add_argument("--dt", type=float, default=1.0e-4, help="stable integration timestep; 1e-4 is suitable for the silicone defaults")
    parser.add_argument("--runs", type=int, default=1, help="independent simulation runs")
    parser.add_argument("--force-repeats", type=int, default=1)
    parser.add_argument("--settings", type=Path, default=Path("output/auto-tuned-settings.json"), help="auto-tuned BFGS JSON settings")
    parser.add_argument("--output", type=Path, default=Path("output/soft_body_scaling.pdf"))
    args = parser.parse_args()
    if args.duration <= 0.0 or args.dt <= 0.0:
        parser.error("duration and dt must be positive")
    steps = int(round(args.duration / args.dt))
    if steps < 1 or not np.isclose(steps * args.dt, args.duration, rtol=1.0e-10, atol=1.0e-14):
        parser.error("duration must be an integer multiple of dt")
    runs = args.runs
    if runs < 1:
        parser.error("runs must be positive")
    if args.min_elements <= 0 or args.max_elements < args.min_elements:
        parser.error("element range must satisfy 0 < min-elements <= max-elements")

    try:
        import jax  # noqa: F401
    except ImportError:
        has_jax = False
        print("JAX unavailable: only NumPy will be profiled")
    else:
        has_jax = True

    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_scenarios = (args.scenario,) if args.scenario else SCENARIOS
    selected_methods = (args.method,) if args.method else METHODS
    settings_path = args.settings if args.settings.is_absolute() else PROJECT_ROOT / args.settings
    with settings_path.open() as stream:
        tuned_settings = json.load(stream)
    tuned_combinations = tuned_settings.get("combinations", {})

    def settings_for(backend: str) -> dict:
        entry = tuned_combinations.get(f"implicit_bfgs/{backend}")
        if entry is not None:
            return dict(entry.get("solver_settings", {}))
        return {}

    print(f"Using BFGS settings from {settings_path}")
    print("Material and geometry: canonical 10 cm skin-like soft-body baselines")
    all_results = {}
    trajectories = {}
    largest_i = mesh_sizes(args.min_elements, args.max_elements, args.num_sizes, args.j, args.k)[-1]

    for scenario_name in selected_scenarios:
        case = load_case(scenario_name)
        all_results[scenario_name] = {}
        print(f"\n=== {scenario_name} ===")
        for method in selected_methods:
            results = []
            for i in mesh_sizes(args.min_elements, args.max_elements, args.num_sizes, args.j, args.k):
                baseline = make_cantilever(i, args.j, args.k, scenario_name)
                mesh, fixed, pressure_faces = baseline.mesh, baseline.fixed, baseline.pressure_faces
                row = {"elements": mesh.tet_count, "nodes": mesh.node_count, "timings": {}}
                row["timings"]["NumPy"] = [benchmark_backend(baseline, case, method, steps, args.force_repeats, args.dt, False, settings_for("NumPy")) for _ in range(runs)]
                if has_jax:
                    row["timings"]["JAX"] = [benchmark_backend(baseline, case, method, steps, args.force_repeats, args.dt, True, settings_for("JAX")) for _ in range(runs)]
                results.append(row)
                print(f"{method:15} {mesh.tet_count:7,} elements ({mesh.node_count:7,} nodes; {runs} runs × {steps} steps)")
                for backend in row["timings"]:
                    mean, q25, q75 = step_summary(row, backend)
                    print(f"  {backend:5} one-step mean={mean:9.3f} ms [{q25:9.3f}, {q75:9.3f}]")
            all_results[scenario_name][method] = results

            baseline = make_cantilever(largest_i, args.j, args.k, scenario_name)
            mesh = baseline.mesh
            for backend_name, use_jax in BACKENDS:
                if use_jax and not has_jax:
                    continue
                label = f"{method}/{backend_name}"
                trajectories.setdefault(scenario_name, {})[label] = trajectory(
                    baseline, case, method, steps, args.dt, use_jax,
                    settings_for(backend_name)
                )

    mesh = make_cantilever(largest_i, args.j, args.k, selected_scenarios[0]).mesh
    plot_scaling(all_results, output, runs, steps)
    states_output = output.with_name(f"{output.stem}_states{output.suffix}")
    plot_trajectory_pages(trajectories, boundary_faces(mesh.elements), states_output)
    print(f"\nWrote scaling plots to {output}")
    print(f"Wrote mesh and energy plots to {states_output}")


if __name__ == "__main__":
    main()
