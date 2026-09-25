"""End-to-end verification of all supported soft-body solver combinations.

The script exercises the canonical bending beam with both materials.  The
semi-implicit method has two material combinations; implicit BFGS has three
directional-residual strategies for each material, giving eight combinations
in total.  It reports trajectory validity, energy behavior, and observed
time-step convergence, and writes a PDF report.

There is no useful closed-form solution for this deliberately nonlinear,
finite-deformation cantilever.  The convergence study therefore compares
successively refined time steps, which is the appropriate experimental
verification for this benchmark.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darerl.simulators.soft import (
    SoftBody,
    create_bending_baseline,
    create_compress_baseline,
    create_stretch_baseline,
    create_twist_baseline,
)

BASELINE_FACTORIES = {
    "bending": create_bending_baseline,
    "twist": create_twist_baseline,
    "compress": create_compress_baseline,
    "stretch": create_stretch_baseline,
}


@dataclass(frozen=True)
class Case:
    method: str
    material: str
    strategy: str | None = None

    @property
    def label(self) -> str:
        material = "SVK" if self.material == "svk" else "SNH"
        if self.method == "semi_implicit":
            return f"semi-implicit / {material}"
        return f"implicit BFGS / {material} / {self.strategy}"


@dataclass
class Run:
    backend: str
    case: Case
    dt: float
    times: np.ndarray
    states: np.ndarray
    energies: dict[str, np.ndarray]
    max_displacement: float
    fixed_position_error: float
    converged: bool


def make_cases() -> list[Case]:
    """Return the complete material/integrator/derivative portfolio."""
    cases = [Case("semi_implicit", material) for material in ("svk", "snh")]
    for material in ("svk", "snh"):
        for strategy in ("tangent_action", "closed_form", "finite_difference"):
            cases.append(Case("implicit_bfgs", material, strategy))
    return cases


def compute_energies(body: SoftBody, gravity: np.ndarray) -> dict[str, float]:
    """Compute kinetic, potential, elastic, and mechanical energies."""
    v = body.get_v()
    x = body.get_x()
    kinetic = 0.5 * np.sum(body.mesh.lumped_mass[:, None] * v * v)
    potential = -float(np.sum(body.mesh.lumped_mass[:, None] * x * gravity[None, :]))
    elastic = body.compute_elastic_energy()
    return {
        "kinetic": float(kinetic),
        "potential": potential,
        "elastic": elastic,
        "mechanical": float(kinetic + potential + elastic),
    }


def run_case(
    baseline,
    case: Case,
    dt: float,
    final_time: float,
    use_jax: bool,
) -> Run:
    """Run one case and validate its state throughout the trajectory."""
    steps = int(round(final_time / dt))
    if steps < 1 or not math.isclose(steps * dt, final_time, rel_tol=1e-10, abs_tol=1e-14):
        raise ValueError("final_time must be an integer multiple of dt")
    body = baseline.create_body(use_jax=use_jax, material_model=case.material)
    initial = body.get_x()
    gravity = np.asarray(baseline.gravity, dtype=np.float64)
    times = np.linspace(0.0, final_time, steps + 1)
    states = np.empty((steps + 1,) + initial.shape, dtype=np.float64)
    energy_names = ("kinetic", "potential", "elastic", "mechanical")
    energies = {name: np.empty(steps + 1, dtype=np.float64) for name in energy_names}

    states[0] = initial
    initial_energy = compute_energies(body, gravity)
    for name in energy_names:
        energies[name][0] = initial_energy[name]

    settings = {
        "max_iterations": 100,
        "history_size": 10,
        "tolerance": 1.0e-6,
        "line_search": True,
        "raise_on_failure": True,
    }
    if case.strategy is not None:
        settings["directional_residual_strategy"] = case.strategy

    for index in range(1, steps + 1):
        if case.method == "implicit_bfgs":
            body.step_implicit(dt, gravity=gravity, settings=settings)
        else:
            body.step(dt, gravity=gravity, sync=not use_jax)
            if use_jax:
                body.synchronize()
        x = body.get_x()
        v = body.get_v()
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)):
            raise RuntimeError(f"{case.label} produced non-finite state at t={times[index]:g}")
        states[index] = x
        current_energy = compute_energies(body, gravity)
        for name in energy_names:
            energies[name][index] = current_energy[name]

    fixed_error = float(np.max(np.abs(body.get_x()[baseline.fixed] - initial[baseline.fixed])))
    if fixed_error > 1.0e-11:
        raise RuntimeError(f"{case.label} violated fixed vertices by {fixed_error:.3e}")
    displacement = np.linalg.norm(states - initial[None, :, :], axis=2)
    return Run("jax" if use_jax else "numpy", case, dt, times, states, energies, float(np.max(displacement)), fixed_error, True)


def observed_convergence(runs: list[Run]) -> tuple[float, list[float]]:
    """Estimate order from successive final-state differences."""
    ordered = sorted(runs, key=lambda run: run.dt, reverse=True)
    differences = [
        float(np.linalg.norm(ordered[index].states[-1] - ordered[index + 1].states[-1]))
        for index in range(len(ordered) - 1)
    ]
    if len(differences) < 2 or min(differences) <= 0.0:
        return float("nan"), differences
    return math.log(differences[0] / differences[1], 2.0), differences


def plot_report(runs_by_case: dict[tuple[str, Case], list[Run]], output: Path, final_time: float) -> None:
    """Write summary, energy, and convergence plots to a multipage PDF."""
    from matplotlib.backends.backend_pdf import PdfPages

    cases = list(runs_by_case)
    with PdfPages(output) as pdf:
        figure, axis = plt.subplots(figsize=(12, max(4.0, 0.42 * len(cases) + 1.5)))
        axis.axis("off")
        rows = []
        for backend, case in cases:
            runs = runs_by_case[(backend, case)]
            finest = min(runs, key=lambda run: run.dt)
            rate, differences = observed_convergence(runs)
            energy = finest.energies["mechanical"]
            drift = float(energy[-1] - energy[0])
            rows.append([backend, case.label, f"{finest.max_displacement:.3e}", f"{drift:.3e}", f"{rate:.2f}", f"{differences[-1]:.3e}"])
        axis.table(
            cellText=rows,
            colLabels=["backend", "case", "max displacement (m)", "mechanical ΔE (J)", "observed order", "finest difference"],
            loc="center",
            cellLoc="left",
        )
        axis.set_title(f"Soft-body verification summary, final time = {final_time:g} s")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        rows = max(1, math.ceil(len(cases) / 2))
        figure, axes = plt.subplots(rows, 2, figsize=(12, max(8, 3.2 * rows)), sharex=True)
        axes = np.atleast_1d(axes).reshape(-1)
        for axis, (backend, case) in zip(axes, cases):
            run = min(runs_by_case[(backend, case)], key=lambda item: item.dt)
            for name in ("kinetic", "potential", "elastic", "mechanical"):
                axis.plot(run.times, run.energies[name], label=name)
            axis.set_title(f"{backend} / {case.label}", fontsize=9)
            axis.grid(True, alpha=0.25)
            axis.set_ylabel("energy (J)")
        axes[-2].set_xlabel("time (s)")
        axes[-1].set_xlabel("time (s)")
        axes[0].legend(fontsize=8)
        figure.suptitle("Energy behavior at the finest tested timestep")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(rows, 2, figsize=(12, max(8, 3.2 * rows)))
        axes = np.atleast_1d(axes).reshape(-1)
        for axis, (backend, case) in zip(axes, cases):
            runs = sorted(runs_by_case[(backend, case)], key=lambda item: item.dt, reverse=True)
            finest = runs[-1]
            errors = [np.linalg.norm(run.states[-1] - finest.states[-1]) for run in runs[:-1]]
            dts = [run.dt for run in runs[:-1]]
            axis.loglog(dts, errors, "o-")
            axis.set_title(f"{backend} / {case.label}", fontsize=9)
            axis.set_xlabel("timestep (s)")
            axis.set_ylabel("final-state error")
            axis.grid(True, which="both", alpha=0.25)
        figure.suptitle("Experimental timestep refinement")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(BASELINE_FACTORIES), default="bending")
    parser.add_argument("--dt", type=float, default=2.0e-4, help="coarsest timestep")
    parser.add_argument("--final-time", type=float, default=2.0e-2)
    parser.add_argument("--refinement-levels", type=int, default=3, choices=range(3, 6))
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--numpy", dest="backend", action="store_const", const="numpy", help="run only the NumPy backend")
    backend.add_argument("--jax", dest="backend", action="store_const", const="jax", help="run only the JAX backend")
    parser.set_defaults(backend=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dt <= 0.0 or args.final_time <= 0.0:
        raise ValueError("dt and final_time must be positive")
    baseline = BASELINE_FACTORIES[args.case](6, 3, 3)
    backends = [args.backend] if args.backend is not None else ["numpy", "jax"]
    cases = make_cases()
    runs_by_case: dict[tuple[str, Case], list[Run]] = {}
    for backend in backends:
        use_jax = backend == "jax"
        for case in cases:
            print(f"=== {backend} / {case.label} ===")
            runs = []
            for level in range(args.refinement_levels):
                dt = args.dt / (2.0**level)
                run = run_case(baseline, case, dt, args.final_time, use_jax)
                runs.append(run)
                print(f"dt={dt:.3e} max_displacement={run.max_displacement:.3e} mechanical_drift={run.energies['mechanical'][-1] - run.energies['mechanical'][0]:+.3e}")
            rate, differences = observed_convergence(runs)
            print(f"observed_order={rate:.3f} final_state_differences={[f'{value:.3e}' for value in differences]}")
            runs_by_case[(backend, case)] = runs
    output = args.output or Path(f"output/verify_soft_{args.case}.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_report(runs_by_case, output, args.final_time)
    print(f"case={args.case}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
