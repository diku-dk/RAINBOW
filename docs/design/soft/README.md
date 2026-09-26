# DARERL soft-body simulator

This directory documents the experimental soft-body simulator in
`darerl/simulators/soft`. The implementation is a contact-free, first-order
tetrahedral FEM solver intended to provide a fast reference implementation
for later collision, contact, and material-model work.

## Scope

The current solver supports:

- linear tetrahedral elements;
- lumped nodal mass;
- Saint Venant--Kirchhoff elasticity;
- Stable Neo-Hookean elasticity from Smith, de Goes, and Kim (2018);
- fixed-vertex Dirichlet conditions;
- triangular-face pressure Neumann conditions;
- persistent nodal external forces;
- semi-implicit Euler integration;
- fully implicit backward Euler solved with matrix-free limited-memory BFGS;
- NumPy and optional JAX execution paths.

Collision detection, contact, friction, damping, adaptive time stepping, and
general multi-body scene management are intentionally outside the prototype.

## Documents

- [Architecture](architecture.md) — modules, data ownership, and execution paths.
- [Theory derivation](theory.md) — continuum mechanics, linear tetrahedra,
  time-steppers, L-BFGS, and directional residuals.
- [Numerical formulation](numerics.md) — FEM forces, materials, boundary conditions, and time integration.
- [Testing and verification](testing.md) — test layers, commands, and known limitations.
- [Canonical baselines](baselines.md) — shared 10 cm stretch, compression, bend, and twist examples.
- [Naming convention](naming.md) — operation-oriented names for public APIs,
  examples, and tests.
- `examples/study_soft_convergence.py` studies implicit L-BFGS residual
  convergence using the auto-tuned settings.

The parameterized verification portfolio can be run from the project root:

```bash
python examples/verify_soft.py
```

It covers both NumPy and JAX backends, both materials, both time-steppers, and
all three implicit directional-residual strategies. By default it runs 16
backend/material/stepper combinations and writes a convergence and energy
report to `output/verify_soft.pdf`. Pass `--numpy` or `--jax` to run only
one backend. The profiling examples require the auto-tuner settings file
described in [testing](testing.md).

## Quick start

```python
from darerl.simulators.soft import SVKMaterial, SoftBody, TetMesh

mesh = TetMesh.from_vertices(vertices, tetrahedra, density=1000.0)
body = SoftBody(mesh, SVKMaterial(2.0e5, 0.3, 1000.0))

body.step(1.0e-4, gravity=(0.0, -9.81, 0.0))
```

For implicit stepping:

```python
body.step(
    1.0e-3,
    method="implicit_bfgs",
    settings={
        "max_iterations": 25,
        "history_size": 10,
        "absolute_tolerance": 1.0e-6,
        "relative_tolerance": 1.0e-6,
    },
)
```

The public API is intentionally small: construct a `TetMesh`, choose an
`SVKMaterial` or `StableNeoHookeanMaterial`, construct a `SoftBody`, configure
fixed vertices/pressure/external loads, query kinematics/forces/energy, and
advance with `step` or `step_implicit`. These paths are exercised by
`unit_tests/test_soft_api.py`.
