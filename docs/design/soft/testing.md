# Testing and verification

## Unit tests

The focused soft-body tests are in
`unit_tests/test_darerl_soft.py` and use Python's `unittest` module.

They cover:

- positive orientation of the NumPy beam mesh;
- zero force in the reference configuration;
- SVK and Stable Neo-Hookean finite/inversion force behavior;
- force as the negative gradient of elastic energy;
- NumPy/JAX force agreement for both material models;
- linear-FEM pressure-face nodal-force distribution;
- fixed-vertex preservation;
- semi-implicit gravity response;
- implicit BFGS settings and convergence under gravity.

Run the focused suite with:

```bash
uv run python -m unittest unit_tests.test_darerl_soft -v
```

## Verification examples

The four material/integrator examples are:

```bash
uv run python -m examples.verify_semi_implicit_svk
uv run python -m examples.verify_semi_implicit_snh
uv run python -m examples.verify_implicit_svk
uv run python -m examples.verify_implicit_snh
```

They run a fixed-root cantilever under gravity and verify finite state and
fixed-node invariants. The implicit examples also print BFGS convergence
diagnostics.

## Profiling reference

`examples/profile_soft_body_cantilever.py` performs one simulation run per
method/backend, with a default duration of three seconds and `dt=0.001`. It
compares NumPy/JAX and both semi-implicit/implicit methods over approximately
10K--100K tetrahedra. The BFGS settings are loaded from
`output/auto-tuned-settings.json` by default.

The profiler also generates mesh-state and energy plots for hanging,
extension, compression, and twisting load cases.

`examples/profile_soft_body_realtime.py` performs the real-time scalability
study. It tunes internal substeps for a 30 FPS frame budget at each mesh size,
then reports frame time and timestep invocations per frame for NumPy/JAX
semi-implicit and implicit BFGS combinations. Invalid candidates are recorded
in the CSV/JSON report rather than aborting the entire mesh sweep.

`examples/autotune-soft-on-bending-beam.py` tunes all available combinations
by default (`--backend both`) and stores them under the `combinations` key in
`output/auto-tuned-settings.json`. Each combination entry contains its tuned
timestep and, for BFGS, its solver settings.

## Current validation limits

- The soft solver enables JAX 64-bit arithmetic so implicit residual
  tolerances are comparable with the NumPy reference path.
- The full repository test suite includes legacy modules requiring optional
  packages such as `igl`, `networkx`, and `pxr`. The focused soft-body suite
  does not depend on those packages.
- Contact, collision, damping, and adaptive timestep behavior are not covered
  because they are outside the current simulator scope.
