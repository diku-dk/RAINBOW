# Testing and verification

## Unit tests

The focused soft-body tests are split into logical `unit_tests/test_soft_*.py`
modules and use Python's `unittest` module. Shared fixtures live in
`unit_tests/test_soft_support.py`.

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
- constitutive stress invariants: rest-state zero stress, energy-gradient
  consistency, linear-elastic limit, objectivity, isotropy, and symmetric
  `P F^T`;
- finite constitutive responses for near-singular and inverted deformation
  gradients;
- restoring force directions for ±10% dilation/compression of a centered
  regular tetrahedron;
- restoring force directions for ±10% scaling of an eight-tet octahedron,
  including axis alignment, zero accumulated center force, and zero net force;
- zero elastic force under coordinate-axis translations and rotations;
- restoring generalized force under positive and negative unit-cube shear,
  for both material models and available NumPy/JAX paths.
- exact single-tetrahedron force assembly against `-V P grad(N)`;
- affine multi-tetrahedron patch-test cancellation at interior nodes;
- global linear and angular momentum balance of internal forces;
- invariance to tetrahedron ordering and rejection of degenerate or inverted
  reference tetrahedra;
- complete multi-tetrahedron NumPy/JAX elastic-force agreement.
- beam mass-lumping checks: nodal masses equal density times barycentric dual
  control volumes;
- analytic total-mass conservation across multiple beam resolutions;
- linear scaling of assembled mass with material density.
- beam body-force assembly checks: nodal force density equals `rho * gravity`,
  integrated resultants equal total mass times gravity, and results are
  independent of mesh resolution.
- pressure Neumann-force checks: current-face area-vector resultants, equal
  one-third nodal surface-FEM distribution, per-face pressure values,
  translation invariance, empty face sets, and NumPy/JAX agreement.
- persistent external nodal-force checks: set/clear behavior, input
  validation, exact additivity with elastic, pressure, and body forces, and
  NumPy/JAX timestep agreement.
- direct kinematics checks: exact affine deformation-gradient recovery,
  translation invariance, rigid-rotation recovery, Green--Lagrange strain
  definition/symmetry, multi-element consistency, and position-shape validation.
- isolated nonlinear-solver checks in `test_soft_nonlinear.py`: L-BFGS
  directions and history, curvature rejection, Armijo acceptance and failure,
  quadratic convergence, and iteration-limit reporting.
- API and integration edge cases: mesh/index/density validation, boundary-face
  cancellation, fixed-vertex updates, timestep/settings validation, implicit
  solves with pressure and external loads, zero-free-DOF systems, directional
  force derivatives, pressure covariance, closed-surface pressure balance, and
  baseline load-resultant semantics.
- backend parity for initially displaced fixed vertices and rejection of
  non-finite state, load, pressure, and gravity inputs.
- deterministic helper checks for profiler mesh scaling, compute_trajectory error
  handling, and autotuner sample alignment in `test_soft_examples.py`.

Run the focused suite with:

```bash
uv run python -m unittest discover -s unit_tests -p 'test_soft_*.py' -v
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

The profiling and comparison examples require this JSON file. If it is
missing, they stop with a command explaining how to generate it. They can be
run either as modules from the project root or directly by filename; direct
execution is supported by adding the project root to the example import path.
The comparison PDF reports runtime for both the equal-step-size and tuned
step-size cases; the CSV retains final displacement as an additional metric.

The public API smoke test in `test_soft_api.py` exercises materials, mesh
constructors, boundary extraction, all baseline factories, NumPy/JAX body
methods, load setters/clearers, synchronization, and both steppers.

Public API names follow the operation-oriented rules in
[naming.md](naming.md); in particular, numerical queries use `compute_*` and
state changes use `set_*`, `add_*`, or `remove_*`.

## Current validation limits

- The soft solver enables JAX 64-bit arithmetic so implicit residual
  tolerances are comparable with the NumPy reference path.
- The full repository test suite includes legacy modules requiring optional
  packages such as `igl`, `networkx`, and `pxr`. The focused soft-body suite
  does not depend on those packages.
- Contact, collision, damping, and adaptive timestep behavior are not covered
  because they are outside the current simulator scope.
