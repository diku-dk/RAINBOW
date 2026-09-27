# Extension register

This page records plausible future extensions of the soft-body simulator. The
main documentation describes the behavior that exists today; this page keeps
forward-looking ideas, design questions, and implementation opportunities in
one place.

Statuses are deliberately broad. “Planned” means that the direction is
identified but not implemented. “Research candidate” means that the expected
benefit or formulation still needs investigation. “Out of scope” means that
it is intentionally excluded from the current prototype rather than being an
immediate implementation commitment.

## Overview

| Extension | Status | Possible gain |
|---|---|---|
| Additional constitutive material models | Planned | Broader material behavior and better physical modeling |
| Material damping | Research candidate | More realistic dissipative motion and improved interactive robustness |
| BDF2 time stepping | Planned | Second-order implicit integration with potentially better efficiency for smooth dissipative dynamics |
| Generalized-α time stepping | Planned | Tunable high-frequency damping while retaining second-order low-frequency accuracy |
| Additional globalization or nonlinear-solver strategies | Research candidate | Better convergence on difficult nonlinear steps or lower solve cost |
| Adaptive time stepping | Research candidate | More work on difficult transients and less work during easy motion |
| Collision detection and contact | Out of scope for current prototype | Interaction with obstacles and other bodies |
| Friction and contact constraints | Out of scope for current prototype | Sliding, gripping, and realistic contact response |
| Continuous inversion/collision handling | Research candidate | Fewer missed crossings and more reliable feasibility protection |
| Multi-body scene management | Out of scope for current prototype | Larger interactive scenes and object-object interaction |
| Additional execution backends | Research candidate | Portability to other accelerators or runtime systems |
| Complete autodiff-kernel performance study | Planned | Evidence for choosing analytical versus autodiff constitutive kernels end to end |
| Higher-order or richer finite elements | Research candidate | Better accuracy per element for smooth deformation fields |
| Production quality gates | Planned | Safer maintenance through CI, coverage, typing, linting, and regression tracking |

## Constitutive models and dissipation

### Additional constitutive material models

The current material interface supplies Lamé parameters and a compact
`model_code`. Adding a material currently requires synchronized NumPy and JAX
branches for the first Piola stress, energy density, and directional force
action, together with backend-agreement and energy-gradient tests. The
interface and current force ownership are described in
[architecture](architecture.md) and [numerics](numerics.md).

The longer-term design question is whether hard-coded model branches should be
replaced by a more extensible constitutive-kernel registry or a richer
material protocol. The possible gain is easier addition of anisotropy,
plasticity, viscoelasticity, and nearly incompressible models without
spreading model identifiers through every backend kernel.

### Material damping

The current examples have no material damping; damping comes only from the
chosen time integrator and solver tolerances. A damping model could introduce
velocity-dependent forces or internal variables. This could make interactive
motion more realistic and reduce persistent oscillations, but it would require
clear energy accounting, a revised implicit residual, and new validation
cases. The current damping discussion is in
[time-stepping](time-stepping.md).

## Time integration and step-size control

### BDF2

BDF2 would use two previous states and a startup step. For a first-order state
$\mathbf y=(\mathbf x,\mathbf v)$, its constant-step residual is

$$
\frac{3\mathbf y^{n+1}-4\mathbf y^n+\mathbf y^{n-1}}{2\Delta t}
=\mathbf G(\mathbf y^{n+1}).
$$

It is second-order and A-stable in the classical constant-step linear
multistep setting, but it is not L-stable or energy-conserving. It could offer
better accuracy per step for smooth dissipative motion, at the cost of startup
logic, history management, and care when the time step changes. See
[time-stepping](time-stepping.md) and [stability terminology](stability.md).

### Generalized-α

Generalized-α methods evaluate inertia and internal forces at different
weighted points. This permits controlled high-frequency numerical damping while
preserving second-order accuracy for low-frequency behavior. The method would
need acceleration history, method parameters, and a documented spectral-radius
policy. It is a candidate for simulations that need both accurate slow motion
and suppression of unresolved high-frequency motion.

### Adaptive time stepping

The current steppers use a caller-selected fixed `dt`. An adaptive controller
could respond to nonlinear iteration counts, residual reduction, feasibility
rejections, estimated local error, or changes in energy and displacement. The
possible gain is fewer expensive small steps during benign motion while
retaining small steps near strong deformation or rapid transients. It would
need reproducible acceptance rules and separate scientific and interactive
policies; it should not silently replace the current fixed-step reference
experiments.

### Additional globalization and nonlinear-solver strategies

The current implicit solver provides L-BFGS directions, residual backtracking,
preconditioned gradient fallback, rescue, and a NumPy watchdog mode. Other
possibilities include trust-region methods, nonlinear preconditioning, damping
or regularization of the quasi-Newton model, and alternative line searches.
The possible gain is improved convergence for cases where the current
direction and fallback sequence stagnate, or lower cost when a more suitable
globalization method is available. Any new strategy must preserve finite,
feasible trial-state handling and comparable diagnostics across backends. The
current control flow is described in [theory](theory.md), while the public
settings are documented in [architecture](architecture.md).

## Interaction and geometric robustness

### Collision, contact, friction, and multi-body scenes

The prototype currently supports fixed vertices, face pressure, and persistent
nodal forces, but no collision detection, contact constraints, friction, or
general multi-body scene management. Adding these features would enable
interactive manipulation and object-object interaction. They would also
require contact search, constraint enforcement, force regularization or
complementarity decisions, and new energy/momentum and robustness tests.

The current module boundaries and boundary-condition ownership are described
in [architecture](architecture.md); the current force and boundary
formulation is described in [numerics](numerics.md).

### Continuous inversion and collision handling

The current trial-state Jacobian guard checks only the endpoint of a proposed
trial. It does not search the segment between the current and trial positions
for an intermediate inversion or collision. A continuous feasibility test,
root solve, or conservative step-length bound could reduce missed crossings,
but would add force evaluations and potentially substantial runtime cost. This
is a separate concern from the current endpoint guard documented in
[numerics](numerics.md).

## Execution and performance

### Additional execution backends

NumPy is the reference path and JAX provides the device-resident path. A new
backend would need to reproduce force evaluation, directional residuals,
implicit iteration behavior, diagnostics, and state synchronization semantics.
The current separation between `time_stepper.py` and `solver.py` is the place
to assess whether a backend interface would reduce duplication.

### Complete autodiff-kernel performance study

The current constitutive benchmark compares analytical and autodiff stress
evaluation in isolation. A complete study should compare the full force path,
including element gathering, constitutive evaluation, nodal scatter, and the
complete timestep over representative mesh sizes and target hardware. The
motivation and current benchmark limitation are documented in
[numerics](numerics.md).

### Higher-order or richer finite elements

The implementation uses first-order tetrahedra with constant deformation
gradients per element. Higher-order tetrahedra or richer discretizations could
improve accuracy per element for smooth fields, but would change mesh
preprocessing, shape-function evaluation, quadrature, force assembly, tangent
actions, and verification baselines. This is a larger discretization extension
rather than a local material or timestepper change.

## Verification and maintenance

The focused soft-body tests verify the current materials, steppers, backends,
and representative baselines. Before treating the simulator as a production
release, the project could add a configured coverage threshold, static typing
and lint gates, a continuous-integration matrix, and a long-running
performance-regression baseline. These would improve maintenance confidence
without changing the numerical method itself. Current verification scope and
limitations are summarized in [testing](testing.md).

## Adding a new extension

For a new constitutive model or time-stepper, the usual path is:

1. define the mathematical update and acceptance criteria;
2. identify the NumPy reference implementation and any JAX kernel that must
   agree with it;
3. add focused unit tests and a verification example;
4. measure performance and diagnostics on both backends where supported;
5. update the implementation-focused documentation only after the behavior is
   implemented, and record the remaining work here until then.
