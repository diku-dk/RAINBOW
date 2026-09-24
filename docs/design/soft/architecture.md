# Architecture

## Package layout

```text
darerl/simulators/soft/
├── types.py      shared array, mesh, and material type exports
├── material.py   constitutive material definitions
├── mesh.py       TetMesh and structured beam/surface-mesh utilities
├── forces.py     stateless NumPy force and energy kernels
├── nonlinear.py  reusable NumPy L-BFGS and Armijo solver routines
├── stepper.py    semi-implicit and implicit-BFGS strategies
├── solver.py     SoftBody orchestration and JAX hot kernels
└── __init__.py  public exports
```

The package deliberately does not import `rainbow.geometry` or `igl`. The
prototype can therefore be used independently of the legacy geometry stack.

## Data flow

```text
reference vertices + tetrahedra
              │
              ▼
          TetMesh
  Dm⁻¹, volume, shape gradients,
  lumped mass, inverse mass
              │
              ▼
          SoftBody
  x, v, fixed vertices, pressure faces,
  external nodal forces, material
              │
       ┌──────┴────────┐
       ▼               ▼
  NumPy path       JAX path
  host arrays      device-resident x,v
       │               │
       └──────┬────────┘
              ▼
       elastic and applied forces
              │
              ▼
       time integration
```

`TetMesh` owns immutable reference-space quantities. `SoftBody` owns mutable
state and boundary-condition data. Material objects contain only material
parameters and a small model identifier; constitutive kernels remain separate
from mesh and integrator state.

## Module layout

The implementation is organized by responsibility:

- `material.py` contains constitutive material data and validation.
- `types.py` contains shared array, mesh, and material type exports.
- `mesh.py` contains `TetMesh` preprocessing and mesh construction helpers.
- `forces.py` contains stateless NumPy force, pressure, tangent, and energy
  kernels used as the reference implementation.
- `nonlinear.py` contains the backend-independent L-BFGS and line-search
  routines.
- `stepper.py` contains the semi-implicit and implicit-BFGS integration
  strategies.
- `solver.py` contains `SoftBody` state, public operations, and the JAX hot
  kernels used by the time steppers. It is the high-level orchestration
  object, not the nonlinear solver itself.
- `baseline.py` contains reusable deformation-mode examples.
- `theory.md` derives the implemented continuum-to-discrete equations and
  directional-residual strategies.

`SoftBody` remains the public orchestration object while the numerical pieces
can be inspected and tested independently.

## JAX execution

When JAX is available and `use_jax=True`, reference arrays and boundary data
are converted once to device arrays. Semi-implicit stepping uses one fused JIT
kernel for element forces, pressure forces, external loads, and integration.
Fully implicit stepping uses a separate JIT-compiled device-resident L-BFGS
kernel. Its residuals, selectable tangent-action/closed-form/finite-difference
directional derivatives, line search, and fixed-size history arrays remain on
the device; only the final state and convergence metadata are synchronized
back to NumPy.

`step(sync=False)` leaves the state device-resident. Call
`body.synchronize()` before reading NumPy state. This distinction is important
for performance measurements because host synchronization can dominate small
steps.

The JAX implicit kernel uses static iteration/history bounds so each distinct
solver configuration is compiled once. Changing settings such as history size
or maximum iterations can therefore trigger a new compilation. JAX 64-bit
mode is enabled because the implicit residual tolerances and NumPy reference
path use float64.

## Extensibility points

Material models implement `compute_lame_parameters()` and `model_code`. New models
must provide:

1. a NumPy first Piola stress expression;
2. a NumPy energy density expression;
3. a matching JAX first Piola stress branch;
4. tests for rest stress, energy-gradient consistency, and backend agreement.

Future constitutive models should avoid coupling material logic to boundary
conditions or timestepper code.

The NumPy implicit stepper delegates L-BFGS direction construction, curvature
history management, Armijo backtracking, and residual iteration to
`nonlinear.py`. The JAX implicit path remains device-resident in `solver.py`;
its fixed iteration bounds and history arrays are expressed directly with JAX
primitives for compilation.
