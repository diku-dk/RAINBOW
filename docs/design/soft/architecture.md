# Architecture

## Package layout

```text
darerl/simulators/soft/
├── mesh.py       NumPy-only structured beam and surface-mesh utilities
├── solver.py     mesh data, materials, forces, boundary conditions, steppers
└── __init__.py   public exports
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

## JAX execution

When JAX is available and `use_jax=True`, reference arrays and boundary data
are converted once to device arrays. Semi-implicit stepping uses one fused JIT
kernel for element forces, pressure forces, external loads, and integration.
Fully implicit stepping uses a separate JIT-compiled device-resident L-BFGS
kernel. Its residuals, JVP directional derivatives, line search, and fixed-size
history arrays remain on the device; only the final state and convergence
metadata are synchronized back to NumPy.

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

Material models implement `lame_parameters()` and `model_code`. New models
must provide:

1. a NumPy first Piola stress expression;
2. a NumPy energy density expression;
3. a matching JAX first Piola stress branch;
4. tests for rest stress, energy-gradient consistency, and backend agreement.

Future constitutive models should avoid coupling material logic to boundary
conditions or timestepper code.
