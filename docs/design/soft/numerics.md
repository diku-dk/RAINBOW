# Numerical formulation

## Linear tetrahedral kinematics

For a tetrahedron with reference vertices `X₀...X₃`, define

```text
Dₘ = [X₁-X₀, X₂-X₀, X₃-X₀]
D  = [x₁-x₀, x₂-x₀, x₃-x₀]
F  = D Dₘ⁻¹
```

`Dₘ⁻¹` and the reference volume are precomputed by `TetMesh`. Reference
orientation must be positive and nondegenerate. The nodal shape-function
gradients are the rows of `Dₘ⁻¹`, with the zeroth gradient obtained by the
partition-of-unity condition.

## Internal elastic forces

For first Piola stress `P`, the element nodal force is

```text
fₐᵉ = -V₀ P ∇Nₐ
```

Element contributions are accumulated into global nodal forces. The NumPy
implementation uses `numpy.bincount`; the JAX implementation uses a segment
sum.

## Materials

The constitutive tests verify more than backend agreement. For both SVK and
stable Neo-Hookean materials they check zero stress in the reference state,
that the first Piola stress is the deformation-gradient derivative of the
energy, frame indifference under left rotations, isotropy under right
rotations, and symmetry of `P F^T`. Inverted and near-singular deformation
gradients are also checked for finite values.

The JAX implementation keeps analytical stress expressions in the runtime
kernel. Energy autodiff is used as an independent test oracle, not as the
production implementation: on the development machine, a 100,000-element
JAX benchmark made autodiff about 2.3x slower for SVK and 14.7x slower for
stable Neo-Hookean. Autodiff remains useful when adding or changing a
material, after which an analytical kernel can be introduced once validated.

## Autodiff benchmark scope

The autodiff comparison was fully JIT compiled. It used
`jax.jit(jax.vmap(jax.grad(energy_density)))`, paid compilation in a warm-up
call, and synchronized device execution with `block_until_ready()` before
recording each runtime sample. There was no Python loop over elements and no
host callback in the measured path. The analytical implementation was also
JIT compiled and measured after warm-up.

The result should nevertheless be interpreted as a constitutive-kernel result,
not as a universal end-to-end solver result. The benchmark compared batched
stress evaluation and did not include element indexing, nodal force scatter,
or the complete timestep. The autodiff input used flattened nine-component
deformation gradients, while the analytical path used matrix-shaped inputs.
The benchmark was run on one CPU backend, one batch size, and one family of
deformation states. Different mesh sizes, accelerators, or memory-pressure
conditions may change the relative result.

The current engineering decision is therefore to retain analytical stress
expressions in the production force kernels and use energy autodiff as an
independent correctness oracle. A future performance study should compare
both approaches through the complete `_jax_forces` path, including element
gathering and nodal scatter, over representative mesh sizes and target
hardware before making a broader performance claim.

### Saint Venant--Kirchhoff

```text
C = FᵀF
E = 1/2 (C-I)
S = λ tr(E) I + 2 μ E
P = F S
```

### Stable Neo-Hookean

The implementation follows Smith, de Goes, and Kim, *Stable Neo-Hookean Flesh
Simulation* (2018):

```text
μ̂ = 4μ/3
λ̂ = λ + 5μ/6
α = 1 + μ̂/λ̂ - μ̂/(4λ̂)

Ψ(F) = μ̂/2 (I_C-3)
     + λ̂/2 (J-α)²
     - μ̂/2 log(I_C+1)
```

where `I_C = tr(FᵀF)` and `J = det(F)`. Its first Piola stress is

```text
P = μ̂ (1 - 1/(I_C+1)) F
  + λ̂ (J-α) ∂J/∂F
```

The regularized `log(I_C+1)` term remains finite for collapsed (`J=0`) and
inverted (`J<0`) elements. This does not make arbitrary simulations
physically valid after inversion; it only removes the singularity targeted by
the model. The exact collapsed and inverted states are covered by the
constitutive and assembled-force tests.

## Boundary conditions

### Dirichlet conditions

`fixed` is a boolean nodal mask. Fixed positions are restored to their
reference positions and fixed velocities are set to zero after every step,
including the JAX semi-implicit path. `set_fixed_vertices(indices)` is the
convenience API for creating the mask. Initial positions, velocities, external
loads, pressures, and accelerations must be finite.

### Pressure Neumann conditions

Pressure faces are oriented triangles. For a face with current area vector

```text
a = 1/2 ((x₁-x₀) × (x₂-x₀))
```

and scalar pressure `p`, the constant traction linear-FEM load is integrated as

```text
f_face = p a
f₀ = f₁ = f₂ = f_face / 3
```

Positive pressure acts in the supplied face-normal direction. Reverse the sign
for inward pressure. The current configuration is used, so this is a follower
pressure load.

Persistent nodal loads can be applied with `set_external_forces`.

## Time integration

### Semi-implicit Euler

```text
vⁿ⁺¹ = vⁿ + Δt M⁻¹ f(xⁿ)
xⁿ⁺¹ = xⁿ + Δt vⁿ⁺¹
```

### Fully implicit backward Euler

The solver finds free positions satisfying

```text
g(x) = M/Δt² (x-xⁿ-Δt vⁿ) - f(x) = 0
```

It uses limited-memory BFGS history vectors. For a direction `s`, the
directional residual is

```text
Jg(x)s = M/Δt² s - Jf(x)s
```

The force action `Jf(x)s` has three selectable implementations through
`directional_residual_strategy`:

- `tangent_action`: JAX uses forward-mode `jax.jvp` on the complete force
  operator. NumPy uses the analytical element tangent, since NumPy has no
  forward-mode autodiff.
- `closed_form`: evaluates the analytical SVK or stable-Neo-Hookean tangent
  and the pressure follower-load derivative directly.
- `finite_difference`: evaluates a configurable forward difference of the
  complete force operator, including pressure loads.

For JAX, all three paths are inside the JIT-compiled implicit kernel; the
finite-difference option does not call back into Python. A backtracking
residual-norm line search controls updates.
Solver diagnostics are available in `body.last_implicit_info`.
