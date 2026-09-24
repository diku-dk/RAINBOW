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

The regularized `log(I_C+1)` term remains finite for inverted elements. This
does not make arbitrary simulations physically valid after inversion; it only
removes the singularity targeted by the model.

## Boundary conditions

### Dirichlet conditions

`fixed` is a boolean nodal mask. Fixed positions are restored to their
reference positions and fixed velocities are set to zero after every step.
`set_fixed_vertices(indices)` is the convenience API for creating the mask.

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
directional derivative is

```text
Jg(x)s = M/Δt² s - Jf(x)s
```

JAX computes `Jf(x)s` with `jax.jvp`; NumPy uses a configurable directional
finite difference. A backtracking residual-norm line search controls updates.
Solver diagnostics are available in `body.last_implicit_info`.
