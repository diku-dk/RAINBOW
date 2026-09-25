# From continuum mechanics to the soft-body time steppers

This note follows the equations implemented in `darerl/simulators/soft`. It
specializes the discussion to first-order tetrahedra, where all shape-function
gradients and the deformation gradient are constant inside each element.

## 1. Continuum starting point

Let `X` be a material point in the reference configuration and `x(X,t)` its
current position. The deformation gradient is

```text
F = ∂x / ∂X
```

The momentum equation is

```text
ρ ẍ = div(P) + ρ g + b,
```

where `ρ` is density, `P` is the first Piola--Kirchhoff stress, `g` is a body
acceleration such as gravity, and `b` is any additional volumetric load.
Prescribed surface tractions are applied on boundary faces. Contact is not
part of this prototype.

## 2. One linear tetrahedron

For reference vertices `X₀,...,X₃` and current vertices `x₀,...,x₃`, define

```text
Dm = [X₁-X₀  X₂-X₀  X₃-X₀]
D  = [x₁-x₀  x₂-x₀  x₃-x₀]
```

The mesh stores `inv_Dm = Dm⁻¹`, so the linear shape functions give

```text
F = D Dm⁻¹.
```

This is `compute_deformation_gradient`. Because the shape functions are
linear, one `F` represents the whole tetrahedron; no quadrature loop is
needed. The reference gradients satisfy

```text
∇N₀ = -(∇N₁ + ∇N₂ + ∇N₃),
∇N₁, ∇N₂, ∇N₃ = rows of Dm⁻¹.
```

With `V = det(Dm)/6`, the mesh stores both `∇Nₐ` and `V∇Nₐ`.

### Elastic forces

Virtual work gives the internal force at vertex `a`:

```text
fᵉₐ = -V P ∇Nₐ.
```

The four element contributions are summed into the global nodal array. The
constitutive branches currently implemented are:

- Saint Venant--Kirchhoff: `E = 1/2(FᵀF-I)`,
  `S = λ tr(E)I + 2μE`, and `P = FS`;
- stable Neo-Hookean: the stable energy and first Piola stress from Smith,
  de Goes, and Kim.

The reference tetrahedra must have positive orientation. Stable Neo-Hookean
forces can still be evaluated for collapsed or inverted *current* elements.

### Mass and body force

Lumped mass assigns one quarter of every element mass to each vertex:

```text
mₐ += ρV/4.
```

Thus a uniform body acceleration produces

```text
fᵇₐ = mₐ g.
```

The total body force is therefore total mass times `g`, independently of mesh
resolution.

### Pressure faces

For an oriented triangular face, its current area vector is

```text
a = 1/2 ((x₁-x₀) × (x₂-x₀)).
```

Constant pressure is integrated with the three linear face shape functions,
so each face vertex receives

```text
fᵖₐ = p a/3.
```

Face ordering determines the normal direction and therefore the sign.

## 3. Semi-implicit Euler

Let `f(x)` contain elastic, pressure, persistent external, and body forces.
With diagonal lumped mass `M`, the semi-implicit update is

```text
vⁿ⁺¹ = vⁿ + Δt M⁻¹ f(xⁿ) + Δt g,
xⁿ⁺¹ = xⁿ + Δt vⁿ⁺¹.
```

The new velocity is used for the position update, so only one force evaluation
is needed and no nonlinear solve is required. Fixed vertices are enforced as

```text
xⁿ⁺¹ₐ = Xₐ,    vⁿ⁺¹ₐ = 0.
```

The NumPy and JAX paths apply the same rule.

## 4. Fully implicit backward Euler

For backward Euler, forces are evaluated at the unknown new position `x`:

```text
vⁿ⁺¹ = (x - xⁿ)/Δt,
M(vⁿ⁺¹-vⁿ)/Δt = f(x).
```

Eliminating velocity gives the position residual used by `step_implicit`:

```text
g(x) = M/Δt² (x - xⁿ - Δt vⁿ) - f(x) = 0.
```

Only free vertex degrees of freedom occur in this residual. Fixed coordinates
are reset to their reference positions and their velocities are zero. For a
free direction `s`, the required Jacobian action is

```text
Jg(x)s = M/Δt² s - Jf(x)s.
```

The global Jacobian is never assembled.

## 5. L-BFGS with a directional residual

Newton's method would solve `Jg(x) Δx = -g(x)`, but constructing and
factorizing this matrix is expensive for large meshes. L-BFGS instead stores a
short history of position and residual changes. After an accepted update:

```text
sᵏ = xᵏ⁺¹ - xᵏ,
yᵏ = g(xᵏ⁺¹) - g(xᵏ).
```

The implementation obtains `yᵏ` with the directional residual

```text
yᵏ = Jg(xᵏ⁺¹)sᵏ.
```

This is the local linearization of the residual in the direction just taken.
It avoids forming a matrix or running another nonlinear solve.

The L-BFGS two-loop recursion uses the history pairs and

```text
ρᵏ = 1 / ((yᵏ)ᵀsᵏ).
```

It starts with `-g`, applies the history backwards and forwards, and scales
the result with the diagonal inverse-mass preconditioner `Δt²M⁻¹`. The number
of stored pairs is limited by `history_size`; invalid or insufficient-curvature
pairs are discarded.

The proposed update is tested with a backtracking line search using

```text
φ(x) = 1/2 g(x)ᵀg(x).
```

The step length is reduced until the Armijo condition is satisfied. If the
quasi-Newton direction is not a descent direction, the diagonal
inverse-mass direction is used. Convergence is declared when

```text
||g(x)|| ≤ tolerance · max(||g(x⁰)||, 1).
```

### Globalization and fallback

The line search globalizes the local quasi-Newton model by accepting a trial
`x + αd` only when its residual merit satisfies

```text
φ(x + αd) ≤ φ(x) + c₁ α gᵀd.
```

Rejected steps reduce `α` by `line_search_reduction`. If the trial fails the
element-Jacobian feasibility policy, it is rejected before its residual is
evaluated. This is a cheap endpoint guard, not a continuous collision or
root-crossing test. Non-finite trial residuals are rejected independently of
whether Armijo backtracking is enabled.

When all L-BFGS trials are rejected, the NumPy solver retries with the
preconditioned negative-gradient direction `d_g = -Dg`. It can then rescue the
best finite trial encountered when that trial has lower merit than the current
state, even if strict Armijo was not satisfied. The event count is returned as
`gradient_fallback_steps`.

The optional `globalization="watchdog"` mode permits bounded non-monotone
progress after strict backtracking fails. Trials with merit at most
`watchdog_growth_factor` times the saved strict merit are allowed for at most
`max_watchdog_steps` iterations. The saved strict iterate is restored and the
L-BFGS history is cleared when that budget is exceeded, since old curvature
pairs no longer describe the restored state. The number of accepted
non-monotone trials is reported as `watchdog_acceptances`.
Both backends also report `direction_fallback_steps`, `gradient_fallback_steps`,
and `rescue_steps`; for JAX, `direction_fallback_steps` counts invalid
quasi-Newton directions, while `gradient_fallback_steps` counts device-side
line-search retries. For JAX, `rescue_steps` counts device-side
best-finite-trial rescues, while NumPy counts rescues after the configured
fallback search.

These controls change only how the residual is reached, not the constitutive
model. The NumPy implementation exposes `globalization`,
`enable_gradient_fallback`, `max_watchdog_steps`, and
`watchdog_growth_factor`. The JAX implementation keeps its fixed-shape search
inside the compiled kernel and implements descent-direction fallback and
best-finite-trial rescue; it does not execute the Python watchdog path.
JAX rejects watchdog-specific settings instead of silently ignoring them.

The JAX version keeps the residual, history, line search, and iteration inside
one JIT-compiled device-resident kernel.

For convergence studies, each implicit invocation reports its residual
reduction factor

```text
r = ||g_final|| / max(||g_initial||, 1).
```

The study script plots `-log10(r)`: larger values indicate stronger residual
reduction during that invocation. It summarizes the distribution of these
values over the simulated trajectory with the mean and 25th--75th percentile
band.

## 6. Three directional-force strategies

All strategies approximate or evaluate the same force derivative:

```text
Jf(x)s = d/dε f(x + εs) at ε=0.
```

The inertial term `M/Δt² s` is then added to obtain `Jg(x)s`.

### Tangent action

With JAX, `tangent_action` uses forward-mode `jax.jvp` on the complete force
operator:

```text
f(x) = fᵉ(x) + fᵖ(x) + fᵉˣᵗ + fᵇ.
```

The JVP propagates `s` through `F`, the constitutive stress, element assembly,
and pressure forces. It is inside the JIT-compiled L-BFGS kernel, so there is
no Python callback or host-side directional loop. NumPy has no built-in
forward-mode autodiff, so its tangent-action path uses the same analytical
local tangent as `closed_form`.

### Closed-form directional force

The `closed_form` path differentiates the linear-tet formulas directly. Since
`F = DDm⁻¹`:

```text
dF = dD Dm⁻¹.
```

For SVK, the implementation uses

```text
dC = dFᵀF + FᵀdF,
dE = 1/2 dC,
dS = λ tr(dE)I + 2μdE,
dP = dF S + F dS.
```

The element derivative is `dfᵉₐ = -V dP∇Nₐ`. Stable Neo-Hookean differentiates
the invariants and cofactor used by its implemented first Piola stress and
then uses the same element assembly.

Pressure is a follower load. If `e₁=x₁-x₀`, `e₂=x₂-x₀`, then

```text
da = 1/2 ((ds₁ × e₂) + (e₁ × ds₂)),
dfᵖₐ = p da/3.
```

Thus pressure directional forces are included, not treated as constant loads.

### Finite difference

The `finite_difference` path uses a forward difference of the complete force:

```text
Jf(x)s ≈ [f(x + hs) - f(x)]/h.
```

The implementation scales `h` from `directional_epsilon`, the current-position
norm, and the direction norm. This makes the perturbation less sensitive to
mesh scale. It requires two complete force evaluations and has truncation and
round-off error. In JAX both evaluations remain inside the JIT kernel; NumPy
performs them on the host.

## 7. Selecting a strategy

JAX production runs normally use `tangent_action`. Use `closed_form` when
profiling or validating an analytical tangent, and `finite_difference` as a
simple cross-check for new material or force terms:

```python
body.step_implicit(
    dt,
    settings={"directional_residual_strategy": "closed_form"},
)
```
