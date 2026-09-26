# Numerical formulation

## Linear tetrahedral kinematics

For a tetrahedron with reference vertices $\mathbf X_0,\ldots,\mathbf X_3$,
define

$$
\mathbf D_m=[\mathbf X_1-\mathbf X_0,\mathbf X_2-\mathbf X_0,\mathbf X_3-\mathbf X_0],
\qquad
\mathbf D=[\mathbf x_1-\mathbf x_0,\mathbf x_2-\mathbf x_0,\mathbf x_3-\mathbf x_0],
\qquad
\mathbf F=\mathbf D\mathbf D_m^{-1}.
$$

`Dₘ⁻¹` and the reference volume are precomputed by `TetMesh`. Reference
orientation must be positive and nondegenerate. The nodal shape-function
gradients are the rows of `Dₘ⁻¹`, with the zeroth gradient obtained by the
partition-of-unity condition.

## Internal elastic forces

For first Piola stress $\mathbf P$, the element nodal force is

$$
\mathbf f_a^{\mathrm e}=-V_0\,\mathbf P\,\nabla N_a.
$$

Element contributions are accumulated into global nodal forces. The NumPy
implementation uses `numpy.bincount`; the JAX implementation uses a segment
sum.

## Materials

The constitutive tests verify more than backend agreement. For both SVK and
stable Neo-Hookean materials they check zero stress in the reference state,
that the first Piola stress is the deformation-gradient derivative of the
energy, frame indifference under left rotations, isotropy under right
rotations, and symmetry of $\mathbf P\mathbf F^{\mathsf T}$. Inverted and
near-singular deformation
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

$$
\mathbf C=\mathbf F^{\mathsf T}\mathbf F,\qquad
\mathbf E=\frac12(\mathbf C-\mathbf I),\qquad
\mathbf S=\lambda\operatorname{tr}(\mathbf E)\mathbf I+2\mu\mathbf E,\qquad
\mathbf P=\mathbf F\mathbf S.
$$

### Stable Neo-Hookean

The implementation follows Smith, de Goes, and Kim, *Stable Neo-Hookean Flesh
Simulation* (2018):

$$
\widehat\mu=\frac{4\mu}{3},\qquad
\widehat\lambda=\lambda+\frac{5\mu}{6},\qquad
\alpha=1+\frac{\widehat\mu}{\widehat\lambda}
 -\frac{\widehat\mu}{4\widehat\lambda},
$$
$$
\Psi(\mathbf F)=\frac{\widehat\mu}{2}(I_C-3)
 +\frac{\widehat\lambda}{2}(J-\alpha)^2
 -\frac{\widehat\mu}{2}\log(I_C+1).
$$

where $I_C=\operatorname{tr}(\mathbf F^{\mathsf T}\mathbf F)$ and
$J=\det(\mathbf F)$. Its first Piola stress is

$$
\mathbf P=\widehat\mu\left(1-\frac{1}{I_C+1}\right)\mathbf F
 +\widehat\lambda(J-\alpha)\frac{\partial J}{\partial\mathbf F}.
$$

The regularized $\log(I_C+1)$ term remains finite for collapsed ($J=0$) and
inverted ($J<0$) elements. This does not make arbitrary simulations
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

$$
\mathbf a=\frac12\bigl((\mathbf x_1-\mathbf x_0)\times(\mathbf x_2-\mathbf x_0)\bigr).
$$

and scalar pressure `p`, the constant traction linear-FEM load is integrated as

$$
\mathbf f_{\mathrm{face}}=p\mathbf a,
\qquad
\mathbf f_0=\mathbf f_1=\mathbf f_2=\frac{\mathbf f_{\mathrm{face}}}{3}.
$$

Positive pressure acts in the supplied face-normal direction. Reverse the sign
for inward pressure. The current configuration is used, so this is a follower
pressure load.

Persistent nodal loads can be applied with `set_external_forces`.

## Time integration

### Semi-implicit Euler

$$
\begin{aligned}
\mathbf v^{n+1}&=\mathbf v^n+\Delta t\,M^{-1}\mathbf f(\mathbf x^n),\\
\mathbf x^{n+1}&=\mathbf x^n+\Delta t\,\mathbf v^{n+1}.
\end{aligned}
$$

### Fully implicit backward Euler

The solver finds free positions satisfying

$$
\mathbf g(\mathbf x)=\frac{M}{\Delta t^2}
 (\mathbf x-\mathbf x^n-\Delta t\,\mathbf v^n)-\mathbf f(\mathbf x)=\mathbf0.
$$

It uses limited-memory BFGS history vectors. For a direction $\mathbf s$, the
directional residual is

$$
J_{\mathbf g}(\mathbf x)\,\mathbf s
=\frac{M}{\Delta t^2}\mathbf s-J_{\mathbf f}(\mathbf x)\,\mathbf s.
$$

The force action $J_{\mathbf f}(\mathbf x)\mathbf s$ has three selectable implementations through
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

### Trial-state Jacobian guard

The line search accepts an optional feasibility callback. For implicit soft-body
steps, the callback evaluates the signed Jacobian of every current linear
tetrahedron in the trial configuration and requires

$$
\det(\mathbf F_e)>\texttt{minimum\_jacobian}.
$$

This is an inexpensive endpoint test, not a continuous collision test or root
solve: intermediate states between the current and trial positions are not
searched. If a trial fails, the line search reduces its step length and tries
again. NumPy skips the residual/force evaluation for rejected trials, and JAX
keeps the same test inside the JIT-compiled kernel.

The default is material-aware. StVK enables the guard because its constitutive
model is not intended to be used through element inversion. Stable
Neo-Hookean disables it because that model is explicitly finite for collapsed
and inverted elements. Set `prevent_inversion` to `True` or `False` to override
this policy, and use `minimum_jacobian` to impose a positive clearance above
zero. A non-feasible current state is rejected immediately when the guard is
enabled.

The NumPy L-BFGS path also supports two globalization layers. The default
backtracking mode first tries the L-BFGS direction, then a preconditioned
negative-gradient direction if the line search is exhausted. A finite trial
with a lower residual may be accepted as a rescue step. Optional watchdog mode
allows a bounded number of non-monotone trials before restoring the saved best
iterate. These controls are exposed through `globalization`,
`enable_gradient_fallback`, `max_watchdog_steps`, and
`watchdog_growth_factor` in implicit solver settings. The JAX kernel always
uses its device-resident backtracking path and accepts the best finite
residual-reducing trial when strict Armijo backtracking is exhausted. Both
backends reject non-finite trial residuals, including when ordinary line
search is disabled. Both backends expose separate
`direction_fallback_steps`, `gradient_fallback_steps`, and `rescue_steps`;
watchdog diagnostics
are zero for JAX because watchdog mode is explicitly unsupported there.
On JAX, watchdog mode and disabling the compiled gradient fallback are rejected
explicitly rather than silently ignored.
