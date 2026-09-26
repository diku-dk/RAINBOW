# From continuum mechanics to the soft-body time steppers

This note follows the equations implemented in `darerl/simulators/soft`. It
specializes the discussion to first-order tetrahedra, where all shape-function
gradients and the deformation gradient are constant inside each element.

## 1. Continuum starting point

Let $\mathbf X$ be a material point in the reference configuration and
$\mathbf x(\mathbf X,t)$ its
current position. The deformation gradient is

$$
\mathbf F = \frac{\partial \mathbf x}{\partial \mathbf X}.
$$

The momentum equation is

$$
\rho\,\ddot{\mathbf x}=\operatorname{div}(\mathbf P)+\rho\,\mathbf g+\mathbf b.
$$

where $\rho$ is density, $\mathbf P$ is the first Piola--Kirchhoff stress,
$\mathbf g$ is a body acceleration such as gravity, and $\mathbf b$ is any
additional volumetric load.
Prescribed surface tractions are applied on boundary faces. Contact is not
part of this prototype.

## 2. One linear tetrahedron

For reference vertices $\mathbf X_0,\ldots,\mathbf X_3$ and current vertices
$\mathbf x_0,\ldots,\mathbf x_3$, define

$$
\mathbf D_m=[\mathbf X_1-\mathbf X_0\;\mathbf X_2-\mathbf X_0\;\mathbf X_3-\mathbf X_0],
\qquad
\mathbf D=[\mathbf x_1-\mathbf x_0\;\mathbf x_2-\mathbf x_0\;\mathbf x_3-\mathbf x_0].
$$

The mesh stores `inv_Dm = Dm⁻¹`, so the linear shape functions give

$$
\mathbf F=\mathbf D\mathbf D_m^{-1}.
$$

This is `compute_deformation_gradient`. Because the shape functions are
linear, one `F` represents the whole tetrahedron; no quadrature loop is
needed. The reference gradients satisfy

$$
\nabla N_0=-(\nabla N_1+\nabla N_2+\nabla N_3),
\qquad
\nabla N_1,\nabla N_2,\nabla N_3
\text{ are the rows of }\mathbf D_m^{-1}.
$$

With $V=\det(\mathbf D_m)/6$, the mesh stores both $\nabla N_a$ and
$V\nabla N_a$.

### Elastic forces

Virtual work gives the internal force at vertex `a`:

$$
\mathbf f^{\mathrm e}_a=-V\,\mathbf P\,\nabla N_a.
$$

The four element contributions are summed into the global nodal array. The
constitutive branches currently implemented are:

- Saint Venant--Kirchhoff:
  $\mathbf E=\tfrac12(\mathbf F^{\mathsf T}\mathbf F-\mathbf I)$,
  $\mathbf S=\lambda\operatorname{tr}(\mathbf E)\mathbf I+2\mu\mathbf E$,
  and $\mathbf P=\mathbf F\mathbf S$;
- stable Neo-Hookean: the stable energy and first Piola stress from Smith,
  de Goes, and Kim.

The reference tetrahedra must have positive orientation. Stable Neo-Hookean
forces can still be evaluated for collapsed or inverted *current* elements.

### Mass and body force

Lumped mass assigns one quarter of every element mass to each vertex:

$$
m_a\mathrel{+}=\frac{\rho V}{4}.
$$

Thus a uniform body acceleration produces

$$
\mathbf f^{\mathrm b}_a=m_a\,\mathbf g.
$$

The total body force is therefore total mass times $\mathbf g$, independently
of mesh resolution.

### Pressure faces

For an oriented triangular face, its current area vector is

$$
\mathbf a=\frac12\bigl((\mathbf x_1-\mathbf x_0)\times(\mathbf x_2-\mathbf x_0)\bigr).
$$

Constant pressure is integrated with the three linear face shape functions,
so each face vertex receives

$$
\mathbf f^{\mathrm p}_a=\frac{p\,\mathbf a}{3}.
$$

Face ordering determines the normal direction and therefore the sign.

## 3. Semi-implicit Euler

Let $\mathbf f(\mathbf x)$ contain elastic, pressure, persistent external, and
body forces.
With diagonal lumped mass `M`, the semi-implicit update is

$$
\begin{aligned}
\mathbf v^{n+1}&=\mathbf v^n+\Delta t\,M^{-1}\mathbf f(\mathbf x^n)+\Delta t\,\mathbf g,\\
\mathbf x^{n+1}&=\mathbf x^n+\Delta t\,\mathbf v^{n+1}.
\end{aligned}
$$

The new velocity is used for the position update, so only one force evaluation
is needed and no nonlinear solve is required. Fixed vertices are enforced as

$$
\mathbf x^{n+1}_a=\mathbf X_a,
\qquad
\mathbf v^{n+1}_a=\mathbf 0.
$$

The NumPy and JAX paths apply the same rule.

## 4. Fully implicit backward Euler

For backward Euler, forces are evaluated at the unknown new position
$\mathbf x$:

$$
\mathbf v^{n+1}=\frac{\mathbf x-\mathbf x^n}{\Delta t},
\qquad
M\frac{\mathbf v^{n+1}-\mathbf v^n}{\Delta t}=\mathbf f(\mathbf x).
$$

Eliminating velocity gives the position residual used by `step_implicit`:

$$
\mathbf g(\mathbf x)=\frac{M}{\Delta t^2}\bigl(\mathbf x-\mathbf x^n-\Delta t\,\mathbf v^n\bigr)-\mathbf f(\mathbf x)=\mathbf 0.
$$

Only free vertex degrees of freedom occur in this residual. Fixed coordinates
are reset to their reference positions and their velocities are zero. For a
free direction $\mathbf s$, the required Jacobian action is

$$
J_{\mathbf g}(\mathbf x)\,\mathbf s
=\frac{M}{\Delta t^2}\mathbf s-J_{\mathbf f}(\mathbf x)\,\mathbf s.
$$

The global Jacobian is never assembled.

## 5. L-BFGS with a directional residual

Newton's method would solve
$J_{\mathbf g}(\mathbf x)\Delta\mathbf x=-\mathbf g(\mathbf x)$, but constructing and
factorizing this matrix is expensive for large meshes. L-BFGS instead stores a
short history of position and residual changes. After an accepted update:

$$
\mathbf s^k=\mathbf x^{k+1}-\mathbf x^k,
\qquad
\mathbf y^k=\mathbf g(\mathbf x^{k+1})-\mathbf g(\mathbf x^k).
$$

The implementation obtains $\mathbf y^k$ with the directional residual

$$
\mathbf y^k=J_{\mathbf g}(\mathbf x^{k+1})\,\mathbf s^k.
$$

This is the local linearization of the residual in the direction just taken.
It avoids forming a matrix or running another nonlinear solve.

The L-BFGS two-loop recursion uses the history pairs and

$$
\rho^k=\frac{1}{(\mathbf y^k)^{\mathsf T}\mathbf s^k}.
$$

It starts with $-\mathbf g$, applies the history backwards and forwards, and
scales the result with the diagonal inverse-mass preconditioner
$\Delta t^2M^{-1}$. The number
of stored pairs is limited by `history_size`; invalid or insufficient-curvature
pairs are discarded.

The proposed update is tested with a backtracking line search using

$$
\phi(\mathbf x)=\frac12\,\mathbf g(\mathbf x)^{\mathsf T}\mathbf g(\mathbf x).
$$

The step length is reduced until the Armijo condition is satisfied. If the
quasi-Newton direction is not a descent direction, the diagonal
inverse-mass direction is used. Convergence is declared when

$$
\lVert\mathbf g(\mathbf x)\rVert_2
\le \texttt{absolute\_tolerance}
+\texttt{relative\_tolerance}\,\lVert\mathbf g(\mathbf x^0)\rVert_2.
$$

### Globalization and fallback

The line search globalizes the local quasi-Newton model by accepting a trial
$\mathbf x+\alpha\mathbf d$ only when its residual merit satisfies

$$
\phi(\mathbf x+\alpha\mathbf d)
\le \phi(\mathbf x)+c_1\alpha\,\mathbf g^{\mathsf T}\mathbf d.
$$

Rejected steps reduce $\alpha$ by `line_search_reduction`. If the trial fails the
element-Jacobian feasibility policy, it is rejected before its residual is
evaluated. This is a cheap endpoint guard, not a continuous collision or
root-crossing test. Non-finite trial residuals are rejected independently of
whether Armijo backtracking is enabled.

When all L-BFGS trials are rejected, the NumPy solver retries with the
preconditioned negative-gradient direction $\mathbf d_g=-D\mathbf g$. It can then rescue the
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

### Complete solve algorithm

The following pseudocode summarizes one fully implicit position solve. It is
intended to show the control flow of the implementation; `free(...)` means
that fixed vertex coordinates are excluded from the solve, and `feasible(...)`
is the optional material-aware Jacobian guard.

```text
solve_implicit(x_previous, v_previous, Δt, settings):
    x ← predictor based on x_previous and v_previous
    g ← residual(x)
    g_initial_norm ← ||g||
    tolerance ← absolute_tolerance
                  + relative_tolerance · g_initial_norm
    history ← empty L-BFGS history
    watchdog_state ← empty

    for iteration = 1 ... max_iterations:
        if ||g|| ≤ tolerance:
            return converged(x, g)

        # 1. Construct a matrix-free quasi-Newton direction.
        d ← L_BFGS_inverse_action(-g, history)
        if d is not finite or dot(g, d) ≥ 0:
            d ← -Dg                         # diagonal preconditioned gradient
            count direction_fallback

        # 2. Try the normal direction with residual backtracking.
        trial ← line_search(
            x, g, d,
            merit = 1/2 ||g||²,
            feasibility = feasible,
            Armijo = enabled,
        )

        # 3. Optional watchdog globalization.
        if trial was rejected and globalization == watchdog:
            if no watchdog episode is active:
                save x, g, and merit as the restoration point
            trial ← line_search(
                x, g, d,
                merit_limit = watchdog_growth_factor · saved_merit,
                feasibility = feasible,
                Armijo = enabled,
            )

        # 4. Gradient fallback if all quasi-Newton trials failed.
        if trial was rejected and enable_gradient_fallback:
            count gradient_fallback
            d_gradient ← -Dg
            trial ← line_search(
                x, g, d_gradient,
                feasibility = feasible,
                Armijo = enabled,
                remember_best_finite_trial = true,
            )

            # Rescue is used only if strict Armijo failed but the best
            # finite trial still reduced the residual merit.
            if trial was rejected and best_trial exists and
               best_trial.merit < 0.5 · ||g||²:
                accept best_trial
                count rescue

        if trial was rejected:
            return failed(x, g)

        # 5. Accept the trial and update the matrix-free approximation.
        x_new ← trial.position
        g_new ← trial.residual
        s ← x_new - x
        y ← directional_residual(x_new, s)
        if dot(s, y) passes curvature test:
            append (s, y, 1/dot(s, y)) to history
            discard oldest pair if history is too long
        x, g ← x_new, g_new

        # 6. Watchdog bookkeeping and possible restoration.
        if watchdog is active:
            if merit(g) > merit(g_before_trial):
                count watchdog_acceptance
                increment watchdog episode length
            if merit(g) < saved_merit:
                close watchdog episode
            if episode length > max_watchdog_steps:
                restore saved x and g
                clear history
                close watchdog episode

    if ||g|| ≤ tolerance:
        return converged(x, g)
    return failed(x, g)
```

The line-search subroutine starts with $\alpha=1$. Each rejected trial replaces
$\alpha$ by `line_search_reduction · α`. A trial is accepted by ordinary
backtracking when it is finite, feasible, and satisfies

$$
\phi(\mathbf x+\alpha\mathbf d)
\le \phi(\mathbf x)+c_1\alpha\,\mathbf g^{\mathsf T}\mathbf d.
$$

When ordinary line search is disabled, the first finite feasible trial is
accepted. The rescue path is different: it deliberately accepts the best
finite residual-reducing trial after the line-search budget is exhausted.
It is therefore a progress mechanism, not a convergence guarantee.

The NumPy implementation follows this control flow directly. The JAX
implementation uses the same logical stages inside a fixed-shape compiled
loop. JAX supports descent-direction fallback and best-finite-trial rescue,
but not the Python watchdog episode and restoration policy.

For convergence studies, each implicit invocation reports its residual
reduction factor

$$
r=\frac{\lVert\mathbf g_{\mathrm{final}}\rVert_2}
{\max\bigl(\lVert\mathbf g_{\mathrm{initial}}\rVert_2,1\bigr)}.
$$

The study script plots $-\log_{10}(r)$: larger values indicate stronger residual
reduction during that invocation. It summarizes the distribution of these
values over the simulated trajectory with the mean and 25th--75th percentile
band.

## 6. Three directional-force strategies

All strategies approximate or evaluate the same force derivative:

$$
J_{\mathbf f}(\mathbf x)\,\mathbf s
=\left.\frac{\mathrm d}{\mathrm d\varepsilon}
\mathbf f(\mathbf x+\varepsilon\mathbf s)\right|_{\varepsilon=0}.
$$

The inertial term $M\mathbf s/\Delta t^2$ is then added to obtain
$J_{\mathbf g}(\mathbf x)\mathbf s$.

### Tangent action

With JAX, `tangent_action` uses forward-mode `jax.jvp` on the complete force
operator:

$$
\mathbf f(\mathbf x)=\mathbf f^{\mathrm e}(\mathbf x)
+\mathbf f^{\mathrm p}(\mathbf x)+\mathbf f^{\mathrm{ext}}+\mathbf f^{\mathrm b}.
$$

The JVP propagates `s` through `F`, the constitutive stress, element assembly,
and pressure forces. It is inside the JIT-compiled L-BFGS kernel, so there is
no Python callback or host-side directional loop. NumPy has no built-in
forward-mode autodiff, so its tangent-action path uses the same analytical
local tangent as `closed_form`.

### Closed-form directional force

The `closed_form` path differentiates the linear-tet formulas directly. Since
$\mathbf F=\mathbf D\mathbf D_m^{-1}$:

$$
\mathrm d\mathbf F=\mathrm d\mathbf D\,\mathbf D_m^{-1}.
$$

For SVK, the implementation uses

$$
\begin{aligned}
\mathrm d\mathbf C&=(\mathrm d\mathbf F)^{\mathsf T}\mathbf F
    +\mathbf F^{\mathsf T}\mathrm d\mathbf F,\\
\mathrm d\mathbf E&=\frac12\,\mathrm d\mathbf C,\\
\mathrm d\mathbf S&=\lambda\,\operatorname{tr}(\mathrm d\mathbf E)\mathbf I
    +2\mu\,\mathrm d\mathbf E,\\
\mathrm d\mathbf P&=\mathrm d\mathbf F\,\mathbf S+\mathbf F\,\mathrm d\mathbf S.
\end{aligned}
$$

The element derivative is
$\mathrm d\mathbf f^{\mathrm e}_a=-V\,\mathrm d\mathbf P\,\nabla N_a$.
Stable Neo-Hookean differentiates
the invariants and cofactor used by its implemented first Piola stress and
then uses the same element assembly.

Pressure is a follower load. If
$\mathbf e_1=\mathbf x_1-\mathbf x_0$ and
$\mathbf e_2=\mathbf x_2-\mathbf x_0$, then

$$
\mathrm d\mathbf a=\frac12\bigl((\mathrm d\mathbf s_1\times\mathbf e_2)
 +(\mathbf e_1\times\mathrm d\mathbf s_2)\bigr),
\qquad
\mathrm d\mathbf f^{\mathrm p}_a=\frac{p\,\mathrm d\mathbf a}{3}.
$$

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
