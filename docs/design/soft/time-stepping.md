# Time-stepping methods

The soft-body prototype advances the semi-discrete equations

$$
M\ddot{x} = F(x),
$$

where $F$ includes elasticity, pressure, external forces, and gravity. The
methods below reuse the same force kernels, directional force actions,
inversion checks, and matrix-free L-BFGS solver. They differ in the residual
posed for the unknown new position.

## Overview

| Method | Order | Implicit solve | Numerical damping | State/history | Intended use |
|---|---:|---|---|---|---|
| Semi-implicit Euler | 1 | No | Low for low modes, conditionally stable | One state | Fine-step reference and inexpensive dynamics |
| Backward Euler (`implicit_bfgs`) | 1 | Yes | Strong, increasing with timestep | One state | Robust stiff or interactive simulation |
| Implicit midpoint (`implicit_midpoint`) | 2 | Yes | Very low for conservative oscillations | One state | Accurate undamped dynamics |
| Trapezoidal (`trapezoidal`) | 2 | Yes | Very low to moderate | One state | General second-order dynamics |
| Newmark average acceleration (`newmark`) | 2 | Yes | Very low to moderate | One state plus acceleration | Structural dynamics and configurable variants |
| BDF2 | 2 | Yes | Moderate; not energy-conserving | Two previous states | Future multistep extension |
| Generalized-α | 2 | Yes | Configurable high-frequency damping | Previous acceleration/state | Future controlled-damping extension |

Trapezoidal integration and Newmark with

$$
\beta=\frac14,\qquad \gamma=\frac12
$$

are equivalent for this position-only force model. They remain separate API
choices because Newmark exposes the structural-dynamics formulation and can
later support other parameter choices.

## Implemented implicit residuals

Let

$$
d(x)=x-x_n-\Delta t\,v_n.
$$

### Backward Euler

$$
R(x)=\frac{M}{\Delta t^2}d(x)-F(x),
\qquad
v_{n+1}=\frac{x_{n+1}-x_n}{\Delta t}.
$$

Backward Euler is first-order, A-stable, and L-stable. Its strong
high-frequency damping is useful for robustness but undesirable when tracing
undamped oscillations accurately.

### Implicit midpoint

With

$$
x_{n+1/2}=\frac{x_n+x_{n+1}}{2},
$$

midpoint solves

$$
R(x)=\frac{2M}{\Delta t^2}d(x)-F\left(\frac{x_n+x}{2}\right),
$$

and recovers velocity as

$$
v_{n+1}=\frac{2(x_{n+1}-x_n)}{\Delta t}-v_n.
$$

For smooth conservative systems it has substantially less numerical damping
than backward Euler.

### Trapezoidal rule

$$
R(x)=\frac{4M}{\Delta t^2}d(x)-F(x_n)-F(x),
$$

with the same velocity recovery as midpoint. For a smooth position-dependent
force this is the same update produced by Newmark average acceleration.

### Newmark

Let $a_n=M^{-1}F(x_n)$. For $\beta>0$ and $\gamma$, define

$$
a_{n+1}(x)=
\frac{x-x_n-\Delta t\,v_n-\Delta t^2(\frac12-\beta)a_n}
{\beta\Delta t^2}.
$$

Newmark solves

$$
R(x)=M a_{n+1}(x)-F(x),
$$

then updates

$$
v_{n+1}=v_n+\Delta t\left[(1-\gamma)a_n+\gamma a_{n+1}\right].
$$

The default is average acceleration,

$$
\beta=\frac14,\qquad \gamma=\frac12.
$$

Other choices
can introduce numerical damping and require separate verification.

## Nonlinear solution and acceptance

Each implicit residual follows the same algorithm:

1. predict a trial position $x_n+\Delta t\,v_n$;
2. evaluate the residual and directional action;
3. compute an L-BFGS or diagonal-gradient direction;
4. apply feasibility checks and globalization;
5. accept a trial position and update the limited-memory history;
6. recover velocity using the method-specific formula.

Nonlinear residual tolerance is not temporal accuracy. A converged nonlinear
solve can still have phase error or numerical damping.

## Scientific and interactive tuning

`autotune-soft-scientific.py` evaluates finite states, bounded energy, and
trajectory error relative to a fine semi-implicit reference. It asks which
method and timestep reproduce the selected validation trajectory.

`autotune-soft-interactive.py` uses graphics-oriented tests: finite positions
and velocities, bounded energy, bounded displacement, bounded speed, and
bounded per-frame motion. It treats trajectory error as a diagnostic rather
than the primary acceptance gate.

Results are written below `output/autotune/scientific/` and
`output/autotune/interactive/`.

## Future extensions

### BDF2

BDF2 applies the second-order backward differentiation formula to the
first-order state $y=(x,v)$:

$$
\frac{3y_{n+1}-4y_n+y_{n-1}}{2\Delta t}=G(y_{n+1}).
$$

It is second-order and A-stable, but requires two previous states and a
startup method. It is not L-stable or energy-conserving, and timestep changes
require a consistent multistep history.

### Generalized-α

Generalized-α methods use separate evaluation points for inertia and internal
force and can selectively damp high-frequency modes while retaining
second-order low-frequency accuracy. They require additional parameters and
acceleration history, so their spectral damping targets should be specified
before implementation.
