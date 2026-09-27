Th # Stability terminology

This page is an optional short reference for the stability terms used in
[Time-stepping methods](time-stepping.md). It is meant as a quick refresher;
the main page does not depend on remembering these definitions.

## The test equation and stability function

Classical time-integration stability is introduced with the scalar linear test
equation

$$
\frac{dy}{dt}=\lambda y,
\qquad \lambda\in\mathbb{C},
\qquad \operatorname{Re}(\lambda)\le 0.
$$

Here a *mode* means one simple pattern of motion or response in a larger
system. For example, a flexible body can move in several characteristic
patterns, such as a low-frequency bending shape or a higher-frequency local
deformation. In a linearized analysis, a complicated motion can be understood
as a combination of such patterns, and each pattern can be studied separately
with a scalar equation like the one above. You do not need to compute these
patterns explicitly to use the stability definitions.

In the equation, $y$ represents the size of one such component or mode,
$\lambda$ describes how that mode evolves, and $h$ is the time step. If
$\operatorname{Re}(\lambda)<0$, the exact mode decays; if
$\operatorname{Re}(\lambda)=0$, it neither grows nor decays. The exact
solution therefore does not grow when the real part of $\lambda$ is
non-positive. Applying a one-step method gives

$$
y_{n+1}=R(z)y_n,
\qquad z=h\lambda,
$$

where $R(z)$ is the method's *stability function*. It tells us how one
numerical step changes the mode: $\lvert R(z)\rvert<1$ damps it,
$\lvert R(z)\rvert=1$ preserves its magnitude, and $\lvert R(z)\rvert>1$
amplifies it.

The set of all values of $z$ for which a numerical mode does not grow is
called the *absolute stability region*:

$$
\mathcal{S}=\{z\in\mathbb{C}:\lvert R(z)\rvert\le 1\}.
$$

Thus, “absolute” here means stability for a fixed problem mode and a fixed
step size; it does not mean that the method is absolutely accurate or that it
handles every nonlinear problem without difficulty.

For multistep methods, such as BDF2, there is another requirement called
*zero-stability*. A multistep method uses several previous values, so even when
the differential equation has no dynamics ($\lambda=0$), small numerical
perturbations must remain bounded as more steps are taken. Zero-stability is
the condition that prevents the method's internal history from amplifying
round-off, starting, or data errors. Absolute stability controls the response
to the physical mode; zero-stability controls the method's memory. Both are
needed for a convergent multistep method.

The stability-function picture above is mainly a simple one-step-method
intuition. For multistep methods, the same ideas are expressed through the
roots of a characteristic polynomial, but A- and L-stability retain the same
practical interpretation given below.

## Accuracy terms that appear with stability

Stability and accuracy answer different questions. Stability asks whether an
error or mode grows uncontrollably. Accuracy asks how closely the numerical
mode follows the exact one.

For an oscillatory mode, the numerical multiplier can have the form

$$
R(z)=\lvert R(z)\rvert e^{\mathrm{i}\theta_{\mathrm{num}}}.
$$

The magnitude $\lvert R(z)\rvert$ describes numerical damping or growth. The
angle $\theta_{\mathrm{num}}$ describes the numerical phase. *Phase error* is
the difference between this numerical angle and the exact phase accumulated
over one step. Small phase error is important when the timing of oscillations
matters, even if the solution remains perfectly bounded. Amplitude error is a
similar mismatch in the size of the oscillation.

This is why an A-stable method is not automatically more accurate than a
conditionally stable method: A-stability prevents a particular class of
growth, but says nothing by itself about phase or amplitude error.

## Conditional stability

A method is *conditionally stable* when stability is guaranteed only under a
restriction on the time step. For a mode with characteristic rate $\lambda$,
the restriction is expressed through $z=h\lambda$: the chosen $h$ must keep
$z$ inside the method's stability region.

For example, an explicit method may be stable for sufficiently small steps
but become unstable when $h$ is too large. The precise limit depends on the
system's fastest mode, so refining the spatial discretization can force a
smaller stable time step. “Conditionally stable” therefore does not mean
“inaccurate”; it means that the method's stability depends on choosing a
suitable step size.

## A-stability

A method is **A-stable** if its stability region contains the entire left half
of the complex plane:

$$
\{z\in\mathbb{C}:\operatorname{Re}(z)\le 0\}\subseteq\mathcal{S}.
$$

For the linear test equation, this means that no step-size restriction is
needed merely to prevent a decaying mode from growing. A-stability is a
stability statement, not an accuracy guarantee: a method can be A-stable and
still have significant phase, amplitude, or nonlinear-solve error.

## L-stability

A method is **L-stable** when it is A-stable and also damps very stiff modes
completely. In terms of the stability function,

$$
\lim_{\lvert z\rvert\to\infty}R(z)=0
$$

in the stiff, decaying part of the complex plane. Backward Euler is L-stable.
By contrast, the trapezoidal rule and implicit midpoint are A-stable but not
L-stable: on the negative real axis their stability function approaches $-1$,
so very stiff components are not removed in one large step.

This distinction is useful when interpreting the soft-body examples. L-stable
damping can improve robustness by suppressing unresolved or high-frequency
motion, but it can also suppress physically meaningful oscillations. For an
undamped scientific trajectory, stability and accurate phase/energy behaviour
are therefore separate acceptance questions.

## Where the methods fit

The following labels use the classical linearized/test-equation meaning:

| Method | Classical stability description |
|---|---|
| Semi-implicit Euler | Conditionally stable; not A-stable in general |
| Backward Euler | A-stable and L-stable |
| Implicit midpoint | A-stable, not L-stable |
| Trapezoidal rule | A-stable, not L-stable |
| Newmark average acceleration | Trapezoidal-equivalent for the formulation used here |
| BDF2 | A-stable for the constant-step linear multistep setting, not L-stable |

These classifications do not guarantee convergence of the nonlinear solve.
Every implicit step still has to be solved successfully, and the practical
behaviour also depends on the spatial discretization, nonlinearities,
step-size, tolerances, and globalization strategy.

## References

- [E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed., Springer](https://link.springer.com/book/10.1007/978-3-642-05221-7)
- [G. Dahlquist, “A special stability problem for linear multistep methods,” *BIT* 3 (1963)](https://doi.org/10.1007/BF01963532)
