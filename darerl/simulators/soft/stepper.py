"""Time-integration strategies for :class:`~.solver.SoftBody`.

The steppers intentionally receive a body rather than owning one.  This keeps
the physical state, loads, and diagnostics on ``SoftBody`` while making the
integration strategy explicit and independently replaceable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .solver import SoftBody


class Stepper(Protocol):
    """Interface implemented by one-step time integrators."""

    def step(self, body: "SoftBody", dt: float, gravity: np.ndarray, **kwargs): ...


class SemiImplicitStepper:
    """Semi-implicit Euler integration strategy."""

    def step(self, body: "SoftBody", dt: float, gravity: np.ndarray, *, sync: bool = True, **kwargs):
        return body._step_semi_implicit(dt, gravity, sync)


class ImplicitBFGSStepper:
    """Backward-Euler strategy using the matrix-free L-BFGS solve."""

    def step(self, body: "SoftBody", dt: float, gravity: np.ndarray, *, settings: dict | None = None, **kwargs):
        return body.step_implicit(dt, gravity, settings)


__all__ = ["ImplicitBFGSStepper", "SemiImplicitStepper", "Stepper"]
