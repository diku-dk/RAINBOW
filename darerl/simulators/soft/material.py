"""Constitutive material definitions for the tetrahedral soft solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Material(Protocol):
    """Constitutive model interface used by the solver kernels."""

    def compute_lame_parameters(self) -> tuple[float, float]: ...

    @property
    def model_code(self) -> int: ...


@dataclass(frozen=True)
class SVKMaterial:
    """Saint Venant--Kirchhoff material parameters."""

    youngs_modulus: float
    poisson_ratio: float
    density: float

    @property
    def model_code(self) -> int:
        return 0

    def compute_lame_parameters(self) -> tuple[float, float]:
        e, nu = self.youngs_modulus, self.poisson_ratio
        if not np.isfinite(e) or e <= 0.0:
            raise ValueError("youngs_modulus must be finite and positive")
        if not (-1.0 < nu < 0.5):
            raise ValueError("poisson_ratio must be in (-1, 0.5)")
        if not np.isfinite(self.density) or self.density <= 0.0:
            raise ValueError("density must be finite and positive")
        return (nu * e / ((1.0 + nu) * (1.0 - 2.0 * nu)), e / (2.0 * (1.0 + nu)))


@dataclass(frozen=True)
class StableNeoHookeanMaterial(SVKMaterial):
    """Stable Neo-Hookean material from Smith, de Goes, and Kim (2018)."""

    @property
    def model_code(self) -> int:
        return 1
