"""Canonical, visibly deforming soft-body benchmark cases.

The factory functions in this module are the single source of truth for the
soft-body examples.  They use a 10 cm characteristic beam size and a soft,
nearly incompressible skin-like material.  Loads are intentionally amplified
where necessary so that short example runs contain measurable nonlinear
deformation; they are benchmark loads, not patient-specific material data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .material import SVKMaterial, StableNeoHookeanMaterial
from .mesh import TetMesh, create_beam_mesh
from .solver import SoftBody


# Approximate properties for soft human skin.  Actual skin varies widely with
# location, age, hydration, strain rate, and test protocol.
SKIN_YOUNGS_MODULUS = 1.0e5  # Pa
SKIN_POISSON_RATIO = 0.49
SKIN_DENSITY = 1100.0  # kg/m^3

# Canonical geometry: 10 cm long, with a 2 cm square section.
BASELINE_LENGTH = 0.10
BASELINE_HEIGHT = 0.02
BASELINE_DEPTH = 0.02

# Standard gravity produces a clearly nonlinear cantilever response at the
# canonical 10 cm scale while remaining compatible with the default fine
# semi-implicit reference timestep used by the autotuner.
BENDING_GRAVITY = (0.0, -9.81, 0.0)
# Pressure loads are chosen to remain stable for the default 1e-4 s
# semi-implicit reference step while still producing measurable axial motion.
STRETCH_PRESSURE = 1.0e3
COMPRESSION_PRESSURE = -1.0e3
TWIST_TORQUE = 5.0e-3


@dataclass(frozen=True)
class SoftBaseline:
    """A canonical mesh, boundary conditions, and loading for one mode."""

    mode: str
    mesh: TetMesh
    fixed: np.ndarray
    pressure_faces: np.ndarray
    pressure: float
    external_forces: np.ndarray
    gravity: tuple[float, float, float]

    def create_body(
        self,
        use_jax: bool = False,
        material_model: str = "stable_neohookean",
    ) -> SoftBody:
        """Create a fresh solver body with this baseline's configuration."""
        material_type = {
            "svk": SVKMaterial,
            "stable_neohookean": StableNeoHookeanMaterial,
            "stable_neo_hookean": StableNeoHookeanMaterial,
            "snh": StableNeoHookeanMaterial,
        }.get(material_model.lower())
        if material_type is None:
            raise ValueError("material_model must be 'svk' or 'stable_neohookean'")
        body = SoftBody(
            self.mesh,
            material_type(SKIN_YOUNGS_MODULUS, SKIN_POISSON_RATIO, SKIN_DENSITY),
            fixed=self.fixed,
            use_jax=use_jax,
        )
        if len(self.pressure_faces):
            body.set_pressure_boundary(self.pressure_faces, self.pressure)
        body.set_external_forces(self.external_forces)
        return body


def _free_end_faces(i: int, j: int, k: int) -> np.ndarray:
    """Return outward-oriented triangles on the positive-x end face."""
    node = lambda ii, jj, kk: (kk * j + jj) * i + ii
    faces = []
    ii = i - 1
    for kk in range(k - 1):
        for jj in range(j - 1):
            a = node(ii, jj, kk)
            b = node(ii, jj + 1, kk)
            c = node(ii, jj, kk + 1)
            d = node(ii, jj + 1, kk + 1)
            faces.extend(((a, b, c), (b, d, c)))
    return np.asarray(faces, dtype=np.int32)


def _create_baseline(
    mode: str,
    i: int,
    j: int,
    k: int,
    pressure: float = 0.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SoftBaseline:
    vertices, elements = create_beam_mesh(
        i, j, k, BASELINE_LENGTH, BASELINE_HEIGHT, BASELINE_DEPTH
    )
    mesh = TetMesh.from_vertices(vertices, elements, density=SKIN_DENSITY)
    fixed = np.isclose(vertices[:, 0], vertices[:, 0].min())
    tip_faces = _free_end_faces(i, j, k)
    forces = np.zeros_like(vertices)
    return SoftBaseline(mode, mesh, fixed, tip_faces if pressure else np.empty((0, 3), dtype=np.int32), pressure, forces, gravity)


def create_bending_baseline(i: int = 6, j: int = 3, k: int = 3) -> SoftBaseline:
    """Create the cantilever baseline with visible gravity-driven bending."""
    return _create_baseline("bend", i, j, k, gravity=BENDING_GRAVITY)


def create_stretch_baseline(i: int = 6, j: int = 3, k: int = 3) -> SoftBaseline:
    """Create the cantilever baseline with positive normal tip pressure."""
    return _create_baseline("stretch", i, j, k, pressure=STRETCH_PRESSURE)


def create_compress_baseline(i: int = 6, j: int = 3, k: int = 3) -> SoftBaseline:
    """Create the cantilever baseline with negative normal tip pressure."""
    return _create_baseline("compress", i, j, k, pressure=COMPRESSION_PRESSURE)


def create_twist_baseline(i: int = 6, j: int = 3, k: int = 3) -> SoftBaseline:
    """Create the cantilever baseline with a distributed tip torque."""
    case = _create_baseline("twist", i, j, k)
    tip = case.mesh.x0[:, 0] > case.mesh.x0[:, 0].max() - 1.0e-12
    center = case.mesh.x0[tip].mean(axis=0)
    radius = case.mesh.x0[tip] - center
    denominator = np.sum(radius[:, 1] ** 2 + radius[:, 2] ** 2)
    forces = case.external_forces.copy()
    scale = TWIST_TORQUE / denominator
    forces[tip, 1] = -scale * radius[:, 2]
    forces[tip, 2] = scale * radius[:, 1]
    return SoftBaseline(case.mode, case.mesh, case.fixed, case.pressure_faces, case.pressure, forces, case.gravity)


__all__ = [
    "BASELINE_DEPTH",
    "BASELINE_HEIGHT",
    "BASELINE_LENGTH",
    "BENDING_GRAVITY",
    "COMPRESSION_PRESSURE",
    "SKIN_DENSITY",
    "SKIN_POISSON_RATIO",
    "SKIN_YOUNGS_MODULUS",
    "STRETCH_PRESSURE",
    "SoftBaseline",
    "TWIST_TORQUE",
    "create_bending_baseline",
    "create_compress_baseline",
    "create_stretch_baseline",
    "create_twist_baseline",
]
