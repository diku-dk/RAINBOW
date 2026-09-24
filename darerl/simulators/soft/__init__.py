"""Fast, contact-free tetrahedral soft-body prototype."""

from .solver import Material, SoftBody, StableNeoHookeanMaterial, TetMesh, SVKMaterial
from .mesh import compute_boundary_faces, create_beam_mesh
from .baseline import (
    BASELINE_DEPTH,
    BASELINE_HEIGHT,
    BASELINE_LENGTH,
    BENDING_GRAVITY,
    COMPRESSION_PRESSURE,
    SKIN_DENSITY,
    SKIN_POISSON_RATIO,
    SKIN_YOUNGS_MODULUS,
    STRETCH_PRESSURE,
    TWIST_TORQUE,
    SoftBaseline,
    create_bending_baseline,
    create_compress_baseline,
    create_stretch_baseline,
    create_twist_baseline,
)

__all__ = [
    "Material", "SoftBody", "StableNeoHookeanMaterial", "SVKMaterial", "TetMesh",
    "compute_boundary_faces", "create_beam_mesh", "SoftBaseline", "create_bending_baseline",
    "create_stretch_baseline", "create_compress_baseline", "create_twist_baseline",
    "BASELINE_LENGTH", "BASELINE_HEIGHT", "BASELINE_DEPTH", "SKIN_YOUNGS_MODULUS",
    "SKIN_POISSON_RATIO", "SKIN_DENSITY", "BENDING_GRAVITY", "STRETCH_PRESSURE",
    "COMPRESSION_PRESSURE", "TWIST_TORQUE",
]
