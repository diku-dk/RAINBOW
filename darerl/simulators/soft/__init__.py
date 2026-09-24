"""Fast, contact-free tetrahedral soft-body prototype."""

from .solver import SoftBody
from .material import Material, StableNeoHookeanMaterial, SVKMaterial
from .mesh import TetMesh, compute_boundary_faces, create_beam_mesh
from .stepper import ImplicitBFGSStepper, SemiImplicitStepper, Stepper
from .types import Array
from .forces import compute_directional_forces, compute_elastic_forces, compute_energy_density, compute_pressure_forces
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
    "Array", "Material", "SoftBody", "StableNeoHookeanMaterial", "SVKMaterial", "TetMesh",
    "Stepper", "SemiImplicitStepper", "ImplicitBFGSStepper",
    "compute_boundary_faces", "create_beam_mesh", "compute_elastic_forces", "compute_pressure_forces",
    "compute_directional_forces", "compute_energy_density", "SoftBaseline", "create_bending_baseline",
    "create_stretch_baseline", "create_compress_baseline", "create_twist_baseline",
    "BASELINE_LENGTH", "BASELINE_HEIGHT", "BASELINE_DEPTH", "SKIN_YOUNGS_MODULUS",
    "SKIN_POISSON_RATIO", "SKIN_DENSITY", "BENDING_GRAVITY", "STRETCH_PRESSURE",
    "COMPRESSION_PRESSURE", "TWIST_TORQUE",
]
