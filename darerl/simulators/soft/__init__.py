"""Fast, contact-free tetrahedral soft-body prototype."""

from .solver import SoftBody
from .material import Material, StableNeoHookeanMaterial, SVKMaterial
from .mesh import TetMesh, compute_boundary_faces, create_beam_mesh
from .time_stepper import step_implicit, step_semi_implicit
from .types import Array, Elements, Forces, Vertices
from .forces import (
    compute_directional_forces,
    compute_elastic_forces,
    compute_energy_density,
    compute_element_jacobians,
    compute_pk1_stress,
    compute_pressure_forces,
)
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
    "Array", "Elements", "Forces", "Vertices", "Material", "SoftBody", "StableNeoHookeanMaterial", "SVKMaterial", "TetMesh",
    "step_implicit", "step_semi_implicit",
    "compute_boundary_faces", "create_beam_mesh", "compute_elastic_forces", "compute_pressure_forces",
    "compute_directional_forces", "compute_energy_density", "compute_element_jacobians", "compute_pk1_stress", "SoftBaseline", "create_bending_baseline",
    "create_stretch_baseline", "create_compress_baseline", "create_twist_baseline",
    "BASELINE_LENGTH", "BASELINE_HEIGHT", "BASELINE_DEPTH", "SKIN_YOUNGS_MODULUS",
    "SKIN_POISSON_RATIO", "SKIN_DENSITY", "BENDING_GRAVITY", "STRETCH_PRESSURE",
    "COMPRESSION_PRESSURE", "TWIST_TORQUE",
]
