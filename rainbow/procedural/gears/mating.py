import numpy as np

import rainbow.math.quaternion as Q

from .types import GearSpec, PlanetaryGearSpec, MatingSpec


def compute_gear_position(spec1: GearSpec, spec2: GearSpec, angle: float) -> np.ndarray:
    return (spec1.rp + spec2.rp) * np.array([np.cos(angle), np.sin(angle), 0])


def compute_gear_orientation(spec1: GearSpec, spec2: GearSpec, angle: float) -> np.ndarray:
    return Q.Rz(compute_rotation_angle(spec1, spec2, angle))


def compute_internal_gear_orientation(spec: GearSpec) -> np.ndarray:
    return Q.Rz(compute_ring_gear_rotation_angle(spec))


def compute_rotation_angle(spec1: GearSpec, spec2: GearSpec, angle: float) -> float:
    rotation_angle = (spec1.rp / spec2.rp + 1) * angle
    if not (spec1.z % 2 == 1 and spec2.z % 2 == 1):
        rotation_angle += spec2.gamma_p
    return rotation_angle % (2 * np.pi)


def compute_ring_gear_rotation_angle(spec: GearSpec) -> float:
    if not spec.is_internal:
        raise ValueError("The gear is not an internal gear.")
    
    if spec.z % 2 == 1:
        return spec.gamma_p
    else:
        return 0
