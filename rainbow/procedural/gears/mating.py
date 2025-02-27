import numpy as np

import rainbow.math.quaternion as Q

from .types import GearSpec, PlanetaryGearSpec, MatingSpec


def compute_gear_position(spec1: GearSpec, spec2: GearSpec, angle: float) -> np.ndarray:
    return (spec1.rp + spec2.rp) * np.array([np.cos(angle), np.sin(angle), 0])


def compute_gear_orientation(spec1: GearSpec, spec2: GearSpec, angle: float) -> np.ndarray:
    return Q.Rz(compute_rotation_angle(spec1, spec2, angle))


def compute_planetary_gear_ring_orientation(spec: PlanetaryGearSpec) -> np.ndarray:
    return Q.Rz(compute_planetary_gear_ring_angle(spec, 0))


def compute_rotation_angle(spec1: GearSpec, spec2: GearSpec, angle: float) -> float:
    rotation_angle = (spec1.rp / spec2.rp + 1) * angle
    if spec2.z % 2 == 0:
        rotation_angle += spec2.gamma_p
    return rotation_angle % (2 * np.pi)


def compute_planetary_gear_ring_angle(spec: PlanetaryGearSpec, angle: float) -> float:
    if spec.planet_spec.z % 2 == 0:
        return spec.ring_spec.gamma_p
    else:
        return 0
