"""
This script contains code to create a dry stone-masonry arch.
"""

import numpy as np

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
from rainbow.simulators.prox_rigid_bodies.types import Engine
from .create_arch import create_arch
from .create_tower import create_tower


def create_colosseum(
        engine: Engine,
        r: np.ndarray,
        q: np.ndarray,
        outer_radius: float,
        inner_radius: float,
        height: float,
        levels: int,
        arches: int,
        density: float,
        material_name: str,
        safety_factor: float = 0.99,
) -> list[str]:
    """
    Creates the Colosseum scene in the engine.

    The model origin of the colosseum scene is the middle point at the first floor.

    :param engine:          The engine that will be used to create the dry stone rigid bodies in.
    :param r:               Translation of the Colosseum into the world.
    :param q:               Rotation of the Colosseum in the world.
    :param outer_radius:    The outer radius of the Colosseum.
    :param inner_radius:    The inner radius of the Colosseum.
    :param height:          The total height of the Colosseum.
    :param levels:          The total levels (floors) of the Colosseum.
    :param arches:          The number of arches to use in each level of the Colosseum.
    :param density:         The mass density to use for all the rigid bodies.
    :param material_name:   The material name to use for all the rigid bodies that are created.
    :param safety_factor:
    :return:                A list with the names of all the rigid bodies that were created.
    """
    center_radius = (outer_radius + inner_radius) / 2.0
    level_height = height / levels
    separation_height = level_height * 0.1
    arch_height = level_height * 0.9
    arch_depth = outer_radius - inner_radius
    delta_phi = (2.0 * np.pi) / arches
    arch_width = (
            2.0
            * (np.sin(delta_phi / 2.0) / np.sin(np.pi / 2.0 - delta_phi / 2.0))
            * inner_radius
            * safety_factor
    )

    body_names = []
    for lvl in range(levels):
        height = lvl * level_height
        for a in range(arches):
            phi = delta_phi * a
            r_l = V3.make(0.0, height, -center_radius)
            q_l = Q.Ry(phi)
            r_l = Q.rotate(q_l, r_l)
            r_g = Q.rotate(q, r_l) + r
            q_g = Q.prod(q, q_l)
            names = create_arch(
                engine,
                r_g,
                q_g,
                arch_width,
                arch_height,
                arch_depth,
                5,
                9,
                density,
                material_name,
            )
            body_names.extend(names)
        r_l = V3.make(0.0, height + arch_height, 0.0)
        q_l = Q.identity()
        r_g = Q.rotate(q, r_l) + r
        q_g = Q.prod(q, q_l)
        names = create_tower(
            engine,
            r_g,
            q_g,
            outer_radius,
            inner_radius,
            separation_height,
            1,
            arches,
            False,
            density,
            material_name,
            safety_factor,
        )
        body_names.extend(names)

    return body_names
