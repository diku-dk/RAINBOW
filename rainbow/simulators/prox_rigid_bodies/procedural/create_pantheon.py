"""
This script contains code to create a pantheon scene.
"""

import numpy as np

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
from rainbow.simulators.prox_rigid_bodies.types import Engine
from .create_dome import create_dome
from .create_tower import create_tower


def create_pantheon(
        engine: Engine,
        r: np.ndarray,
        q: np.ndarray,
        outer_radius: float,
        inner_radius: float,
        height: float,
        layers: int,
        segments: int,
        density: float,
        material_name: str,
        safety_factor=0.99,
) -> list[str]:
    """
    This function creates a pantheon scene of rigid bodies.

    :param engine:          The engine that will be used to create the dry stone rigid bodies in.
    :param r:               Translation of the Pantheon scene into the world.
    :param q:               Rotation of the Pantheon scene in the world.
    :param outer_radius:    The outer radius of the Pantheon walls.
    :param inner_radius:    The inner radius of the Pantheon walls.
    :param height:          The total height of the Pantheon building.
    :param layers:          The total layers (bricks) of the Pantheon building.
    :param segments:        The total number of bricks to use in each bricklayer.
    :param density:         The mass density to use for all the rigid bodies.
    :param material_name:   The material name to use for all the rigid bodies that are created.
    :param safety_factor:
    :return:                A list with the names of all the rigid bodies that were created.
    """
    wall_height = height - outer_radius
    wall_layers = layers
    dome_layers = layers

    if wall_height <= 0.0:
        raise RuntimeWarning("dome height must be larger than outer radius")

    body_names = []

    names = create_tower(
        engine,
        r,
        q,
        outer_radius,
        inner_radius,
        wall_height,
        wall_layers,
        segments,
        False,
        density,
        material_name,
        safety_factor,
    )
    body_names.extend(names)

    r_l = V3.make(0.0, wall_height, 0.0)
    q_l = Q.identity()
    r_g = Q.rotate(q, r_l) + r
    q_g = Q.prod(q, q_l)
    names = create_dome(
        engine,
        r_g,
        q_g,
        outer_radius,
        inner_radius,
        dome_layers,
        segments,
        density,
        material_name,
    )
    body_names.extend(names)

    return body_names
