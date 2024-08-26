"""
This script contains code to create a tower scene.
"""
import numpy as np

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
from rainbow.simulators.prox_rigid_bodies.types import Engine


def create_tower(
        engine: Engine,
        r: np.ndarray,
        q: np.ndarray,
        outer_radius: float,
        inner_radius: float,
        height: float,
        layers: int,
        segments: int,
        use_cubes: bool,
        density: float,
        material_name: str,
        safety_factor: float = 0.99
) -> list[str]:
    """
    Create a tower scene.

    :param engine:          The engine that will be used to create the dry stone rigid bodies in.
    :param r:               Translation of the stone tower into the world.
    :param q:               Rotation of the stone tower in the world.
    :param outer_radius:    The outer radius of the tower wall.
    :param inner_radius:    The inner radius of the tower wall.
    :param height:          The total height of the tower.
    :param layers:          The number of bricklayers in the tower.
    :param segments:        The number of segments per layer in the tower.
    :param use_cubes:       Boolean value that determines whether the tower will be created with cube bricks or not.
    :param density:         The mass density to use for all the rigid bodies.
    :param material_name:   The material name to use for all the rigid bodies that are created.
    :param safety_factor:   The safety factor is a down scale of brick shapes to ensure that all bricks are
                             initially separated.
    :return:                A list with the names of all the rigid bodies that were created.
    """
    center_radius = (outer_radius + inner_radius) / 2.0
    brick_height = height / layers
    brick_depth = outer_radius - inner_radius
    delta_phi = (2.0 * np.pi) / segments

    shape_name = API.generate_unique_name("brick_shape")
    if use_cubes:
        brick_width = (
                2.0
                * (np.sin(delta_phi / 2.0) / np.sin(np.pi / 2.0 - delta_phi / 2.0))
                * inner_radius
                * safety_factor
        )

        V, T = MESH.create_box(brick_width, brick_height, brick_depth)
        mesh = API.create_mesh(V, T)
        API.create_shape(engine, shape_name, mesh)

    else:
        brick_inner_half_width = (
                (np.sin(delta_phi / 2.0) / np.sin(np.pi / 2.0 - delta_phi / 2.0))
                * inner_radius
                * safety_factor
        )
        brick_outer_half_width = (
                (np.sin(delta_phi / 2.0) / np.sin(np.pi / 2.0 - delta_phi / 2.0))
                * outer_radius
                * safety_factor
        )
        p0 = V3.make(-brick_inner_half_width, -brick_height / 2.0, brick_depth / 2.0)
        p1 = V3.make(brick_inner_half_width, -brick_height / 2.0, brick_depth / 2.0)
        p2 = V3.make(brick_inner_half_width, brick_height / 2.0, brick_depth / 2.0)
        p3 = V3.make(-brick_inner_half_width, brick_height / 2.0, brick_depth / 2.0)
        p4 = V3.make(-brick_outer_half_width, -brick_height / 2.0, -brick_depth / 2.0)
        p5 = V3.make(brick_outer_half_width, -brick_height / 2.0, -brick_depth / 2.0)
        p6 = V3.make(brick_outer_half_width, brick_height / 2.0, -brick_depth / 2.0)
        p7 = V3.make(-brick_outer_half_width, brick_height / 2.0, -brick_depth / 2.0)

        V, T = MESH.create_cuboid(p0, p1, p2, p3, p4, p5, p6, p7)
        mesh = API.create_mesh(V, T)
        API.create_shape(engine, shape_name, mesh)

    body_names = []
    for layer in range(layers):
        height = brick_height / 2.0 + layer * brick_height
        for segment in range(segments):

            body_name = API.generate_unique_name("body")
            body_names.append(body_name)
            API.create_rigid_body(engine, body_name)
            API.connect_shape(engine, body_name, shape_name)

            phi = delta_phi * segment
            if (layer % 2) == 0:
                phi += delta_phi / 2.0

            r_l = V3.make(0.0, height, -center_radius)
            q_l = Q.Ry(phi)
            r_l = Q.rotate(q_l, r_l)

            r_g = Q.rotate(q, r_l) + r
            q_g = Q.prod(q, q_l)

            API.set_position(engine, body_name, r_g, True)
            API.set_orientation(engine, body_name, q_g, True)

    for body_name in body_names:
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    return body_names
