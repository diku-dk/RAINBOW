"""
This script contains code to create a dry stone-masonry pillar.
"""

import numpy as np

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
from rainbow.simulators.prox_rigid_bodies.types import Engine


def create_pillar(
        engine: Engine,
        r: np.ndarray,
        q: np.ndarray,
        width: float,
        height: float,
        depth: float,
        stones: int,
        density: float,
        material_name: str) -> list[str]:
    """
    This function creates a dry stone-masonry pillar.

    The origin of the pillar is at the mid-point of the base.

    :param engine:          The engine that will be used to create the dry stone rigid bodies in.
    :param r:               Translation of the stone pillar into the world.
    :param q:               Rotation of the stone pillar in the world.
    :param width:           The total width of the pillar.
    :param height:          The total height of the pillar.
    :param depth:           The depth of the stones used to crate the pillar.
    :param stones:          The number of stones in the pillar column.
    :param density:         The mass density to use for all the rigid bodies.
    :param material_name:   The material name to use for all the rigid bodies that are created.
    :return:                A list with the names of all the rigid bodies that were created.
    """
    pillar_width = width
    pillar_depth = depth
    pillar_height = height
    pillar_stones = stones

    body_names = []

    base_height = pillar_height / 15
    top_height = pillar_height / 20
    pillar_base_radius = 0.9 * (pillar_width / 2.0)
    pillar_top_radius = 0.8 * (pillar_width / 2.0)
    pillar_radius_start = 0.7 * pillar_base_radius
    pillar_radius_stop = 0.5 * pillar_base_radius
    cap_height = base_height * 0.9
    cup_height = base_height * 0.7

    shape_name = API.generate_unique_name("socle_shape")
    V, T = MESH.create_box(pillar_width, base_height, pillar_depth)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("socle_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r_l = V3.make(0.0, base_height / 2.0, 0.0)
    q_l = Q.identity()
    r_g = Q.rotate(q, r_l) + r
    q_g = Q.prod(q, q_l)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    shape_name = API.generate_unique_name("abacus_shape")
    V, T = MESH.create_box(pillar_width, top_height, pillar_depth)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("abacus_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r_l = V3.make(0.0, pillar_height - top_height / 2.0, 0.0)
    q_l = Q.identity()
    r_g = Q.rotate(q, r_l) + r
    q_g = Q.prod(q, q_l)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    shape_name = API.generate_unique_name("base_shape")
    V, T = MESH.create_conical(pillar_base_radius, pillar_radius_start, cap_height)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("base_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r_l = V3.make(0.0, base_height, 0.0)
    q_l = Q.identity()
    r_g = Q.rotate(q, r_l) + r
    q_g = Q.prod(q, q_l)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    shape_name = API.generate_unique_name("echinus_shape")
    V, T = MESH.create_conical(pillar_radius_stop, pillar_top_radius, cup_height)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("echinus_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r_l = V3.make(0.0, pillar_height - top_height - cup_height, 0.0)
    q_l = Q.identity()
    r_g = Q.rotate(q, r_l) + r
    q_g = Q.prod(q, q_l)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    start_y = base_height + cap_height
    end_y = pillar_height - top_height - cup_height
    dy = (end_y - start_y) / pillar_stones
    slope = (pillar_radius_stop - pillar_radius_start) / (end_y - start_y)
    for stone in range(pillar_stones):
        bottom_y = start_y + stone * dy
        top_y = bottom_y + dy
        bottom_radius = pillar_radius_start + slope * (bottom_y - start_y)
        top_radius = pillar_radius_start + slope * (top_y - start_y)

        shape_name = API.generate_unique_name("shaft_stone")
        V, T = MESH.create_conical(bottom_radius, top_radius, dy)
        mesh = API.create_mesh(V, T)
        API.create_shape(engine, shape_name, mesh)

        body_name = API.generate_unique_name("shaft_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        r_l = V3.make(0.0, bottom_y, 0.0)
        q_l = Q.identity()
        r_g = Q.rotate(q, r_l) + r
        q_g = Q.prod(q, q_l)

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

    for body_name in body_names:
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    return body_names
