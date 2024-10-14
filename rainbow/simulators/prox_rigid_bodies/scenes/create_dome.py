"""
This script contains code to create a dry stone-masonry dome.
"""

import numpy as np

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
from rainbow.simulators.prox_rigid_bodies.types import Engine


def create_dome(
        engine: Engine,
        r: np.ndarray,
        q: np.ndarray,
        outer_radius: float,
        inner_radius: float,
        layers: int,
        segments: int,
        density: float,
        material_name: str
        ) -> list[str]:
    """
    Create a stone dome scene.

    :param engine:          The engine that will be used to create the dry stone rigid bodies in.
    :param r:               Translation of the stone dome into the world.
    :param q:               Rotation of the stone dome in the world.
    :param outer_radius:    The outer radius of the dome wall.
    :param inner_radius:    The inner radius of the dome wall.
    :param layers:          The number of bricklayers in the dome.
    :param segments:        The number of segments per layer in the dome.
    :param density:         The mass density to use for all the rigid bodies.
    :param material_name:   The material name to use for all the rigid bodies that are created.
    :return:                A list with the names of all the rigid bodies that were created.
    """
    r_i = inner_radius
    r_o = outer_radius
    d_theta = (2.0 * np.pi) / segments
    d_phi = (np.pi / 2.0) / (layers + 1)

    body_names = []
    for layer in range(layers):
        theta = 0.0
        phi = d_phi * layer
        shape_name = API.generate_unique_name("brick_shape")

        q0 = Q.prod(Q.Ry(theta), Q.Rz(phi))
        q1 = Q.prod(Q.Ry(theta + d_theta), Q.Rz(phi))
        q2 = Q.prod(Q.Ry(theta + d_theta), Q.Rz(phi + d_phi))
        q3 = Q.prod(Q.Ry(theta), Q.Rz(phi + d_phi))
        p0 = Q.rotate(q0, V3.make(r_o, 0.0, 0.0))
        p1 = Q.rotate(q1, V3.make(r_o, 0.0, 0.0))
        p2 = Q.rotate(q2, V3.make(r_o, 0.0, 0.0))
        p3 = Q.rotate(q3, V3.make(r_o, 0.0, 0.0))
        p4 = Q.rotate(q0, V3.make(r_i, 0.0, 0.0))
        p5 = Q.rotate(q1, V3.make(r_i, 0.0, 0.0))
        p6 = Q.rotate(q2, V3.make(r_i, 0.0, 0.0))
        p7 = Q.rotate(q3, V3.make(r_i, 0.0, 0.0))

        V, T = MESH.create_cuboid(p0, p1, p2, p3, p4, p5, p6, p7)
        mesh = API.create_mesh(V, T)
        API.create_shape(engine, shape_name, mesh)

        for s in range(segments):
            theta = d_theta * s

            body_name = API.generate_unique_name("body")
            body_names.append(body_name)
            API.create_rigid_body(engine, body_name)
            API.connect_shape(engine, body_name, shape_name)

            q_l = Q.Ry(theta)
            r_l = Q.rotate(q_l, V3.make(0.0, 0.0, 0.0))
            r_g = Q.rotate(q, r_l) + r
            q_g = Q.prod(q, q_l)

            API.set_position(engine, body_name, r_g, True)
            API.set_orientation(engine, body_name, q_g, True)

    for body_name in body_names:
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    return body_names
