"""
This script contains code to create a sandbox scene.
"""
import numpy as np

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
from rainbow.simulators.prox_rigid_bodies.types import Engine


def create_cube_hinge_chain(
        engine: Engine,
        cube_edge_length: float,
        K_cubes: int,
        density: float,
        material_name: str,
) -> list[str]:
    """
    This function creates a scene with a chain of cubes connected by hinge joints.

    :param engine:                  The engine that will be used to create the dry stone rigid bodies in.
    :param cube_edge_length:        The edge length of all the cubes.
    :param K_cubes:                 The number of cubes that will be stacked.
    :param density:                 The mass density to use for all the rigid bodies.
    :param material_name:           The material name to use for all the rigid bodies that are created.
    :return:                        A list with the names of all the rigid bodies that were created.
    """

    diameter = np.sqrt(2 * cube_edge_length ** 2)
    radius = diameter / 2.0
    margin = 0.1 * radius
    chain_length = K_cubes * diameter + (K_cubes - 1) * margin

    shape_name = API.generate_unique_name("cube")
    V, T = MESH.create_box(cube_edge_length, cube_edge_length, cube_edge_length)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_names = []

    for k in range(K_cubes):
        body_name = API.generate_unique_name("link")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        displacement = k * (diameter + margin) + radius - chain_length / 2.0
        r = V3.make(displacement, 0.0, 0.0)

        radians = np.pi / 4
        q = Q.Rz(radians)

        API.set_position(engine, body_name, r, True)
        API.set_orientation(engine, body_name, q, True)
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    API.set_body_type(engine, body_names[0], "fixed")

    for k in range(K_cubes - 1):
        parent_name = body_names[k]
        child_name = body_names[k + 1]

        hinge_name = parent_name + "_" + child_name
        API.create_hinge(engine, hinge_name)

        s_world = V3.k()
        r_parent = API.get_position(engine, parent_name)
        r_child = API.get_position(engine, child_name)
        o_world = (r_parent + r_child) / 2

        API.set_hinge(
            engine=engine,
            hinge_name=hinge_name,
            parent_name=parent_name,
            child_name=child_name,
            origin=o_world,
            axis=s_world,
            mode="world"
        )

    return body_names
