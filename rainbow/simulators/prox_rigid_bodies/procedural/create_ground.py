"""
This script contains code to create a ground box.
"""
import numpy as np

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
from rainbow.simulators.prox_rigid_bodies.types import Engine


def create_ground(engine: Engine,
                  r: np.ndarray,
                  q: np.ndarray,
                  density: float,
                  material_name: str) -> None:
    """
    Creates a ground box in the world.

    :param engine:          The engine that will be used to create the rigid bodies in.
    :param r:               Translation of the ground box in the world.
    :param q:               Rotation of the ground box in the world.
    :param density:         The mass density to use for all the rigid bodies.
    :param material_name:   The material name to use for all the rigid bodies that are created.
    :return:                None.
    """
    shape_name = API.generate_unique_name("ground_shape")

    V, T = MESH.create_box(100.0, 1.0, 100.0)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("ground_body")
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r_l = V3.make(0.0, -0.5, 0.0)
    q_l = Q.identity()
    r_g = Q.rotate(q, r_l) + r
    q_g = Q.prod(q, q_l)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    API.set_body_type(engine, body_name, "fixed")
    API.set_body_material(engine, body_name, material_name)
    API.set_mass_properties(engine, body_name, density)
