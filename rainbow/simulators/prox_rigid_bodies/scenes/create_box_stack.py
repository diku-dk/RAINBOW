"""
This script contains code to create a sandbox scene.
"""
import numpy as np

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
from rainbow.simulators.prox_rigid_bodies.types import Engine


def create_box_stack(
        engine: Engine,
        box_width: float,
        box_height: float,
        box_depth: float,
        K_boxes: int,
        density: float,
        material_name: str,
) -> list[str]:
    """
    This function creates a box stack scene.

    :param engine:                  The engine that will be used to create the dry stone rigid bodies in.
    :param box_width:               The width of the boxes.
    :param box_height:              The height of the boxes.
    :param box_depth:               The depth of the boxes.
    :param K_boxes:                 The number of boxes that will be stacked.
    :param density:                 The mass density to use for all the rigid bodies.
    :param material_name:           The material name to use for all the rigid bodies that are created.
    :return:                        A list with the names of all the rigid bodies that were created.
    """

    shape_name = API.generate_unique_name("box")
    V, T = MESH.create_box(box_width, box_height, box_depth)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_names = []

    for k in range(K_boxes):

        body_name = API.generate_unique_name("body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        height = k*box_height + box_height/2.0
        r = V3.make(0.0, height, 0.0)

        radians = k*np.pi/4
        q = Q.Ry(radians)

        API.set_position(engine, body_name, r, True)
        API.set_orientation(engine, body_name, q, True)
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    return body_names
