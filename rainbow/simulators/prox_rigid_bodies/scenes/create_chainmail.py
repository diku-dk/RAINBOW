"""
This script contains code to create chain-mail ring structure.
"""

import numpy as np

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
from rainbow.simulators.prox_rigid_bodies.types import Engine


def create_chainmail(
        engine: Engine,
        major_radius: float,
        minor_radius: float,
        columns: int,
        rows: int,
        stretch: float = 0.5,
        density: float = 1.0,
        material_name: str = "default",
        pinned: bool = True
        ) -> list[str]:
    """
    Create a square chainmail.

    :param engine:           The engine that will be used to create the rigid bodies in.
    :param major_radius:     The major radius of an individual ring.
    :param minor_radius:     The minor radius of an individual ring.
    :param columns:          The number of columns in the chainmail.
    :param rows:             The number of rows the chainmail.
    :param stretch:          A stretch factor used to move rings apart.
    :param density:          Mass density of the rings in the chainmail.
    :param material_name:    The material name to use for all the rings in the chainmail.
    :param pinned:           Boolean flag to indicate if the corners of the chainmail are pinned down or not.
    :return:                 A list of all the body names that are created.
    """
    stretch = np.clip(stretch, 0.0, 1.0)

    R = major_radius
    r = minor_radius

    shape_name = API.generate_unique_name("torus_shape")
    V, T = MESH.create_torus(R, r, segments=24, slices=24)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_names = []

    # Maximum clearance between two rings
    tau = 2 * R - 4 * r
    # Maximum distance between two ring centers along any axis.
    delta = 2 * R + tau * stretch

    q_m2w = Q.Rx(-np.pi / 2)  # Needed to change the z-up direction to the y-up direction.
    r_m2w = V3.make(- delta * columns / 2,
                    0,
                    delta * rows / 2
                    )  # Needed to center chain mail at origin of the world coordinate system.

    for j in range(rows + 1):
        for i in range(columns + 1):
            body_name = API.generate_unique_name("body")
            body_names.append(body_name)
            API.create_rigid_body(engine, body_name)
            API.connect_shape(engine, body_name, shape_name)

            # Place chain ring in local model space
            r_l = V3.make(i * delta, j * delta, 0)
            q_l = Q.identity()

            # Transform from local model space into global world space
            r_w = Q.rotate(q_m2w, r_l) + r_m2w  # World position (y-up)
            q_w = Q.prod(q_m2w, q_l)  # World orientation (y-up)

            API.set_position(engine, body_name, r_w, True)
            API.set_orientation(engine, body_name, q_w, True)

    for j in range(rows + 1):
        for i in range(columns):
            body_name = API.generate_unique_name("body")
            body_names.append(body_name)
            API.create_rigid_body(engine, body_name)
            API.connect_shape(engine, body_name, shape_name)

            # Place chain ring in local model space
            r_l = V3.make((i + 0.5) * delta, j * delta, 0)
            q_l = Q.Rx(np.pi / 2)

            # Transform from local model space into global world space
            r_w = Q.rotate(q_m2w, r_l) + r_m2w  # World position (y-up)
            q_w = Q.prod(q_m2w, q_l)  # World orientation (y-up)

            API.set_position(engine, body_name, r_w, True)
            API.set_orientation(engine, body_name, q_w, True)

    for j in range(rows):
        for i in range(columns + 1):
            body_name = API.generate_unique_name("body")
            body_names.append(body_name)
            API.create_rigid_body(engine, body_name)
            API.connect_shape(engine, body_name, shape_name)

            # Place chain ring in local model space
            r_l = V3.make(i * delta, (j + 0.5) * delta, 0)
            q_l = Q.Ry(np.pi / 2)

            # Transform from local model space into global world space
            r_w = Q.rotate(q_m2w, r_l) + r_m2w  # World position (y-up)
            q_w = Q.prod(q_m2w, q_l)  # World orientation (y-up)

            API.set_position(engine, body_name, r_w, True)
            API.set_orientation(engine, body_name, q_w, True)

    for body_name in body_names:
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    if pinned:
        south_west_ring = body_names[0]
        south_east_ring = body_names[columns]
        north_west_ring = body_names[(rows + 1) * (columns + 1) - columns - 1]
        north_east_ring = body_names[(rows + 1) * (columns + 1) - 1]
        API.set_body_type(engine, south_west_ring, "fixed")
        API.set_body_type(engine, south_east_ring, "fixed")
        API.set_body_type(engine, north_west_ring, "fixed")
        API.set_body_type(engine, north_east_ring, "fixed")

    return body_names
