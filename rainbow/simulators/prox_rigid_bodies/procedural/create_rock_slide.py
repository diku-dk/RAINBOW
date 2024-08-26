"""
This script contains code to create a rock slide scene.
"""

import numpy as np
import igl

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
from rainbow.simulators.prox_rigid_bodies.types import Engine
from .create_lattice import create_lattice


def create_rock_slide(engine: Engine,
                      pile_width: float,
                      pile_height: float,
                      pile_depth: float,
                      I_rocks: int,
                      J_rocks: int,
                      K_rocks: int,
                      density: float = 1.0,
                      material_name: str = "default"
                      ) -> list[str]:
    """
    This function creates a rock slide scene.

    :param engine:                  The engine that will be used to create the dry stone rigid bodies in.
    :param pile_width:              The width of the rock pile.
    :param pile_height:             The height of the rock pile.
    :param pile_depth:              The depth of the rock pile..
    :param I_rocks:                 The number of poles along the width-axis.
    :param J_rocks:                 The number of poles along the height-axis.
    :param K_rocks:                 The number of poles along the depth-axis.
    :param density:                 The mass density to use for all the rigid bodies.
    :param material_name:           The material name to use for all the rigid bodies that are created.
    :return:                        A list with the names of all the rigid bodies that were created.
    """
    body_names = []

    shape_name = API.generate_unique_name("landscape")
    V, T = igl.read_triangle_mesh("./data/rock_slide/mountains.obj", dtypef=np.float64)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)
    body_name = API.generate_unique_name("mountains")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)
    API.set_position(engine, body_name, V3.zero(), True)
    API.set_orientation(engine, body_name, Q.identity(), True)
    API.set_body_type(engine, body_name, "fixed")
    API.set_body_material(engine, body_name, material_name)
    API.set_mass_properties(engine, body_name, density)

    building_files = [
        "./data/rock_slide/building_01.obj",
        "./data/rock_slide/building_02.obj",
        "./data/rock_slide/building_03.obj",
        "./data/rock_slide/building_04.obj",
        "./data/rock_slide/building_05.obj",
        "./data/rock_slide/building_06.obj",
        "./data/rock_slide/building_07.obj",
        "./data/rock_slide/building_08.obj",
        "./data/rock_slide/building_09.obj",
        "./data/rock_slide/building_10.obj",
        "./data/rock_slide/building_11.obj",
        "./data/rock_slide/building_12.obj",
        "./data/rock_slide/building_13.obj",
        "./data/rock_slide/building_14.obj",
        "./data/rock_slide/building_15.obj",
        "./data/rock_slide/building_16.obj",
        "./data/rock_slide/building_17.obj",
        "./data/rock_slide/building_18.obj",
        "./data/rock_slide/building_19.obj",
        "./data/rock_slide/building_20.obj"
    ]

    for filename in building_files:
        shape_name = API.generate_unique_name("shape")
        V, T = igl.read_triangle_mesh(filename, dtypef=np.float64)
        mesh = API.create_mesh(V, T)
        API.create_shape(engine, shape_name, mesh)
        body_name = API.generate_unique_name("building")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)
        API.set_position(engine, body_name, V3.zero(), True)
        API.set_orientation(engine, body_name, Q.identity(), True)
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    shape_names = []
    for n in range(20):
        shape_name = API.generate_unique_name("rock")
        points = [V3.rand(-1.0, 1.0) for _ in range(20)]

        V, T = MESH.create_convex_hull(points)
        mesh = API.create_mesh(V, T)
        MESH.scale_to_unit(mesh)
        s = (
                min(pile_width / I_rocks, pile_height / J_rocks, pile_depth / K_rocks) * 0.9
        )  # The 0.9 scaling ensure some padding to avoid initial contact
        MESH.scale(mesh, s, s, s)
        API.create_shape(engine, shape_name, mesh)

        shape_names.append(shape_name)

    r = V3.make(-10, 16, 5)
    q = Q.Ry(3 * np.pi / 4)
    body_names.append(
        create_lattice(
            engine,
            r,
            q,
            shape_names,
            pile_width,
            pile_height,
            pile_depth,
            I_rocks,
            J_rocks,
            K_rocks,
            density,
            material_name,
            True
        )
    )
    return body_names
