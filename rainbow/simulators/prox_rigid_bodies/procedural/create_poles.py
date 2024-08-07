"""
This script contains code to create a scene of a bunch of random objects falling onto a grid of poles.
"""

import numpy as np
import igl   # Only using igl.read_triangle_mesh to read obj files.

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
from rainbow.simulators.prox_rigid_bodies.types import Engine
from .create_grid import create_grid


def create_poles(
        engine: Engine,
        pole_height: float,
        pole_radius: float,
        I_poles: int,
        K_poles: int,
        grid_width: float,
        grid_height: float,
        grid_depth: float,
        I_grid: int,
        J_grid: int,
        K_grid: int,
        density: float,
        material_name: str,
        use_random_orientation=True,
) -> list[str]:
    """
    This function creates the pole scene.


    :param engine:                  The engine that will be used to create the dry stone rigid bodies in.
    :param pole_height:             The height of the poles.
    :param pole_radius:             The radius of the poles.
    :param I_poles:                 The number of poles along the x-axis.
    :param K_poles:                 The number of poles along the y-axis.
    :param grid_width:              The width of a lattice with objects-shapes falling down onto the poles.
    :param grid_height:             The height of a lattice with objects-shapes falling down onto the poles.
    :param grid_depth:              The depth of a lattice with objects-shapes falling down onto the poles.
    :param I_grid:                  The number of objects along the width-direction of the lattice.
    :param J_grid:                  The number of objects along the height-direction of the lattice.
    :param K_grid:                  The number of objects along the depth-direction of the lattice.
    :param density:                 The mass density to use for all the rigid bodies.
    :param material_name:           The material name to use for all the rigid bodies that are created.
    :param use_random_orientation:  Boolean flag used to tell whether jack shapes should be randomly oriented or not.
    :return:                        A list with the names of all the rigid bodies that were created.
    """
    height = grid_height
    width = grid_width
    depth = grid_depth

    shape_names = []
    obj_files = [
        "../data/cow.obj",
        "../data//dragon.obj",
        "../data//knot.obj",
        "../data//lamp.obj",
        "../data//propella.obj",
        "../data//shuttle.obj",
    ]

    for file in obj_files:
        shape_name = API.generate_unique_name("small_obj_shape")
        V, T = igl.read_triangle_mesh(file, dtypef=np.float64)
        mesh = API.create_mesh(V, T)
        MESH.scale_to_unit(mesh)
        s = min(width / I_grid, height / J_grid, depth / K_grid)
        MESH.scale(mesh, s, s, s)
        API.create_shape(engine, shape_name, mesh)
        shape_names.append(shape_name)

    r = V3.make(-width / 2.0, 2.0 * pole_height, -depth / 2.0)
    q = Q.identity()
    body_names = create_grid(
        engine,
        r,
        q,
        shape_names,
        width,
        height,
        depth,
        I_grid,
        J_grid,
        K_grid,
        density,
        material_name,
        use_random_orientation,
    )

    shape_name = API.generate_unique_name("pole")
    V, T = MESH.create_cylinder(pole_radius, pole_height)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    r = V3.make(0.0, 0.0, 0.0)
    q = Q.identity()
    dx = width / I_poles
    dz = depth / K_poles
    offset_x = -width / 2.0 + pole_radius
    offset_z = -depth / 2.0 + pole_radius
    for i in range(I_poles):
        for k in range(K_poles):
            body_name = API.generate_unique_name("body")
            body_names.append(body_name)
            API.create_rigid_body(engine, body_name)
            API.connect_shape(engine, body_name, shape_name)

            r_l = V3.make(i * dx + offset_x, pole_height / 2.0, k * dz + offset_z)
            q_l = Q.identity()
            r_g = Q.rotate(q, r_l) + r
            q_g = Q.prod(q, q_l)

            API.set_position(engine, body_name, r_g, True)
            API.set_orientation(engine, body_name, q_g, True)

            API.set_body_type(engine, body_name, "fixed")
            API.set_body_material(engine, body_name, material_name)
            API.set_mass_properties(engine, body_name, density)

    return body_names
