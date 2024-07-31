"""
This script contains code to create a funnel scene.
"""

import numpy as np
import igl   # Only using igl.read_triangle_mesh to read obj files.

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
from rainbow.simulators.prox_rigid_bodies.types import Engine
from .create_grid import _create_grid


def create_funnel(
        engine: Engine,
        funnel_height,
        funnel_radius,
        grid_width,
        grid_height,
        grid_depth,
        I,
        J,
        K,
        density: float,
        material_name: str,
        use_random_orientation: bool = True
        ) -> list[str]:
    """

    :param engine:
    :param funnel_height:
    :param funnel_radius:
    :param grid_width:
    :param grid_height:
    :param grid_depth:
    :param I:
    :param J:
    :param K:
    :param density:
    :param material_name:
    :param use_random_orientation:
    :return:
    """
    height = grid_height
    width = grid_width
    depth = grid_depth

    shape_names = []
    shape_name = API.generate_unique_name("jack")

    V, T = igl.read_triangle_mesh(filename="../data/jack.obj", dtypef=np.float64)
    mesh = API.create_mesh(V, T)
    MESH.scale_to_unit(mesh)
    s = (
            min(width / I, height / J, depth / K) * 0.9
    )  # The 0.9 scaling ensure some padding to avoid initial contact
    MESH.scale(mesh, s, s, s)
    API.create_shape(engine, shape_name, mesh)

    shape_names.append(shape_name)

    r = V3.make(-width / 2.0, 3.0 * funnel_height, -depth / 2.0)
    q = Q.identity()
    body_names = _create_grid(
        engine,
        r,
        q,
        shape_names,
        width,
        height,
        depth,
        I,
        J,
        K,
        density,
        material_name,
        use_random_orientation,
    )

    shape_name = API.generate_unique_name("funnel")
    V, T = igl.read_triangle_mesh(filename="../data/funnel.obj", dtypef=np.float64)
    mesh = API.create_mesh(V, T)

    MESH.scale_to_unit(mesh)
    (l, u) = MESH.aabb(mesh)
    extends = u - l
    MESH.scale(
        mesh,
        (funnel_radius * 2.0) / extends[0],
        funnel_height / extends[1],
        (funnel_radius * 2.0) / extends[2],
    )
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r = V3.make(0.0, 1.5 * funnel_height, 0.0)
    q = Q.Rx(np.pi)
    API.set_position(engine, body_name, r, True)
    API.set_orientation(engine, body_name, q, True)

    API.set_body_type(engine, body_name, "fixed")
    API.set_body_material(engine, body_name, material_name)
    API.set_mass_properties(engine, body_name, density)

    return body_names
