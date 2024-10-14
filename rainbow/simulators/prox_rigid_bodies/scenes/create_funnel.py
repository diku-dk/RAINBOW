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
from .create_lattice import create_lattice


def create_funnel(
        engine: Engine,
        funnel_height: float,
        funnel_radius: float,
        grid_width: float,
        grid_height: float,
        grid_depth: float,
        I: int,
        J: int,
        K: int,
        density: float,
        material_name: str,
        use_random_orientation: bool = True
        ) -> list[str]:
    """
    This function is used to create a funnel scene.

    :param engine:                    The engine that will be used to create the dry stone rigid bodies in.
    :param funnel_height:             The height of the funnel.
    :param funnel_radius:             The radius of the funnel.
    :param grid_width:                The width of a lattice with jack-shapes falling down into the funnel.
    :param grid_height:               The height of a lattice with jack-shapes falling down into the funnel.
    :param grid_depth:                The depth of a lattice with jack-shapes falling down into the funnel.
    :param I:                         The number of jacks along the width-direction of the lattice.
    :param J:                         The number of jacks along the height-direction of the lattice.
    :param K:                         The number of jacks along the depth-direction of the lattice.
    :param density:                   The mass density to use for all the rigid bodies.
    :param material_name:             The material name to use for all the rigid bodies that are created.
    :param use_random_orientation:    Boolean flag used to tell whether jack shapes should be randomly oriented or not.
    :return:                          A list with the names of all the rigid bodies that were created.
    """
    height = grid_height
    width = grid_width
    depth = grid_depth

    shape_names = []
    shape_name = API.generate_unique_name("jack")

    V, T = igl.read_triangle_mesh(filename="./data/jack.obj", dtypef=np.float64)
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
    body_names = create_lattice(
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
    V, T = igl.read_triangle_mesh(filename="./data/funnel.obj", dtypef=np.float64)
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
