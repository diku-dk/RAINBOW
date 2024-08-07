"""
This script contains code to create a falling glasses scene.
"""

import igl  # Only using igl.read_triangle_mesh to read obj files.
import numpy as np

import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
from rainbow.simulators.prox_rigid_bodies.types import Engine
from .create_grid import create_grid


def create_jack_grid(
        engine: Engine,
        r: np.ndarray,
        q: np.ndarray,
        width: float,
        height: float,
        depth: float,
        I: int,
        J: int,
        K: int,
        density: float = 1.0,
        material_name: str = "default",
        use_random_orientation: bool = False
) -> list[str]:
    """
    This function is used to create a lattice of jack shapes.

    :param engine:                    The engine that will be used to create the dry stone rigid bodies in.
    :param r:                         Translation of the lattice in the world.
    :param q:                         Rotation of the lattice in the world.
    :param width:                     The width of a lattice with jack-shapes.
    :param height:                    The height of a lattice with jack-shapes.
    :param depth:                     The depth of a lattice with jack-shapes.
    :param I:                         The number of jacks along the width-direction of the lattice.
    :param J:                         The number of jacks along the height-direction of the lattice.
    :param K:                         The number of jacks along the depth-direction of the lattice.
    :param density:                   The mass density to use for all the rigid bodies.
    :param material_name:             The material name to use for all the rigid bodies that are created.
    :param use_random_orientation:    Boolean flag used to tell whether jack shapes should be randomly oriented or not.
    :return:                          A list with the names of all the rigid bodies that were created.

    """
    shape_names = []
    shape_name = API.generate_unique_name("jack")

    V, T = igl.read_triangle_mesh("../data/jack.obj", dtypef=np.float64)
    mesh = API.create_mesh(V, T)
    MESH.scale_to_unit(mesh)
    s = (
            min(width / I, height / J, depth / K) * 0.9
    )  # The 0.9 scaling ensure some padding to avoid initial contact
    MESH.scale(mesh, s, s, s)
    API.create_shape(engine, shape_name, mesh)

    shape_names.append(shape_name)

    body_names = create_grid(
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
    return body_names
