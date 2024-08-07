"""
This script contains code to create a lattice of given shapes.
"""
from typing import Optional

import numpy as np

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.simulators.prox_rigid_bodies.api as API
from rainbow.simulators.prox_rigid_bodies.types import Engine


def create_grid(
        engine: Engine,
        r: np.ndarray,
        q: np.ndarray,
        shape_names: list[str],
        width: float,
        height: float,
        depth: float,
        I: int,
        J: int,
        K: int,
        density: float,
        material_name: str,
        use_random_orientation=True,
        q_body: Optional[np.ndarray] = None,
) -> list[str]:
    """
    This function is used to create a lattice of given shapes.

    :param engine:          The engine that will be used to create the rigid bodies in.
    :param r:               Translation of the lattice in the world.
    :param q:               Rotation of the lattice in the world.
    :param shape_names:     A list of shape names to use to create the objects in the lattice.
    :param width:           The width of the lattice.
    :param height:          The height of the lattice.
    :param depth:           The depth of the lattice.
    :param I:               The number of shapes in the lattice along the x-axis.
    :param J:               The number of shapes in the lattice along the y-axis.
    :param K:               The number of shapes in the lattice along the z-axis.
    :param density:                   The mass density to use for all the rigid bodies.
    :param material_name:             The material name to use for all the rigid bodies that are created.
    :param use_random_orientation:    Boolean flag used to tell whether jack shapes should be randomly oriented or not.
    :param q_body:                    An optional initial quaternion ot initialize the orientation of all the shapes.
    :return:                          A list with the names of all the rigid bodies that were created.
    """
    body_names = []
    dW = width / I
    dH = height / J
    dD = depth / K
    N = len(shape_names)
    for i in range(I):
        for j in range(J):
            for k in range(K):
                shape_choice = (i + j + k) % N
                shape_name = shape_names[shape_choice]

                body_name = API.generate_unique_name("body")
                body_names.append(body_name)
                API.create_rigid_body(engine, body_name)
                API.connect_shape(engine, body_name, shape_name)

                r_l = V3.make(i * dW, j * dH, k * dD) + V3.make(dW, dH, dD) / 2.0

                q_l = Q.identity()
                if use_random_orientation:
                    q_l = Q.rand()
                elif q_body is not None:
                    q_l = q_body

                r_g = Q.rotate(q, r_l) + r
                q_g = Q.prod(q, q_l)

                API.set_position(engine, body_name, r_g, False)
                API.set_orientation(engine, body_name, q_g, False)

    for body_name in body_names:
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    return body_names
