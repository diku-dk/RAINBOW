"""
This script contains code to create a dry stone-masonry arch.
"""

import numpy as np

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
from rainbow.simulators.prox_rigid_bodies.types import Engine


def create_arch(
        engine: Engine,
        r: np.ndarray,
        q: np.ndarray,
        width: float,
        height: float,
        depth: float,
        pier_stones: int,
        arch_stones: int,
        density: float,
        material_name: str
        ) -> list[str]:
    """
    This function creates a dry stone-masonry arch.

    The origin of the arch is at the mid-point of the base.

    :param engine:          The engine that will be used to create the dry stone rigid bodies in.
    :param r:               Translation of the stone arch into the world.
    :param q:               Rotation of the stone arch in the world.
    :param width:           The total width of the arch.
    :param height:          The total height of the arch.
    :param depth:           The depth of the stones used to crate the arch.
    :param pier_stones:     The number of stones in each pier.
    :param arch_stones:     The number of stones in the arch.
    :param density:         The mass density to use for all the rigid bodies.
    :param material_name:   The material name to use for all the rigid bodies that are created.
    :return:                A list with the names of all the rigid bodies that were created.
    """
    arch_width = width
    arch_height = height
    arch_depth = depth
    pier_height = arch_height - arch_width / 2.0
    pier_brick_width = arch_width / 5.0
    pier_brick_depth = arch_depth
    pier_brick_height = pier_height / pier_stones

    if arch_width <= 0.0:
        raise RuntimeWarning("Arch width must be positive")
    if arch_height <= 0.0:
        raise RuntimeWarning("Arch height must be positive")
    if arch_depth <= 0.0:
        raise RuntimeWarning("Arch depth must be positive")
    if pier_stones <= 0:
        raise RuntimeWarning("Must have at least one pier stone")
    if arch_stones <= 2:
        raise RuntimeWarning("Must have at least 3 arch stones")
    if (arch_stones % 2) == 0:
        raise RuntimeWarning(
            "Arch stones was an even number, so no key stone will be created"
        )
    if density <= 0.0:
        raise RuntimeWarning("Density must be positive")
    if pier_height <= 0.0:
        raise RuntimeWarning(
            "arch height must be larger than half arch width, height > width/2.0"
        )

    body_names = []

    shape_name = API.generate_unique_name("pier_brick_shape")

    V, T = MESH.create_box(pier_brick_width, pier_brick_height, pier_brick_depth)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    for stone in range(pier_stones):
        height = pier_brick_height / 2.0 + stone * pier_brick_height

        body_name = API.generate_unique_name("left_pier_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        r_l = V3.make((-arch_width / 2.0) + (pier_brick_width / 2.0), height, 0.0)
        q_l = Q.identity()
        r_g = Q.rotate(q, r_l) + r
        q_g = Q.prod(q, q_l)

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

        body_name = API.generate_unique_name("right_pier_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        r_l = V3.make((arch_width / 2.0 - pier_brick_width / 2.0), height, 0.0)
        q_l = Q.identity()
        r_g = Q.rotate(q, r_l) + r
        q_g = Q.prod(q, q_l)

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

    shape_name = API.generate_unique_name("arch_brick_shape")

    delta_phi = np.pi / arch_stones
    outer_radius = arch_width / 2.0
    inner_radius = outer_radius - pier_brick_width
    center_radius = outer_radius - (pier_brick_width / 2.0)

    r0 = V3.make(center_radius, 0.0, arch_depth / 2.0)
    p0 = V3.make(inner_radius, 0.0, arch_depth / 2.0)
    p0 = Q.rotate(Q.Rz(-delta_phi / 2.0), p0)
    p1 = V3.make(outer_radius, 0.0, arch_depth / 2.0)
    p1 = Q.rotate(Q.Rz(-delta_phi / 2.0), p1)
    p2 = V3.make(outer_radius, 0.0, arch_depth / 2.0)
    p2 = Q.rotate(Q.Rz(delta_phi / 2.0), p2)
    p3 = V3.make(inner_radius, 0.0, arch_depth / 2.0)
    p3 = Q.rotate(Q.Rz(delta_phi / 2.0), p3)

    p0 -= r0
    p1 -= r0
    p2 -= r0
    p3 -= r0

    p4 = p0 - V3.make(0.0, 0.0, arch_depth)
    p5 = p1 - V3.make(0.0, 0.0, arch_depth)
    p6 = p2 - V3.make(0.0, 0.0, arch_depth)
    p7 = p3 - V3.make(0.0, 0.0, arch_depth)

    V, T = MESH.create_cuboid(p0, p1, p2, p3, p4, p5, p6, p7)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    for stone in range(arch_stones):
        body_name = API.generate_unique_name("arch_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        q_l = Q.Rz((delta_phi / 2.0) + (stone * delta_phi))
        r_l = Q.rotate(q_l, r0) + V3.make(0.0, pier_height, 0.0)
        r_g = Q.rotate(q, r_l) + r
        q_g = Q.prod(q, q_l)

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

    for body_name in body_names:
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    return body_names
