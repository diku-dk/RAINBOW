"""
This script contains code to create a temple scene.
"""
import numpy as np

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
from rainbow.simulators.prox_rigid_bodies.types import Engine
from .create_pillar import create_pillar


def create_temple(
        engine: Engine,
        I_pillars: int,
        K_pillars: int,
        pillar_width: float,
        pillar_height: float,
        pillar_depth: float,
        pillar_stones: int,
        density: float,
        material_name: str,
) -> list[str]:
    """
    Create a temple scene.

    :param engine:              The engine that will be used to create the dry stone rigid bodies in.
    :param I_pillars:           The number of pillars along the width-axis.
    :param K_pillars:           The number of pillars along the height-axis.
    :param pillar_width:        The width of a single pillar.
    :param pillar_height:       The height of a single pillar.
    :param pillar_depth:        The depth of a single pillar.
    :param pillar_stones:       How many stones in the colum of a single pillar.
    :param density:             The mass density to use for all the rigid bodies.
    :param material_name:       The material name to use for all the rigid bodies that are created.
    :return:                    A list with the names of all the rigid bodies that were created.
    """
    if (K_pillars - 1) % 3 != 0:
        raise RuntimeError(
            "K pillars must be equal to 3*n+1 for a given positive integer n"
        )

    temple_width = (2 * I_pillars - 1) * pillar_width
    temple_depth = (2 * K_pillars - 1) * pillar_depth
    stairs_height = 0.1 * pillar_height

    body_names = []

    # Create base/floor of temple
    # shape_name = API.generate_unique_name(engine, 'bottom_step_shape')
    shape_name = API.generate_unique_name("regular_epistyle_brick_shape")
    V, T = MESH.create_box(
        temple_width + 6 * pillar_width, stairs_height, temple_depth + 6 * pillar_depth
    )
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("bottom_step_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    API.set_position(engine, body_name, V3.make(0.0, stairs_height * 0.5, 0.0), True)
    API.set_orientation(engine, body_name, Q.identity(), True)

    shape_name = API.generate_unique_name("middle_step_shape")
    V, T = MESH.create_box(
        temple_width + 4 * pillar_width, stairs_height, temple_depth + 4 * pillar_depth
    )
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("middle_step_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    API.set_position(engine, body_name, V3.make(0.0, stairs_height * 1.5, 0.0), True)
    API.set_orientation(engine, body_name, Q.identity(), True)

    shape_name = API.generate_unique_name("top_step_shape")
    V, T = MESH.create_box(
        temple_width + 2 * pillar_width, stairs_height, temple_depth + 2 * pillar_depth
    )
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("top_step_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    API.set_position(engine, body_name, V3.make(0.0, stairs_height * 2.5, 0.0), True)
    API.set_orientation(engine, body_name, Q.identity(), True)

    shape_name = API.generate_unique_name("interior_step_shape")
    V, T = MESH.create_box(
        temple_width - 2.5 * pillar_width,
        stairs_height,
        temple_depth - 2.5 * pillar_depth,
    )
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("interior_step_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    API.set_position(engine, body_name, V3.make(0.0, stairs_height * 3.5, 0.0), True)
    API.set_orientation(engine, body_name, Q.identity(), True)

    # Create pillars
    y = stairs_height * 3.0
    for i in range(I_pillars):
        x = 2 * i * pillar_width - temple_width / 2.0 + pillar_width / 2.0

        z_front = temple_depth / 2.0 - pillar_depth / 2.0
        r_g = V3.make(x, y, z_front)
        q_g = Q.identity()
        names = create_pillar(
            engine,
            r_g,
            q_g,
            pillar_width,
            pillar_height,
            pillar_depth,
            pillar_stones,
            density,
            material_name,
        )
        body_names.extend(names)

        z_back = -temple_depth / 2.0 + pillar_depth / 2.0
        r_g = V3.make(x, y, z_back)
        q_g = Q.identity()
        names = create_pillar(
            engine,
            r_g,
            q_g,
            pillar_width,
            pillar_height,
            pillar_depth,
            pillar_stones,
            density,
            material_name,
        )
        body_names.extend(names)

    for k in range(1, K_pillars - 1):
        z = 2 * k * pillar_depth - temple_depth / 2.0 + pillar_depth / 2.0

        x_right = temple_width / 2.0 - pillar_width / 2.0
        r_g = V3.make(x_right, y, z)
        q_g = Q.identity()
        names = create_pillar(
            engine,
            r_g,
            q_g,
            pillar_width,
            pillar_height,
            pillar_depth,
            pillar_stones,
            density,
            material_name,
        )
        body_names.extend(names)

        x_left = -temple_width / 2.0 + pillar_width / 2.0
        r_g = V3.make(x_left, y, z)
        q_g = Q.identity()
        names = create_pillar(
            engine,
            r_g,
            q_g,
            pillar_width,
            pillar_height,
            pillar_depth,
            pillar_stones,
            density,
            material_name,
        )
        body_names.extend(names)

    y = stairs_height * 4.0
    for k in range(3, K_pillars - 3, 3):
        z = 2 * k * pillar_depth - temple_depth / 2.0 + pillar_depth / 2.0
        for i in range(1, I_pillars - 1):
            x = 2 * i * pillar_width - temple_width / 2.0 + pillar_width / 2.0

            r_g = V3.make(x, y, z)
            q_g = Q.identity()
            names = create_pillar(
                engine,
                r_g,
                q_g,
                pillar_width,
                pillar_height - stairs_height,
                pillar_depth,
                pillar_stones,
                density,
                material_name,
            )
            body_names.extend(names)

    # Create epistyle (bricklayer onto the pillars)
    epistyle_height = 2.0 * stairs_height

    shape_name = API.generate_unique_name("regular_epistyle_brick_shape")
    V, T = MESH.create_box(2.0 * pillar_width, epistyle_height, pillar_depth)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    y = stairs_height * 3.0 + pillar_height + epistyle_height / 2.0
    for k in range(0, K_pillars, 3):
        z = 2 * k * pillar_depth - temple_depth / 2.0 + pillar_depth / 2.0
        for i in range(1, I_pillars - 2):
            body_name = API.generate_unique_name("epistyle_brick_body")
            body_names.append(body_name)
            API.create_rigid_body(engine, body_name)
            API.connect_shape(engine, body_name, shape_name)

            x = 2 * i * pillar_width - temple_width / 2.0 + 1.5 * pillar_width
            r_g = V3.make(x, y, z)
            q_g = Q.identity()

            API.set_position(engine, body_name, r_g, True)
            API.set_orientation(engine, body_name, q_g, True)

    for k in range(1, K_pillars, 3):
        z = 2 * k * pillar_depth - temple_depth / 2.0 + pillar_depth * 1.5

        body_name = API.generate_unique_name("epistyle_brick_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        x_right = temple_width / 2.0 - pillar_width / 2.0
        r_g = V3.make(x_right, y, z)
        q_g = Q.Ry(np.pi / 2.0)

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

        body_name = API.generate_unique_name("epistyle_brick_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        x_left = -temple_width / 2.0 + pillar_width / 2.0
        r_g = V3.make(x_left, y, z)
        q_g = Q.Ry(np.pi / 2.0)

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

    shape_name = API.generate_unique_name("corner_epistyle_brick_shape")
    p0 = V3.make(-1.5 * pillar_width, epistyle_height / 2.0, pillar_depth / 2.0)
    p1 = V3.make(1.0 * pillar_width, epistyle_height / 2.0, pillar_depth / 2.0)
    p2 = V3.make(1.0 * pillar_width, epistyle_height / 2.0, -pillar_depth / 2.0)
    p3 = V3.make(-0.5 * pillar_width, epistyle_height / 2.0, -pillar_depth / 2.0)
    p4 = p0 - V3.make(0.0, epistyle_height, 0.0)
    p5 = p1 - V3.make(0.0, epistyle_height, 0.0)
    p6 = p2 - V3.make(0.0, epistyle_height, 0.0)
    p7 = p3 - V3.make(0.0, epistyle_height, 0.0)

    V, T = MESH.create_cuboid(p0, p1, p2, p3, p4, p5, p6, p7)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("epistyle_brick_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    x = -temple_width / 2.0 + pillar_width * 1.5
    z = temple_depth / 2.0 - pillar_depth * 0.5
    r_g = V3.make(x, y, z)
    q_g = Q.identity()

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    body_name = API.generate_unique_name("epistyle_brick_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    x = temple_width / 2.0 - pillar_width * 1.5
    z = temple_depth / 2.0 - pillar_depth * 0.5
    r_g = V3.make(x, y, z)
    q_g = Q.Rz(np.pi)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    body_name = API.generate_unique_name("epistyle_brick_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    x = -temple_width / 2.0 + pillar_width * 1.5
    z = -temple_depth / 2.0 + pillar_depth * 0.5
    r_g = V3.make(x, y, z)
    q_g = Q.Rx(np.pi)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    body_name = API.generate_unique_name("epistyle_brick_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    x = temple_width / 2.0 - pillar_width * 1.5
    z = -temple_depth / 2.0 + pillar_depth * 0.5
    r_g = V3.make(x, y, z)
    q_g = Q.Ry(-np.pi)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    body_name = API.generate_unique_name("epistyle_brick_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    x = temple_width / 2.0 - pillar_width * 0.5
    z = temple_depth / 2.0 - pillar_depth * 1.5
    r_g = V3.make(x, y, z)
    q_g = Q.Ry(np.pi / 2.0)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    body_name = API.generate_unique_name("epistyle_brick_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    x = -temple_width / 2.0 + pillar_width * 0.5
    z = temple_depth / 2.0 - pillar_depth * 1.5
    r_g = V3.make(x, y, z)
    q_g = Q.prod(Q.Ry(np.pi / 2.0), Q.Rx(np.pi))

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    body_name = API.generate_unique_name("epistyle_brick_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    x = -temple_width / 2.0 + pillar_width * 0.5
    z = -temple_depth / 2.0 + pillar_depth * 1.5
    r_g = V3.make(x, y, z)
    q_g = Q.Ry(-np.pi / 2.0)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    body_name = API.generate_unique_name("epistyle_brick_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    x = temple_width / 2.0 - pillar_width * 0.5
    z = -temple_depth / 2.0 + pillar_depth * 1.5
    r_g = V3.make(x, y, z)
    q_g = Q.prod(Q.Ry(-np.pi / 2.0), Q.Rx(np.pi))

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    shape_name = API.generate_unique_name("wedge_epistyle_brick_shape")
    p0 = V3.make(-0.5 * pillar_width, epistyle_height / 2.0, pillar_depth / 2.0)
    p1 = V3.make(1.0 * pillar_width, epistyle_height / 2.0, pillar_depth / 2.0)
    p2 = V3.make(1.0 * pillar_width, epistyle_height / 2.0, -pillar_depth / 2.0)
    p3 = V3.make(-0.5 * pillar_width, epistyle_height / 2.0, -pillar_depth / 2.0)
    p4 = V3.make(-1.5 * pillar_width, epistyle_height / 2.0, 0.0)
    p5 = p0 - V3.make(0.0, epistyle_height, 0.0)
    p6 = p1 - V3.make(0.0, epistyle_height, 0.0)
    p7 = p2 - V3.make(0.0, epistyle_height, 0.0)
    p8 = p3 - V3.make(0.0, epistyle_height, 0.0)
    p9 = p4 - V3.make(0.0, epistyle_height, 0.0)

    V, T = MESH.create_convex_hull([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9])
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    for k in range(3, K_pillars - 3, 3):
        z = 2 * k * pillar_depth - temple_depth / 2.0 + pillar_depth / 2.0

        body_name = API.generate_unique_name("epistyle_brick_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        x = -temple_width / 2.0 + pillar_width * 1.5
        r_g = V3.make(x, y, z)
        q_g = Q.identity()

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

        body_name = API.generate_unique_name("epistyle_brick_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        x = temple_width / 2.0 - pillar_width * 1.5
        r_g = V3.make(x, y, z)
        q_g = Q.Ry(np.pi)

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

    shape_name = API.generate_unique_name("junction_epistyle_brick_shape")
    p0 = V3.make(-0.5 * pillar_width, epistyle_height / 2.0, pillar_depth)
    p1 = V3.make(0.5 * pillar_width, epistyle_height / 2.0, pillar_depth)
    p2 = V3.make(0.5 * pillar_width, epistyle_height / 2.0, -pillar_depth / 2.0)
    p3 = V3.make(-0.5 * pillar_width, epistyle_height / 2.0, -pillar_depth)
    p4 = p0 - V3.make(0.0, epistyle_height, 0.0)
    p5 = p1 - V3.make(0.0, epistyle_height, 0.0)
    p6 = p2 - V3.make(0.0, epistyle_height, 0.0)
    p7 = p3 - V3.make(0.0, epistyle_height, 0.0)

    V, T = MESH.create_cuboid(p0, p1, p2, p3, p4, p5, p6, p7)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    for k in range(3, K_pillars - 3, 3):
        z_junction = 2 * k * pillar_depth - temple_depth / 2.0 + pillar_depth / 2.0
        z_back = z_junction - pillar_depth
        z_front = z_junction + pillar_depth
        x_left = -temple_width / 2.0 + pillar_width * 0.5
        x_right = temple_width / 2.0 - pillar_width * 0.5

        body_name = API.generate_unique_name("epistyle_brick_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        r_g = V3.make(x_left, y, z_front)
        q_g = Q.identity()

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

        body_name = API.generate_unique_name("epistyle_brick_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        r_g = V3.make(x_left, y, z_back)
        q_g = Q.Rx(np.pi)

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

        body_name = API.generate_unique_name("epistyle_brick_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        r_g = V3.make(x_right, y, z_front)
        q_g = Q.Rz(np.pi)

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

        body_name = API.generate_unique_name("epistyle_brick_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        r_g = V3.make(x_right, y, z_back)
        q_g = Q.Ry(np.pi)

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

    # Next we create the gables
    # y_epistyle = 3.0 * stairs_height + pillar_height + epistyle_height
    # roof_thickness = 0.5 * stairs_height
    # roof_angle = pi / 6.0
    # gable_offset = sqrt((1.0 + (sin(pi / 2 - roof_angle) / sin(roof_angle)) ** 2)) * roof_thickness

    gable_brick_height = stairs_height
    gable_brick_width = 2.0 * pillar_width
    gable_brick_depth = 0.5 * pillar_depth

    shape_name_regular = API.generate_unique_name("gable_brick_shape")
    x_left = -gable_brick_width / 2.0
    x_right = gable_brick_width / 2.0
    y_bottom = -gable_brick_height / 2.0
    y_top = gable_brick_height / 2.0
    z_front = gable_brick_depth / 2.0
    z_back = -gable_brick_depth / 2.0
    p0 = V3.make(x_left, y_bottom, z_front)
    p1 = V3.make(x_right, y_bottom, z_front)
    p2 = V3.make(x_right, y_top, z_front)
    p3 = V3.make(x_left, y_top, z_front)
    p4 = V3.make(x_left, y_bottom, z_back)
    p5 = V3.make(x_right, y_bottom, z_back)
    p6 = V3.make(x_right, y_top, z_back)
    p7 = V3.make(x_left, y_top, z_back)

    V, T = MESH.create_cuboid(p0, p1, p2, p3, p4, p5, p6, p7)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name_regular, mesh)

    shape_name_edge = API.generate_unique_name("gable_edge_brick_shape")
    p0 = V3.make(0.0, y_bottom, z_front)
    p1 = V3.make(x_right, y_bottom, z_front)
    p2 = V3.make(x_right, y_top, z_front)
    p3 = V3.make(0.0, y_bottom, z_back)
    p4 = V3.make(x_right, y_bottom, z_back)
    p5 = V3.make(x_right, y_top, z_back)

    V, T = MESH.create_convex_hull([p0, p1, p2, p3, p4, p5])
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name_edge, mesh)

    for k in range(0, K_pillars, 3):
        z_gable = 2 * k * pillar_depth - temple_depth / 2.0 + pillar_depth / 2.0

        for j in range(I_pillars - 1):
            for i in range(I_pillars - j):

                x_brick = (
                        -temple_width / 2.0
                        + pillar_width / 2.0
                        + i * gable_brick_width
                        + j * gable_brick_width * 0.5
                )
                y_brick = (
                        3.0 * stairs_height
                        + pillar_height
                        + epistyle_height
                        + gable_brick_height / 2.0
                        + j * gable_brick_height
                )

                r_g = V3.make(x_brick, y_brick, z_gable)
                q_g = Q.identity()
                if i == 0 and j < I_pillars - 1:
                    shape_name = shape_name_edge
                elif i == I_pillars - j - 1 and j < I_pillars - 1:
                    q_g = Q.Ry(np.pi)
                    shape_name = shape_name_edge
                else:
                    shape_name = shape_name_regular

                body_name = API.generate_unique_name("gable_brick_body")
                body_names.append(body_name)
                API.create_rigid_body(engine, body_name)
                API.connect_shape(engine, body_name, shape_name)

                API.set_position(engine, body_name, r_g, True)
                API.set_orientation(engine, body_name, q_g, True)

    y_bottom = 0.0
    y_top = gable_brick_height

    shape_name_left_long = API.generate_unique_name("long_left_plank_shape")
    x_left_bottom = 0.0
    x_right_bottom = pillar_width / 2.0
    x_left_top = pillar_width
    x_right_top = pillar_width * 1.5
    z_front = 0.0
    z_back = -6.5 * pillar_depth
    p0 = V3.make(x_left_bottom, y_bottom, z_front)
    p1 = V3.make(x_right_bottom, y_bottom, z_front)
    p2 = V3.make(x_right_top, y_top, z_front)
    p3 = V3.make(x_left_top, y_top, z_front)
    p4 = V3.make(x_left_bottom, y_bottom, z_back)
    p5 = V3.make(x_right_bottom, y_bottom, z_back)
    p6 = V3.make(x_right_top, y_top, z_back)
    p7 = V3.make(x_left_top, y_top, z_back)

    V, T = MESH.create_cuboid(p0, p1, p2, p3, p4, p5, p6, p7)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name_left_long, mesh)

    shape_name_left_short = API.generate_unique_name("short_left_plank_shape")
    x_left_bottom = 0.0
    x_right_bottom = pillar_width / 2.0
    x_left_top = pillar_width
    x_right_top = pillar_width * 1.5
    z_front = 0.0
    z_back = -6.0 * pillar_depth
    p0 = V3.make(x_left_bottom, y_bottom, z_front)
    p1 = V3.make(x_right_bottom, y_bottom, z_front)
    p2 = V3.make(x_right_top, y_top, z_front)
    p3 = V3.make(x_left_top, y_top, z_front)
    p4 = V3.make(x_left_bottom, y_bottom, z_back)
    p5 = V3.make(x_right_bottom, y_bottom, z_back)
    p6 = V3.make(x_right_top, y_top, z_back)
    p7 = V3.make(x_left_top, y_top, z_back)

    V, T = MESH.create_cuboid(p0, p1, p2, p3, p4, p5, p6, p7)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name_left_short, mesh)

    shape_name_right_long = API.generate_unique_name("long_right_plank_shape")
    x_right_bottom = 0.0
    x_left_bottom = -pillar_width / 2.0
    x_right_top = -pillar_width
    x_left_top = -pillar_width * 1.5
    z_front = 0.0
    z_back = -6.5 * pillar_depth
    p0 = V3.make(x_left_bottom, y_bottom, z_front)
    p1 = V3.make(x_right_bottom, y_bottom, z_front)
    p2 = V3.make(x_right_top, y_top, z_front)
    p3 = V3.make(x_left_top, y_top, z_front)
    p4 = V3.make(x_left_bottom, y_bottom, z_back)
    p5 = V3.make(x_right_bottom, y_bottom, z_back)
    p6 = V3.make(x_right_top, y_top, z_back)
    p7 = V3.make(x_left_top, y_top, z_back)

    V, T = MESH.create_cuboid(p0, p1, p2, p3, p4, p5, p6, p7)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name_right_long, mesh)

    shape_name_right_short = API.generate_unique_name("short_right_plank_shape")
    x_right_bottom = 0.0
    x_left_bottom = -pillar_width / 2.0
    x_right_top = -pillar_width
    x_left_top = -pillar_width * 1.5
    z_front = 0.0
    z_back = -6.0 * pillar_depth
    p0 = V3.make(x_left_bottom, y_bottom, z_front)
    p1 = V3.make(x_right_bottom, y_bottom, z_front)
    p2 = V3.make(x_right_top, y_top, z_front)
    p3 = V3.make(x_left_top, y_top, z_front)
    p4 = V3.make(x_left_bottom, y_bottom, z_back)
    p5 = V3.make(x_right_bottom, y_bottom, z_back)
    p6 = V3.make(x_right_top, y_top, z_back)
    p7 = V3.make(x_left_top, y_top, z_back)

    V, T = MESH.create_cuboid(p0, p1, p2, p3, p4, p5, p6, p7)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name_right_short, mesh)

    for k in range(0, K_pillars - 1, 3):

        if k == 0:
            shape_name_left = shape_name_left_long
            shape_name_right = shape_name_right_long
        elif k == K_pillars - 4:
            shape_name_left = shape_name_left_long
            shape_name_right = shape_name_right_long
        else:
            shape_name_left = shape_name_left_short
            shape_name_right = shape_name_right_short

        if k == 0:
            z_plank = -k * 2.0 * pillar_depth + temple_depth / 2.0
        else:
            z_plank = -k * 2.0 * pillar_depth - pillar_depth / 2.0 + temple_depth / 2.0

        for i in range(I_pillars - 1):
            body_name = API.generate_unique_name("plank_body")
            body_names.append(body_name)
            API.create_rigid_body(engine, body_name)
            API.connect_shape(engine, body_name, shape_name_left)

            y_plank = (
                    3.0 * stairs_height
                    + pillar_height
                    + epistyle_height
                    + i * gable_brick_height
            )
            q_g = Q.identity()

            x_left_plank = -temple_width / 2.0 + i * gable_brick_width * 0.5
            r_g = V3.make(x_left_plank, y_plank, z_plank)

            API.set_position(engine, body_name, r_g, True)
            API.set_orientation(engine, body_name, q_g, True)

            body_name = API.generate_unique_name("plank_body")
            body_names.append(body_name)
            API.create_rigid_body(engine, body_name)
            API.connect_shape(engine, body_name, shape_name_right)

            x_right_plank = temple_width / 2.0 - i * gable_brick_width * 0.5
            r_g = V3.make(x_right_plank, y_plank, z_plank)

            API.set_position(engine, body_name, r_g, True)
            API.set_orientation(engine, body_name, q_g, True)

    shape_name = API.generate_unique_name("ridge_brick_shape")
    p0 = V3.make(-pillar_width / 2.0, 0.0, 0.0)
    p1 = V3.make(pillar_width / 2.0, 0.0, 0.0)
    p2 = V3.make(0.0, gable_brick_height / 2.0, 0.0)
    p3 = V3.make(-pillar_width / 2.0, 0.0, pillar_depth)
    p4 = V3.make(pillar_width / 2.0, 0.0, pillar_depth)
    p5 = V3.make(0.0, gable_brick_height / 2.0, pillar_depth)

    V, T = MESH.create_convex_hull([p0, p1, p2, p3, p4, p5])
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    for k in range(K_pillars * 2 - 1):
        body_name = API.generate_unique_name("ridge_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        z = temple_depth / 2.0 - (k + 1) * pillar_depth
        x = 0
        y = (
                3.0 * stairs_height
                + pillar_height
                + epistyle_height
                + (I_pillars - 1) * gable_brick_height
        )
        r = V3.make(x, y, z)
        q = Q.identity()

        API.set_position(engine, body_name, r, True)
        API.set_orientation(engine, body_name, q, True)

    for body_name in body_names:
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    return body_names
