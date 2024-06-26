import numpy as np
import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
from math import pi, sin, sqrt
import igl


def create_ground(engine, r, q, density, material_name):
    """

    :param engine:
    :param r:
    :param q:
    :param density:
    :param material_name:
    :return:
    """
    shape_name = API.generate_unique_name("ground_shape")

    V, T = MESH.create_box(100.0, 1.0, 100.0)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("ground_body")
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r_l = V3.make(0.0, -0.5, 0.0)
    q_l = Q.identity()
    r_g = Q.rotate(q, r_l) + r
    q_g = Q.prod(q, q_l)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    API.set_body_type(engine, body_name, "fixed")
    API.set_body_material(engine, body_name, material_name)
    API.set_mass_properties(engine, body_name, density)


def create_pillar(engine, r, q, width, height, depth, stones, density, material_name):
    """

    :param engine:
    :param r:
    :param q:
    :param width:
    :param height:
    :param depth:
    :param stones:
    :param density:
    :param material_name:
    :return:
    """
    pillar_width = width
    pillar_depth = depth
    pillar_height = height
    pillar_stones = stones

    body_names = []

    base_height = pillar_height / 15
    top_height = pillar_height / 20
    pillar_base_radius = 0.9 * (pillar_width / 2.0)
    pillar_top_radius = 0.8 * (pillar_width / 2.0)
    pillar_radius_start = 0.7 * pillar_base_radius
    pillar_radius_stop = 0.5 * pillar_base_radius
    cap_height = base_height * 0.9
    cup_height = base_height * 0.7

    shape_name = API.generate_unique_name("socle_shape")
    V, T = MESH.create_box(pillar_width, base_height, pillar_depth)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("socle_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r_l = V3.make(0.0, base_height / 2.0, 0.0)
    q_l = Q.identity()
    r_g = Q.rotate(q, r_l) + r
    q_g = Q.prod(q, q_l)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    shape_name = API.generate_unique_name("abacus_shape")
    V, T = MESH.create_box(pillar_width, top_height, pillar_depth)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("abacus_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r_l = V3.make(0.0, pillar_height - top_height / 2.0, 0.0)
    q_l = Q.identity()
    r_g = Q.rotate(q, r_l) + r
    q_g = Q.prod(q, q_l)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    shape_name = API.generate_unique_name("base_shape")
    V, T = MESH.create_conical(pillar_base_radius, pillar_radius_start, cap_height)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("base_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r_l = V3.make(0.0, base_height, 0.0)
    q_l = Q.identity()
    r_g = Q.rotate(q, r_l) + r
    q_g = Q.prod(q, q_l)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    shape_name = API.generate_unique_name("echinus_shape")
    V, T = MESH.create_conical(pillar_radius_stop, pillar_top_radius, cup_height)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("echinus_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r_l = V3.make(0.0, pillar_height - top_height - cup_height, 0.0)
    q_l = Q.identity()
    r_g = Q.rotate(q, r_l) + r
    q_g = Q.prod(q, q_l)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    start_y = base_height + cap_height
    end_y = pillar_height - top_height - cup_height
    dy = (end_y - start_y) / pillar_stones
    slope = (pillar_radius_stop - pillar_radius_start) / (end_y - start_y)
    for stone in range(pillar_stones):
        bottom_y = start_y + stone * dy
        top_y = bottom_y + dy
        bottom_radius = pillar_radius_start + slope * (bottom_y - start_y)
        top_radius = pillar_radius_start + slope * (top_y - start_y)

        shape_name = API.generate_unique_name("shaft_stone")
        V, T = MESH.create_conical(bottom_radius, top_radius, dy)
        mesh = API.create_mesh(V, T)
        API.create_shape(engine, shape_name, mesh)

        body_name = API.generate_unique_name("shaft_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        r_l = V3.make(0.0, bottom_y, 0.0)
        q_l = Q.identity()
        r_g = Q.rotate(q, r_l) + r
        q_g = Q.prod(q, q_l)

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

    for body_name in body_names:
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    return body_names


def create_arch(
        engine,
        r,
        q,
        width,
        height,
        depth,
        pier_stones,
        arch_stones,
        density,
        material_name,
):
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

    delta_phi = pi / arch_stones
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


def create_dome(
        engine, r, q, outer_radius, inner_radius, layers, segments, density, material_name,
):
    r_i = inner_radius
    r_o = outer_radius
    d_theta = (2.0 * pi) / segments
    d_phi = (pi / 2.0) / (layers + 1)

    body_names = []
    for l in range(layers):
        theta = 0.0
        phi = d_phi * l
        shape_name = API.generate_unique_name("brick_shape")

        q0 = Q.prod(Q.Ry(theta), Q.Rz(phi))
        q1 = Q.prod(Q.Ry(theta + d_theta), Q.Rz(phi))
        q2 = Q.prod(Q.Ry(theta + d_theta), Q.Rz(phi + d_phi))
        q3 = Q.prod(Q.Ry(theta), Q.Rz(phi + d_phi))
        p0 = Q.rotate(q0, V3.make(r_o, 0.0, 0.0))
        p1 = Q.rotate(q1, V3.make(r_o, 0.0, 0.0))
        p2 = Q.rotate(q2, V3.make(r_o, 0.0, 0.0))
        p3 = Q.rotate(q3, V3.make(r_o, 0.0, 0.0))
        p4 = Q.rotate(q0, V3.make(r_i, 0.0, 0.0))
        p5 = Q.rotate(q1, V3.make(r_i, 0.0, 0.0))
        p6 = Q.rotate(q2, V3.make(r_i, 0.0, 0.0))
        p7 = Q.rotate(q3, V3.make(r_i, 0.0, 0.0))

        V, T = MESH.create_cuboid(p0, p1, p2, p3, p4, p5, p6, p7)
        mesh = API.create_mesh(V, T)
        API.create_shape(engine, shape_name, mesh)

        for s in range(segments):
            theta = d_theta * s

            body_name = API.generate_unique_name("body")
            body_names.append(body_name)
            API.create_rigid_body(engine, body_name)
            API.connect_shape(engine, body_name, shape_name)

            q_l = Q.Ry(theta)
            r_l = Q.rotate(q_l, V3.make(0.0, 0.0, 0.0))
            r_g = Q.rotate(q, r_l) + r
            q_g = Q.prod(q, q_l)

            API.set_position(engine, body_name, r_g, True)
            API.set_orientation(engine, body_name, q_g, True)

    for body_name in body_names:
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    return body_names


def create_tower(
        engine,
        r,
        q,
        outer_radius,
        inner_radius,
        height,
        layers,
        segments,
        use_cubes,
        density,
        material_name,
        safety_factor=0.99,
):
    center_radius = (outer_radius + inner_radius) / 2.0
    brick_height = height / layers
    brick_depth = outer_radius - inner_radius
    delta_phi = (2.0 * pi) / segments

    shape_name = API.generate_unique_name("brick_shape")
    if use_cubes:
        brick_width = (
                2.0
                * (sin(delta_phi / 2.0) / sin(pi / 2.0 - delta_phi / 2.0))
                * inner_radius
                * safety_factor
        )

        V, T = MESH.create_box(brick_width, brick_height, brick_depth)
        mesh = API.create_mesh(V, T)
        API.create_shape(engine, shape_name, mesh)

    else:
        brick_inner_half_width = (
                (sin(delta_phi / 2.0) / sin(pi / 2.0 - delta_phi / 2.0))
                * inner_radius
                * safety_factor
        )
        brick_outer_half_width = (
                (sin(delta_phi / 2.0) / sin(pi / 2.0 - delta_phi / 2.0))
                * outer_radius
                * safety_factor
        )
        p0 = V3.make(-brick_inner_half_width, -brick_height / 2.0, brick_depth / 2.0)
        p1 = V3.make(brick_inner_half_width, -brick_height / 2.0, brick_depth / 2.0)
        p2 = V3.make(brick_inner_half_width, brick_height / 2.0, brick_depth / 2.0)
        p3 = V3.make(-brick_inner_half_width, brick_height / 2.0, brick_depth / 2.0)
        p4 = V3.make(-brick_outer_half_width, -brick_height / 2.0, -brick_depth / 2.0)
        p5 = V3.make(brick_outer_half_width, -brick_height / 2.0, -brick_depth / 2.0)
        p6 = V3.make(brick_outer_half_width, brick_height / 2.0, -brick_depth / 2.0)
        p7 = V3.make(-brick_outer_half_width, brick_height / 2.0, -brick_depth / 2.0)

        V, T = MESH.create_cuboid(p0, p1, p2, p3, p4, p5, p6, p7)
        mesh = API.create_mesh(V, T)
        API.create_shape(engine, shape_name, mesh)

    body_names = []
    for layer in range(layers):
        height = brick_height / 2.0 + layer * brick_height
        for segment in range(segments):

            body_name = API.generate_unique_name("body")
            body_names.append(body_name)
            API.create_rigid_body(engine, body_name)
            API.connect_shape(engine, body_name, shape_name)

            phi = delta_phi * segment
            if (layer % 2) == 0:
                phi += delta_phi / 2.0

            r_l = V3.make(0.0, height, -center_radius)
            q_l = Q.Ry(phi)
            r_l = Q.rotate(q_l, r_l)

            r_g = Q.rotate(q, r_l) + r
            q_g = Q.prod(q, q_l)

            API.set_position(engine, body_name, r_g, True)
            API.set_orientation(engine, body_name, q_g, True)

    for body_name in body_names:
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    return body_names


def create_colosseum(
        engine,
        r,
        q,
        outer_radius,
        inner_radius,
        height,
        levels,
        arches,
        density,
        material_name,
        safety_factor=0.99,
):
    center_radius = (outer_radius + inner_radius) / 2.0
    level_height = height / levels
    separation_height = level_height * 0.1
    arch_height = level_height * 0.9
    arch_depth = outer_radius - inner_radius
    delta_phi = (2.0 * pi) / arches
    arch_width = (
            2.0
            * (sin(delta_phi / 2.0) / sin(pi / 2.0 - delta_phi / 2.0))
            * inner_radius
            * safety_factor
    )

    body_names = []
    for l in range(levels):
        height = l * level_height
        for a in range(arches):
            phi = delta_phi * a
            r_l = V3.make(0.0, height, -center_radius)
            q_l = Q.Ry(phi)
            r_l = Q.rotate(q_l, r_l)
            r_g = Q.rotate(q, r_l) + r
            q_g = Q.prod(q, q_l)
            names = create_arch(
                engine,
                r_g,
                q_g,
                arch_width,
                arch_height,
                arch_depth,
                5,
                9,
                density,
                material_name,
            )
            body_names.extend(names)
        r_l = V3.make(0.0, height + arch_height, 0.0)
        q_l = Q.identity()
        r_g = Q.rotate(q, r_l) + r
        q_g = Q.prod(q, q_l)
        names = create_tower(
            engine,
            r_g,
            q_g,
            outer_radius,
            inner_radius,
            separation_height,
            1,
            arches,
            False,
            density,
            material_name,
            safety_factor,
        )
        body_names.extend(names)

    return body_names


def create_pantheon(
        engine,
        r,
        q,
        outer_radius,
        inner_radius,
        height,
        layers,
        segments,
        density,
        material_name,
        safety_factor=0.99,
):
    wall_height = height - outer_radius
    wall_layers = layers
    dome_layers = layers

    if wall_height <= 0.0:
        raise RuntimeWarning("dome height must be larger than outer radius")

    body_names = []

    names = create_tower(
        engine,
        r,
        q,
        outer_radius,
        inner_radius,
        wall_height,
        wall_layers,
        segments,
        False,
        density,
        material_name,
        safety_factor,
    )
    body_names.extend(names)

    r_l = V3.make(0.0, wall_height, 0.0)
    q_l = Q.identity()
    r_g = Q.rotate(q, r_l) + r
    q_g = Q.prod(q, q_l)
    names = create_dome(
        engine,
        r_g,
        q_g,
        outer_radius,
        inner_radius,
        dome_layers,
        segments,
        density,
        material_name,
    )
    body_names.extend(names)

    return body_names


def _create_grid(
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
        use_random_orientation=True,
        q_body=None,
):
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


def create_funnel(
        engine,
        funnel_height,
        funnel_radius,
        grid_width,
        grid_height,
        grid_depth,
        I,
        J,
        K,
        density,
        material_name,
        use_random_orientation=True,
):
    height = grid_height
    width = grid_width
    depth = grid_depth

    shape_names = []
    shape_name = API.generate_unique_name("jack")

    V, T = igl.read_triangle_mesh("../data/jack.obj")
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
    V, T = igl.read_triangle_mesh("../data/funnel.obj")
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
    q = Q.Rx(pi)
    API.set_position(engine, body_name, r, True)
    API.set_orientation(engine, body_name, q, True)

    API.set_body_type(engine, body_name, "fixed")
    API.set_body_material(engine, body_name, material_name)
    API.set_mass_properties(engine, body_name, density)

    return body_names


def create_glasses(
        engine,
        glass_height,
        glass_radius,
        grid_width,
        grid_height,
        grid_depth,
        I,
        J,
        K,
        density,
        material_name,
        use_random_orientation=True,
):
    height = grid_height
    width = grid_width
    depth = grid_depth

    shape_names = []
    shape_name = API.generate_unique_name("small_glass")
    V, T = igl.read_triangle_mesh("../data/glass.obj")
    mesh = API.create_mesh(V, T)

    MESH.scale_to_unit(mesh)
    s = (
            min(width / I, height / J, depth / K) * 0.9
    )  # The 0.9 scaling ensure some padding to avoid initial contact
    MESH.scale(mesh, s, s, s)

    API.create_shape(engine, shape_name, mesh)
    shape_names.append(shape_name)

    r = V3.make(-width / 2.0, 1.25 * glass_height, -depth / 2.0)
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
        Q.Rx(-pi / 2.0),
    )

    shape_name = API.generate_unique_name("glass")
    V, T = igl.read_triangle_mesh("../data/glass.obj")
    mesh = API.create_mesh(V, T)

    MESH.scale_to_unit(mesh)
    (l, u) = MESH.aabb(mesh)
    extends = u - l
    MESH.scale(
        mesh,
        (glass_radius * 2.0) / extends[0],
        glass_height / extends[1],
        (glass_radius * 2.0) / extends[2],
    )
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r = V3.make(0.0, glass_height / 2.0, 0.0)
    API.set_position(engine, body_name, r, True)
    API.set_orientation(engine, body_name, q, True)

    API.set_body_type(engine, body_name, "fixed")
    API.set_body_material(engine, body_name, material_name)
    API.set_mass_properties(engine, body_name, density)

    return body_names


def create_poles(
        engine,
        pole_height,
        pole_radius,
        I_poles,
        K_poles,
        grid_width,
        grid_height,
        grid_depth,
        I_grid,
        J_grid,
        K_grid,
        density,
        material_name,
        use_random_orientation=True,
):
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
        V, T = igl.read_triangle_mesh(file)
        mesh = API.create_mesh(V, T)
        MESH.scale_to_unit(mesh)
        s = min(width / I_grid, height / J_grid, depth / K_grid)
        MESH.scale(mesh, s, s, s)
        API.create_shape(engine, shape_name, mesh)
        shape_names.append(shape_name)

    r = V3.make(-width / 2.0, 2.0 * pole_height, -depth / 2.0)
    q = Q.identity()
    body_names = _create_grid(
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


def create_temple(
        engine,
        I_pillars,
        K_pillars,
        pillar_width,
        pillar_height,
        pillar_depth,
        pillar_stones,
        density,
        material_name,
):
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
        q_g = Q.Ry(pi / 2.0)

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

        body_name = API.generate_unique_name("epistyle_brick_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        x_left = -temple_width / 2.0 + pillar_width / 2.0
        r_g = V3.make(x_left, y, z)
        q_g = Q.Ry(pi / 2.0)

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
    q_g = Q.Rz(pi)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    body_name = API.generate_unique_name("epistyle_brick_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    x = -temple_width / 2.0 + pillar_width * 1.5
    z = -temple_depth / 2.0 + pillar_depth * 0.5
    r_g = V3.make(x, y, z)
    q_g = Q.Rx(pi)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    body_name = API.generate_unique_name("epistyle_brick_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    x = temple_width / 2.0 - pillar_width * 1.5
    z = -temple_depth / 2.0 + pillar_depth * 0.5
    r_g = V3.make(x, y, z)
    q_g = Q.Ry(-pi)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    body_name = API.generate_unique_name("epistyle_brick_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    x = temple_width / 2.0 - pillar_width * 0.5
    z = temple_depth / 2.0 - pillar_depth * 1.5
    r_g = V3.make(x, y, z)
    q_g = Q.Ry(pi / 2.0)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    body_name = API.generate_unique_name("epistyle_brick_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    x = -temple_width / 2.0 + pillar_width * 0.5
    z = temple_depth / 2.0 - pillar_depth * 1.5
    r_g = V3.make(x, y, z)
    q_g = Q.prod(Q.Ry(pi / 2.0), Q.Rx(pi))

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    body_name = API.generate_unique_name("epistyle_brick_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    x = -temple_width / 2.0 + pillar_width * 0.5
    z = -temple_depth / 2.0 + pillar_depth * 1.5
    r_g = V3.make(x, y, z)
    q_g = Q.Ry(-pi / 2.0)

    API.set_position(engine, body_name, r_g, True)
    API.set_orientation(engine, body_name, q_g, True)

    body_name = API.generate_unique_name("epistyle_brick_body")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    x = temple_width / 2.0 - pillar_width * 0.5
    z = -temple_depth / 2.0 + pillar_depth * 1.5
    r_g = V3.make(x, y, z)
    q_g = Q.prod(Q.Ry(-pi / 2.0), Q.Rx(pi))

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
        q_g = Q.Ry(pi)

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
        q_g = Q.Rx(pi)

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

        body_name = API.generate_unique_name("epistyle_brick_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        r_g = V3.make(x_right, y, z_front)
        q_g = Q.Rz(pi)

        API.set_position(engine, body_name, r_g, True)
        API.set_orientation(engine, body_name, q_g, True)

        body_name = API.generate_unique_name("epistyle_brick_body")
        body_names.append(body_name)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        r_g = V3.make(x_right, y, z_back)
        q_g = Q.Ry(pi)

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
                    q_g = Q.Ry(pi)
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


def create_sandbox(
        engine,
        box_width,
        box_height,
        box_depth,
        I_grains,
        J_grains,
        K_grains,
        density,
        material_name,
):
    thickness = 0.1 * box_height

    shape_names = []
    for n in range(20):
        shape_name = API.generate_unique_name("grain")
        points = [V3.rand(-1.0, 1.0) for _ in range(20)]

        V, T = MESH.create_convex_hull(points)
        mesh = API.create_mesh(V, T)
        MESH.scale_to_unit(mesh)
        s = (
                min(box_width / I_grains, box_height / J_grains, box_depth / K_grains) * 0.9
        )  # The 0.9 scaling ensure some padding to avoid initial contact
        MESH.scale(mesh, s, s, s)
        API.create_shape(engine, shape_name, mesh)

        shape_names.append(shape_name)

    r = V3.make(-box_width / 2.0, 1.5 * box_height + thickness, -box_depth / 2.0)
    q = Q.identity()
    body_names = _create_grid(
        engine,
        r,
        q,
        shape_names,
        box_width,
        box_height,
        box_depth,
        I_grains,
        J_grains,
        K_grains,
        density,
        material_name,
        use_random_orientation=True
    )

    shape_name = API.generate_unique_name("box_left")
    V, T = MESH.create_box(thickness, box_height, box_depth)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("body_left")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r = V3.make(-box_width / 2.0 - thickness / 2.0, box_height / 2.0 + thickness, 0.0)
    q = Q.identity()

    API.set_position(engine, body_name, r, True)
    API.set_orientation(engine, body_name, q, True)
    API.set_body_type(engine, body_name, "fixed")
    API.set_body_material(engine, body_name, material_name)
    API.set_mass_properties(engine, body_name, density)

    shape_name = API.generate_unique_name("box_right")
    V, T = MESH.create_box(thickness, box_height, box_depth)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("body_right")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r = V3.make(box_width / 2.0 + thickness / 2.0, box_height / 2.0 + thickness, 0.0)
    q = Q.identity()

    API.set_position(engine, body_name, r, True)
    API.set_orientation(engine, body_name, q, True)
    API.set_body_type(engine, body_name, "fixed")
    API.set_body_material(engine, body_name, material_name)
    API.set_mass_properties(engine, body_name, density)

    shape_name = API.generate_unique_name("box_front")
    V, T = MESH.create_box(thickness * 2.0 + box_width, box_height, thickness)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("body_front")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r = V3.make(0.0, box_height / 2.0 + thickness, box_depth / 2.0 + thickness / 2.0)
    q = Q.identity()

    API.set_position(engine, body_name, r, True)
    API.set_orientation(engine, body_name, q, True)
    API.set_body_type(engine, body_name, "fixed")
    API.set_body_material(engine, body_name, material_name)
    API.set_mass_properties(engine, body_name, density)

    shape_name = API.generate_unique_name("box_back")
    V, T = MESH.create_box(thickness * 2.0 + box_width, box_height, thickness)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("body_back")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r = V3.make(0.0, box_height / 2.0 + thickness, -box_depth / 2.0 - thickness / 2.0)
    q = Q.identity()

    API.set_position(engine, body_name, r, True)
    API.set_orientation(engine, body_name, q, True)
    API.set_body_type(engine, body_name, "fixed")
    API.set_body_material(engine, body_name, material_name)
    API.set_mass_properties(engine, body_name, density)

    shape_name = API.generate_unique_name("base")
    V, T = MESH.create_box(
        thickness * 2.0 + box_width, thickness, box_depth + 2.0 * thickness
    )
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("base")
    body_names.append(body_name)
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r = V3.make(0.0, thickness / 2.0, 0.0)
    q = Q.identity()

    API.set_position(engine, body_name, r, True)
    API.set_orientation(engine, body_name, q, True)
    API.set_body_type(engine, body_name, "fixed")
    API.set_body_material(engine, body_name, material_name)
    API.set_mass_properties(engine, body_name, density)

    return body_names


def create_chainmail(
        engine,
        major_radius,
        minor_radius,
        width,
        height,
        stretch: float = 0.5,
        density: float = 1.0,
        material_name: str = "default",
        pinned: bool = True
):
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

    q_m2w = Q.Rx(-np.pi / 2)  # Needed to change z-up direction to y-up direction
    r_m2w = V3.make(- delta * (width) / 2
                    , 0
                    , delta * (height) / 2
                    )  # Needed to center chain mail at origin of world

    for j in range(height + 1):
        for i in range(width + 1):
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

    for j in range(height + 1):
        for i in range(width):
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

    for j in range(height):
        for i in range(width + 1):
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
        south_east_ring = body_names[width]
        north_west_ring = body_names[(height + 1) * (width + 1) - width - 1]
        north_east_ring = body_names[(height + 1) * (width + 1) - 1]
        API.set_body_type(engine, south_west_ring, "fixed")
        API.set_body_type(engine, south_east_ring, "fixed")
        API.set_body_type(engine, north_west_ring, "fixed")
        API.set_body_type(engine, north_east_ring, "fixed")

    return body_names


def create_jack_grid(engine,
                     r,
                     q,
                     width: float,
                     height: float,
                     depth: float,
                     I: int,
                     J: int,
                     K: int,
                     density: float = 1.0,
                     material_name: str = "default",
                     use_random_orientation: bool = False
                     ):
    shape_names = []
    shape_name = API.generate_unique_name("jack")

    V, T = igl.read_triangle_mesh("../data/jack.obj")
    mesh = API.create_mesh(V, T)
    MESH.scale_to_unit(mesh)
    s = (
            min(width / I, height / J, depth / K) * 0.9
    )  # The 0.9 scaling ensure some padding to avoid initial contact
    MESH.scale(mesh, s, s, s)
    API.create_shape(engine, shape_name, mesh)

    shape_names.append(shape_name)

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


class GearFactory:

    @staticmethod
    def _roll_angle(r_base: float, r_top: float):
        """
        Compute the roll angle.

        :param r_base:  The radius of the inner circle.
        :param r_top:   The radius of the outer circle.
        :return:        The roll angle of the inner circle that will create an involute curve connecting to the outer circle.
        """
        return np.sqrt((r_top / r_base) ** 2 - 1)

    @staticmethod
    def _span_angle(roll_angle: float):
        """
        This function computes the span angle of the involute curve. That is if
        projected onto its base circle how big an angle does the resulting
        circle arch span?

        :param roll_angle:
        :return:
        """
        x = (np.cos(roll_angle) + roll_angle * np.sin(roll_angle))
        y = (np.sin(roll_angle) - roll_angle * np.cos(roll_angle))
        return np.arctan2(y, x)

    @staticmethod
    def _make_involute_curve(r_base: float, r_top: float, shift: float = 0, reverse: bool = False):
        """
        This function is used to create involute curve.
        Let us look at the involute curve going in the ccw direction (we call this the forward direction).
        The opposing direction we refer to the reverse direction.

        The direction that the involute curve should wound around the inner circle. CCW is forward and CW is reserve.


        A point on the involute curve is given by

            q(theta) = r_base (cos(theta), sin(theta))^T + r_base theta (sin(theta), -cos(theta))^T

        Solving for the root of

            norm(q(beta))^2 - r_top^2 = 0

        Gives us the roll angle beta. This function create the involute curve going from 0 to beta which will
        connect the circle with radius r_base to the circle with radius r_top.

        :param r_base:  The radius of the inner circle.
        :param r_top:   The radius of the outer circle.
        :param shift:   The shifting angle for where the involute curve should start on the inner circle.
        :param reverse: A boolean flag indicating if the involute curve should be reversed.
        :return:        A curve represented as tuple of two numpy arrays. The First one is x-coordinates, and the second is y-coordinates of points on the curve.
        """
        theta = np.linspace(0, GearFactory._roll_angle(r_base, r_top), 12)
        if reverse:
            theta = -theta[::-1]
        ix = r_base * (np.cos(theta + shift) + theta * np.sin(theta + shift))
        iy = r_base * (np.sin(theta + shift) - theta * np.cos(theta + shift))
        return ix, iy

    @staticmethod
    def make_gear_specs(m: float = 1.0, Z: int = 12, alpha: float = 20.0):
        """
        This function is used to create a gear geometry profile.

        For assembling gears remember that mating gears must share the same module. They
        must also have the same pressure angle. Their pitch circles must be tangent.

        The code is based on descriptions from https://www.stlgears.com/ and https://khkgears.net/new/gear_knowledge/

        :param m: Module value
        :param Z:  Number of teeth of the gear
        :param alpha: The pressure angle given in units of degrees

        :return: A tuple with the first element being the pitch circle radius.
        """
        # First, we compute a lot of gear parameters that are used for defining the geometry of the gear.
        rad = np.pi * alpha / 180.0  # Convert pressure angle to radians
        h_a = m  # The distance from the reference circle to tooth tip
        h_d = 1.25 * m  # Distance from reference circle to root of the teeth
        h = 2.25 * m  # Total tooth height
        p = np.pi * m  # The distance between corresponding points on reference circle for adjacent tooths
        s = p / 2  # Tooth thickness on reference circle
        R_p = m * Z / 2.0  # Radius of the reference circle also known as the pitch circle
        R_r = R_p - h_d  # Radius of the dedendum circle (root circle)
        R_t = R_p + h_a  # Radius of the addendum circle (tip circle)
        R_b = R_p * np.cos(rad)  # Radius of the base circle which is the basis for the involute curve

        # Generate teeth profile
        pitch_roll = GearFactory._roll_angle(R_b, R_p)
        top_roll = GearFactory._roll_angle(R_b, R_t)

        delta = GearFactory._span_angle(
            pitch_roll)  # Angle spanned by involute curve going from base circle to pitch circle
        beta = np.pi / Z  # Angle spanned by tooth thickness at the pitch circle
        gamma = GearFactory._span_angle(
            top_roll) - delta  # Angle spanned by involute curve going from pitch circle to top circle

        corners = []
        gear_x_list = []
        gear_y_list = []
        for z in range(Z):
            shift = z * beta * 2
            forward_shift = shift - delta
            reverse_shift = shift + delta + beta
            # Generate the involute curve going from base circle to the top circle
            forward_x, forward_y = GearFactory._make_involute_curve(r_base=R_b, r_top=R_t, shift=forward_shift)
            # Generate top circle arch between the forward and reverse involute curves
            start_top_theta = shift + gamma
            end_top_theta = shift + beta - gamma
            theta = np.linspace(start_top_theta, end_top_theta, 6)
            top_x = R_t * np.cos(theta)
            top_y = R_t * np.sin(theta)
            # Generate the involute curve going from top circle to the base circle
            rev_x, rev_y = GearFactory._make_involute_curve(r_base=R_b, r_top=R_t, shift=reverse_shift, reverse=True)
            # Generate the circular arch on root circle from this tooth end to the next tooth start point.
            start_root_theta = shift + beta + delta
            end_root_theta = shift + 2 * beta - delta
            theta = np.linspace(start_root_theta, end_root_theta, 6)
            root_x = R_r * np.cos(theta)
            root_y = R_r * np.sin(theta)
            # Piece all the tooth curves together to one curve
            tooth_x = np.concatenate((forward_x, top_x[1:-1], rev_x, root_x))
            tooth_y = np.concatenate((forward_y, top_y[1:-1], rev_y, root_y))
            # Put indices of tooth root vertices into a list for later
            tooth_start_idx = len(gear_x_list) - 1
            tooth_end_idx = len(gear_x_list) + len(tooth_x) - len(root_x)
            corners.append([tooth_start_idx, tooth_end_idx])
            # Append the tooth curve to all previous generated teeth
            gear_x_list.extend(tooth_x)
            gear_y_list.extend(tooth_y)

        gear_x = np.array(gear_x_list, dtype=float)
        gear_y = np.array(gear_y_list, dtype=float)

        # Store all the technical specifications about the gear
        specs = {}
        specs['pitch radius'] = R_p
        specs['top radius'] = R_t
        specs['base radius'] = R_b
        specs['root radius'] = R_r
        specs['tooth thickness'] = s
        specs['pitch'] = p
        specs['tooth height'] = h
        specs['module'] = m
        specs['number of teeth'] = Z
        specs['pressure angle'] = alpha
        specs['beta'] = beta  # Tooth thickness angle span on pitch circle
        specs['gamma'] = gamma  # Involute angle span from pitch to top circle
        specs['delta'] = delta  # Involute angle span from base to pitch circle
        specs['x'] = gear_x  # The x coordinates of the gear profile
        specs['y'] = gear_y  # The y coordinates of the gear profile
        specs['corners'] = np.array(corners, dtype=int)
        return specs

    @staticmethod
    def make_gears_assembly(drive_gear: dict
                            , driven_gear: dict
                            , cx: float
                            , cy: float
                            , theta: float
                            , omega: float
                            ):
        """

        :param drive_gear:   A dictionary with specifications of the driving gear.
        :param driven_gear:  A dictionary with specifications of the driven gear.
        :param cx:           The world x coordinate of the center position of the drive gear.
        :param cy:           The world y coordinate of the center position of the drive gear.
        :param theta:        Current rotation of the drive gear
        :param omega:        The connection angle to the drive gear
        :return:
        """
        PR1 = drive_gear['pitch radius']
        PR2 = driven_gear['pitch radius']
        M1 = drive_gear['module']
        M2 = driven_gear['module']
        PA1 = drive_gear['pressure angle']
        PA2 = driven_gear['pressure angle']
        Z1 = drive_gear['number of teeth']
        Z2 = driven_gear['number of teeth']
        if M1 != M2:
            raise ValueError("Gears are not compatible, must have same module")
        if PA1 != PA2:
            raise ValueError("Gears are not compatible, must have same pressure angles")

        R = PR1 + PR2
        tx = np.cos(omega) * R + cx
        ty = np.sin(omega) * R + cy
        # If gear 1 rotates an angle beta1, then gear 2 will rotate beta2 = - ratio * beta1
        # where ration= Z1/Z2
        #
        # Pitch circle 1 has radius
        #
        #   r1 =  m * Z1 / 2.0
        #
        # Pitch circle 2 has radius
        #
        #   r2 =  m * Z2 / 2.0
        #
        # It must hold that if contact is not slipping between the two gears
        # then they "travel" the same distance as they rotate. So we must have
        #
        #   beta2 r2 = - beta1 r1
        #
        # The minus sign is there because the two gears are always rotating
        # opposite. So if beta1>0 then beta2<0 and vice versa.
        # Now substitution gives us
        #
        #   beta2 Z2 = - beta1 Z1
        #
        # And we find
        #
        #   beta2 = - (Z1/Z2) beta1
        #
        ratio = Z1 / Z2
        # If both gears are in neutral positions then rotating gear 2 by pi will align teeth of both gears.
        # However, gear 1 is not in neutral position, and gear 2 is not connecting to gear 1 with an angle of zero.
        # Hence, we must compute the relative connection angle of gear 2 to the current world location
        # of gear1's neutral position.
        phi = np.pi - ratio * (theta - omega) + omega
        return tx, ty, phi

    @staticmethod
    def make_gear_mesh(gear: dict, face_width: float = 1.0):
        x = gear['x']
        y = gear['y']
        z = np.ones_like(x) * face_width / 2

        V = np.row_stack(
            (
                np.column_stack((x, y, -z))
                , np.column_stack((x, y, z))
            )
        )

        N = len(x)
        T = []
        # Create mesh of teeth profile of the gear
        for i in range(N):
            j = (i + 1) % N
            k = N + j
            m = N + i
            T.append([i, j, k])
            T.append([i, k, m])

        # Create top and bottom caps of gear
        C = gear['corners']
        for start, stop in C:
            # First, we do bottom teeth
            indices = np.arange(start, stop + 1, 1, dtype=int)
            indices[indices < 0] += N

            tooth_center = np.mean(V[indices, :], axis=0)
            V = np.row_stack((V, tooth_center))
            idx_tooth_center = V.shape[0] - 1
            M = len(indices)
            for m in range(M):
                j = indices[m]
                k = indices[(m + 1) % M]
                T.append([idx_tooth_center, k, j])

            # Second, we do top teeth
            indices += N
            tooth_center = np.mean(V[indices, :], axis=0)
            V = np.row_stack((V, tooth_center))
            idx_tooth_center = V.shape[0] - 1
            M = len(indices)
            for m in range(M):
                j = indices[m]
                k = indices[(m + 1) % M]
                T.append([idx_tooth_center, j, k])

        cx = np.mean(x)
        cy = np.mean(y)
        bottom_center = np.array([cx, cy, -face_width / 2]).T
        top_center = np.array([cx, cy, face_width / 2]).T
        V = np.row_stack((V, bottom_center, top_center))
        idx_top = V.shape[0] - 1
        idx_bot = V.shape[0] - 2
        for start, stop in C:
            if start < 0:
                start += N
            T.append([idx_bot, stop, start])
            T.append([idx_top, start + N, stop + N])

        K = len(C)
        for k in range(K):
            start = C[k, 1]
            stop = C[(k + 1) % K, 0]
            if start < 0:
                start += N
            if stop < 0:
                stop += N
            for i in range(start, stop):
                j = i + 1
                T.append([idx_bot, j, i])
            for i in range(start, stop):
                j = i + 1
                T.append([idx_top, i + N, j + N])
        T = np.array(T, dtype=int)
        return V, T


def create_gear_train(engine
                      , N: int
                      , density: float = 1.0
                      , material_name: str = "default"
                      ):
    body_names = []
    gear_specs = []
    gear_names = []

    q_m2w = Q.Rx(-np.pi / 2)  # Needed to change z-up direction to y-up direction

    m = 1.0  # Gear module
    numbers = [7, 12, 15, 28]  # Possible gear teeth to use
    Z = np.random.choice(numbers, size=(N,))  # Random list of N numbers of gear teeth
    alpha = 20  # Pressure angle
    face_width = 10.0  # Width of the gear

    for i in range(N):
        specs = GearFactory.make_gear_specs(m=m, Z=Z[i], alpha=alpha)
        shape_name = API.generate_unique_name("shape")
        body_name = API.generate_unique_name("body")

        body_names.append(body_name)
        gear_names.append(shape_name)
        gear_specs.append(specs)

        V, T = GearFactory.make_gear_mesh(specs, face_width=face_width)

        mesh = API.create_mesh(V, T)
        API.create_shape(engine, shape_name, mesh)

        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        r_m = V3.make(0.0, 0.0, 0.0)  # Model space position of gear (z-up)
        q_m = Q.identity()  # Model space orientation of gear (z-up)
        r_w = Q.rotate(q_m2w, r_m)  # World position of gear (y-up)
        q_w = Q.prod(q_m2w, q_m)  # World orientation of gear (y-up)

        API.set_position(engine, body_name, r_w, True)
        API.set_orientation(engine, body_name, q_w, True)
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    cx = 0
    cy = 0
    cz = 0
    theta = 0
    for i in range(N - 1):
        drive_gear = gear_specs[i]
        driven_gear = gear_specs[i + 1]
        omega = np.random.uniform(0, 2 * np.pi)
        cx, cy, theta = GearFactory.make_gears_assembly(drive_gear, driven_gear, cx, cy, theta, omega)
        cz += face_width / 2
        r_m = V3.make(cx, cy, cz)  # Model space position of gear (z-up)
        q_m = Q.Rz(theta)  # Model space orientation of gear (z-up)
        r_w = Q.rotate(q_m2w, r_m)  # World position of gear (y-up)
        q_w = Q.prod(q_m2w, q_m)  # World orientation of gear (y-up)
        driven_gear_body_name = body_names[i + 1]
        API.set_position(engine, driven_gear_body_name, r_w, True)
        API.set_orientation(engine, driven_gear_body_name, q_w, True)

    return body_names


def create_rockslide(engine
                     , pile_width: float
                     , pile_height: float
                     , pile_depth: float
                     , I_rocks: int
                     , J_rocks: int
                     , K_rocks: int
                     , density: float = 1.0
                     , material_name: str = "default"
                     ):
    body_names = []

    shape_name = API.generate_unique_name("landscape")
    V, T = igl.read_triangle_mesh("../data/rockslide/mountains.obj")
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
        "../data/rockslide/building_01.obj",
        "../data/rockslide/building_02.obj",
        "../data/rockslide/building_03.obj",
        "../data/rockslide/building_04.obj",
        "../data/rockslide/building_05.obj",
        "../data/rockslide/building_06.obj",
        "../data/rockslide/building_07.obj",
        "../data/rockslide/building_08.obj",
        "../data/rockslide/building_09.obj",
        "../data/rockslide/building_10.obj",
        "../data/rockslide/building_11.obj",
        "../data/rockslide/building_12.obj",
        "../data/rockslide/building_13.obj",
        "../data/rockslide/building_14.obj",
        "../data/rockslide/building_15.obj",
        "../data/rockslide/building_16.obj",
        "../data/rockslide/building_17.obj",
        "../data/rockslide/building_18.obj",
        "../data/rockslide/building_19.obj",
        "../data/rockslide/building_20.obj"
    ]

    for filename in building_files:
        shape_name = API.generate_unique_name("shape")
        V, T = igl.read_triangle_mesh(filename)
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
        _create_grid(
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
