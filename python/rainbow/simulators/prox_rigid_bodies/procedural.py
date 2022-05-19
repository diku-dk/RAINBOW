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
    API.create_shape(engine, shape_name, mesh)

    shape_name_edge = API.generate_unique_name("gable_edge_brick_shape")
    p0 = V3.make(0.0, y_bottom, z_front)
    p1 = V3.make(x_right, y_bottom, z_front)
    p2 = V3.make(x_right, y_top, z_front)
    p3 = V3.make(0.0, y_bottom, z_back)
    p4 = V3.make(x_right, y_bottom, z_back)
    p5 = V3.make(x_right, y_top, z_back)

    V, T = MESH.create_convex_hull([p0, p1, p2, p3, p4, p5])
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

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
    API.create_shape(engine, shape_name, mesh)

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
    API.create_shape(engine, shape_name, mesh)

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
    API.create_shape(engine, shape_name, mesh)

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
    API.create_shape(engine, shape_name, mesh)

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
        False,
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
