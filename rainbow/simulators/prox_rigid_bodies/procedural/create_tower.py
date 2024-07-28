
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

