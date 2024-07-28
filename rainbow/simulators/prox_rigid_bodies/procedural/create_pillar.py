
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

