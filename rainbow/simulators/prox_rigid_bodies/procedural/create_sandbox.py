
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

