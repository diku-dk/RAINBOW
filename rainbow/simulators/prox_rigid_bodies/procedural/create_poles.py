

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

