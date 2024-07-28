
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

