
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
