
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

