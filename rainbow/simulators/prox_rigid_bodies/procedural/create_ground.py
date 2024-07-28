
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

