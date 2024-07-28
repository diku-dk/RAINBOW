
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

