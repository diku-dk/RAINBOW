
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

