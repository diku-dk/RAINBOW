from soft_application import SoftBodyApplication


def run():
    """
    This function provides a demonstration of soft body simulation, specifically,
    a beam, fixed to a wall on one side, is affected by gravity, causing it to droop or lower on the unfixed side over time.

    Usage:
        To run the simulation, simply execute this function. Adjust parameters within to modify simulation behaviors,
        such as the material properties of the beam or the strength of gravity.
    """
    app = SoftBodyApplication()

    # Create a soft material named "default_material"
    app.create_soft_material("default_material", model_name="SNH")
    app.set_friction_coefficient("default_material", "default_material", 0.5)

    # Add the beam and wall into the scene
    app.create_soft_body("beam", "default_material", (12, 3, 3, 4.0, 1.0, 1.0))
    app.create_soft_body("wall", "default_material", (2, 2, 2, 0.1, 8.0, 8.0), is_free=False, center_position=(-2.05, 0, 0))

    # Set dirichlet conditions to fix the beam on the wall
    app.create_dirichlet_conditions("beam", lambda x: x[0] + 1.9)

    # Do simulation loop
    app.initialize_viewer()
    app.initialize_usd('./hanging_beam_simulation.usda')
    app.simulate(T = 5.0, update_viewer = False, non_update_bodies = ["wall"])


if __name__ == "__main__":
    run()