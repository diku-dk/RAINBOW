import numpy as np
from soft_application import SoftBodyApplication


def run():
    """
    A simulation scenario in which a sheet has both ends fixed. A cube is located on the sheet. 

    Usage:
        To run the simulation, simply execute this function. Adjust parameters within to modify simulation behaviors,
        such as the material properties of the beam or the strength of gravity.
    """
    app = SoftBodyApplication(time_step=0.0001)

    # Create a soft material named "default_material"
    app.create_soft_material("default_material")
    app.set_friction_coefficient("default_material", "default_material", 0.5)
    app.set_damping_coefficient("default_material", 0.5)

    # Add the cube and surface into the scene
    app.create_soft_body("cube", "default_material", (4, 4, 4, 0.5, 0.5, 0.5))
    app.create_soft_body("sheet", "default_material", (20, 2, 10, 4, 0.05, 4), center_position=(0, 0.275, 0))
    
    # Set dirichlet conditions to fix dual ends of the beam
    app.create_dirichlet_conditions("sheet", lambda x: x[0] + 1.9)
    app.create_dirichlet_conditions("sheet", lambda x: 1.9 - x[0])

    # Do simulation loop
    app.initialize_viewer()
    app.initialize_usd('./soft_sheet.usda')
    app.simulate(T = 5.0, update_viewer = False)


if __name__ == "__main__":
    run()