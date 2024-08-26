import numpy as np
from soft_application import SoftBodyApplication


def run():
    """
    A simulation scenario in which a beam has both ends fixed. A traction force is applied 
    to the middle section of the beam.

    Usage:
        To run the simulation, simply execute this function. Adjust parameters within to modify simulation behaviors,
        such as the material properties of the beam or the strength of gravity.
    """
    app = SoftBodyApplication()

    # Create a soft material named "default_material"
    app.create_soft_material("default_material", model_name="SNH")
    app.set_friction_coefficient("default_material", "default_material", 0.5)

    # Add the beam into the scene
    app.create_soft_body("beam", "default_material", (20, 2, 2, 4.0, 0.1, 0.1), gravity= 0.0)

    # Set traction conditions to push midpoint of the beam upward
    def center_beam(x):
        return -1 if x[0] < 0.05 or x[0] > -0.05 else 1
    app.create_traction_conditions("beam", center_beam, np.array([0, 2000, 0], dtype=np.float64))
    
    # Set dirichlet conditions to fix dual ends of the beam
    app.create_dirichlet_conditions("beam", lambda x: x[0] + 1.9)
    app.create_dirichlet_conditions("beam", lambda x: 1.9 - x[0])

    # Do simulation loop
    app.initialize_viewer()
    app.initialize_usd('./midpoint_traction_beam_simulation.usda')
    app.simulate(T = 5.0, update_viewer = False)


if __name__ == "__main__":
    run()