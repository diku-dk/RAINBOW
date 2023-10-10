import os
import sys
import numpy as np
from numpy.typing import ArrayLike
from typing import Callable, Sequence

# tqdm is used for generating an progress bar in for loops, 
# enhancing visual feedback. It will be used to visualize the similation progress.
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.math.vector3 as V3
import rainbow.geometry.volume_mesh as VM
import rainbow.simulators.prox_soft_bodies.api as API
import rainbow.simulators.prox_soft_bodies.solver as SOLVER
import rainbow.util.viewer as VIEWER
from rainbow.util.USD import USD


class SoftBodyApplication:
    """
    SoftBodyApplication represents a soft body simulation application.

    This class provides an interface to create and manipulate soft bodies in a 
    simulated scene. It supports functionalities such as creating materials, 
    setting physical properties, adding traction forces, and rendering the simulation 
    in a viewer or exporting it to a USD (Universal Scene Description) format.

    Attributes:
        MODEL_MAP (dict): A mapping between elasticity model names and their respective APIs.
        
    Methods:
        create_soft_material: Define a new soft material with specific physical properties.
        set_damping_coefficient: Set damping coefficient for a material.
        set_friction_coefficient: Define friction between two materials.
        create_soft_body: Create a soft body with specific attributes in the scene.
        initialize_viewer: Initialize the 3D viewer with desired properties.
        update_viewer: Update the viewer to reflect changes in the simulation.
        initialize_usd: Initialize the Universal Scene Description (USD) for the simulation.
        update_usd: Update the USD to reflect changes in the simulation at a specific timestamp.
        create_dirichlet_conditions: Specify Dirichlet (fixed) conditions for a body based on a condition function.
        create_traction_conditions: Apply a traction force to a body based on a condition function.
        simulate: Run the simulation for a specified duration and render it at a given fps.
    """
     
    MODEL_MAP = {
        'SVK': API.SVK,
        'COR': API.COR,
        'SNH': API.SNH
    }

    def __init__(self, time_step: float = 0.001) -> None:
        self.scene = API.create_engine()
        self.viewer = VIEWER.Viewer()
        self.scene.time_step = time_step
        self.usd = None
    
    def create_soft_material(self, material_name: str, E: float = 10e5, nu: float = 0.3, rho: float = 1000, model_name: str = 'SVK', viscosity: float = 0.5) -> None:
        """ Create a soft material in the scene

        Args:
            material_name (str): The name of the material, such as "rubber"
            E (float, optional): Young's modulus. Defaults to 10e5.
            nu (float, optional): Poisson's Ratio. Defaults to 0.3.
            rho (float, optional): Density. Defaults to 1000.
            model_name (str, optional): The Elasticity model name. Defaults to 'SVK', Valid model names can be found in the MODEL_MAP dictionary..
            viscosity (float, optional): The dynamic viscosity of the material. Defaults to 0.5, Must be a non-negative value.
        """
        if model_name not  in self.MODEL_MAP:
            raise ValueError("Invalid elasticity model name.")
        if viscosity <= 0:
            raise ValueError("viscosity must be a positive floating-point number.")
        model = self.MODEL_MAP[model_name]
        API.create_material(self.scene, material_name)
        API.set_elasticity(self.scene, material_name, E, nu)
        API.set_mass_density(self.scene, material_name, rho)
        API.set_constitutive_model(self.scene, material_name, model)
    
    def set_damping_coefficient(self, material_name: str, damping: float):
        """ Set the damping coefficient of a material

        Args:
            material_name (str): The name of the material
            damping (float): The damping coefficient value
        """
        self.scene.materials[material_name].c = damping

    def set_friction_coefficient(self, material_name_1: str, material_name_2: str, friction: float):
        """ Set the friction coefficient between two materials

        Args:
            material_name_1 (str): The name of the first material
            material_name_2 (str): The name of the second material
            friction (float): The friction coefficient value
        """
        API.create_surfaces_interaction(self.scene, material_name_1, material_name_2, friction)

    def create_soft_body(self, body_name: str, material_name: str, mesh: tuple, gravity: float = -9.8, is_free: bool = True, center_position: tuple = (0, 0, 0)) -> None:
        """ Create a soft body in the scene

        Args:
            body_name (str): The name of the body, such as "rubber_beam"
            material_name (str): The name of the material
            mesh (tuple): The mesh of the body, such as a cube can be define as (2, 2, 2, 1, 1, 1)
            gravity (float, optional): The gravity value for this body. Defaults to -9.8.
            is_free (bool, optional): The body should be fixed if the value is false. Defaults to True.
            center_position (tuple, optional): The center position of the body. Defaults to (0, 0, 0), maens origin point in the scene.
        """
        V, T = VM.create_beam(*mesh)
        x_transform, y_transform, z_transform = center_position
        V[:, 0] += x_transform
        V[:, 1] += y_transform
        V[:, 2] += z_transform
        API.create_soft_body(self.scene, body_name, V, T)
        API.set_type(self.scene, body_name, "Free" if is_free else "Fixed")
        API.set_gravity(self.scene, body_name, (0,gravity,0))
        API.set_material(self.scene, body_name, material_name)
        
    def initialize_viewer(self, opacity: float = 0.5, color: Sequence[float] = V3.make(0.1, 0.8, 0.1)) -> None:
        """ Initialize the viewer

        Args:
            opacity (float, optional): a opacity value for all bodies in the scene . Defaults to 0.5.
            color (Sequence[float], optional): a color value for all bodies in the scene. Defaults to V3.make(0.1, 0.8, 0.1).
        """
        for body in self.scene.bodies.values():
            F = VM.to_triangles(body.T)
            self.viewer.create_mesh(body.name, body.x0, F, color, opacity)   
            self.viewer.create_scatter(body.name + '.DBC')
            self.viewer.create_quiver(body.name + '.Ft')

        self.viewer.show()

    def upadte_viewer(self, non_update_bodies: Sequence[str] = []):
        """ This function handles the interactive (real-time) rendering of each step in the simulation.

        Note:
        It's essential to differentiate between two types of rendering happening in our simulation:
        - Interactive rendering: Which is handled by the `update_viewer` function, wherein the viewer tries to 
        keep up with the simulator's time-step, providing a real-time visual feedback during the simulation process.
        - Offline rendering: Which is not performed in this function but it si supported with the `USD` class, 
        which refers to saving (or "baking") the simulation data into a USD file for potential future rendering or movie creation. 
        The FPS (frames per second) control in this context would pertain to the movie creation process from the saved USD file. 
        For example, we can bake the simulation data at 60 FPS to a USD file and then import it into Blender to render a movie at 
        60 FPS.
        
        Args:
            non_update_bodies (Sequence[str], optional): Those bodies should not be update in the scene. Defaults to [].
        """
        for body in self.scene.bodies.values():
            if body.name in non_update_bodies:
                continue
            self.viewer.update_mesh(body.name, body.x)
            if body.Ft is not None:
                self.viewer.update_quiver(body.name + '.Ft', V=body.x, N=body.Ft, scale=0.1, color=[0.7, 0.7, 0.1])
            if len(body.dirichlet_conditions)>0:
                indices = [bc.idx for bc in body.dirichlet_conditions]
                V = body.x[indices]
                self.viewer.update_scatter(body.name + '.DBC', V=V, scale=0.5, color=[0.4, 0.4, 0.4])
            
    def initialize_usd(self, file_path: str) -> None:
        """ Initialize the USD instance

        Args:
            file_path (str): The file path to save the USD file, such as "./test.usda"
        """
        self.usd = USD(file_path)
        for body in self.scene.bodies.values():
            F = VM.to_triangles(body.T)
            self.usd.add_mesh(body.name, body.x0, F)
    
    def upadte_usd(self, time: float, non_update_bodies: Sequence[str] = []) -> None:
        """ Update the vertices position of USD

        Args:
            time (float): the time stamp of the scene
            non_update_bodies (Sequence[str], optional): Those bodies should not be update in the scene. Defaults to [].
        """
        for body in self.scene.bodies.values():
            if body.name in non_update_bodies:
                continue
            self.usd.set_mesh_positions(body.name, body.x, time)
    
    def create_dirichlet_conditions(self, body_name: str, condition_func: Callable):
        """ Create dirichlet conditions for a body

        Args:
            body_name (str): The name of the body
            condition_func (Callable): A function that returns a boolean value, which indicates whether a vertex should be fixed, such as: `lambda x: x[0] > 1.0` would fix vertices with x-coordinates greater than 1.0.
        """
        API.create_dirichlet_conditions(self.scene, body_name, condition_func)
    
    def create_traction_conditions(self, body_name: str, traction_condition_func: Callable, force: ArrayLike):
        """ Create traction force for a body

        Args:
            body_name (str): The name of the body
            force (ArrayLike): The traction force
        """
        API.create_traction_conditions(self.scene, body_name, traction_condition_func, force)

    def simulate(self, T: float = 1.0, render_fps: float = 60.0, update_viewer: bool = True, update_usd: bool = True,non_update_bodies: Sequence[str] = []):
        """ Simulates the scene for a given duration, `T`, at a specific frame rate, `render_fps`.
        
        Args:
            T (float, optional): The total duration of the simulation in seconds. Defaults to 1.0.
            render_fps (float, optional): The frames per second at which to render the simulation. Defaults to 60.0.
            update_viewer (bool, optional): Whether to update the viewer during the simulation. Defaults to True.
            update_usd (bool, optional): Whether to update the Universal Scene Description (USD) animation during the simulation. Defaults to True.
            non_update_bodies (Sequence[str], optional): List of body names that shouldn't be updated during the simulation. Defaults to an empty list.

        Raises:
            ValueError: If USD updates are requested but the USD instance isn't initialized.

        Returns:
            dict: A log or report of the simulation, detailing various aspects of the scene after the simulation.
        """
        sim_dt = self.scene.params.time_step
        render_dt = 1.0 / render_fps
        render_steps = int(np.round(T / render_dt))
        if update_usd:
            if self.usd is None:
                raise ValueError("USD is not initialized.")
            self.usd.set_animation_time(T*render_fps)

        # A nested loop for simulation and rendering, 
        # outer loop for rendering, inner loop for simulation computation.
        for step in tqdm(range(render_steps), desc="Rendering Loop:"):
            dT = render_dt 
            while dT > 0:
                dt = min(dT, sim_dt) 
                API.simulate(self.scene, dt, debug_on=True)
                dT -= dt
                if update_viewer:
                    self.upadte_viewer(non_update_bodies) # Interactive rendering the simulation process with the simulation time step.
            if update_usd:
                self.upadte_usd(step*render_dt*render_fps, non_update_bodies) # Offline rendering the simulation process with the render time step. The default rendering fps is 60.

        if update_usd:
            self.usd.save()

        return API.get_log(self.scene)