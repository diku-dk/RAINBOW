import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm
import rainbow.math.vector3 as V3
import rainbow.geometry.volume_mesh as VM
import rainbow.simulators.prox_soft_bodies.api as API
import rainbow.simulators.prox_soft_bodies.solver as SOLVER
import rainbow.util.viewer as VIEWER
from rainbow.util.USD import USD


# This block includes the helper functions to create the soft beam and the wall, and to initialize the viewer, and to simulate the scene.

# colors for visualization
COLOR_MAP = {
    'beam': V3.make(0.1, 0.8, 0.1),
    'surface': V3.make(0.1, 0.8, 0.1),
    'wall': V3.make(224/255, 208/255, 193/255),
    'ground': V3.make(224/255, 208/255, 193/255),
    'cube':V3.make(224/255, 208/255, 193/255),
}

MODEL_MAP = {
    'SVK': API.SVK,
    'COR': API.COR,
    'SNH': API.SNH
}

# create a soft material into a scene
def create_soft_material(engine, name=None, model_name='SVK', friction=0.5):
    model = MODEL_MAP[model_name]
    E, nu, rho = API.create_material_parameters(name) # if name == None, default values , E = 10e5 nu = 0.3 rho = 1000  

    API.create_material(engine, 'soft_mat1')
    API.set_elasticity(engine, 'soft_mat1', E, nu)
    API.set_mass_density(engine, 'soft_mat1', rho)
    API.set_constitutive_model(engine, 'soft_mat1', model)
    API.set_viscosity(engine, 'soft_mat1', 0.5)
    API.create_surfaces_interaction(engine,'soft_mat1','soft_mat1', friction)

# create the soft beam into a scene
def create_soft_beam(engine, geo, gravity=-10, material='soft_mat1', name='beam', y_offset=0.0):
    V_beam, T_beam = VM.create_beam(*geo) # geometry 
    V_beam[:,1] += y_offset
    API.create_soft_body(engine, name, V_beam, T_beam)
    API.set_type(engine, name, "Free")
    API.set_gravity(engine, name, (0,gravity,0))
    API.set_material(engine, name, material)

# create the wall into a scene
def create_wall(engine):
    V_wall, T_wall = VM.create_beam(2, 2, 2, 0.1, 8.0, 8.0)
    V_wall[:,0] -= 2.05
    API.create_soft_body(engine, 'wall', V_wall, T_wall)
    API.set_type(engine, 'wall', "Fixed")
    API.set_gravity(engine, 'wall', (0,0,0))
    API.set_material(engine,'wall','soft_mat1')

# initialize the viewer
def initialize_viewer(engine, viewer, usd):
    # viewer.create_frame("origin")
    for body in engine.bodies.values():    
        opacity = 0.5 if body.name == "beam" else 1.0
        F = VM.to_triangles(body.T)
        color = COLOR_MAP[body.name]
        viewer.create_mesh(body.name, body.x0, F, color, opacity)   
        viewer.create_scatter(body.name + '.DBC')
        viewer.create_quiver(body.name + '.Ft')
        # viewer.hide_quiver(body.name + '.Ft')
        # add mesh to usd
        usd.add_mesh(body.name, body.x0, F)

    viewer.show()

def upadte_viewer(engine, viewer, non_update_bodies=[]):
    for body in engine.bodies.values():
        if body.name in non_update_bodies:
            continue
        viewer.update_mesh(body.name, body.x)
        if body.Ft is not None:
            viewer.update_quiver(body.name + '.Ft', V=body.x, N=body.Ft, scale=0.1, color=[0.7, 0.7, 0.1])
        if len(body.dirichlet_conditions)>0:
            indices = [bc.idx for bc in body.dirichlet_conditions]
            V = body.x[indices]
            viewer.update_scatter(body.name + '.DBC', V=V, scale=0.5, color=[0.4, 0.4, 0.4])

def upadte_usd(engine, usd, time, non_update_bodies=[]):
    for body in engine.bodies.values():
        if body.name in non_update_bodies:
            continue
        usd.update_mesh_positions(body.name, body.x, time)

def simulate(engine, viewer, usd, T = 1.0, render_fps=60.0, non_update_bodies=[], update_traction=False):
    sim_dt = engine.params.time_step
    sim_fps = 1.0 / sim_dt
    render_dt = 1 / render_fps
    render_steps = int(np.round(T / render_dt))
    usd.set_animation_time(T*render_fps) # set the simulation ending time  in usd

    for step in tqdm(range(render_steps), desc="Rendering Loop:"):
        dT = render_dt 
        while dT > 0: # do simulation block
            dt = min(dT, sim_dt) # generally, sim_dt < render_dt
            API.simulate(engine, dt, debug_on=True)
            dT -= dt
            # upadte_viewer(engine, viewer, non_update_bodies)
        upadte_usd(engine, usd, step*render_dt*render_fps, non_update_bodies)

    return API.get_log(engine)