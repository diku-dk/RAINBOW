from soft_body_sim_helper import *

def create_scene(material=None, model='SVK', T = 10.0, usd_save_path="soft_body_surface.usda", time_step=0.001):
    scene = API.create_engine()
    scene.params.time_step = time_step
    viewer = VIEWER.Viewer()

    create_soft_material(scene, material, model) 

    create_soft_beam(scene, (2, 2, 2, 0.5, 0.5, 0.5), name="cube")
    create_soft_beam(scene, (20, 2, 10, 4, 0.05, 4), name="surface")
    
    create_wall(scene)

    API.create_dirichlet_conditions(scene, 'surface', lambda x: x[0] + 1.9)
    API.create_dirichlet_conditions(scene, 'surface', lambda x: 1.9 - x[0])

    usd = USD(usd_save_path)

    initialize_viewer(scene, viewer, usd)
    stats = simulate(scene, viewer, usd, T = T)

    usd.save()

def run():
    create_scene(material=None, model='SVK', usd_save_path="soft_body_surface_svk.usda")

if __name__ == '__main__':
    run()