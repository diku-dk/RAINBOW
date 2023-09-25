from soft_body_sim_helper import *

def create_scene(material=None, model='SVK', T = 10.0, usd_save_path="soft_body_surface.usda", time_step=0.0001):
    scene = API.create_engine()
    scene.params.time_step = time_step
    viewer = VIEWER.Viewer()

    create_soft_material(scene, material, model) 
    scene.materials['soft_mat1'].c = 0.5

    create_soft_beam(scene, (4, 4, 4, 0.5, 0.5, 0.5), name="cube", y_offset=0.275)
    create_soft_beam(scene, (10, 2, 10, 4, 0.05, 4), name="surface")

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