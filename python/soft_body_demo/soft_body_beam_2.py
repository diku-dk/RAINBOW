from soft_body_sim_helper import *

def left_beam(x):
    return x[0] + 1.9

def right_beam(x):
    return 1.9 - x[0]

def center_beam(x):
    return -1 if x[0] < 0.05 or x[0] > -0.05 else 1

def create_scene(material=None, usd_save_path="soft_body_beam_default.usda"):
    # scene2 setup
    scene = API.create_engine()
    scene.params.time_step = 0.001
    viewer = VIEWER.Viewer()

    create_soft_material(scene, material, 'SNH') 
    create_soft_beam(scene, (20, 2, 2, 4.0, 0.1, 0.1), gravity=0)
    
    # API.create_dirichlet_conditions(scene, 'beam', center_beam)
    # API.create_traction_conditions(scene, 'beam', left_beam, load=V3.make(2000, -2000, 0))
    API.create_traction_conditions(scene, 'beam', center_beam, load=V3.make(0, 2000, 0))

    API.create_dirichlet_conditions(scene, 'beam', lambda x: x[0] + 1.9)
    API.create_dirichlet_conditions(scene, 'beam', lambda x: 1.9 - x[0])

    usd = USD(usd_save_path)

    initialize_viewer(scene, viewer, usd)
    stats = simulate(scene, viewer, usd, T = 5.0, non_update_bodies=['wall'])

    usd.save()

def run():
    materials = [None]
    # materials = ['concrete']

    for material in materials:
        create_scene(material,usd_save_path=f"soft_body_beam_2_snh_{material}.usda")

if __name__ == '__main__':
    run()
    

  