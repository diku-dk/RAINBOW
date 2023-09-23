from soft_body_sim_helper import *

# scene1 setup

def create_scene(material=None, model='SVK', T = 10.0, usd_save_path="soft_body_beam_default.usda", time_step=0.0001):
    scene = API.create_engine()
    scene.params.time_step = time_step
    viewer = VIEWER.Viewer()

    create_soft_material(scene, material, model) 
    create_soft_beam(scene, (12, 3, 3, 4.0, 1.0, 1.0))
    create_wall(scene)

    def left_wall(x):
        return x[0] + 1.9

    API.create_dirichlet_conditions(scene, 'beam', left_wall)

    usd = USD(usd_save_path)

    initialize_viewer(scene, viewer, usd)
    stats = simulate(scene, viewer, usd, T = T, non_update_bodies=['wall'])

    usd.save()


def run():
    materials = [
        {
            'material': None,
            'model': 'SNH',
            'time_step': 0.001,
        },
        # {
        #     'material': 'rubber',
        #     'model': 'SVK',
        #     'time_step': 0.0001,
        # }
    ]

    for material in materials:
        create_scene(material=material['material'], model=material['model'], T = 2.0, usd_save_path=f"soft_body_beam_1_{material['model']}.usda", time_step=material['time_step'])
    
    # create_scene(material='rubber', model='SNH', usd_save_path=f"soft_body_beam_snh.usda")
    # create_scene(material='rubber', model='COR', usd_save_path=f"soft_body_beam_cor.usda")



if __name__ == '__main__':
    run()
    
