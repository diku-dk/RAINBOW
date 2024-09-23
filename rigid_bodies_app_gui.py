import logging

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.simulators.prox_rigid_bodies.procedural as PROC

from rainbow.util.USD import USD

app_params = {}  # Dictionary used to control parameters that affect application
usd_scene: USD | None = None 


def plotting(profiling_data):
    logger = logging.getLogger("main.plotting")
    logger.info(f"Starting plotting sub-routine")

    import matplotlib.pyplot as plt
    colors = [
        '#e6194b',
        '#3cb44b',
        '#ffe119',
        '#4363d8',
        '#f58231',
        '#911eb4',
        '#46f0f0',
        '#f032e6',
        '#bcf60c',
        '#fabebe',
        '#008080',
        '#e6beff',
        '#9a6324',
        '#fffac8',
        '#800000',
        '#aaffc3',
        '#808000',
        '#ffd8b1',
        '#000075',
        '#808080',
        '#ffffff',
        '#000000'
    ]

    plt.figure()
    ax = plt.subplot(111)
    ax.set_title('Convergence rates')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Merit')
    plt.grid(True)
    for i in range(len(profiling_data)):
        data = profiling_data[i]
        if 'residuals' in data.keys():
            residuals = data['residuals']
            ax.semilogy(residuals,
                        color=colors[i % 20],
                        linewidth=2
                        )
    plt.show()

    time_update_bvh = [profiling_data[i]['update_bvh'] for i in range(len(profiling_data))]
    time_narrow_phase = [profiling_data[i]['narrow_phase'] for i in range(len(profiling_data))]
    time_contact_determination = [profiling_data[i]['contact_determination'] for i in range(len(profiling_data))]
    time_contact_point_reduction = [profiling_data[i]['contact_point_reduction'] for i in range(len(profiling_data))]
    time_collision_detection = [profiling_data[i]['collision_detection_time'] for i in range(len(profiling_data))]

    time_stepper = [profiling_data[i]['stepper_time'] for i in range(len(profiling_data))]

    plt.figure()
    ax = plt.subplot(111)
    ax.set_title('Profiling Timings')
    ax.set_xlabel('Step')
    ax.set_ylabel('Time [s]')
    plt.grid(True)
    ax.plot(time_update_bvh, label='Update bvh', color=colors[6])
    ax.plot(time_narrow_phase, label='Narrow phase', color=colors[7])
    ax.plot(time_contact_determination, label='Contact determination', color=colors[8])
    ax.plot(time_contact_point_reduction, label='Contact reduction', color=colors[9])
    ax.plot(time_collision_detection, label='Collision Detection', color=colors[10])
    ax.plot(time_stepper, label='Stepper', color=colors[11])
    ax.legend()
    plt.show()

    number_of_overlaps = [profiling_data[i]['number_of_overlaps'] for i in range(1, len(profiling_data))]
    step_sizes = [profiling_data[i]['dt'] for i in range(1, len(profiling_data))]
    number_of_contact_points = [profiling_data[i]['contact_points'] for i in range(1, len(profiling_data))]
    penetrations = [profiling_data[i]['max_penetration'] for i in range(1, len(profiling_data))]

    plt.figure()
    ax = plt.subplot(111)
    ax.set_title('Profiling data')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    plt.grid(True)
    ax.plot(number_of_overlaps, label='Overlaps', color=colors[0])
    ax.plot(step_sizes, label='Step size', color=colors[1])
    ax.plot(number_of_contact_points, label='Contacts', color=colors[2])
    ax.plot(penetrations, label='Penetrations', color=colors[6])
    ax.legend()
    plt.show()

    kinetic_energy = [profiling_data[i]['kinetic_energy'] for i in range(len(profiling_data))]
    potential_energy = [profiling_data[i]['potential_energy'] for i in range(len(profiling_data))]

    plt.figure()
    ax = plt.subplot(111)
    ax.set_title('Energy Plots')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    plt.grid(True)
    ax.plot(kinetic_energy, label='Kinetic Energy', color=colors[4])
    ax.plot(potential_energy, label='Potential Energy', color=colors[5])
    ax.legend()
    plt.show()
    logger.info(f"Completed plotting")


def create_visual_geometry(engine):
    logger = logging.getLogger("main.create_visual_geometry")
    ps.remove_all_structures()
    logger.info(f"Removed all visual geometries")

    for body in engine.bodies.values():
        transparency = 1.0

        color = V3.make(1.0, 0.1, 0.1)
        if body.is_fixed:
            color = V3.make(0.1, 0.1, 1.0)
        ps.register_surface_mesh(
            body.name,
            body.shape.mesh.V,
            body.shape.mesh.T,
            smooth_shade=False,
            color=color,
            transparency=transparency
        )

        T = np.eye(4)
        T[:3, :3] = Q.to_matrix(body.q)
        T[:3, 3] = body.r
        ps.get_surface_mesh(body.name).set_transform(T)
    logger.info(f"Done creating new visual geometries")


def create_gui():
    logger = logging.getLogger("main.create_visual_geometry")

    global app_params, usd_scene

    changed, app_params['simulate'] = psim.Checkbox('Simulate', app_params['simulate'])
    if changed:
        logger.info(f"Application will run simulation loop = {app_params['simulate']}")
    changed, app_params['xml'] = psim.Checkbox('Save xml', app_params['xml'])
    if changed:
        logger.info(f"Application will save XML files = {app_params['xml']}")

    changed, app_params['profiling'] = psim.Checkbox('Show profiling', app_params['profiling'])
    if changed:
        logger.info(f"Application will show profiling and plots = {app_params['profiling']}")
    changed, app_params['selected'] = psim.Combo('Scene', app_params['selected'], app_params['names'])
    if changed:
        logger.info(f"Selected scene = {app_params['selected']}")
    if psim.Button('Create scene'):
        scene_name = app_params['names'][app_params['selected']]
        logger.info(f"Creating scene = {scene_name}")

        engine = API.create_engine()
        engine.params.time_step = 0.0001

        total_time = 5.0
        steps = int(np.round(total_time / engine.params.time_step))
        app_params['total time'] = total_time
        app_params['steps'] = steps
        app_params['step'] = 0

        PROC.create_scene(engine=engine, scene_name=scene_name)

        if app_params['xml']:
            scene_name = app_params['names'][app_params['selected']]
            PROC.write_xml(engine, scene_name + '.xml')

        create_visual_geometry(engine=engine)

        app_params['engine'] = engine

def simulate() -> None:
    global usd_scene
    
    logger = logging.getLogger("main.simulate")
    engine: API.Engine = app_params['engine']

    if engine is None:
        return

    if not app_params['simulate']:
        return

    if app_params['step'] >= app_params['steps']:
        if usd_scene is not None:
            usd_scene.save()
            usd_scene = None
        return
    
    if app_params['step'] == 0:
        usd_scene = USD(f'./animation.usda')
        usd_scene.set_frames_per_second(app_params['total time'] / app_params['steps'])
        usd_scene.set_animation_time(app_params['steps'])
        for body in engine.bodies.values():
            usd_scene.add_rigid_body(body)
            usd_scene.update_rigid_body(body, 0)

    logger.info(f"Running simulation step {app_params['step']}")
    for body in engine.bodies.values():
        T = np.eye(4)
        T[:3, :3] = Q.to_matrix(body.q)
        T[:3, 3] = body.r
        ps.get_surface_mesh(body.name).set_transform(T)
        
        if usd_scene is not None:
            usd_scene.update_rigid_body(body, app_params['step'])

    API.simulate(engine=engine, T=engine.params.time_step, profiling_on=True)

    app_params['step'] += 1
    logger.info(f"Completed simulation step")


def callback():
    create_gui()
    simulate()


def main():
    logging.basicConfig(level=logging.INFO)

    ps.set_up_dir('y_up')
    ps.init()
    ps.set_build_default_gui_panels(True)
    ps.set_ground_plane_mode('none')
    ps.look_at((0., 0., 100.), (0., 0., 0.))

    app_params['engine'] = None
    app_params['simulate'] = False
    app_params['xml'] = False
    app_params['profiling'] = False
    app_params['selected'] = 0
    app_params['names'] = PROC.get_scene_names()

    ps.set_user_callback(callback)

    ps.show()

    if app_params["profiling"]:
        engine = app_params["engine"]
        profiling_data = API.get_log(engine)
        plotting(profiling_data)


if __name__ == '__main__':
    main()
