"""
This script contains a high-level function to create different scenes.
"""
import logging

import rainbow.simulators.prox_rigid_bodies.types as TYPES
import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.simulators.prox_rigid_bodies.scenes as PROC
import rainbow.math.quaternion as Q
import rainbow.math.vector3 as V3


def get_scene_names() -> list[str]:
    """
    Get the list of all scene names that can be created by the create_scene function.

    :return: List of scene names.
    """
    names = [
        "pillar",
        "arch",
        "dome",
        "tower",
        "colosseum",
        "pantheon",
        "funnel",
        "glasses",
        "poles",
        "temple",
        "chainmail",
        "gear_train",
        "rock_slide",
        "sandbox",
        "box_stack",
        "cube_hinge_chain",
        "ur5"
    ]
    return names


def create_scene(engine: TYPES.Engine, scene_name: str) -> None:
    """
    Create a scene in a rigid body simulator.

    :param engine:        The engine that should contain the scene.
    :param scene_name:    The name of the scene is to be created.
    """
    logger = logging.getLogger("main.setup_scene")
    logger.info(f"Setting up: {scene_name}")

    scene_names = get_scene_names()

    if scene_name == scene_names[0]:
        PROC.create_ground(engine, V3.zero(), Q.identity(), density=1.0, material_name='default')
        PROC.create_pillar(engine,
                           r=V3.zero(),
                           q=Q.identity(),
                           width=1.0,
                           height=5.0,
                           depth=1.0,
                           stones=3,
                           density=1.0,
                           material_name='default'
                           )
    elif scene_name == scene_names[1]:
        PROC.create_ground(engine, V3.zero(), Q.identity(), density=1.0, material_name='default')
        PROC.create_arch(engine,
                         r=V3.zero(),
                         q=Q.identity(),
                         width=2.0,
                         height=3.0,
                         depth=0.5,
                         pier_stones=3,
                         arch_stones=5,
                         density=1.0,
                         material_name='default'
                         )
    elif scene_name == scene_names[2]:
        PROC.create_ground(engine, V3.zero(), Q.identity(), density=1.0, material_name='default')
        PROC.create_dome(engine,
                         r=V3.zero(),
                         q=Q.identity(),
                         outer_radius=5.0,
                         inner_radius=4.0,
                         layers=4,
                         segments=11,
                         density=1.0,
                         material_name='default'
                         )
    elif scene_name == scene_names[3]:
        PROC.create_ground(engine, V3.zero(), Q.identity(), density=1.0, material_name='default')
        PROC.create_tower(engine,
                          r=V3.zero(),
                          q=Q.identity(),
                          outer_radius=5.0,
                          inner_radius=4.0,
                          height=8.0,
                          layers=6,
                          segments=11,
                          use_cubes=False,
                          density=1.0,
                          material_name='default'
                          )
    elif scene_name == scene_names[4]:
        PROC.create_ground(engine, V3.zero(), Q.identity(), density=1.0, material_name='default')
        PROC.create_colosseum(engine,
                              r=V3.zero(),
                              q=Q.identity(),
                              outer_radius=5.0,
                              inner_radius=4.0,
                              height=7.0,
                              levels=3,
                              arches=12,
                              density=1.0,
                              material_name='default'
                              )
    elif scene_name == scene_names[5]:
        PROC.create_ground(engine, V3.zero(), Q.identity(), density=1.0, material_name='default')
        PROC.create_pantheon(engine,
                             r=V3.zero(),
                             q=Q.identity(),
                             outer_radius=5.0,
                             inner_radius=4.0,
                             height=8.0,
                             layers=4,
                             segments=11,
                             density=1.0,
                             material_name='default'
                             )
    elif scene_name == scene_names[6]:
        PROC.create_ground(engine, V3.zero(), Q.identity(), density=1.0, material_name='default')
        PROC.create_funnel(engine,
                           funnel_height=4.0,
                           funnel_radius=4.0,
                           grid_width=4.0,
                           grid_height=8.0,
                           grid_depth=4.0,
                           I=10,
                           J=20,
                           K=10,
                           density=1.0,
                           material_name='default'
                           )
    elif scene_name == scene_names[7]:
        PROC.create_ground(engine, V3.zero(), Q.identity(), density=1.0, material_name='default')
        PROC.create_glasses(engine,
                            glass_height=4.0,
                            glass_radius=2.0,
                            grid_width=3.0,
                            grid_height=3.0,
                            grid_depth=3.0,
                            I=4,
                            J=4,
                            K=4,
                            density=1.0,
                            material_name='default'
                            )
    elif scene_name == scene_names[8]:
        PROC.create_ground(engine, V3.zero(), Q.identity(), density=1.0, material_name='default')
        PROC.create_poles(engine,
                          pole_height=2.0,
                          pole_radius=0.1,
                          I_poles=6,
                          K_poles=6,
                          grid_width=4.0,
                          grid_height=4.0,
                          grid_depth=4.0,
                          I_grid=4,
                          J_grid=4,
                          K_grid=4,
                          density=1.0,
                          material_name='default'
                          )
    elif scene_name == scene_names[9]:
        PROC.create_ground(engine, V3.zero(), Q.identity(), density=1.0, material_name='default')
        PROC.create_temple(engine,
                           I_pillars=4,
                           K_pillars=7,
                           pillar_width=1.0,
                           pillar_height=3.0,
                           pillar_depth=1.0,
                           pillar_stones=3,
                           density=1.0,
                           material_name='default'
                           )
    elif scene_name == scene_names[10]:
        PROC.create_chainmail(
            engine,
            major_radius=2,
            minor_radius=0.5,
            columns=10,
            rows=10,
            stretch=0.75,
            density=1.0,
            material_name='default'
        )
        PROC.create_jack_lattice(
            engine,
            r=V3.make(-20, 40, -20),
            q=Q.identity(),
            width=40.0,
            height=40.0,
            depth=40.0,
            I=5,
            J=5,
            K=5,
            density=1.0,
            material_name='default',
            use_random_orientation=True
        )
    elif scene_name == scene_names[11]:
        PROC.create_gear_train(
            engine,
            N=14,
            density=1.0,
            material_name='default'
        )
    elif scene_name == scene_names[12]:
        PROC.create_rock_slide(
            engine,
            pile_width=8,
            pile_height=4,
            pile_depth=2,
            I_rocks=20,
            J_rocks=10,
            K_rocks=5,
            density=1.0,
            material_name='default'
        )
    elif scene_name == scene_names[13]:
        PROC.create_sandbox(
            engine,
            box_width=10,
            box_height=10,
            box_depth=10,
            I_grains=20,
            J_grains=20,
            K_grains=20,
            density=1.0,
            material_name='default'
        )
    elif scene_name == scene_names[14]:
        PROC.create_ground(
            engine,
            V3.zero(),
            Q.identity(),
            density=1.0,
            material_name='default'
        )
        PROC.create_box_stack(
            engine,
            box_width=1.0,
            box_height=1.0,
            box_depth=1.0,
            K_boxes=5,
            density=1.0,
            material_name='default'
        )
    elif scene_name == scene_names[15]:
        PROC.create_cube_hinge_chain(
            engine,
            cube_edge_length=10.0,
            K_cubes=5,
            density=1.0,
            material_name='default'
        )
    elif scene_name == scene_names[16]:
        package_folder = "/Users/kennyerleben/Documents/GitHub/"
        urdf_file_path = 'example-robot-data/robots/ur_description/urdf/ur5_robot.urdf'
        PROC.read_urdf(
            engine,
            package_folder,
            urdf_file_path
        )

    API.create_gravity_force(engine=engine, force_name="earth", g=9.81, up=V3.j())
    API.create_damping_force(engine=engine, force_name="air", alpha=0.01, beta=0.01)
    for body in engine.bodies.values():
        API.connect_force(engine=engine, body_name=body.name, force_name="earth")
        API.connect_force(engine=engine, body_name=body.name, force_name="air")
    logger.info(f"Done with creating {scene_name}")
