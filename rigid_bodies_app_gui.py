import logging

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.simulators.prox_rigid_bodies.procedural as PROC

app_params = {}         # Dictionary used to control parameters that affect application


def setup_scene(engine, scene_name: str):
    logger = logging.getLogger("main.setup_scene")
    logger.info(f"Setting up: {scene_name}")

    if scene_name == "pillar":
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
    elif scene_name == "arch":
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
    elif scene_name == "dome":
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
    elif scene_name == "tower":
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
    elif scene_name == "colosseum":
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
    elif scene_name == "pantheon":
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
    elif scene_name == "funnel":
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
    elif scene_name == "glasses":
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
    elif scene_name == "poles":
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
    elif scene_name == "temple":
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
    elif scene_name == "chainmail":
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
    elif scene_name == "gear_train":
        PROC.create_gear_train(
            engine,
            N=14,
            density=1.0,
            material_name='default'
        )
    elif scene_name == "rock_slide":
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
    elif scene_name == "sandbox":
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

    API.create_gravity_force(engine=engine, force_name="earth", g=9.81, up=V3.j())
    API.create_damping_force(engine=engine, force_name="air", alpha=0.01, beta=0.01)
    for body in engine.bodies.values():
        API.connect_force(engine=engine, body_name=body.name, force_name="earth")
        API.connect_force(engine=engine, body_name=body.name, force_name="air")
    logger.info(f"Done with creating {scene_name}")


def export_to_xml(engine, xml_filename):
    logger = logging.getLogger("main.export_to_xml")
    logger.info(f"Converting scene to xml file: {xml_filename}")

    import xml.etree.ElementTree as ET

    root = ET.Element("scene")

    bodies_xml_node = ET.SubElement(root, "bodies")
    for body in engine.bodies.values():
        body_xml_node = ET.SubElement(bodies_xml_node, "body")
        body_xml_node.set("name", body.name)
        body_xml_node.set("idx", str(body.idx))
        body_xml_node.set("shape", body.shape.name)
        body_xml_node.set("pos", np.array2string(body.r))
        body_xml_node.set("rot", np.array2string(body.q))
        body_xml_node.set("vel", np.array2string(body.v))
        body_xml_node.set("spin", np.array2string(body.w))
        body_xml_node.set("mass", str(body.mass))
        body_xml_node.set("inertia", np.array2string(body.inertia))
        body_xml_node.set("is_fixed", str(body.is_fixed))
        body_xml_node.set("material", body.material)
        for force in body.forces:
            apply_force_xml_node = ET.SubElement(body_xml_node, "apply")
            apply_force_xml_node.set("name", force.name)

    forces_xml_node = ET.SubElement(root, "forces")
    for force in engine.forces.values():
        if force.force_type == "Gravity":
            gravity_xml_node = ET.SubElement(forces_xml_node, "gravity")
            gravity_xml_node.set("name", force.name)
            # print(force.force_type)
            gravity_xml_node.set("g", str(force.g))
            gravity_xml_node.set("up", np.array2string(force.up))
        if force.force_type == "Damping":
            damping_xml_node = ET.SubElement(forces_xml_node, "damping")
            damping_xml_node.set("name", force.name)
            # print(force.force_type)
            damping_xml_node.set("alpha", str(force.alpha))
            damping_xml_node.set("beta", str(force.beta))

    shapes_xml_node = ET.SubElement(root, "shapes")
    for shape in engine.shapes.values():
        shape_xml_node = ET.SubElement(shapes_xml_node, "shape")
        shape_xml_node.set("name", shape.name)
        vertices_xml_node = ET.SubElement(shape_xml_node, "vertices")
        vertices_xml_node.set("count", str(len(shape.mesh.V)))
        for idx, v in enumerate(shape.mesh.V):
            vertex_xml_node = ET.SubElement(vertices_xml_node, "v")
            vertex_xml_node.set("idx", str(idx))
            vertex_xml_node.set("p", np.array2string(v))
        triangles_xml_node = ET.SubElement(shape_xml_node, "triangles")
        triangles_xml_node.set("count", str(len(shape.mesh.T)))
        for idx, t in enumerate(shape.mesh.T):
            triangle_xml_node = ET.SubElement(triangles_xml_node, "t")
            triangle_xml_node.set("idx", str(idx))
            triangle_xml_node.set("labels", np.array2string(t))

    materials_xml_node = ET.SubElement(root, "materials")
    for key, interaction in engine.surfaces_interactions.storage.items():
        interaction_xml_node = ET.SubElement(materials_xml_node, "interaction")
        interaction_xml_node.set("materials", key)
        interaction_xml_node.set("friction", np.array2string(interaction.mu))
        interaction_xml_node.set("restitution", str(interaction.epsilon))

    params_xml_node = ET.SubElement(root, "params")
    settings = dir(engine.params)
    for name in settings:
        if not name.startswith("__"):  # Skip built-in attributes
            param_xml_node = ET.SubElement(params_xml_node, "param")
            value = getattr(engine.params, name)
            param_xml_node.set("name", str(name))
            param_xml_node.set("value", str(value))
            param_xml_node.set("type", type(value).__name__)

    tree = ET.ElementTree(root)
    ET.indent(tree, '  ', level=0)
    ET.indent(tree, '  ', level=1)
    ET.indent(tree, '  ', level=2)
    ET.indent(tree, '  ', level=3)

    tree.write(
        xml_filename,
        encoding="utf-8",
        xml_declaration=True,
        method="xml",
        short_empty_elements=True
    )
    logger.info(f"Done writing file: {xml_filename}")


def plotting(stats):
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
    for i in range(len(stats)):
        data = stats[i]
        if 'residuals' in data.keys():
            residuals = data['residuals']
            reject = data['reject']
            ax.plot(residuals[np.where(reject == False)])
    plt.show()

    time_update_bvh = [stats[i]['update_bvh'] for i in range(len(stats))]
    time_narrow_phase = [stats[i]['narrow_phase'] for i in range(len(stats))]
    time_contact_determination = [stats[i]['contact_determination'] for i in range(len(stats))]
    time_contact_point_reduction = [stats[i]['contact_point_reduction'] for i in range(len(stats))]
    time_collision_detection = [stats[i]['collision_detection_time'] for i in range(len(stats))]

    time_stepper = [stats[i]['stepper_time'] for i in range(len(stats))]

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

    number_of_overlaps = [stats[i]['number_of_overlaps'] for i in range(1, len(stats))]
    step_sizes = [stats[i]['dt'] for i in range(1, len(stats))]
    number_of_contact_points = [stats[i]['contact_points'] for i in range(1, len(stats))]
    penetrations = [stats[i]['max_penetration'] for i in range(1, len(stats))]

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

    kinetic_energy = [stats[i]['kinetic_energy'] for i in range(len(stats))]
    potential_energy = [stats[i]['potential_energy'] for i in range(len(stats))]

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
    ps.remove_all_structures()

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


def create_gui():
    global app_params
    # if psim.BeginMainMenuBar():
    #    if psim.BeginMenu("My Custom Menu"):
    #        if psim.MenuItem("Item 1"):
    #            print("Item 1 pressed")
    #        if psim.MenuItem("Item 2"):
    #            print("Item 2 pressed")
    #        psim.EndMenu()
    #    psim.EndMainMenuBar()
    changed, app_params["simulate"] = psim.Checkbox("Simulate", app_params["simulate"])
    if changed:
        print("Simulate = ", app_params["simulate"])
    changed, app_params["xml"] = psim.Checkbox("Save xml", app_params["xml"])
    if changed:
        print("Save XML = ", app_params["xml"])

    changed, app_params["stats"] = psim.Checkbox("Show stats", app_params["stats"])
    if changed:
        print("Show stats = ", app_params["stats"])
    changed, app_params["selected"] = psim.Combo("Scene", app_params["selected"], app_params["names"])
    if changed:
        print("Selected scene = ", app_params["selected"])
    if psim.Button("Create scene"):
        scene_name = app_params["names"][app_params["selected"]]
        print("Creating scene:", scene_name)

        engine = API.create_engine()

        total_time = 0.5
        steps = int(np.round(total_time / engine.params.time_step))
        app_params["total time"] = total_time
        app_params["steps"] = steps
        app_params["step"] = 0

        setup_scene(engine=engine, scene_name=scene_name)

        if app_params["xml"]:
            scene_name = app_params["names"][app_params['selected']]
            export_to_xml(engine, scene_name + ".xml")

        create_visual_geometry(engine=engine)

        app_params["engine"] = engine


def simulate() -> None:
    engine = app_params["engine"]

    if engine is None:
        return

    if not app_params["simulate"]:
        return

    if app_params["step"] >= app_params["steps"]:
        return

    for body in engine.bodies.values():
        T = np.eye(4)
        T[:3, :3] = Q.to_matrix(body.q)
        T[:3, 3] = body.r
        ps.get_surface_mesh(body.name).set_transform(T)

    API.simulate(engine=engine, T=engine.params.time_step, profiling_on=True)

    app_params["step"] += 1


def callback():
    create_gui()
    simulate()


def main():
    logging.basicConfig(level=logging.INFO)

    ps.set_up_dir('y_up')
    ps.init()
    ps.set_build_default_gui_panels(False)
    ps.set_ground_plane_mode("none")
    ps.look_at((0., 0., 100.), (0., 0., 0.))

    app_params["engine"] = None
    app_params["simulate"] = False
    app_params["xml"] = False
    app_params["stats"] = False
    app_params["selected"] = 0
    app_params["names"] = [
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
        "sandbox"
    ]

    ps.set_user_callback(callback)

    ps.show()

    if app_params["stats"]:
        engine = app_params["engine"]
        stats = API.get_log(engine)
        plotting(stats)


if __name__ == '__main__':
    main()
