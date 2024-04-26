"""
This script is used to generate rigid body scene XML files. It is based on the rainbow
 software. That can be found here:

 https://diku-dk.github.io/RAINBOW/

 Copyright 2024 Kenny Erleben
"""
import numpy as np
import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.simulators.prox_rigid_bodies.procedural as PROC


def setup_scene(engine, scene_name: str):
    PROC.create_ground(engine, V3.zero(), Q.identity(), density=1.0, material_name='default');

    if scene_name == "pillar":
        PROC.create_pillar(engine,
                           r=V3.zero(),
                           q=Q.identity(),
                           width=1.0,
                           height=5.0,
                           depth=1.0,
                           stones=3,
                           density=1.0,
                           material_name='default'
                           );
    elif scene_name == "arch":
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
        PROC.create_funnel(engine,
                           funnel_height=4.0,
                           funnel_radius=4.0,
                           grid_width=2.0,
                           grid_height=2.0,
                           grid_depth=2.0,
                           I=4,
                           J=4,
                           K=4,
                           density=1.0,
                           material_name='default'
                           )
    elif scene_name == "glasses":
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

    API.create_gravity_force(engine=engine, force_name="earth", g=9.81, up=V3.k())
    API.create_damping_force(engine=engine, force_name="air", alpha=0.01, beta=0.01)
    for body in engine.bodies.values():
        API.connect_force(engine=engine, body_name=body.name, force_name="earth")
        API.connect_force(engine=engine, body_name=body.name, force_name="air")


def explort_to_xml(engine, xml_filename):
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
            #print(force.force_type)
            gravity_xml_node.set("g", str(force.g))
            gravity_xml_node.set("up", np.array2string(force.up))
        if force.force_type == "Damping":
            damping_xml_node = ET.SubElement(forces_xml_node, "damping")
            damping_xml_node.set("name", force.name)
            #print(force.force_type)
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
            param_xml_node.set(name, str(value))

    tree = ET.ElementTree(root)
    ET.indent(tree, '  ', level=0)
    ET.indent(tree, '  ', level=1)
    ET.indent(tree, '  ', level=2)
    ET.indent(tree, '  ', level=3)
    tree.write(xml_filename
               , encoding="utf-8"
               , xml_declaration=True
               , method = "xml"
               , short_empty_elements = True
               )


if __name__ == "__main__":
    engine = API.create_engine()
    setup_scene(engine, "pillar")
    explort_to_xml(engine, "pillar.xml")

    engine = API.create_engine()
    setup_scene(engine, "arch")
    explort_to_xml(engine, "arch.xml")

    engine = API.create_engine()
    setup_scene(engine, "dome")
    explort_to_xml(engine, "dome.xml")

    engine = API.create_engine()
    setup_scene(engine, "tower")
    explort_to_xml(engine, "tower.xml")

    engine = API.create_engine()
    setup_scene(engine, "colosseum")
    explort_to_xml(engine, "colosseum.xml")

    engine = API.create_engine()
    setup_scene(engine, "pantheon")
    explort_to_xml(engine, "pantheon.xml")

    engine = API.create_engine()
    setup_scene(engine, "funnel")
    explort_to_xml(engine, "funnel.xml")

    engine = API.create_engine()
    setup_scene(engine, "glasses")
    explort_to_xml(engine, "glasses.xml")

    engine = API.create_engine()
    setup_scene(engine, "poles")
    explort_to_xml(engine, "poles.xml")

    engine = API.create_engine()
    setup_scene(engine, "temple")
    explort_to_xml(engine, "temple.xml")
