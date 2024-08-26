"""
This file contains a function to export the scene content of a rigid body engine to a file.
"""
import logging

import numpy as np

import rainbow.math.quaternion as Q
import rainbow.simulators.prox_rigid_bodies.types as TYPES


def write_xml(engine: TYPES.Engine, xml_filename: str) -> None:
    """
    Write a scene to xml-file.

    :param engine:         The engine containing the scene.
    :param xml_filename:   The filename of the output xml file.
    """
    logger = logging.getLogger("rainbow.simulators.prox_rigid_bodies.procedural.write_xml")
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

    hinges_xml_node = ET.SubElement(root, "hinges")
    for idx, hinge in enumerate(engine.hinges.values()):
        hinge_xml_node = ET.SubElement(hinges_xml_node, "hinge")
        hinge_xml_node.set("name", hinge.name)
        hinge_xml_node.set("idx", str(idx))
        hinge_xml_node.set("parent", hinge.parent.name)
        hinge_xml_node.set("parent-idx", str(hinge.parent.idx))
        hinge_xml_node.set("child", hinge.child.name)
        hinge_xml_node.set("child-idx", str(hinge.child.idx))
        o_world = Q.rotate(hinge.parent.q, hinge.arm_p) + hinge.parent.r
        s_world = Q.rotate(hinge.parent.q, hinge.axis_p)
        hinge_xml_node.set("origin", np.array2string(o_world))
        hinge_xml_node.set("axis", np.array2string(s_world))
        hinge_xml_node.set("mode", "world")

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
