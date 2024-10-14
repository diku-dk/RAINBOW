"""
This file contains a function to imports the scene content of a rigid body engine from an xml file.
"""

import logging
import xml.etree.ElementTree as ET

import numpy as np

import rainbow.math.quaternion as Q
import rainbow.simulators.prox_rigid_bodies.types as TYPES


def read_xml(xml_filename: str) -> TYPES.Engine:
    """
    Read a scene from an XML file and construct an Engine object.

    :param xml_filename: The filename of the input XML file.
    :return: An Engine object containing the scene.
    """
    logger = logging.getLogger("rainbow.simulators.prox_rigid_bodies.scenes.read_xml")
    logger.info(f"Reading scene from XML file: {xml_filename}")

    tree = ET.parse(xml_filename)
    root = tree.getroot()

    engine = TYPES.Engine()

    # Read Bodies
    bodies_xml_node = root.find("bodies")
    if bodies_xml_node is not None:
        for body_xml_node in bodies_xml_node.findall("body"):
            name = body_xml_node.get("name")
            idx = int(body_xml_node.get("idx"))
            shape_name = body_xml_node.get("shape")
            pos = np.fromstring(body_xml_node.get("pos"), sep=' ')
            rot = np.fromstring(body_xml_node.get("rot"), sep=' ')
            vel = np.fromstring(body_xml_node.get("vel"), sep=' ')
            spin = np.fromstring(body_xml_node.get("spin"), sep=' ')
            mass = float(body_xml_node.get("mass"))
            inertia = np.fromstring(body_xml_node.get("inertia"), sep=' ')
            is_fixed = body_xml_node.get("is_fixed") == 'True'
            material = body_xml_node.get("material")

            body = TYPES.Body(name, idx, shape_name, pos, rot, vel, spin, mass, inertia, is_fixed, material)

            # Add forces to the body
            for apply_force_xml_node in body_xml_node.findall("apply"):
                force_name = apply_force_xml_node.get("name")
                force = TYPES.Force(name=force_name)
                body.forces.append(force)

            engine.bodies[body.name] = body

    # Read Forces
    forces_xml_node = root.find("forces")
    if forces_xml_node is not None:
        for gravity_xml_node in forces_xml_node.findall("gravity"):
            name = gravity_xml_node.get("name")
            g = float(gravity_xml_node.get("g"))
            up = np.fromstring(gravity_xml_node.get("up"), sep=' ')
            force = TYPES.Force(name=name, force_type="Gravity", g=g, up=up)
            engine.forces[force.name] = force

        for damping_xml_node in forces_xml_node.findall("damping"):
            name = damping_xml_node.get("name")
            alpha = float(damping_xml_node.get("alpha"))
            beta = float(damping_xml_node.get("beta"))
            force = TYPES.Force(name=name, force_type="Damping", alpha=alpha, beta=beta)
            engine.forces[force.name] = force

    # Read Shapes
    shapes_xml_node = root.find("shapes")
    if shapes_xml_node is not None:
        for shape_xml_node in shapes_xml_node.findall("shape"):
            name = shape_xml_node.get("name")
            vertices = []
            vertices_xml_node = shape_xml_node.find("vertices")
            if vertices_xml_node is not None:
                for vertex_xml_node in vertices_xml_node.findall("v"):
                    p = np.fromstring(vertex_xml_node.get("p"), sep=' ')
                    vertices.append(p)

            triangles = []
            triangles_xml_node = shape_xml_node.find("triangles")
            if triangles_xml_node is not None:
                for triangle_xml_node in triangles_xml_node.findall("t"):
                    labels = np.fromstring(triangle_xml_node.get("labels"), sep=' ')
                    triangles.append(labels)

            shape = TYPES.Shape(name=name, mesh=TYPES.Mesh(vertices, triangles))
            engine.shapes[shape.name] = shape

    # Read Material Interactions
    materials_xml_node = root.find("materials")
    if materials_xml_node is not None:
        for interaction_xml_node in materials_xml_node.findall("interaction"):
            materials = interaction_xml_node.get("materials")
            friction = np.fromstring(interaction_xml_node.get("friction"), sep=' ')
            restitution = float(interaction_xml_node.get("restitution"))

            interaction = TYPES.SurfaceInteraction(materials, friction, restitution)
            engine.surfaces_interactions.storage[materials] = interaction

    # Read Hinges
    hinges_xml_node = root.find("hinges")
    if hinges_xml_node is not None:
        for hinge_xml_node in hinges_xml_node.findall("hinge"):
            name = hinge_xml_node.get("name")
            idx = int(hinge_xml_node.get("idx"))
            parent_name = hinge_xml_node.get("parent")
            child_name = hinge_xml_node.get("child")
            parent = engine.bodies[parent_name]
            child = engine.bodies[child_name]
            origin = np.fromstring(hinge_xml_node.get("origin"), sep=' ')
            axis = np.fromstring(hinge_xml_node.get("axis"), sep=' ')

            hinge = TYPES.Hinge(name, idx, parent, child, origin, axis)
            engine.hinges[hinge.name] = hinge

    # Read Parameters
    params_xml_node = root.find("params")
    if params_xml_node is not None:
        for param_xml_node in params_xml_node.findall("param"):
            name = param_xml_node.get("name")
            value = param_xml_node.get("value")
            type_name = param_xml_node.get("type")

            # Convert the value to the correct type
            if type_name == "int":
                value = int(value)
            elif type_name == "float":
                value = float(value)
            elif type_name == "bool":
                value = value == "True"

            setattr(engine.params, name, value)

    logger.info(f"Done reading file: {xml_filename}")
    return engine