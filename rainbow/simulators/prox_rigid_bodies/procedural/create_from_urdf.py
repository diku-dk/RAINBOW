"""
This file contains a function to import the scene content from an URDF file.
"""
import logging
import numpy as np

import rainbow.math.quaternion as Q
import rainbow.simulators.prox_rigid_bodies.types as TYPES

import xml.etree.ElementTree as ET


def _parse_geometry(element):
    if element is None:
        return None

    geometry = element.find('geometry')
    if geometry is not None:
        shape_element = list(geometry)[0]  # e.g., 'box', 'cylinder', 'sphere', 'mesh'
        shape = shape_element.tag

        if shape == 'mesh':
            filename = shape_element.attrib.get('filename', '')
            scale = shape_element.attrib.get('scale', '1 1 1')
            return {'shape': 'mesh', 'filename': filename, 'scale': scale}
        else:
            dimensions = shape_element.attrib
            return {'shape': shape, 'dimensions': dimensions}
    return None


def _parse_inertial(inertial):
    if inertial is None:
        return None

    origin = inertial.find('origin').attrib if inertial.find('origin') is not None else {}
    mass = inertial.find('mass').attrib if inertial.find('mass') is not None else {}
    inertia = inertial.find('inertia').attrib if inertial.find('inertia') is not None else {}
    return {'origin': origin, 'mass': mass, 'inertia': inertia}







def create_from_urdf(engine: TYPES.Engine, urdf_filename: str) :
    """

    :param engine:
    :param urdf_filename:
    :return:
    """
    logger = logging.getLogger("rainbow.simulators.prox_rigid_bodies.procedural.create_from_urdf")
    logger.info(f"Creating from urdf file: {urdf_filename}")

    tree = ET.parse(urdf_filename)
    root = tree.getroot()
    robot_name = root.attrib.get('name')
    links = []
    joints = []

    for link in root.findall('link'):
        link_name = link.attrib['name']
        visual = link.find('visual')
        collision = link.find('collision')
        inertial = link.find('inertial')

        link_info = {
            'name': link_name,
            'visual': _parse_geometry(visual),
            'collision': _parse_geometry(collision),
            'inertial': _parse_inertial(inertial)
        }
        links.append(link_info)

    for joint in root.findall('joint'):
        joint_name = joint.attrib['name']
        joint_type = joint.attrib['type']
        parent = joint.find('parent').attrib['link']
        child = joint.find('child').attrib['link']
        origin = joint.find('origin').attrib if joint.find('origin') is not None else {}
        axis = joint.find('axis').attrib if joint.find('axis') is not None else {}
        limit = joint.find('limit').attrib if joint.find('limit') is not None else {}

        joint_info = {
            'name': joint_name,
            'type': joint_type,
            'parent': parent,
            'child': child,
            'origin': origin,
            'axis': axis,
            'limit': limit
        }
        joints.append(joint_info)

    return robot_name, links, joints


if __name__ == "__main__":
    engine = TYPES.Engine()

    urdf_filename = '/Users/kenny/Documents/GitHub/libRAINBOW/data/urdf/ur_description/ur5e.urdf'

    name, links, joints = create_from_urdf(engine, urdf_filename)

    print(f"robot name is {name}")
    print("Links:", links)
    print("Joints:", joints)
