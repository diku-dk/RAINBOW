"""
This file contains a function to import the scene content from an URDF file.
"""
import logging
import os
import numpy as np
import igl

import rainbow.math.quaternion as Q
import rainbow.math.vector3 as V3
import rainbow.simulators.prox_rigid_bodies.types as TYPES
import rainbow.simulators.prox_rigid_bodies.api as API


import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element


def _read_numbers_from_string(string: str) -> list[float]:
    # Split the string into a list of number strings
    number_strings = string.split()

    # Convert the list of number strings to a list of floats
    numbers = [float(num) for num in number_strings]
    return numbers


def _convert_frame(frame: dict) -> tuple:
    rpy = [0.0, 0.0, 0.0]
    xyz = [0.0, 0.0, 0.0]
    if 'rpy' in frame:
        rpy = _read_numbers_from_string(frame['rpy'])
    if 'xyz' in frame:
        xyz = _read_numbers_from_string(frame['xyz'])

    Qx = Q.Rx(rpy[0])
    Qy = Q.Ry(rpy[1])
    Qz = Q.Rz(rpy[2])

    Qtot = Q.prod(Qz, Q.prod(Qy, Qx))
    Ttot = V3.make(xyz[0], xyz[1], xyz[2])
    return Ttot, Qtot


def _convert_mass(mass: dict) -> float:
    value = 0
    if 'value' in mass:
        value = float(mass['value'])
    return value


def _convert_inertia_tensor(inertia: dict) ->list[float]:
    ixx = 0.0
    iyy = 0.0
    izz = 0.0
    ixy = 0.0
    ixz = 0.0
    iyz = 0.0
    if 'ixx' in inertia:
        ixx = float(inertia['ixx'])
    if 'iyy' in inertia:
        iyy = float(inertia['iyy'])
    if 'izz' in inertia:
        izz = float(inertia['izz'])
    if 'ixy' in inertia:
        ixy = float(inertia['ixy'])
    if 'ixz' in inertia:
        ixz = float(inertia['ixz'])
    if 'iyz' in inertia:
        iyz = float(inertia['iyz'])
    return [ixx, iyy, izz, ixy, ixz, iyz]


def _parse_geometry_tag(package_folder: str, element_tag: Element) -> dict:
    if element_tag is None:
        return None

    geometry_tag = element_tag.find('geometry')
    if geometry_tag is None:
        return None

    shape_element = list(geometry_tag)[0]  # e.g., 'box', 'cylinder', 'sphere', 'mesh'
    shape_tag = shape_element.tag

    if shape_tag != 'mesh':
        return None

    origin_dict = element_tag.find('origin').attrib if element_tag.find('origin') is not None else {}
    origin_frame = _convert_frame(origin_dict)

    scale = _read_numbers_from_string(
        shape_element.attrib.get('scale', '1 1 1')
    )

    file_url = shape_element.attrib.get('filename', '')
    if file_url.startswith('package://'):
        filename = os.path.join(package_folder, file_url.replace('package://', ''))
    else:
        filename = file_url

    # Get the file extension
    _, extension = os.path.splitext(filename)

    # List of allowed extensions
    supported_extensions = {'.stl', '.obj', '.ply'}

    # Check if the extension is in the list
    if not extension.lower() in supported_extensions:
        return None

    V, T = igl.read_triangle_mesh(filename=filename, dtypef=np.float64)

    return {
        'shape': 'mesh',
        'origin': origin_frame,
        'filename': filename,
        'scale': scale,
        'vertices': V,
        'faces': T
    }


def _parse_inertial_tag(inertial_tag: Element) -> dict:
    if inertial_tag is None:
        return None

    origin_dict = inertial_tag.find('origin').attrib if inertial_tag.find('origin') is not None else {}
    origin_frame = _convert_frame(origin_dict)

    mass_dict = inertial_tag.find('mass').attrib if inertial_tag.find('mass') is not None else {}
    mass_value = _convert_mass(mass_dict)

    inertia_dict = inertial_tag.find('inertia').attrib if inertial_tag.find('inertia') is not None else {}
    inertia = _convert_inertia_tensor(inertia_dict)

    return {'origin': origin_frame, 'mass': mass_value, 'inertia': inertia}


def create_from_urdf(engine: TYPES.Engine, package_folder: str, urdf_file_path: str):
    """

    :param engine:
    :param package_folder:
    :param urdf_file_path:
    :return:
    """
    logger = logging.getLogger("rainbow.simulators.prox_rigid_bodies.procedural.create_from_urdf")
    logger.info(f"Creating from urdf file path: {urdf_file_path} in package folder {package_folder}")

    if os.path.isabs(urdf_file_path):
        logger.info(f"The urdf file path was absolute trying to convert it to relative path")
        urdf_file_path = urdf_file_path.lstrip('/')

    filename = os.path.join(package_folder, urdf_file_path)

    tree = ET.parse(filename)
    root = tree.getroot()
    robot_name = root.attrib.get('name')
    links = []
    joints = []

    for link_tag in root.findall('link'):
        link_name = link_tag.attrib['name']
        visual_tag = link_tag.find('visual')
        collision_tag = link_tag.find('collision')
        inertial_tag = link_tag.find('inertial')

        link_info = {
            'name': link_name,
            'visual': _parse_geometry_tag(package_folder, visual_tag),
            'collision': _parse_geometry_tag(package_folder, collision_tag),
            'inertial': _parse_inertial_tag(inertial_tag)
        }
        links.append(link_info)

    for joint_tag in root.findall('joint'):
        joint_name = joint_tag.attrib['name']
        joint_type = joint_tag.attrib['type']
        parent = joint_tag.find('parent').attrib['link']
        child = joint_tag.find('child').attrib['link']
        origin_dict = joint_tag.find('origin').attrib if joint_tag.find('origin') is not None else {}
        axis_dict = joint_tag.find('axis').attrib if joint_tag.find('axis') is not None else { 'xyz': '1 0 0'}
        limit = joint_tag.find('limit').attrib if joint_tag.find('limit') is not None else {}

        origin_frame = _convert_frame(origin_dict)
        axis = V3.unit(_convert_frame(axis_dict)[0])

        joint_info = {
            'name': joint_name,
            'type': joint_type,
            'parent': parent,
            'child': child,
            'origin': origin_frame,
            'axis': axis,
            'limit': limit
        }
        joints.append(joint_info)

    for link in links:
        body_name = link['name']
        if link['collision'] is None:
            logger.info(f'Body {body_name} did not have any shape, likely virtual body, ignoring it')
            continue
        V = link['collision']['vertices']
        F = link['collision']['faces']
        mesh = API.create_mesh(V, F)
        shape_name = body_name + "shape"
        API.create_shape(engine, shape_name, mesh)

        # Transform from link frame to body frame
        r_g = link['collision']['origin'][0]
        q_g = link['collision']['origin'][1]

        mass = link['inertial']['mass']
        ixx, iyy, izz, ixy, ixz, iyz = link['inertial']['inertia']

        # Transform from link frame to body frame
        r_b = link['inertial']['origin'][0]
        q_b = link['inertial']['origin'][1]

        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)
        # Overriding API.set_mass_properties(engine, body_name, density)
        engine.bodies[body_name].mass = mass
        engine.bodies[body_name].inertia = V3.make(ixx, iyy, izz)

        API.set_position(engine, body_name, r_b, use_model_frame=True)
        API.set_orientation(engine, body_name, q_b, use_model_frame=True)

    for joint in joints:
        joint_name = joint['name']
        joint_type = joint['type']
        if type!='revolute':
            logger.info(f'Joint {joint_name} has {joint_type} type that is not supported, ignoring it')
            continue
        API.create_hinge(engine, joint_name)

        parent_name = joint['parent']
        child_name = joint['child']
        r_j = joint['origin'][0]   # Transform from parent link to child link
        q_j = joint['origin'][1]
        axis = joint['axis']       # Unit vector in the joint frame

        #API.set_hinge(engine, joint_name, parent_name, child_name, origin, axis, mode="model")

    return robot_name, links, joints


if __name__ == "__main__":
    engine = TYPES.Engine()

    # Data is from https://github.com/Gepetto/example-robot-data
    package_folder = "/Users/kenny/Documents/GitHub/"
    urdf_file_path = 'example-robot-data/robots/ur_description/urdf/ur5_robot.urdf'

    name, links, joints = create_from_urdf(engine, package_folder, urdf_file_path)

    print(f"robot name is {name}")
    print("Links:", links)
    print("Joints:", joints)
