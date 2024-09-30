"""
This file contains a function to import the scene content from an URDF file.
"""
import logging
import os
import numpy as np
import igl

import rainbow.math.quaternion as Q
import rainbow.math.vector3 as V3
import rainbow.math.coordsys as X
import rainbow.simulators.prox_rigid_bodies.types as TYPES
import rainbow.simulators.prox_rigid_bodies.api as API

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element


class URDF:

    @staticmethod
    def _read_numbers_from_string(string: str) -> list[float]:
        # Split the string into a list of number strings
        number_strings = string.split()

        # Convert the list of number strings to a list of floats
        numbers = [float(num) for num in number_strings]
        return numbers

    @staticmethod
    def _convert_frame(frame: dict) -> tuple:
        rpy = [0.0, 0.0, 0.0]
        xyz = [0.0, 0.0, 0.0]
        if 'rpy' in frame:
            rpy = URDF._read_numbers_from_string(frame['rpy'])
        if 'xyz' in frame:
            xyz = URDF._read_numbers_from_string(frame['xyz'])

        Qx = Q.Rx(rpy[0])
        Qy = Q.Ry(rpy[1])
        Qz = Q.Rz(rpy[2])

        Qtot = Q.prod(Qz, Q.prod(Qy, Qx))
        Ttot = V3.make(xyz[0], xyz[1], xyz[2])
        return Ttot, Qtot

    @staticmethod
    def _convert_mass(mass: dict) -> float:
        value = 0
        if 'value' in mass:
            value = float(mass['value'])
        return value

    @staticmethod
    def _convert_inertia_tensor(inertia: dict) -> list[float]:
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

    @staticmethod
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
        origin_frame = URDF._convert_frame(origin_dict)

        scale = URDF._read_numbers_from_string(
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

    @staticmethod
    def _parse_inertial_tag(inertial_tag: Element) -> dict:
        if inertial_tag is None:
            return None

        origin_dict = inertial_tag.find('origin').attrib if inertial_tag.find('origin') is not None else {}
        origin_frame = URDF._convert_frame(origin_dict)

        mass_dict = inertial_tag.find('mass').attrib if inertial_tag.find('mass') is not None else {}
        mass_value = URDF._convert_mass(mass_dict)

        inertia_dict = inertial_tag.find('inertia').attrib if inertial_tag.find('inertia') is not None else {}
        inertia = URDF._convert_inertia_tensor(inertia_dict)

        return {'origin': origin_frame, 'mass': mass_value, 'inertia': inertia}

    @staticmethod
    def parse_urdf_file(package_folder: str, urdf_file_path: str):
        logger = logging.getLogger("rainbow.simulators.prox_rigid_bodies.procedural.create_from_urdf.URDF.parse_urdf_file")
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
                'visual': URDF._parse_geometry_tag(package_folder, visual_tag),
                'collision': URDF._parse_geometry_tag(package_folder, collision_tag),
                'inertial': URDF._parse_inertial_tag(inertial_tag)
            }
            links.append(link_info)
        for joint_tag in root.findall('joint'):
            joint_name = joint_tag.attrib['name']
            joint_type = joint_tag.attrib['type']
            parent = joint_tag.find('parent').attrib['link']
            child = joint_tag.find('child').attrib['link']
            origin_dict = joint_tag.find('origin').attrib if joint_tag.find('origin') is not None else {}
            axis_dict = joint_tag.find('axis').attrib if joint_tag.find('axis') is not None else {'xyz': '1 0 0'}
            limit = joint_tag.find('limit').attrib if joint_tag.find('limit') is not None else {}

            origin_frame = URDF._convert_frame(origin_dict)
            axis = V3.unit(URDF._convert_frame(axis_dict)[0])

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
        return robot_name, links, joints


class LinkBody:

    def __init__(self, name: str, X_b2r):
        self.name = name
        self.X_b2r = X_b2r   # Transform from body frame to link frame
        self.children = []
        self.parent = None
        self.joints = []     # Joint frame information


def create_from_urdf(engine: TYPES.Engine, package_folder: str, urdf_file_path: str):
    """

    :param engine:
    :param package_folder:
    :param urdf_file_path:
    :return:
    """
    logger = logging.getLogger("rainbow.simulators.prox_rigid_bodies.procedural.create_from_urdf")
    logger.info(f"Creating from urdf file path: {urdf_file_path} in package folder {package_folder}")

    robot_name, links, joints = URDF.parse_urdf_file(
        package_folder=package_folder,
        urdf_file_path=urdf_file_path
    )

    robot = {}

    for link in links:
        body_name = link['name']
        if link['collision'] is None:
            logger.info(f'Body {body_name} did not have any shape, likely virtual body, ignoring it')
            continue

        # Coordinate transform from the collision frame to link reference frame
        r_c2r = link['collision']['origin'][0]
        q_c2r = link['collision']['origin'][1]

        # Transform from body frame to link reference frame
        r_b2r = link['inertial']['origin'][0]
        q_b2r = link['inertial']['origin'][1]

        X_c2r = X.make(r_c2r, q_c2r)
        X_b2r = X.make(r_b2r, q_b2r)
        logger.info(f'{body_name} has body frame to reference frame transform ({X_b2r.r}, {X_b2r.q})')

        # This transform brings vertices from collision space into body space
        X_c2b = X.make_coordsys_from_to(X_c2r, X_b2r)
        logger.info(f'{body_name} has collision frame to body from translation {X_c2b.r} and rotation {X_c2b.q}')

        V = link['collision']['vertices']
        F = link['collision']['faces']

        # Next, we will make sure the mesh data lives in the body-space of the link
        # This is only necessary to do if X_c2b is not identity transform
        if not np.array_equal(X_c2b.r, V3.zero()) or not np.array_equal(X_c2b.q, Q.identity()):
            for idx, v in enumerate(V):
                V[idx, :] = X.xform_point(X_c2b, v)
            logger.info(f'Transformed all vertices on {body_name}')

        mesh = API.create_mesh(V, F)
        shape_name = body_name + "shape"
        API.create_shape(engine, shape_name, mesh)

        mass = link['inertial']['mass']
        ixx, iyy, izz, ixy, ixz, iyz = link['inertial']['inertia']

        if ixy != 0 or ixz != 0 or ixz != 0:
            logger.info(
                f'Inertia tensor of {body_name} was not wrt body frame orientation, this is not currently supported'
            )
        else:
            # It is a little unclear if one needs to transform the inertia tensor
            # to get it wrt the body frame of the rigid body?
            logger.info(
                f'Inertia tensor products of {body_name} are {ixx}, {iyy}, and {izz}. '
            )

        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)
        # Overriding API.set_mass_properties(engine, body_name, density)
        engine.bodies[body_name].mass = mass
        engine.bodies[body_name].inertia = V3.make(ixx, iyy, izz)

        logger.info(f'Completed URDF creation of {body_name} rigid body')
        robot[body_name] = LinkBody(body_name, X_b2r)

    for joint in joints:
        joint_name = joint['name']
        joint_type = joint['type']
        if joint_type != 'revolute':
            logger.info(f'Joint {joint_name} has {joint_type} type that is not supported, ignoring it')
            continue
        API.create_hinge(engine, joint_name)

        parent_name = joint['parent']
        child_name = joint['child']

        # Transformation from the joint frame (aka the child reference frame) to the parent reference frame
        r_j2r = joint['origin'][0]
        q_j2r = joint['origin'][1]
        X_j2r = X.make(r_j2r, q_j2r)

        axis = joint['axis']       # Unit vector in the joint frame

        logger.info(f'{joint_name} has child frame to parent frame  translation {X_j2r.r} and rotation {X_j2r.q}')
        logger.info(f'{joint_name} has axis  {axis}')

        # Create the links, so we can more easily traverse the robot mechanism
        robot[parent_name].children.append(child_name)
        robot[child_name].parent = parent_name

        robot[parent_name].joints.append( {'name': joint_name, 'X': X_j2r, 'axis': axis} )

    # We need to make sure we know which link is the root so we can traverse the robot mechanism in correct order
    root = None
    leaves = []
    for link in robot.values():
        if link.parent is None:
            if root is not None:
                logger.error(f'Found two roots of the robot, suggest error in URDF file')
            root = link
        if len(link.children) == 0:
            if link.parent is None:
                logger.error(f'Found leaf link with no parent, suggest a robot with one rigid body')
            leaves.append(link)
    logger.info(f'Found {len(leaves)} leaves with root {root.name}')

    # Finally, we can go from root to leaves and compute world transforms of rigid bodies and joints
    stack = [root]

    API.set_position(
        engine,
        root.name,
        root.X_b2r.r,
        use_model_frame=False
    )
    API.set_orientation(
        engine,
        root.name,
        root.X_b2r.q,
        use_model_frame=False
    )

    while stack:
        parent = stack.pop()
        parent_name = parent.name

        X_r2w = parent.X_b2r  # The parent reference frame wrt to world coordinate system

        for idx, child_name in enumerate(parent.children):
            child = robot[child_name]
            joint = parent.joints[idx]

            joint_name = joint['name']
            axis = joint['axis']
            X_j2r = joint['X']          # The joint frame (aka the child reference frame) location wrt the parent reference frame
            X_b2j = child.X_b2r         # The child body frame wrt the joint frame

            # Compute transforms from body frame and joint frame to the world frame.
            X_j2w = X.concat(X_r2w, X_j2r)
            X_b2w = X.concat(X_j2w,  X_b2j)

            # We need to make sure the child reference frame is updated to reflect is known world location
            # Such that when we traverse the children of the child then we can locate their world placement.
            child.X_b2r = X_j2w

            # Before we can set the joint parameters, we must transform them into world space too.
            o_w = X.xform_point(X_j2w, V3.zero())
            s_w = X.xform_vector(X_j2w, axis)

            # We now know the child location in the world, so we can set this before setting the joint to the parent.
            API.set_position(
                engine,
                child_name,
                X_b2w.r,
                use_model_frame=False
            )
            API.set_orientation(
                engine,
                child_name,
                X_b2w.q,
                use_model_frame=False
            )
            # We can now assume both parent and child rigid bodies are placed correctly in the world prior to
            # setting the joint parameters
            API.set_hinge(
                engine,
                joint_name,
                parent_name,
                child_name,
                o_w,     # This is the joint origin position in the world
                s_w,    # This is the joint axis direction in the world
                mode="world"
            )

            stack.append(robot[child_name])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main.main")
    app_folder = os.path.dirname(__file__)
    logger.info(f"Running from {app_folder}")

    engine = TYPES.Engine()

    # Data is from https://github.com/Gepetto/example-robot-data
    package_folder = "/Users/kenny/Documents/GitHub/"
    urdf_file_path = 'example-robot-data/robots/ur_description/urdf/ur5_robot.urdf'

    create_from_urdf(engine, package_folder, urdf_file_path)
