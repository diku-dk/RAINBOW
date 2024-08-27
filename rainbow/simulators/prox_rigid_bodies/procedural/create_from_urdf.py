"""
This file contains a function to import the scene content from an URDF file.
"""
import logging
import numpy as np
import xml.etree.ElementTree as ET

import rainbow.math.quaternion as Q
import rainbow.simulators.prox_rigid_bodies.types as TYPES


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

    # Parse links
    links = []
    for link in root.findall('link'):
        link_name = link.get('name')
        links.append(link_name)

    # Parse joints
    joints = []
    for joint in root.findall('joint'):
        #
        # <joint name="shoulder_lift_joint" type="revolute">
        #     <parent link="shoulder_link"/>
        #     <child link="upper_arm_link"/>
        #     <origin rpy="0.0 1.570796325 0.0" xyz="0.0 0.13585 0.0"/>
        #     <axis xyz="0 1 0"/>
        #     <limit effort="150.0" lower="-3.14159265" upper="3.14159265" velocity="3.15"/>
        #   </joint>
        #
        # The rpy attribute stands for roll, pitch and yaw. These are naming conventions that are defined as follows:
        #
        # Roll: Rotation around the x-axis
        # Pitch: Rotation around the y-axis
        # Yaw: Rotation around the z-axis
        #
        # The URDF tells us the rotation angle in radiants, so in order to get to the third coordinate frame we must
        # rotate 1.570796325 rad (90 degrees) around the y-axis. This scheme is continued until we end up at the last
        # frame at the last robot joint.
        #
        joint_name = joint.get('name')
        parent_link = joint.find('parent').get('link')
        child_link = joint.find('child').get('link')
        joint_type = joint.get('type')
        joints.append({
            'name': joint_name,
            'parent': parent_link,
            'child': child_link,
            'type': joint_type
        })

    return links, joints





# Example usage:
urdf_file = 'path_to_urdf_file.urdf'
links, joints = parse_urdf(urdf_file)

print("Links:")
for link in links:
    print(f"  {link}")

print("\nJoints:")
for joint in joints:
    print(f"  Joint Name: {joint['name']}")
    print(f"    Parent Link: {joint['parent']}")
    print(f"    Child Link: {joint['child']}")
    print(f"    Joint Type: {joint['type']}")