import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.simulators.prox_rigid_bodies.mass as MASS
from rainbow.simulators.prox_rigid_bodies.types import Shape


def get_cube_inertia_matrix(mass, x, y, z):
    """
    Given mass and dimensions of a cube return inertia matrix.
    :return: ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz
    FROM https://gist.github.com/awesomebytes/39a4ba6c64956a1aa9bd
    From https://www.wolframalpha.com/input/?i=moment+of+inertia+cube"""
    ixx = (1.0 / 12.0) * (y ** 2 + z ** 2) * mass
    iyy = (1.0 / 12.0) * (x ** 2 + z ** 2) * mass
    izz = (1.0 / 12.0) * (x ** 2 + y ** 2) * mass
    ixy = 0.0
    ixz = 0.0
    iyz = 0.0
    return [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]


class TestInertiaTensor(unittest.TestCase):

    def test_compute_inertia_tensor_1(self):
        width, height, depth = 4., 5., 6.
        V, T = MESH.create_box(width, height, depth)
        # Translate box on all two axes and verify we get body-frame inertia correctly computed.
        translate = V3.make(0, 1, 1)
        V = V + translate
        mesh = API.create_mesh(V, T)
        shape = Shape("box_shape")
        shape.mesh = mesh
        prop1 = MASS.compute_mass_properties(mesh.V, mesh.T, 1.0)
        _r_bf2mf, _q_bf2mf, m, I_body = MASS.xform_model_2_body_space(prop1)
        inertia_tensor = np.array(get_cube_inertia_matrix(120, width, height, depth))
        self.assertEqual(inertia_tensor[0, 0], I_body[0])
        self.assertEqual(inertia_tensor[1, 1], I_body[1])
        self.assertEqual(inertia_tensor[2, 2], I_body[2])

    def test_compute_inertia_tensor_2(self):
        width, height, depth = 4., 5., 6.
        V, T = MESH.create_box(width, height, depth)
        # Translate box on all three axes and verify we get body-frame inertia correctly computed.
        translate = V3.make(2, 1, 1)
        V = V + translate
        mesh = API.create_mesh(V, T)
        shape = Shape("box_shape")
        shape.mesh = mesh
        prop1 = MASS.compute_mass_properties(mesh.V, mesh.T, 1.0)
        _r_bf2mf, _q_bf2mf, m, I_body = MASS.xform_model_2_body_space(prop1)
        inertia_tensor = np.array(get_cube_inertia_matrix(120, width, height, depth))
        self.assertEqual(inertia_tensor[0, 0], I_body[0])
        self.assertEqual(inertia_tensor[1, 1], I_body[1])
        self.assertEqual(inertia_tensor[2, 2], I_body[2])

    def test_compute_inertia_tensor_3(self):
        width, height, depth = 4., 5., 6.
        V, T = MESH.create_box(width, height, depth)
        # Translate box on all three axes and verify we get body-frame inertia correctly computed.
        translate = V3.make(2, 10, -1)
        V = V + translate
        mesh = API.create_mesh(V, T)
        shape = Shape("box_shape")
        shape.mesh = mesh
        prop1 = MASS.compute_mass_properties(mesh.V, mesh.T, 1.0)
        _r_bf2mf, _q_bf2mf, m, I_body = MASS.xform_model_2_body_space(prop1)
        inertia_tensor = np.array(get_cube_inertia_matrix(120, width, height, depth))
        self.assertEqual(inertia_tensor[0, 0], I_body[0])
        self.assertEqual(inertia_tensor[1, 1], I_body[1])
        self.assertEqual(inertia_tensor[2, 2], I_body[2])

    def test_compute_inertia_tensor_4(self):
        width, height, depth = 4., 5., 6.
        V, T = MESH.create_box(width, height, depth)
        # Rotate box around world x-axis and verify we get body-frame inertia correctly computed.
        q = Q.Rx(0.3 * np.pi)
        V = Q.rotate_array(q, V)
        mesh = API.create_mesh(V, T)
        shape = Shape("box_shape")
        shape.mesh = mesh
        shape.q = q
        prop1 = MASS.compute_mass_properties(mesh.V, mesh.T, 1.0)
        _r_bf2mf, _q_bf2mf, m, I_body = MASS.xform_model_2_body_space(prop1)
        inertia_tensor = np.array(get_cube_inertia_matrix(120, width, height, depth))
        self.assertAlmostEqual(inertia_tensor[0, 0], I_body[0])
        self.assertAlmostEqual(inertia_tensor[1, 1], I_body[1])
        self.assertAlmostEqual(inertia_tensor[2, 2], I_body[2])

    def test_compute_inertia_tensor_5(self):
        width, height, depth = 4., 5., 6.
        V, T = MESH.create_box(width, height, depth)
        # Rotate box around world y-axis and verify we get body-frame inertia correctly computed.
        q = Q.Ry(0.3 * np.pi)
        V = Q.rotate_array(q, V)
        mesh = API.create_mesh(V, T)
        shape = Shape("box_shape")
        shape.mesh = mesh
        shape.q = q
        prop1 = MASS.compute_mass_properties(mesh.V, mesh.T, 1.0)
        _r_bf2mf, _q_bf2mf, m, I_body = MASS.xform_model_2_body_space(prop1)
        inertia_tensor = np.array(get_cube_inertia_matrix(120, width, height, depth))
        self.assertAlmostEqual(inertia_tensor[0, 0], I_body[0])
        self.assertAlmostEqual(inertia_tensor[1, 1], I_body[1])
        self.assertAlmostEqual(inertia_tensor[2, 2], I_body[2])

    def test_compute_inertia_tensor_6(self):
        width, height, depth = 4., 5., 6.
        V, T = MESH.create_box(width, height, depth)
        # Rotate box around world z-axis and verify we get body-frame inertia correctly computed.
        q = Q.Rz(0.3 * np.pi)
        V = Q.rotate_array(q, V)
        mesh = API.create_mesh(V, T)
        shape = Shape("box_shape")
        shape.mesh = mesh
        shape.q = q
        prop1 = MASS.compute_mass_properties(mesh.V, mesh.T, 1.0)
        _r_bf2mf, _q_bf2mf, m, I_body = MASS.xform_model_2_body_space(prop1)
        inertia_tensor = np.array(get_cube_inertia_matrix(120, width, height, depth))
        self.assertAlmostEqual(inertia_tensor[0, 0], I_body[0])
        self.assertAlmostEqual(inertia_tensor[1, 1], I_body[1])
        self.assertAlmostEqual(inertia_tensor[2, 2], I_body[2])
