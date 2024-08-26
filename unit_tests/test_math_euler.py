import unittest
from cmath import sin
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.math.euler as EULER
import rainbow.math.quaternion as Q
import rainbow.math.matrix3 as M3


def compute_orientation_zyx(matrix):
    """
    This is an auxiliary function used to set up test-cases in a more convenient way. It is based on

    https://www.geometricTEST.com/Documentation/EulerAngles.pdf

    :param matrix: A rotation matrix.
    :return: The three Euler angles.
    """
    r20 = matrix[2, 0]
    r12 = matrix[1, 2]
    r11 = matrix[1, 1]

    r10 = matrix[1, 0]
    r00 = matrix[0, 0]
    r21 = matrix[2, 1]
    r22 = matrix[2, 2]

    rz = None
    ry = None
    rx = None

    if r20 >= 1.0:
        rz = np.arctan2(-r12, r11)
        ry = -np.pi / 2
        rx = 0
    elif r20 <= -1.0:
        rz = -np.arctan2(-r12, r11)
        ry = np.pi / 2
        rx = 0
    else:
        rz = np.arctan2(r10, r00)
        ry = np.arcsin(-r20)
        rx = np.arctan2(r21, r22)

    return rx, ry, rz


class TestEulerAPI(unittest.TestCase):
    def test_make_euler_xyz_from_matrix_1(self):
        radians_x = -1.2
        radians_y = np.pi / 2
        radians_z = 1.2

        matrix = np.dot(M3.Rx(radians_z), np.dot(M3.Ry(radians_y), M3.Rz(radians_x)))
        euler_angles = EULER.make_euler_xyz_from_matrix(matrix)
        expected_alpha, expected_beta, expected_gamma = compute_orientation_zyx(matrix)

        actual_alpha = euler_angles.alpha
        actual_beta = euler_angles.beta
        actual_gamma = euler_angles.gamma

        self.assertAlmostEqual(actual_alpha, expected_alpha, 4)
        self.assertAlmostEqual(actual_beta, expected_beta, 4)
        self.assertAlmostEqual(actual_gamma, expected_gamma, 4)

    def test_make_euler_xyz_from_matrix_2(self):
        radians_x = -1.2
        radians_y = np.pi / 2 + np.pi
        radians_z = 1.2

        matrix = np.dot(M3.Rz(radians_z), np.dot(M3.Ry(radians_y), M3.Rx(radians_x)))
        euler_angles = EULER.make_euler_xyz_from_matrix(matrix)
        expected_alpha, expected_beta, expected_gamma = compute_orientation_zyx(matrix)

        actual_alpha = euler_angles.alpha
        actual_beta = euler_angles.beta
        actual_gamma = euler_angles.gamma

        self.assertAlmostEqual(actual_alpha, expected_alpha, 4)
        self.assertAlmostEqual(actual_beta, expected_beta, 4)
        self.assertAlmostEqual(actual_gamma, expected_gamma, 4)

    def test_make_euler_xyz_from_matrix_3(self):
        radians_x = -1.2
        radians_y = 1.1
        radians_z = 1.2

        matrix = np.dot(M3.Rz(radians_z), np.dot(M3.Ry(radians_y), M3.Rx(radians_x)))
        euler_angles = EULER.make_euler_xyz_from_matrix(matrix)
        expected_alpha, expected_beta, expected_gamma = compute_orientation_zyx(matrix)

        actual_alpha = euler_angles.alpha
        actual_beta = euler_angles.beta
        actual_gamma = euler_angles.gamma

        self.assertAlmostEqual(actual_alpha, expected_alpha, 4)
        self.assertAlmostEqual(actual_beta, expected_beta, 4)
        self.assertAlmostEqual(actual_gamma, expected_gamma, 4)

    def test_make_euler_xyz_from_matrix_4(self):
        radians_x = -1.2
        radians_y = -1.1
        radians_z = 1.2

        matrix = np.dot(M3.Rz(radians_z), np.dot(M3.Ry(radians_y), M3.Rx(radians_x)))
        euler_angles = EULER.make_euler_xyz_from_matrix(matrix)
        expected_alpha, expected_beta, expected_gamma = compute_orientation_zyx(matrix)

        actual_alpha = euler_angles.alpha
        actual_beta = euler_angles.beta
        actual_gamma = euler_angles.gamma

        self.assertAlmostEqual(actual_alpha, expected_alpha, 4)
        self.assertAlmostEqual(actual_beta, expected_beta, 4)
        self.assertAlmostEqual(actual_gamma, expected_gamma, 4)

    def test_make_euler_xyz_from_matrix_5(self):
        radians_x = -1.2
        radians_y = -1.1
        radians_z = 1.2

        matrix = np.dot(M3.Rz(radians_z), np.dot(M3.Ry(radians_y), M3.Rx(radians_x)))
        euler_angles = EULER.make_euler_xyz_from_matrix(matrix)
        expected_alpha, expected_beta, expected_gamma = radians_x, radians_y, radians_z

        actual_alpha = euler_angles.alpha
        actual_beta = euler_angles.beta
        actual_gamma = euler_angles.gamma

        self.assertEqual(actual_alpha, expected_alpha)
        self.assertEqual(actual_beta, expected_beta)
        self.assertEqual(actual_gamma, expected_gamma)

    def test_make_euler_xyz_from_matrix_6(self):
        radians_x = -1.3
        radians_y = -1.2
        radians_z = 0.3

        matrix = np.dot(M3.Rz(radians_z), np.dot(M3.Ry(radians_y), M3.Rx(radians_x)))
        euler_angles = EULER.make_euler_xyz_from_matrix(matrix)
        expected_alpha, expected_beta, expected_gamma = radians_x, radians_y, radians_z

        actual_alpha = euler_angles.alpha
        actual_beta = euler_angles.beta
        actual_gamma = euler_angles.gamma

        self.assertAlmostEqual(actual_alpha, expected_alpha, 4)
        self.assertAlmostEqual(actual_beta, expected_beta, 4)
        self.assertAlmostEqual(actual_gamma, expected_gamma, 4)

    def test_make_euler_xyz_from_matrix_7(self):
        radians_x = 0.0
        radians_y = np.pi / 2
        radians_z = 0.0

        matrix = np.dot(M3.Rz(radians_z), np.dot(M3.Ry(radians_y), M3.Rx(radians_x)))
        euler_angles = EULER.make_euler_xyz_from_matrix(matrix)
        expected_alpha, expected_beta, expected_gamma = radians_x, radians_y, radians_z

        actual_alpha = euler_angles.alpha
        actual_beta = euler_angles.beta
        actual_gamma = euler_angles.gamma

        self.assertAlmostEqual(actual_alpha, expected_alpha, 4)
        self.assertAlmostEqual(actual_beta, expected_beta, 4)
        self.assertAlmostEqual(actual_gamma, expected_gamma, 4)

    def test_make_euler_xyz_from_matrix_8(self):
        radians_x = 0.0
        radians_y = np.pi / 2 + np.pi
        radians_z = 0.0

        matrix = np.dot(M3.Rz(radians_z), np.dot(M3.Ry(radians_y), M3.Rx(radians_x)))
        euler_angles = EULER.make_euler_xyz_from_matrix(matrix)
        expected_alpha, _, expected_gamma = radians_x, radians_y, radians_z

        actual_alpha = euler_angles.alpha
        actual_beta = sin(euler_angles.beta)
        actual_gamma = euler_angles.gamma

        self.assertAlmostEqual(actual_alpha, expected_alpha, 4)
        self.assertAlmostEqual(actual_beta, -1, 4)
        self.assertAlmostEqual(actual_gamma, expected_gamma, 4)

    def test_euler_xyz_from_quaternion_1(self):
        radians_x = 0.0
        radians_y = np.pi / 2 + np.pi
        radians_z = 0.0

        matrix = np.dot(M3.Rz(radians_z), np.dot(M3.Ry(radians_y), M3.Rx(radians_x)))
        q = Q.from_matrix(matrix)

        euler_angles = EULER.make_euler_xyz_from_quaternion(q)
        expected_alpha, _, expected_gamma = radians_x, radians_y, radians_z

        actual_alpha = euler_angles.alpha
        actual_beta = sin(euler_angles.beta)
        actual_gamma = euler_angles.gamma

        self.assertAlmostEqual(actual_alpha, expected_alpha, 4)
        self.assertAlmostEqual(actual_beta, -1, 4)
        self.assertAlmostEqual(actual_gamma, expected_gamma, 4)

    def test_euler_xyz_from_quaternion_2(self):
        radians_x = 0.4
        radians_y = 1.1
        radians_z = -1.3

        matrix = np.dot(M3.Rz(radians_z), np.dot(M3.Ry(radians_y), M3.Rx(radians_x)))
        q = Q.from_matrix(matrix)

        euler_angles = EULER.make_euler_xyz_from_quaternion(q)
        expected_alpha, expected_beta, expected_gamma = radians_x, radians_y, radians_z

        actual_alpha = euler_angles.alpha
        actual_beta = euler_angles.beta
        actual_gamma = euler_angles.gamma

        self.assertAlmostEqual(actual_alpha, expected_alpha, 4)
        self.assertAlmostEqual(actual_beta, expected_beta, 4)
        self.assertAlmostEqual(actual_gamma, expected_gamma, 4)

    def test_euler_xyz_from_quaternion_3(self):
        radians_x = -0.4
        radians_y = 0.5
        radians_z = 1.7

        matrix = np.dot(M3.Rz(radians_z), np.dot(M3.Ry(radians_y), M3.Rx(radians_x)))
        q = Q.from_matrix(matrix)

        euler_angles = EULER.make_euler_xyz_from_quaternion(q)
        expected_alpha, expected_beta, expected_gamma = radians_x, radians_y, radians_z

        actual_alpha = euler_angles.alpha
        actual_beta = euler_angles.beta
        actual_gamma = euler_angles.gamma

        self.assertAlmostEqual(actual_alpha, expected_alpha, 4)
        self.assertAlmostEqual(actual_beta, expected_beta, 4)
        self.assertAlmostEqual(actual_gamma, expected_gamma, 4)

    def test_euler_xyz_from_quaternion_4(self):
        radians_x = 0.0
        radians_y = np.pi / 2
        radians_z = 0.0

        matrix = np.dot(M3.Rz(radians_z), np.dot(M3.Ry(radians_y), M3.Rx(radians_x)))
        q = Q.from_matrix(matrix)

        euler_angles = EULER.make_euler_xyz_from_quaternion(q)
        expected_alpha, _, expected_gamma = radians_x, radians_y, radians_z

        actual_alpha = euler_angles.alpha
        actual_beta = sin(euler_angles.beta)
        actual_gamma = euler_angles.gamma

        self.assertAlmostEqual(actual_alpha, expected_alpha, 4)
        self.assertAlmostEqual(actual_beta, 1, 4)
        self.assertAlmostEqual(actual_gamma, expected_gamma, 4)
