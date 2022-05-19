import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.math.quaternion as Q
import rainbow.util.test_tools as TEST


class TestQuaternionAPI(unittest.TestCase):
    def test_quaterion_make_1(self):
        # First define a quaterion q:
        # [s, (x,y,z)]
        # p. 600, Physics-Based Animation, K. Erleben et al

        expected = np.array([1, 2, 3, 4], dtype=np.float64)
        actual = Q.make(1, 2, 3, 4)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_quaterion_make_2(self):
        # First define a quaterion q:
        # [s, (x,y,z)]
        # p. 600, Physics-Based Animation, K. Erleben et al

        expected = np.array([1, 2, 3, 4], dtype=np.float64)
        actual = Q.make(1, -2, 3, 4)
        self.assertTrue(TEST.is_array_not_equal(actual, expected))

    def test_quaterion_prod_1(self):
        # First define a quaterion q:
        # [s, (x,y,z)]
        # p. 600, Physics-Based Animation, K. Erleben et al

        q = np.array([1, 2, 3, 4])

        # Using definition 18.41
        to_c_v = np.array([1, -1, -1, -1], dtype=np.float64)
        q_c = to_c_v * q
        qq_c = np.ones(4)
        qq_c[0] = q[0] * q_c[0] - np.dot(q[1:], q_c[1:])
        qq_c[1:] = q[0] * q_c[1:] + q_c[0] * q[1:] + np.cross(q[1:], q_c[1:], axis=0)
        #

        expected = qq_c
        actual = Q.prod(q, q_c)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_quaterion_prod_2(self):
        # First define a quaterion q:
        # [s, (x,y,z)]
        # p. 600, Physics-Based Animation, K. Erleben et al

        q = np.array([3, 1, 6, -4])

        # Using definition 18.41
        to_c_v = np.array([1, -1, -1, -1], dtype=np.float64)
        q_c = to_c_v * q
        qq_c = np.ones(4)
        qq_c[0] = q[0] * q_c[0] - np.dot(q[1:], q_c[1:])
        qq_c[1:] = q[0] * q_c[1:] + q_c[0] * q[1:] + np.cross(q[1:], q_c[1:], axis=0)
        #

        expected = qq_c
        actual = Q.prod(q, q_c)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_quaterion_prod_3(self):
        # First define a quaterion q:
        # [s, (x,y,z)]
        # p. 600, Physics-Based Animation, K. Erleben et al

        q = np.array([0.3, 0.10000, 0.00006, -8])

        to_c_v = np.array([1, -1, -1, -1], dtype=np.float64)
        q_c = to_c_v * q
        qq_c = np.ones(4)
        qq_c[0] = q[0] * q_c[0] - np.dot(q[1:], q_c[1:])
        qq_c[1:] = q[0] * q_c[1:] + q_c[0] * q[1:] + np.cross(q[1:], q_c[1:], axis=0)
        #

        expected = qq_c
        actual = Q.prod(q, q_c)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_quaterion_unit_1(self):
        # Using def 18.44
        q = np.array([2, 2, 3, 4], dtype=np.float64)
        q_c = Q.conjugate(q)
        # Remember: ai * (-ai) = -aaii = a^2 
        power_two = (q[0] * (q_c[0])
                     + q[1] * (-1 * q_c[1])
                     + q[2] * (-1 * q_c[2])
                     + q[3] * (-1 * q_c[3]))
        norm = np.sqrt(power_two)
        expected = q / norm
        actual = Q.unit(q)

        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_quaterion_unit_2(self):
        # Using def 18.44
        q = np.array([0.1, 2.3, -1000, 0.004], dtype=np.float64)
        q_c = Q.conjugate(q)
        # Remember: ai * (-ai) = -aaii = a^2 
        power_two = (q[0] * (q_c[0])
                     + q[1] * (-1 * q_c[1])
                     + q[2] * (-1 * q_c[2])
                     + q[3] * (-1 * q_c[3]))
        norm = np.sqrt(power_two)
        expected = q / norm
        actual = Q.unit(q)

        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_quaterion_unit_3(self):
        # Using def 18.44
        q = np.array([0, 0, 0, 0], dtype=np.float64)
        expected = np.array([1, 0, 0, 0], dtype=np.float64)
        actual = Q.unit(q)

        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_quaterion_unit_4(self):
        # Using def 18.44
        q = np.array([2, 2, 3, 4], dtype=np.float64)
        q_c = Q.conjugate(q)
        # Remember: ai * (-ai) = -aaii = a^2 

        norm = np.sqrt(Q.prod(q, q_c)[0])
        expected = q / norm
        actual = Q.unit(q)

        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_rotate_1(self):
        n = np.array([1, 0, 0], dtype=np.float64)
        theta = 2 * np.pi
        q = Q.make(np.cos(theta), np.sin(n[0]), np.sin(n[1]), np.sin(n[2]))
        p = np.array([0, 2, 2, 2], dtype=np.float64)

        qr = np.array([0, p[0], p[1], p[2]], dtype=np.float64)

        expected = Q.prod(Q.prod(q, qr), Q.conjugate(q))
        actual = Q.rotate(q, p)
        self.assertTrue(TEST.is_array_equal(actual, expected[1:]))

    def test_rotate_2(self):
        q = Q.unit(Q.make(1, 2, 3, 4))
        # n     = np.array([1,0,0], dtype=np.float64)
        # theta = 2*np.pi
        # q     = quat.make(np.cos(theta), np.sin(n[0]), np.sin(n[1]), np.sin(n[2]))
        p = np.array([0, 2, 2, 2], dtype=np.float64)

        qr = np.array([0, p[0], p[1], p[2]], dtype=np.float64)

        expected = Q.prod(Q.prod(q, qr), Q.conjugate(q))
        actual = Q.rotate(q, p)
        self.assertTrue(TEST.is_array_equal(actual, expected[1:]))

    def test_rotate_3(self):
        r = np.array([1, 0, 0])
        n = np.array([0, 1, 0])
        radian = (0.5 * np.pi) / 2
        q = np.array([np.cos(radian), n[0] * np.sin(radian), n[1] * np.sin(radian), n[2] * np.sin(radian)],
                     dtype=np.float64)
        expected = np.array([0, 0, -1])
        actual = Q.rotate(q, r)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_rotate_4(self):
        r = np.array([1, 0, 0])
        n = np.array([0, 0, 1])
        radian = (0.5 * np.pi) / 2
        q = np.array([np.cos(radian), n[0] * np.sin(radian), n[1] * np.sin(radian), n[2] * np.sin(radian)],
                     dtype=np.float64)
        expected = np.array([0, 1, 0])
        actual = Q.rotate(q, r)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_rotate_5(self):
        r = np.array([1, 0, 0])
        n = np.array([1, 0, 0])
        radian = (0.5 * np.pi) / 2
        q = np.array([np.cos(radian), n[0] * np.sin(radian), n[1] * np.sin(radian), n[2] * np.sin(radian)],
                     dtype=np.float64)
        expected = np.array([1, 0, 0])
        actual = Q.rotate(q, r)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_rotate_6(self):
        r = np.array([0, 1, 0])
        n = np.array([0, 0, 1])
        radian = (0.5 * np.pi) / 2
        q = np.array([np.cos(radian), n[0] * np.sin(radian), n[1] * np.sin(radian), n[2] * np.sin(radian)],
                     dtype=np.float64)
        expected = np.array([-1, 0, 0])
        actual = Q.rotate(q, r)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_rotate_7(self):
        r = np.array([0, 1, 0])
        n = np.array([1, 0, 0])
        radian = (0.5 * np.pi) / 2
        q = np.array([np.cos(radian), n[0] * np.sin(radian), n[1] * np.sin(radian), n[2] * np.sin(radian)],
                     dtype=np.float64)
        expected = np.array([0, 0, 1])
        actual = Q.rotate(q, r)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_rotate_8(self):
        r = np.array([0, 1, 0])
        n = np.array([0, 1, 0])
        radian = (0.5 * np.pi) / 2
        q = np.array([np.cos(radian), n[0] * np.sin(radian), n[1] * np.sin(radian), n[2] * np.sin(radian)],
                     dtype=np.float64)
        expected = np.array([0, 1, 0])
        actual = Q.rotate(q, r)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_rotate_9(self):
        r = np.array([0, 0, 1])
        n = np.array([1, 0, 0])
        radian = (0.5 * np.pi) / 2
        q = np.array([np.cos(radian), n[0] * np.sin(radian), n[1] * np.sin(radian), n[2] * np.sin(radian)],
                     dtype=np.float64)
        expected = np.array([0, -1, 0])
        actual = Q.rotate(q, r)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_rotate_10(self):
        r = np.array([0, 0, 1])
        n = np.array([0, 1, 0])
        radian = (0.5 * np.pi) / 2
        q = np.array([np.cos(radian), n[0] * np.sin(radian), n[1] * np.sin(radian), n[2] * np.sin(radian)],
                     dtype=np.float64)
        expected = np.array([1, 0, 0])
        actual = Q.rotate(q, r)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_rotate_11(self):
        r = np.array([0, 0, 1])
        n = np.array([0, 0, 1])
        radian = (0.5 * np.pi) / 2
        q = np.array([np.cos(radian), n[0] * np.sin(radian), n[1] * np.sin(radian), n[2] * np.sin(radian)],
                     dtype=np.float64)
        expected = np.array([0, 0, 1])
        actual = Q.rotate(q, r)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_string_1(self):
        expected = np.array([1, 0, 0, 0], dtype=np.float64)
        actual = Q.from_string("identity")
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_string_2(self):
        degree = 90
        expected_radian = (0.5 * np.pi) / 2
        expected = np.array([np.cos(expected_radian), np.sin(expected_radian), 0, 0], dtype=np.float64)
        actual = Q.from_string(f"rx: {degree}")
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_string_3(self):
        degree = 90
        expected_radian = (0.5 * np.pi) / 2
        expected = np.array([np.cos(expected_radian), np.sin(expected_radian), 0, 0], dtype=np.float64)
        actual = Q.from_string(f"Rx: {degree}")
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_string_3b(self):
        degree = 90
        expected_radian = (0.5 * np.pi) / 2
        expected = np.array([np.cos(expected_radian), np.sin(expected_radian), 0, 0], dtype=np.float64)
        actual = Q.from_string(f"RX: {degree}")
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_string_3c(self):
        degree = 90
        expected_radian = (0.5 * np.pi) / 2
        expected = np.array([np.cos(expected_radian), np.sin(expected_radian), 0, 0], dtype=np.float64)
        actual = Q.from_string(f"rX: {degree}")
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_string_4(self):
        degree = 90
        expected_radian = (0.5 * np.pi) / 2
        expected = np.array([np.cos(expected_radian), 0, np.sin(expected_radian), 0], dtype=np.float64)
        actual = Q.from_string(f"ry: {degree}")
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_string_5(self):
        degree = 90
        expected_radian = (0.5 * np.pi) / 2
        expected = np.array([np.cos(expected_radian), 0, 0, np.sin(expected_radian)], dtype=np.float64)
        actual = Q.from_string(f"rz: {degree}")
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_string_6(self):
        degree = 90
        expected_radian = (0.5 * np.pi) / 2
        axis = "[1,0,0]"
        expected = np.array([np.cos(expected_radian), np.sin(expected_radian), 0, 0, ], dtype=np.float64)
        actual = Q.from_string(f"ru: {degree}:{axis}")
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_string_7(self):
        degree = 90
        expected_radian = (0.5 * np.pi) / 2
        axis = "[0,1,0]"
        expected = np.array([np.cos(expected_radian), 0, np.sin(expected_radian), 0, ], dtype=np.float64)
        actual = Q.from_string(f"ru: {degree}:{axis}")
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_string_8(self):
        degree = 90
        expected_radian = (0.5 * np.pi) / 2
        axis = "[0,0,1]"
        expected = np.array([np.cos(expected_radian), 0, 0, np.sin(expected_radian)], dtype=np.float64)
        actual = Q.from_string(f"ru: {degree}:{axis}")
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_string_9(self):
        expected = np.array([1, 2, 3, 4], dtype=np.float64)
        actual = Q.from_string(f"[1,2,3,4,5,6,7]")
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_string_10(self):
        with self.assertRaises(AssertionError):
            Q.from_string(f"[1,2,3]")

    def test_from_string_11(self):
        with self.assertRaises(AssertionError):
            Q.from_string(f"rx: [1,2,3]")

    def test_from_vector3_1(self):
        expected = np.array([0, 1, 2, 3], dtype=np.float64)
        actual = Q.from_vector3(np.array([1, 2, 3]))
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_vector3_2(self):
        expected = np.array([0, 1, 2, 3], dtype=np.float64)
        actual = Q.from_vector3(np.array([1, 2, 3, 4]))
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_vector3_3(self):
        with self.assertRaises(IndexError):
            Q.from_vector3(np.array([1, 2]))

    def test_to_matrix_1(self):
        radian = (0.5 * np.pi) / 2
        expected = Q.make(np.cos(radian), np.sin(radian), 0, 0)
        actual = Q.from_matrix(Q.to_matrix(expected))
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_to_matrix_2(self):
        radian = (0.5 * np.pi) / 2
        axis = np.array([1, 1, 0])
        axis_unit = axis / np.linalg.norm(axis)
        expected = Q.make(np.cos(radian), axis_unit[0] * np.sin(radian), axis_unit[1] * np.sin(radian),
                          axis_unit[2] * np.sin(radian))
        actual = Q.from_matrix(Q.to_matrix(expected))
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_to_matrix_3(self):
        '''
            p. 602, theorem 18.44
        '''
        radian = (0.5 * np.pi) / 2
        q = np.array([np.cos(radian), np.sin(radian), 0, 0])
        expected = np.array([
            [1, 0, 0],
            [0, (1 - 2 * (np.sin(radian) ** 2)), -2 * np.cos(radian) * np.sin(radian)],
            [0, 2 * np.cos(radian) * np.sin(radian), 1 - 2 * (np.sin(radian) ** 2)]
        ])

        actual = Q.to_matrix(q)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_matrix_1(self):
        '''
         Use Theorem 18.44, however use reverted indexing to support
         numpy arrays
        '''
        radian = (0.5 * np.pi) / 2
        M = np.array([
            [1, 0, 0],
            [0, (1 - 2 * (np.sin(radian) ** 2)), -2 * np.cos(radian) * np.sin(radian)],
            [0, 2 * np.cos(radian) * np.sin(radian), 1 - 2 * (np.sin(radian) ** 2)]
        ])
        s = 0.5 * np.sqrt(M[0, 0] + M[1, 1] + M[2, 2] + 1)
        x = (M[2, 1] - M[1, 2]) / (4 * s)
        y = (M[2, 0] - M[0, 2]) / (4 * s)
        z = (M[0, 1] - M[1, 0]) / (4 * s)
        expected = np.array([s, x, y, z], dtype=np.float64)
        actual = Q.from_matrix(M)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_matrix_1(self):
        '''
         Use Theorem 18.44, however use reverted indexing to support
         numpy arrays
        '''
        radian = (0.5 * np.pi) / 2
        M = np.array([
            [1, 0, 0],
            [0, (1 - 2 * (np.sin(radian) ** 2)), -2 * np.cos(radian) * np.sin(radian)],
            [0, 2 * np.cos(radian) * np.sin(radian), 1 - 2 * (np.sin(radian) ** 2)]
        ])
        s = 0.5 * np.sqrt(M[0, 0] + M[1, 1] + M[2, 2] + 1)
        x = (M[2, 1] - M[1, 2]) / (4 * s)
        y = (M[2, 0] - M[0, 2]) / (4 * s)
        z = (M[0, 1] - M[1, 0]) / (4 * s)
        expected = np.array([s, x, y, z], dtype=np.float64)
        actual = Q.from_matrix(M)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_matrix_2(self):
        radian = np.pi / 2
        q = np.array([np.cos(radian), np.sin(radian), 0, 0], dtype=np.float64)
        M = Q.to_matrix(q)
        x = (np.sqrt(M[0, 0] - M[1, 1] - M[2, 2] + 1.0)) / (2)
        y = (M[0, 1] + M[1, 0]) / (4 * x)
        z = (M[2, 0] + M[0, 2]) / (4 * x)
        s = (M[1, 2] - M[2, 1]) / (4 * x)
        expected = np.array([s, x, y, z], dtype=np.float64)
        actual = Q.from_matrix(M)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_matrix_3(self):
        radian = np.pi / 2
        axis = [0, 1, 0]
        q = np.array([np.cos(radian), axis[0] * np.sin(radian), axis[1] * np.sin(radian), axis[2] * np.sin(radian)],
                     dtype=np.float64)
        M = Q.to_matrix(q)
        expected = q
        actual = Q.from_matrix(M)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_matrix_4(self):
        radian = np.pi / 2
        axis = [0, 0, 1]
        q = np.array([np.cos(radian), axis[0] * np.sin(radian), axis[1] * np.sin(radian), axis[2] * np.sin(radian)],
                     dtype=np.float64)
        M = Q.to_matrix(q)
        expected = q
        actual = Q.from_matrix(M)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_rotate_array_1(self):
        radian = (0.5 * np.pi) / 2
        axis = np.array([0, 1, 0])
        axis_unit = axis / np.linalg.norm(axis)
        q = Q.make(np.cos(radian), axis_unit[0] * np.sin(radian), axis_unit[1] * np.sin(radian),
                   axis_unit[2] * np.sin(radian))
        rs = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        expected = np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]
        ], dtype=np.float64)
        actual = Q.rotate_array(q, rs)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_prod_array_1(self):
        radian = (0.5 * np.pi) / 2
        axis = np.array([0, 1, 0])
        axis_unit = axis / np.linalg.norm(axis)
        rs = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float64)

        with self.assertRaises(NotImplementedError):
            Q.prod_array(rs, rs)

    def test_prod_array_3(self):
        wrong_input = np.array([])
        with self.assertRaises(ValueError):
            Q.prod_array(wrong_input, wrong_input)

    def test_angle_axis_1(self):
        radian = (0.5 * np.pi) / 2
        axis = np.array([0, 1, 0])
        axis_unit = axis / np.linalg.norm(axis)
        q = Q.make(np.cos(radian), axis_unit[0] * np.sin(radian), axis_unit[1] * np.sin(radian),
                   axis_unit[2] * np.sin(radian))
        expected_theta = radian * 2
        expected_axis = axis_unit
        actual_theta, actual_axis = Q.to_angle_axis(q)
        self.assertTrue(TEST.is_array_equal(actual_theta, expected_theta))
        self.assertTrue(TEST.is_array_equal(actual_axis, expected_axis))

    def test_angle_axis_2(self):
        radian = (0.75 * np.pi) / 2
        axis = np.array([1, 1, 0])
        axis_unit = axis / np.linalg.norm(axis)
        q = Q.make(np.cos(radian), axis_unit[0] * np.sin(radian), axis_unit[1] * np.sin(radian),
                   axis_unit[2] * np.sin(radian))
        expected_theta = radian * 2
        expected_axis = axis_unit
        actual_theta, actual_axis = Q.to_angle_axis(q)
        self.assertTrue(TEST.is_array_equal(actual_theta, expected_theta))
        self.assertTrue(TEST.is_array_equal(actual_axis, expected_axis))

    def test_angle_axis_2b(self):
        radian = (0.75 * np.pi) / 2
        axis = np.array([1, 1, 1])
        axis_unit = axis / np.linalg.norm(axis)
        q = Q.make(np.cos(radian), axis_unit[0] * np.sin(radian), axis_unit[1] * np.sin(radian),
                   axis_unit[2] * np.sin(radian))
        expected_theta = radian * 2
        expected_axis = axis_unit
        actual_theta, actual_axis = Q.to_angle_axis(q)
        self.assertTrue(TEST.is_array_equal(actual_theta, expected_theta))
        self.assertTrue(TEST.is_array_equal(actual_axis, expected_axis))

    def test_angle_axis_3(self):
        radian = (0.00001 * np.pi) / 2
        axis = np.array([0.001, 0, 0])
        axis_unit = axis / np.linalg.norm(axis)
        q = Q.make(np.cos(radian), axis_unit[0] * np.sin(radian), axis_unit[1] * np.sin(radian),
                   axis_unit[2] * np.sin(radian))
        expected_theta = radian * 2
        expected_axis = axis_unit
        actual_theta, actual_axis = Q.to_angle_axis(q)
        self.assertTrue(TEST.is_array_equal(actual_theta, expected_theta))
        self.assertTrue(TEST.is_array_equal(actual_axis, expected_axis))

    def test_angle_axis_4(self):
        radian = np.pi / 2
        axis = np.array([0, 0, 0])
        axis_unit = axis  # axis / np.linalg.norm(axis)
        q = Q.make(np.cos(radian), axis_unit[0] * np.sin(radian), axis_unit[1] * np.sin(radian),
                   axis_unit[2] * np.sin(radian))
        expected_theta = [0.0, radian * 4]
        expected_axis = axis_unit
        actual_theta, actual_axis = Q.to_angle_axis(q)
        self.assertTrue(actual_theta in expected_theta)
        self.assertTrue(TEST.is_array_equal(actual_axis, expected_axis))

    def test_to_angle_1(self):
        radian = (0.5 * np.pi) / 2
        axis = np.array([0, 1, 0])
        axis_unit = axis / np.linalg.norm(axis)
        q = Q.make(np.cos(radian), axis_unit[0] * np.sin(radian), axis_unit[1] * np.sin(radian),
                   axis_unit[2] * np.sin(radian))
        expected = radian * 2
        actual = Q.to_angle(q, axis_unit)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_to_angle_2(self):
        radian = (1.5 * np.pi) / 2
        axis = np.array([0, 1, 0])
        axis_unit = axis / np.linalg.norm(axis)
        q = Q.make(np.cos(radian), axis_unit[0] * np.sin(radian), axis_unit[1] * np.sin(radian),
                   axis_unit[2] * np.sin(radian))
        expected = np.pi - (radian * 2)
        actual = Q.to_angle(q, axis_unit)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_hat_1(self):
        radian = (0.5 * np.pi) / 2
        axis = np.array([0, 1, 0])
        axis_unit = axis / np.linalg.norm(axis)
        q = Q.make(np.cos(radian), axis_unit[0] * np.sin(radian), axis_unit[1] * np.sin(radian),
                   axis_unit[2] * np.sin(radian))
        expected = 0.5 * np.pi
        actual, _ = Q.to_angle_axis(Q.hat(q))
        self.assertTrue(TEST.is_array_equal(actual / 2, expected))

    def test_hat_2(self):
        radian = (0.25 * np.pi) / 2
        axis = np.array([0, 1, 0])
        axis_unit = axis / np.linalg.norm(axis)
        q = Q.make(np.cos(radian), axis_unit[0] * np.sin(radian), axis_unit[1] * np.sin(radian),
                   axis_unit[2] * np.sin(radian))
        expected = 0.5 * np.pi
        actual, _ = Q.to_angle_axis(Q.hat(q))
        self.assertTrue(TEST.is_array_equal(actual / 2, expected))

    def test_lerp_1(self):
        radian = (0.5 * np.pi) / 2
        axis_1 = np.array([0, 1, 0])
        axis_2 = np.array([0, 1, 0])
        axis_unit_1 = axis_1 / np.linalg.norm(axis_1)
        axis_unit_2 = axis_2 / np.linalg.norm(axis_2)
        q_1 = Q.make(np.cos(radian), axis_unit_1[0] * np.sin(radian), axis_unit_1[1] * np.sin(radian),
                     axis_unit_1[2] * np.sin(radian))
        q_2 = Q.make(np.cos(radian), axis_unit_2[0] * np.sin(radian), axis_unit_2[1] * np.sin(radian),
                     axis_unit_2[2] * np.sin(radian))
        h = 0.5
        expected = q_1 * (1 - h) + q_2 * h
        actual = Q.lerp(q_1, q_2, h)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_lerp_2(self):
        radian = (0.5 * np.pi) / 2
        axis_1 = np.array([0, 1, 0])
        axis_2 = np.array([1, 0, 0])
        axis_unit_1 = axis_1 / np.linalg.norm(axis_1)
        axis_unit_2 = axis_2 / np.linalg.norm(axis_2)
        q_1 = Q.make(np.cos(radian), axis_unit_1[0] * np.sin(radian), axis_unit_1[1] * np.sin(radian),
                     axis_unit_1[2] * np.sin(radian))
        q_2 = Q.make(np.cos(radian), axis_unit_2[0] * np.sin(radian), axis_unit_2[1] * np.sin(radian),
                     axis_unit_2[2] * np.sin(radian))
        h = 0.7
        expected = q_1 * (1 - h) + q_2 * h
        actual = Q.lerp(q_1, q_2, h)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_slerp_1(self):
        radian = (0.5 * np.pi) / 2
        axis_1 = np.array([0, 1, 0])
        axis_2 = np.array([1, 0, 0])
        axis_unit_1 = axis_1 / np.linalg.norm(axis_1)
        axis_unit_2 = axis_2 / np.linalg.norm(axis_2)
        q_1 = Q.make(np.cos(radian), axis_unit_1[0] * np.sin(radian), axis_unit_1[1] * np.sin(radian),
                     axis_unit_1[2] * np.sin(radian))
        q_2 = Q.make(np.cos(radian), axis_unit_2[0] * np.sin(radian), axis_unit_2[1] * np.sin(radian),
                     axis_unit_2[2] * np.sin(radian))
        h = 0.0
        expected = q_1
        actual = Q.slerp(q_1, q_2, h)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_slerp_2(self):
        radian = (0.5 * np.pi) / 2
        axis_1 = np.array([0, 1, 0])
        axis_2 = np.array([1, 0, 0])
        axis_unit_1 = axis_1 / np.linalg.norm(axis_1)
        axis_unit_2 = axis_2 / np.linalg.norm(axis_2)
        q_1 = Q.make(np.cos(radian), axis_unit_1[0] * np.sin(radian), axis_unit_1[1] * np.sin(radian),
                     axis_unit_1[2] * np.sin(radian))
        q_2 = Q.make(np.cos(radian), axis_unit_2[0] * np.sin(radian), axis_unit_2[1] * np.sin(radian),
                     axis_unit_2[2] * np.sin(radian))
        h = 1.0
        expected = q_2
        actual = Q.slerp(q_1, q_2, h)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_slerp_3(self):
        radian = (0.5 * np.pi) / 2
        axis_1 = np.array([0, 1, 0])
        axis_2 = np.array([0, 1, 0])
        axis_unit_1 = axis_1 / np.linalg.norm(axis_1)
        axis_unit_2 = axis_2 / np.linalg.norm(axis_2)
        q_1 = Q.make(np.cos(radian), axis_unit_1[0] * np.sin(radian), -axis_unit_1[1] * np.sin(radian),
                     axis_unit_1[2] * np.sin(radian))
        q_2 = Q.make(-np.cos(radian), axis_unit_2[0] * np.sin(radian), axis_unit_2[1] * np.sin(radian),
                     axis_unit_2[2] * np.sin(radian))
        h = 1.0
        expected = q_2
        actual = Q.slerp(q_1, q_2, h)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_from_string_err_1(self):
        input_ = "hello world"
        with self.assertRaises(AssertionError):
            Q.from_string(input_)

    def test_from_string_err_2(self):
        input_ = "rx: 32a"
        with self.assertRaises(AssertionError):
            Q.from_string(input_)

    def test_from_string_err_3(self):
        input_ = "ry: 2+2"
        with self.assertRaises(AssertionError):
            Q.from_string(input_)

    def test_from_string_err_4(self):
        input_ = "[1,a,3]"
        with self.assertRaises(AssertionError):
            Q.from_string(input_)

    def test_from_string_err_5(self):
        input_ = "a [1,3,3]"
        with self.assertRaises(AssertionError):
            Q.from_string(input_)

    def test_from_string_err_6(self):
        input_ = "ru:1.3:[1,3,3]a"
        with self.assertRaises(AssertionError):
            Q.from_string(input_)

    def test_from_string_err_7(self):
        input_ = "rru:1.3:[1,3,3]"
        with self.assertRaises(AssertionError):
            Q.from_string(input_)

    def test_from_string_err_8(self):
        input_ = "[01,3]"
        with self.assertRaises(AssertionError):
            Q.from_string(input_)
