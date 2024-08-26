import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.math.coordsys as CS
import rainbow.math.vector3 as V3
import rainbow.math.matrix3 as M3
import rainbow.math.quaternion as Q
import rainbow.util.test_tools as TEST


class TestCoordSys(unittest.TestCase):
    def test_make_1(self):
        x, y, z = 1, 2, 3
        radians = np.pi * 0.5
        axis = np.array([1, 1, 0])
        r = V3.make(x, y, z)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        actual_r = coordsys.r
        actual_q = coordsys.q
        expected_r = r
        expected_q = q
        self.assertTrue(TEST.is_array_equal(actual_r, expected_r))
        self.assertTrue(TEST.is_array_equal(actual_q, expected_q))

    def test_xform_point_1(self):
        x, y, z = 0, 0, 0
        radians = np.pi * 0.5
        axis = np.array([0, 0, 1])
        r = V3.make(x, y, z)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        point = np.array([1, 0, 0])
        point_rot = CS.xform_point(coordsys, point)
        actual = point_rot
        expected = np.array([0, 1, 0])
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_xform_point_2(self):
        x, y, z = 1, 1, 1
        radians = np.pi * 0.5
        axis = np.array([0, 0, 1])
        r = V3.make(x, y, z)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        point = np.array([1, 0, 0])
        point_rot = CS.xform_point(coordsys, point)
        actual = point_rot
        expected = np.array([1, 2, 1])
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_xform_point_3(self):
        x, y, z = 0, -2, 0
        radians = np.pi / 2
        axis = np.array([0, 0, 1])
        r = V3.make(x, y, z)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        point = np.array([0, 2, 0])
        point_rot = CS.xform_point(coordsys, point)
        actual = point_rot
        expected = np.array([-2, -2, 0])
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_xform_vector_1(self):
        # just a test of rotate, therefore i assume it is already tested
        x, y, z = 0, -2, 0
        radians = np.pi / 2
        axis = np.array([0, 0, 1])
        r = V3.make(x, y, z)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        vec = np.array([0, 2, 0])
        vec_rot = CS.xform_vector(coordsys, vec)
        actual = vec_rot
        expected = np.array([-2, 0, 0])
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_xform_matrix_1(self):
        x, y, z = 1, 0, 0
        radians = np.pi / 2
        axis = np.array([0, 0, 1])
        r = V3.make(x, y, z)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        identity_basis = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        actual = CS.xform_matrix(coordsys, identity_basis)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_xform_matrix_2(self):
        x, y, z = 1, 0, 0
        radians = np.pi / 2
        axis = np.array([0, 1, 0])
        r = V3.make(x, y, z)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        identity_basis = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        expected = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        actual = CS.xform_matrix(coordsys, identity_basis)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_xform_matrix_3(self):
        x, y, z = 1, 0, 0
        radians = np.pi / 2
        axis = np.array([1, 0, 0])
        r = V3.make(x, y, z)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        identity_basis = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        expected = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        actual = CS.xform_matrix(coordsys, identity_basis)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_concat_1(self):
        x, y, z = 1, 0, 0
        radians = np.pi / 4
        axis = np.array([1, 0, 0])
        r = V3.make(x, y, z)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        identity_basis = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        expected = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        concat_transform = CS.concat(coordsys, coordsys)
        actual = CS.xform_matrix(concat_transform, identity_basis)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_concat_2(self):
        x, y, z = 1, 0, 0
        radians_1 = np.pi
        radians_2 = np.pi / 2
        axis = np.array([1, 0, 0])
        r = V3.make(x, y, z)
        q_1 = Q.Ru(radians_1, axis)
        q_2 = Q.Ru(radians_2, axis)
        coordsys_1 = CS.make(r, q_1)
        coordsys_2 = CS.make(r, q_2)
        identity_basis = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        expected = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        concat_transform = CS.concat(coordsys_1, coordsys_2)
        actual = CS.xform_matrix(concat_transform, identity_basis)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_concat_3(self):
        x, y, z = 1, 0, 0
        radians_1 = np.pi
        radians_2 = np.pi / 2
        axis_1 = np.array([0, 1, 0])
        axis_2 = np.array([0, 0, 1])
        r = V3.make(x, y, z)
        q_1 = Q.Ru(radians_1, axis_1)
        q_2 = Q.Ru(radians_2, axis_2)
        coordsys_1 = CS.make(r, q_1)
        coordsys_2 = CS.make(r, q_2)
        identity_basis = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        expected = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
        concat_transform = CS.concat(coordsys_1, coordsys_2)
        actual = CS.xform_matrix(concat_transform, identity_basis)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_inverse_1(self):
        radians = 2 * np.pi
        r = V3.make(0, 0, 0)
        axis = V3.make(1, 0, 0)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        coordsys_inverse = CS.inverse(coordsys)
        expected = M3.make(1, 0, 0, 0, 1, 0, 0, 0, 1)
        actual = np.dot(Q.to_matrix(coordsys_inverse.q), Q.to_matrix(q))
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_inverse_2(self):
        radians = 2 * np.pi
        r = V3.make(0, 0, 0)
        axis = V3.make(1, 0, 0)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        coordsys_inverse = CS.inverse(CS.inverse(coordsys))
        expected = q
        actual = coordsys_inverse.q
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_inverse_3(self):
        radians = 2 * np.pi
        r = V3.make(1, 2, 3)
        axis = V3.make(1, 0, 0)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        coordsys_inverse = CS.inverse(CS.inverse(coordsys))
        expected = r
        actual = coordsys_inverse.r
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_inverse_4(self):
        radians = 2 * np.pi
        r = V3.make(-1, 0.2, 3.5)
        axis = V3.make(1, 0, 0)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        coordsys_inverse = CS.inverse(CS.inverse(coordsys))
        expected = r
        actual = coordsys_inverse.r
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_inverse_5(self):
        radians = 2 * np.pi
        r = V3.make(-1, 0.2, 3.5)
        axis = V3.make(1, 0, 0)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        coordsys_inverse = CS.inverse(coordsys)
        expected = -r
        actual = coordsys_inverse.r
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_inverse_6(self):
        radians = 2 * np.pi
        r = V3.make(-1, 0.2, 3.5)
        axis = V3.make(1, 0, 1)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        coordsys_inverse = CS.inverse(coordsys)
        expected = -r
        actual = coordsys_inverse.r
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_inverse_7(self):
        x, y, z = 0, 0, 0
        radians = np.pi * 0.5
        axis = np.array([0, 0, 1])
        r = V3.make(x, y, z)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        point = np.array([1, 0, 0])
        point_rot = CS.xform_point(coordsys, point)
        coordsys_inverse = CS.inverse(coordsys)
        actual = CS.xform_point(coordsys_inverse, point_rot)
        expected = point
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_inverse_8(self):
        x, y, z = 1, 0, 5
        radians = np.pi * 0.75
        axis = np.array([0, -0.5, 1])
        r = V3.make(x, y, z)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        point = np.array([1, 0, 0])
        point_rot = CS.xform_point(coordsys, point)
        coordsys_inverse = CS.inverse(coordsys)
        actual = CS.xform_point(coordsys_inverse, point_rot)
        expected = point
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_inverse_9(self):
        x, y, z = -1, 0, 5
        radians = np.pi * 0.75
        axis = np.array([0, -0.5, 1])
        r = V3.make(x, y, z)
        q = Q.Ru(radians, axis)
        coordsys = CS.make(r, q)
        point = np.array([1, 0.43, -77])
        point_rot = CS.xform_point(coordsys, point)
        coordsys_inverse = CS.inverse(coordsys)
        actual = CS.xform_point(coordsys_inverse, point_rot)
        expected = point
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_make_coordsys_from_to_1(self):
        radians_A = np.pi * 0.5
        radians_B = np.pi * 0.75
        radians_W = np.pi * 0.33
        axis_A = np.array([0, 0, 1])
        axis_B = np.array([1, 0, 0])
        axis_W = np.array([0, 1, 0])
        r_A = V3.make(0, 1, 0)
        r_B = V3.make(1, 0, 1)
        r_W = V3.make(0, 1, 1)
        q_A = Q.Ru(radians_A, axis_A)
        q_B = Q.Ru(radians_B, axis_B)
        q_W = Q.Ru(radians_W, axis_W)
        bf_1 = CS.make(r_A, q_A)
        bf_2 = CS.make(r_B, q_B)
        wcs = CS.make(r_W, q_W)
        A = CS.make_coordsys_from_to(bf_1, wcs)
        B = CS.make_coordsys_from_to(bf_2, wcs)
        point = V3.make(1, 0, 0)
        coord_A2B = CS.make_coordsys_from_to(A, B)
        coord_B2A = CS.make_coordsys_from_to(B, A)
        expected = point
        actual = CS.xform_point(coord_B2A, CS.xform_point(coord_A2B, point))
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_make_coordsys_from_to_2(self):
        radians_A = np.pi * 0.5
        radians_B = np.pi * 0.75
        radians_W = np.pi * 0.33
        axis_A = np.array([0, 0, 1])
        axis_B = np.array([1, 0, 0])
        axis_W = np.array([0, 1, 0])
        r_A = V3.make(0, 1, 0)
        r_B = V3.make(1, 0, 1)
        r_W = V3.make(0, 1, 1)
        q_A = Q.Ru(radians_A, axis_A)
        q_B = Q.Ru(radians_B, axis_B)
        q_W = Q.Ru(radians_W, axis_W)
        bf_1 = CS.make(r_A, q_A)
        bf_2 = CS.make(r_B, q_B)
        wcs = CS.make(r_W, q_W)
        A = CS.make_coordsys_from_to(bf_1, wcs)
        B = CS.make_coordsys_from_to(bf_2, wcs)
        point = V3.make(1.5, -700, 0.001)
        coord_A2B = CS.make_coordsys_from_to(A, B)
        coord_B2A = CS.make_coordsys_from_to(B, A)
        expected = point
        actual = CS.xform_point(coord_B2A, CS.xform_point(coord_A2B, point))
        self.assertTrue(TEST.is_array_equal(actual, expected))
