import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.math.quaternion as Q
import rainbow.math.matrix3 as M3
import rainbow.util.test_tools as TEST


class TestMatrix3API(unittest.TestCase):
    def test_diag_from_array_1(self):
        v = np.array([1, 2, 3])
        expected = np.diag(v)
        actual = M3.diag_from_array(v)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_diag_1(self):
        a, b, c = 1, 2, 3
        expected = np.diag(np.array([a, b, c]))
        actual = M3.diag(a, b, c)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_diag_2(self):
        a, b, c = 1, 2, "three"
        with self.assertRaises(ValueError):
            M3.diag(a, b, c)

    def test_matrix3_make_1(self):
        a, b, c, d, e, f, h, i, j = 1, 2, 3, 4, 5, 6, 7, 8, 9
        expected = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        )
        actual = M3.make(a, b, c, d, e, f, h, i, j)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_make_from_row_1(self):
        a, d, h = (np.array([1, 2, 3]),
                   np.array([4, 5, 6]),
                   np.array([7, 8, 9])
                   )
        expected = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        )
        actual = M3.make_from_rows(a, d, h)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_make_from_col_1(self):
        a, d, h = (np.array([1, 4, 7]),
                   np.array([2, 5, 8]),
                   np.array([3, 6, 9])
                   )
        expected = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        )
        actual = M3.make_from_cols(a, d, h)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_star_1(self):
        rand_array = TEST.psu_rand_array_gen((2, 3))
        v1, v2 = rand_array[0], rand_array[1]
        expected = np.cross(v1, v2)
        actual = np.dot(M3.star(v1), v2)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_orthonomalize_1(self):
        M = np.random.rand(3, 3)
        M_orthonomalized = M3.ortonormalize(M)
        expected = M3.identity()
        actual = np.dot(np.transpose(M_orthonomalized), M_orthonomalized)
        expected_false_actual = np.dot(np.transpose(M), M)
        self.assertTrue(TEST.is_array_equal(actual, expected))
        self.assertTrue(TEST.is_array_not_equal(expected_false_actual, expected))

    def test_matrix3_rotation_1(self):
        v = np.array([1, 0, 0])
        radian_x = 0.5 * np.pi
        radian_y = 0.0
        radian_z = 0.0
        rotation_matrix = np.dot(M3.Rx(radian_x), np.dot(M3.Ry(radian_y), M3.Rz(radian_z)))
        expected = v
        actual = np.dot(rotation_matrix, v)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_matrix3_rotation_2(self):
        v = np.array([1, 0, 0])
        radian_x = 0.0
        radian_y = 0.0
        radian_z = 0.5 * np.pi
        rotation_matrix = np.dot(M3.Rx(radian_x), np.dot(M3.Ry(radian_y), M3.Rz(radian_z)))
        expected = np.array([0, 1, 0])
        actual = np.dot(rotation_matrix, v)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_matrix3_rotation_3(self):
        v = np.array([1, 0, 0])
        radian_x = 0.0
        radian_y = 0.5 * np.pi
        radian_z = 0.0
        rotation_matrix = np.dot(M3.Rx(radian_x), np.dot(M3.Ry(radian_y), M3.Rz(radian_z)))
        expected = np.array([0, 0, -1])
        actual = np.dot(rotation_matrix, v)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_matrix3_rotation_4(self):
        v = np.array([0, 1, 0])
        radian_x = 0.0
        radian_y = 0.5 * np.pi
        radian_z = 0.0
        rotation_matrix = np.dot(M3.Rx(radian_x), np.dot(M3.Ry(radian_y), M3.Rz(radian_z)))
        expected = v
        actual = np.dot(rotation_matrix, v)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_matrix3_rotation_5(self):
        v = np.array([0, 1, 0])
        radian_x = 0.5 * np.pi
        radian_y = 0.0
        radian_z = 0.0
        rotation_matrix = np.dot(M3.Rx(radian_x), np.dot(M3.Ry(radian_y), M3.Rz(radian_z)))
        expected = np.array([0, 0, 1])
        actual = np.dot(rotation_matrix, v)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_matrix3_rotation_6(self):
        v = np.array([0, 1, 0])
        radian_x = 0.0
        radian_y = 0.0
        radian_z = 0.5 * np.pi
        rotation_matrix = np.dot(M3.Rx(radian_x), np.dot(M3.Ry(radian_y), M3.Rz(radian_z)))
        expected = np.array([-1, 0, 0])
        actual = np.dot(rotation_matrix, v)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_matrix3_rotation_7(self):
        v = np.array([0, 0, 1])
        radian_x = 0.0
        radian_y = 0.0
        radian_z = 0.5 * np.pi
        rotation_matrix = np.dot(M3.Rx(radian_x), np.dot(M3.Ry(radian_y), M3.Rz(radian_z)))
        expected = v
        actual = np.dot(rotation_matrix, v)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_matrix3_rotation_8(self):
        v = np.array([0, 0, 1])
        radian_x = 0.0
        radian_y = 0.5 * np.pi
        radian_z = 0.0
        rotation_matrix = np.dot(M3.Rx(radian_x), np.dot(M3.Ry(radian_y), M3.Rz(radian_z)))
        expected = np.array([1, 0, 0])
        actual = np.dot(rotation_matrix, v)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_matrix3_rotation_9(self):
        v = np.array([0, 0, 1])
        radian_x = 0.5 * np.pi
        radian_y = 0.0
        radian_z = 0.0
        rotation_matrix = np.dot(M3.Rx(radian_x), np.dot(M3.Ry(radian_y), M3.Rz(radian_z)))
        expected = np.array([0, -1, 0])
        actual = np.dot(rotation_matrix, v)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_polar_decompostion_1(self):
        M = np.random.rand(3, 3)
        R, S = M3.polar_decomposition(M)

        actual_1 = S
        actual_2 = np.dot(R, np.transpose(R))
        actual_3 = M

        expected_1 = np.transpose(S)
        expected_2 = M3.identity()
        expected_3 = np.dot(R, S)

        self.assertTrue(TEST.is_array_equal(actual_1, expected_1))
        self.assertTrue(TEST.is_array_equal(actual_2, expected_2))
        self.assertTrue(TEST.is_array_equal(actual_3, expected_3))

    def test_polar_decompostion_2(self):
        M = M3.identity()
        M[0, 0] = 0
        M[1, 1] = -2
        M[2, 2] = -1

        with self.assertRaises(ValueError):
            M3.polar_decomposition(M)

    def test_ru_1(self):
        radians = 0.5 * np.pi
        axis = np.array([1, 0, 0])
        radian_x = 0.5 * np.pi
        radian_y = 0.0
        radian_z = 0.0
        rotation_matrix = np.dot(M3.Rx(radian_x), np.dot(M3.Ry(radian_y), M3.Rz(radian_z)))
        expected = rotation_matrix
        actual = M3.Ru(radians, axis)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_ru_2(self):
        radians = 0.5 * np.pi
        axis = np.array([0, 1, 0])
        radian_x = 0.0
        radian_y = 0.5 * np.pi
        radian_z = 0.0
        rotation_matrix = np.dot(M3.Rx(radian_x), np.dot(M3.Ry(radian_y), M3.Rz(radian_z)))
        expected = rotation_matrix
        actual = M3.Ru(radians, axis)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_ru_3(self):
        radians = 0.5 * np.pi
        axis = np.array([0, 0, 1])
        radian_x = 0.0
        radian_y = 0.0
        radian_z = 0.5 * np.pi
        rotation_matrix = np.dot(M3.Rx(radian_x), np.dot(M3.Ry(radian_y), M3.Rz(radian_z)))
        expected = rotation_matrix
        actual = M3.Ru(radians, axis)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_ru_3(self):
        n = np.array([1, 1, 0])
        axis = n / np.linalg.norm(n)
        radians = (0.5 * np.pi) / 2
        q = np.array([np.cos(radians), axis[0] * np.sin(radians), axis[1] * np.sin(radians), axis[2] * np.sin(radians)],
                     dtype=np.float64)
        expected = Q.to_matrix(q)
        actual = M3.Ru(radians * 2, axis)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_ru_4(self):
        n = np.array([1, 1, 1])
        axis = n / np.linalg.norm(n)
        radians = (0.75 * np.pi) / 2
        q = np.array([np.cos(radians), axis[0] * np.sin(radians), axis[1] * np.sin(radians), axis[2] * np.sin(radians)],
                     dtype=np.float64)
        expected = Q.to_matrix(q)
        actual = M3.Ru(radians * 2, axis)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_ru_5(self):
        n = np.array([1, 0.56, 1])
        axis = n / np.linalg.norm(n)
        radians = (1.75 * np.pi) / 2
        q = np.array([np.cos(radians), axis[0] * np.sin(radians), axis[1] * np.sin(radians), axis[2] * np.sin(radians)],
                     dtype=np.float64)
        expected = Q.to_matrix(q)
        actual = M3.Ru(radians * 2, axis)
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_polar_decompostion_array_1(self):
        number_of_M = 4
        shape = (number_of_M, 3, 3)
        Ms = np.random.rand(shape[0], shape[1], shape[2])
        Rs, Ss = M3.polar_decomposition_array(Ms)

        actual_1 = Ss
        actual_2 = np.array([np.dot(r, np.transpose(r)) for r in Rs])
        actual_3 = Ms

        expected_1 = np.array([np.transpose(s) for s in Ss])
        expected_2 = np.array([M3.identity() for _ in range(number_of_M)])
        expected_3 = np.array([np.dot(r, s) for r, s in zip(Rs, Ss)])

        self.assertTrue(TEST.is_array_equal(actual_1, expected_1))
        self.assertTrue(TEST.is_array_equal(actual_2, expected_2))
        self.assertTrue(TEST.is_array_equal(actual_3, expected_3))

    def test_polar_decompostion_array_2(self):
        number_of_M = 4
        shape = (number_of_M, 3, 3)
        Ms = np.zeros(shape)
        Ms[:, 0, 0] = 1

        with self.assertRaises(ValueError):
            M3.polar_decomposition_array(Ms)
