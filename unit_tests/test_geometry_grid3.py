import unittest
import os
import sys
import numpy as np
import igl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.math.vector3 as V3
import rainbow.geometry.grid3 as GRID
import rainbow.util.test_tools as TEST


def simpelfunc(coord):
    _, _, z = coord[0], coord[1], coord[2]
    return z


def simpelfunc2(coord):
    _, _, z = coord[0], coord[1], coord[2]
    return 2 * z


def simpelfunc3(coord):
    _, _, z = coord[0], coord[1], coord[2]
    return z ** 2


def simpelfunc4(coord):
    _, _, z = coord[0], coord[1], coord[2]
    return z * 2 + 3


def simpelfunc5(coord):
    x, _, _ = coord[0], coord[1], coord[2]
    return x


def simpelfunc6(coord):
    _, y, _ = coord[0], coord[1], coord[2]
    return y


def simpelfunc7(coord):
    x, y, _ = coord[0], coord[1], coord[2]
    return 2 * x + 3 * y


def simpelfunc8(coord):
    x, y, z = coord[0], coord[1], coord[2]
    return 2 * x + 3 * y - 4 * z


class TestGrid(unittest.TestCase):

    def test_get_enclosing_cell_idx(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 4
        J = 4
        K = 4

        A = GRID.Grid(min_coord, max_coord, I, J, K)

        p = V3.make(0, 0, 0)
        idx = A.get_enclosing_cell_idx(p)
        self.assertTrue(TEST.is_array_equal(idx, V3.make(1, 1, 1)))

        p = V3.make(-0.5, -0.5, -0.5)
        idx = A.get_enclosing_cell_idx(p)
        self.assertTrue(TEST.is_array_equal(idx, V3.make(0, 0, 0)))

        p = V3.make(0.5, 0.5, 0.5)
        idx = A.get_enclosing_cell_idx(p)
        self.assertTrue(TEST.is_array_equal(idx, V3.make(2, 2, 2)))

        p = V3.make(-1, -1, -1)
        idx = A.get_enclosing_cell_idx(p)
        self.assertTrue(TEST.is_array_equal(idx, V3.make(0, 0, 0)))

        p = V3.make(1, 1, 1)
        idx = A.get_enclosing_cell_idx(p)
        self.assertTrue(TEST.is_array_equal(idx, V3.make(2, 2, 2)))

        p = V3.make(-2, -2, -2)
        idx = A.get_enclosing_cell_idx(p)
        self.assertTrue(TEST.is_array_equal(idx, V3.make(0, 0, 0)))

        p = V3.make(2, 2, 2)
        idx = A.get_enclosing_cell_idx(p)
        self.assertTrue(TEST.is_array_equal(idx, V3.make(2, 2, 2)))

    def test_valid_node_idx(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 4
        J = 4
        K = 4
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        self.assertFalse(A.is_valid_node_index(-1, 0, 0))
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        self.assertTrue(A.is_valid_node_index(0, 0, 0))
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        self.assertTrue(A.is_valid_node_index(0, 0, 3) )
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        self.assertFalse(A.is_valid_node_index(0, 4, 3))

    def test_get_value(self):

        def func(p):
            return p[0]

        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, func)

        p = V3.make(0, 0, 0)
        val = GRID.get_value(A, p)
        self.assertAlmostEqual(func(p), val)

        p = V3.make(-0.5, -0.5, -0.5)
        val = GRID.get_value(A, p)
        self.assertAlmostEqual(func(p), val)

        p = V3.make(0.5, 0.5, 0.5)
        val = GRID.get_value(A, p)
        self.assertAlmostEqual(func(p), val)

        p = V3.make(-1, -1, -1)
        val = GRID.get_value(A, p)
        self.assertAlmostEqual(func(p), val)

        p = V3.make(1, 1, 1)
        val = GRID.get_value(A, p)
        self.assertAlmostEqual(func(p), val)

        p = V3.make(-2, -2, -2)
        val = GRID.get_value(A, p)
        self.assertAlmostEqual(func(p), val)

        p = V3.make(2, 2, 2)
        val = GRID.get_value(A, p)
        self.assertAlmostEqual(func(p), val)

    def test_get_gradient(self):

        def func(p):
            return p[0]

        def grad(p):
            return np.array([1, 0, 0])

        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, func)

        p = V3.make(0, 0, 0)
        g = GRID.get_gradient(A, p)
        self.assertTrue(TEST.is_array_equal(grad(p), g))

        p = V3.make(-0.5, -0.5, -0.5)
        g = GRID.get_gradient(A, p)
        self.assertTrue(TEST.is_array_equal(grad(p), g))

        p = V3.make(0.5, 0.5, 0.5)
        g = GRID.get_gradient(A, p)
        self.assertTrue(TEST.is_array_equal(grad(p), g))

        p = V3.make(-1, -1, -1)
        g = GRID.get_gradient(A, p)
        self.assertTrue(TEST.is_array_equal(grad(p), g))

        p = V3.make(1, 1, 1)
        g = GRID.get_gradient(A, p)
        self.assertTrue(TEST.is_array_equal(grad(p), g))

        p = V3.make(-2, -2, -2)
        g = GRID.get_gradient(A, p)
        self.assertTrue(TEST.is_array_equal(grad(p), g))

        p = V3.make(2, 2, 2)
        g = GRID.get_gradient(A, p)
        self.assertTrue(TEST.is_array_equal(grad(p), g))

    def test_is_inside(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 4
        J = 4
        K = 4
        A = GRID.Grid(min_coord, max_coord, I, J, K)

        boundary = 0
        p = V3.make(-1, -1, -1)
        self.assertTrue(GRID.is_inside(A, p, boundary))

        boundary = 0
        p = V3.make(0, 0, 0)
        self.assertTrue(GRID.is_inside(A, p, boundary))

        boundary = 0
        p = V3.make(1, 1, 1)
        self.assertTrue(GRID.is_inside(A, p, boundary))

        boundary = 0
        p = V3.make(-1.1, -1.1, -1.1)
        self.assertFalse(GRID.is_inside(A, p, boundary))

        boundary = 0
        p = V3.make(1.1, 1.1, 1.1)
        self.assertFalse(GRID.is_inside(A, p, boundary))

        boundary = 0.5
        p = V3.make(-1, -1, -1)
        self.assertFalse(GRID.is_inside(A, p, boundary))

        boundary = 0.5
        p = V3.make(0, 0, 0)
        self.assertTrue(GRID.is_inside(A, p, boundary))

        boundary = 0.5
        p = V3.make(1, 1, 1)
        self.assertFalse(GRID.is_inside(A, p, boundary))

        boundary = 0.5
        p = V3.make(-1.1, -1.1, -1.1)
        self.assertFalse(GRID.is_inside(A, p, boundary))

        boundary = 0.5
        p = V3.make(1.1, 1.1, 1.1)
        self.assertFalse(GRID.is_inside(A, p, boundary))

    def test_read_write_matlab(self):
        V, F = igl.read_triangle_mesh(TEST.get_base_folder() + "/data/armadillo.obj")
        A = GRID.create_signed_distance(V, F, 98, 98, 98, 0.5)
        GRID.write_matlab_file(TEST.get_base_folder() + "/data/test.mat", A)
        B = GRID.read_matlab_file(TEST.get_base_folder() + "/data/test.mat")

        self.assertEqual(A.I, B.I)
        self.assertEqual(A.J, B.J)
        self.assertEqual(A.K, B.K)
        self.assertTrue(TEST.is_array_equal(A.min_coord, B.min_coord))
        self.assertTrue(TEST.is_array_equal(A.max_coord, B.max_coord))
        self.assertTrue(TEST.is_array_equal(A.spacing, B.spacing))
        self.assertTrue(TEST.is_array_equal(A.values, B.values))

    def test_get_value_2(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc)
        expected = 1.0
        actual = GRID.get_value(A, (0, 0, 1))[0]
        self.assertEqual(actual, expected)

    def test_get_value_1(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc)
        expected = 0.0
        actual = GRID.get_value(A, (0, 0, 0))[0]
        self.assertEqual(actual, expected)

    def test_get_value_3(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc)
        expected = 2.0
        actual = GRID.get_value(A, (0, 0, 2))[0]
        self.assertEqual(actual, expected)

    def test_get_value_4(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc)
        expected = 3.0
        actual = GRID.get_value(A, (0, 0, 3))[0]
        self.assertEqual(actual, expected)

    def test_get_value_5(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc2)
        expected = 2.0
        actual = GRID.get_value(A, (0, 0, 1))[0]
        self.assertEqual(actual, expected)

    def test_get_value_6(self):
        min_coord = V3.make(0, 0, 0)
        max_coord = V3.make(2, 2, 2)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc3)
        expected = 4.0
        actual = GRID.get_value(A, (0, 0, 2))[0]
        self.assertEqual(actual, expected)

    def test_get_value_7(self):
        min_coord = V3.make(0, 0, 0)
        max_coord = V3.make(2 * 2, 2 * 2, 2 * 2)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc3)
        expected = 9.0
        actual = GRID.get_value(A, (0, 0, 3))[0]
        self.assertEqual(actual, expected)

    def test_get_value_9(self):
        min_coord = V3.make(0, 0, 0)
        max_coord = V3.make(4 * 2 + 3, 4 * 2 + 3, 4 * 2 + 3)
        I = 10
        J = 10
        K = 10
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc4)
        expected = 2 * 2 + 3
        actual = GRID.get_value(A, (0, 0, 2))[0]
        self.assertEqual(actual, expected)

    def test_get_value_10(self):
        min_coord = V3.make(0, 0, 0)
        max_coord = V3.make(4 * 2 + 3, 4 * 2 + 3, 4 * 2 + 3)
        I = 10
        J = 10
        K = 10
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc4)
        expected = 4 * 2 + 3
        actual = GRID.get_value(A, (0, 0, 4))[0]
        self.assertEqual(actual, expected)

    def test_get_value_11(self):
        min_coord = V3.make(0, 0, 0)
        max_coord = V3.make(10 * 2 + 3, 10 * 2 + 3, 10 * 2 + 3)
        I = 20
        J = 20
        K = 20
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc4)
        expected = 7 * 2 + 3
        actual = GRID.get_value(A, (0, 0, 7))[0]
        self.assertEqual(actual, expected)

    def test_get_value_12(self):
        min_coord = V3.make(-2 * 2 + 3, -2 * 2 + 3, -2 * 2 + 3)
        max_coord = V3.make(2 * 2 + 3, 2 * 2 + 3, 2 * 2 + 3)
        I = 2 * 7
        J = 2 * 7
        K = 2 * 7
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc4)
        expected = 9 * 2 + 3
        actual = GRID.get_value(A, (0, 0, 9))[0]
        self.assertAlmostEqual(actual, expected)

    def test_gradient_1(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc)
        expected = np.array([0, 0, 1])
        actual = GRID.get_gradient(A, (0, 0, 1))
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_gradient_2(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc2)
        expected = np.array([0, 0, 2])
        actual = GRID.get_gradient(A, (0, 0, 1))
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_gradient_3(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc4)
        expected = np.array([0, 0, 2])
        actual = GRID.get_gradient(A, (0, 0, 1))
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_gradient_4(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc5)
        expected = np.array([1, 0, 0])
        actual = GRID.get_gradient(A, (1, 0, 0))
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_gradient_5(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc5)
        expected = np.array([1, 0, 0])
        actual = GRID.get_gradient(A, (3, 0, 0))
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_gradient_6(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc6)
        expected = np.array([0, 1, 0])
        actual = GRID.get_gradient(A, (0, 3, 0))
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_gradient_7(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc6)
        expected = np.array([0, 1, 0])
        actual = GRID.get_gradient(A, (1, 3, 1))
        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_gradient_8(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc7)
        expected = np.array([2, 3, 0])
        actual = GRID.get_gradient(A, (1, 0, 0))

        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_gradient_9(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc7)
        expected = np.array([2, 3, 0])
        actual = GRID.get_gradient(A, (1, 1, 0))

        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_gradient_10(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc8)
        expected = np.array([2, 3, -4])
        actual = GRID.get_gradient(A, (1, 2, 3))

        self.assertTrue(TEST.is_array_equal(actual, expected))

    def test_is_valid_node_index_1(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        self.assertTrue(A.is_valid_node_index(3, 3, 3))

    def test_is_valid_node_index_2(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        self.assertTrue(A.is_valid_node_index(0, 0, 0))

    def test_is_valid_node_index_3(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        self.assertFalse(A.is_valid_node_index(0, -1, 0))

    def test_is_valid_node_index_4(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        self.assertFalse(A.is_valid_node_index(0, 5, 0))

    def test_get_node_value_1(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        with self.assertRaises(TypeError):
            A.get_node_value(0, 0, 0)

    def test_get_node_value_2(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc)
        expected = 1
        actual = A.get_node_value(0, 0, 4)
        self.assertEqual(actual, expected)

    def test_get_node_value_2(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(A, simpelfunc)
        expected = -1
        actual = A.get_node_value(0, 0, 0)
        self.assertEqual(actual, expected)
