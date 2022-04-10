import unittest
import os
import sys
import numpy as np
import igl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isl.math.vector3 as V3
import isl.geometry.grid3 as GRID
import isl.util.test_tools as TEST


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
        V, F = igl.read_triangle_mesh("../../data/armadillo.obj")
        A = GRID.create_signed_distance(V, F, 98, 98, 98, 0.5)
        GRID.write_matlab_file("../../data/test.mat", A)
        B = GRID.read_matlab_file("../../data/test.mat")

        self.assertEqual(A.I, B.I)
        self.assertEqual(A.J, B.J)
        self.assertEqual(A.K, B.K)
        self.assertTrue(TEST.is_array_equal(A.min_coord, B.min_coord))
        self.assertTrue(TEST.is_array_equal(A.max_coord, B.max_coord))
        self.assertTrue(TEST.is_array_equal(A.spacing, B.spacing))
        self.assertTrue(TEST.is_array_equal(A.values, B.values))
