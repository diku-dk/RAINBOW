from telnetlib import OUTMRK
import unittest
import os
import sys
import numpy as np
import math as m

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../../")
import isl.geometry.grid3 as g3
import isl.math.vector3 as V3
import isl.test.test_common as utils

def simpelfunc(coord):
    _, _, z    = coord[0], coord[1], coord[2]
    return z

def simpelfunc2(coord):
    _, _, z    = coord[0], coord[1], coord[2]
    return 2*z

def simpelfunc3(coord):
    _, _, z    = coord[0], coord[1], coord[2]
    return z**2

def simpelfunc4(coord):
    _, _, z    = coord[0], coord[1], coord[2]
    return z*2+3

def simpelfunc5(coord):
    x, _, _    = coord[0], coord[1], coord[2]
    return x

def simpelfunc6(coord):
    _, y, _    = coord[0], coord[1], coord[2]
    return y

def simpelfunc7(coord):
    x, y, _    = coord[0], coord[1], coord[2]
    return 2*x+3*y

def simpelfunc8(coord):
    x, y, z    = coord[0], coord[1], coord[2]
    return 2*x+3*y-4*z

class TestGeometryAPI(unittest.TestCase):
    def test_get_value_1(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc)
        expected = 0.
        actual    = g3.get_value(A, (0,0,0))[0]
        self.assertEqual(actual, expected) 

    def test_get_value_2(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc)
        expected = 1.
        actual    = g3.get_value(A, (0,0,1))[0]
        self.assertEqual(actual, expected)

    def test_get_value_3(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc)
        expected = 2.
        actual    = g3.get_value(A, (0,0,2))[0]
        self.assertEqual(actual, expected)

    def test_get_value_4(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc)
        expected = 3.
        actual    = g3.get_value(A, (0,0,3))[0]
        self.assertEqual(actual, expected)

    def test_get_value_5(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc2)
        expected = 2.
        actual    = g3.get_value(A, (0,0,1))[0]
        self.assertEqual(actual, expected)

    def test_get_value_6(self):
        min_coord = V3.make(0, 0, 0)
        max_coord = V3.make(2, 2, 2)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc3)
        expected = 4.
        actual    = g3.get_value(A, (0,0,2))[0]
        self.assertEqual(actual, expected)

    def test_get_value_7(self):
        min_coord = V3.make(0, 0, 0)
        max_coord = V3.make(2*2, 2*2, 2*2)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc3)
        expected = 9.
        actual    = g3.get_value(A, (0,0,3))[0]
        self.assertEqual(actual, expected)

    def test_get_value_9(self):
        min_coord = V3.make(0, 0, 0)
        max_coord = V3.make(4*2+3, 4*2+3, 4*2+3)
        I = 10
        J = 10
        K = 10
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc4)
        expected = 2*2+3
        actual    = g3.get_value(A, (0,0,2))[0]
        self.assertEqual(actual, expected)

    def test_get_value_10(self):
        min_coord = V3.make(0, 0, 0)
        max_coord = V3.make(4*2+3, 4*2+3, 4*2+3)
        I = 10
        J = 10
        K = 10
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc4)
        expected = 4*2+3
        actual    = g3.get_value(A, (0,0,4))[0]
        self.assertEqual(actual, expected)

    def test_get_value_11(self):
        min_coord = V3.make(0, 0, 0)
        max_coord = V3.make(10*2+3, 10*2+3, 10*2+3)
        I = 20
        J = 20
        K = 20
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc4)
        expected = 7*2+3
        actual    = g3.get_value(A, (0,0,7))[0]
        self.assertEqual(actual, expected)
    
    def test_get_value_12(self):
        min_coord = V3.make(-2*2+3, -2*2+3, -2*2+3)
        max_coord = V3.make(2*2+3, 2*2+3, 2*2+3)
        I = 2*7
        J = 2*7
        K = 2*7
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc4)
        expected  = 9*2+3
        actual    = g3.get_value(A, (0,0,9))[0]
        self.assertAlmostEqual(actual, expected)

    def test_gradient_1(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc)
        expected = np.array([0,0,1])
        actual   = g3.get_gradient(A, (0,0,1))
        self.assertTrue(utils.assert_array_equal(actual, expected))

    def test_gradient_2(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc2)
        expected = np.array([0,0,2])
        actual   = g3.get_gradient(A, (0,0,1))
        self.assertTrue(utils.assert_array_equal(actual, expected))

    def test_gradient_3(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc4)
        expected = np.array([0,0,2])
        actual   = g3.get_gradient(A, (0,0,1))
        self.assertTrue(utils.assert_array_equal(actual, expected))

    def test_gradient_4(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc5)
        expected = np.array([1,0,0])
        actual   = g3.get_gradient(A, (1,0,0))
        self.assertTrue(utils.assert_array_equal(actual, expected))

    def test_gradient_5(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc5)
        expected = np.array([1,0,0])
        actual   = g3.get_gradient(A, (3,0,0))
        self.assertTrue(utils.assert_array_equal(actual, expected))

    def test_gradient_6(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc6)
        expected = np.array([0,1,0])
        actual   = g3.get_gradient(A, (0,3,0))
        self.assertTrue(utils.assert_array_equal(actual, expected))

    def test_gradient_7(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc6)
        expected = np.array([0,1,0])
        actual   = g3.get_gradient(A, (1,3,1))
        self.assertTrue(utils.assert_array_equal(actual, expected))

    def test_gradient_8(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc7)
        expected = np.array([2,3,0])
        actual   = g3.get_gradient(A, (1,0,0))

        self.assertTrue(utils.assert_array_equal(actual, expected))

    def test_gradient_9(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc7)
        expected = np.array([2,3,0])
        actual   = g3.get_gradient(A, (1,1,0))

        self.assertTrue(utils.assert_array_equal(actual, expected))

    def test_gradient_10(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc8)
        expected = np.array([2,3,-4])
        actual   = g3.get_gradient(A, (1,2,3))

        self.assertTrue(utils.assert_array_equal(actual, expected))

    def test_is_valid_node_index_1(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        self.assertTrue(A.is_valid_node_index(3,3,3))

    def test_is_valid_node_index_2(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        self.assertTrue(A.is_valid_node_index(0,0,0))

    def test_is_valid_node_index_3(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        self.assertFalse(A.is_valid_node_index(0,-1,0))

    def test_is_valid_node_index_4(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        self.assertFalse(A.is_valid_node_index(0,5,0))

    def test_get_node_value_1(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        with self.assertRaises(TypeError):
            A.get_node_value(0,0,0)

    def test_get_node_value_2(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc)
        expected = 1
        actual   = A.get_node_value(0,0,4)
        self.assertEqual(actual, expected)

    def test_get_node_value_2(self):
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = g3.Grid(min_coord, max_coord, I, J, K)
        g3.eval_on_grid(A, simpelfunc)
        expected = -1
        actual   = A.get_node_value(0,0,0)
        self.assertEqual(actual, expected)