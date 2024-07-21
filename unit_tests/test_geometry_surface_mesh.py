import unittest
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.math.vector3 as V3
import rainbow.geometry.surface_mesh as MESH
import rainbow.util.test_tools as TEST


class TestConvexHull(unittest.TestCase):

    def test_tetrahedral_hull(self):
        points = []
        p5 = V3.make(0.1, 0.1, 0.1)
        points.append(p5)
        p6 = V3.make(0.2, 0.2, 0.2)
        points.append(p6)
        p1 = V3.make(1, 0, 0)
        p2 = V3.make(0, 1, 0)
        p3 = V3.make(0, 0, 1)
        p4 = V3.make(0, 0, 0)
        points.append(p1)
        points.append(p2)
        points.append(p3)
        points.append(p4)
        V_actual, T_actual = MESH.create_convex_hull(points)
        V_expected = np.array([[1., 0., 0.],[0., 1., 0.],[0., 0., 1.],[0., 0., 0.]], dtype=np.float32)
        T_expected = np.array([[2, 0, 1],[0, 3, 1],[3, 2, 1], [2, 3, 0]], dtype=int)
        self.assertTrue(TEST.is_array_equal(V_actual, V_expected))
        self.assertTrue(TEST.is_array_equal(T_actual, T_expected))


