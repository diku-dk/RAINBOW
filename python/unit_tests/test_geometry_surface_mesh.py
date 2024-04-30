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

        V, T = MESH.create_convex_hull(points)
        print(V)
        print(T)
        #self.assertAlmostEqual(u1, w1)
        #self.assertAlmostEqual(u2, w2)
        #self.assertAlmostEqual(u3, w3)
        #self.assertAlmostEqual(u4, w4)

