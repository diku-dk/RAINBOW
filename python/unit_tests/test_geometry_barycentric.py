import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.math.vector3 as V3
import rainbow.geometry.barycentric as B
import rainbow.util.test_tools as TEST


class TestBarycentric(unittest.TestCase):

    def test_compute_barycentric_1(self):
        x1 = V3.make(-1, 0, 0)
        x2 = V3.make(1, 0, 0)
        u1 = 0.1
        u2 = 1.0 - u1
        p = u1 * x1 + u2 * x2
        w1, w2 = B.compute_barycentric_edge(x1, x2, p)
        self.assertAlmostEqual(u1, w1)
        self.assertAlmostEqual(u2, w2)

    def test_compute_barycentric_2(self):
        x1 = V3.make(-1, 0, 0)
        x2 = V3.make(1, 0, 0)
        u1 = -0.1
        u2 = 1.0 - u1
        p = u1 * x1 + u2 * x2
        w1, w2 = B.compute_barycentric_edge(x1, x2, p)
        self.assertAlmostEqual(u1, w1)
        self.assertAlmostEqual(u2, w2)

    def test_compute_barycentric_3(self):
        x1 = V3.make(-1, 0, 0)
        x2 = V3.make(1, 0, 0)
        u1 = 1.1
        u2 = 1.0 - u1
        p = u1 * x1 + u2 * x2
        w1, w2 = B.compute_barycentric_edge(x1, x2, p)
        self.assertAlmostEqual(u1, w1)
        self.assertAlmostEqual(u2, w2)

    def test_compute_barycentric_4(self):
        x1 = V3.make(1, 0, 0)
        x2 = V3.make(0, 1, 0)
        x3 = V3.make(0, 0, 0)
        u1 = 0.1
        u2 = 0.1
        u3 = 1.0 - u1 - u2
        p = u1 * x1 + u2 * x2 + u3 * x3
        w1, w2, w3 = B.compute_barycentric_triangle(x1, x2, x3, p)
        self.assertAlmostEqual(u1, w1)
        self.assertAlmostEqual(u2, w2)
        self.assertAlmostEqual(u3, w3)

    def test_compute_barycentric_5(self):
        x1 = V3.make(1, 0, 0)
        x2 = V3.make(0, 1, 0)
        x3 = V3.make(0, 0, 0)
        u1 = -0.1
        u2 = 0.1
        u3 = 1.0 - u1 - u2
        p = u1 * x1 + u2 * x2 + u3 * x3
        w1, w2, w3 = B.compute_barycentric_triangle(x1, x2, x3, p)
        self.assertAlmostEqual(u1, w1)
        self.assertAlmostEqual(u2, w2)
        self.assertAlmostEqual(u3, w3)

    def test_compute_barycentric_6(self):
        x1 = V3.make(1, 0, 0)
        x2 = V3.make(0, 1, 0)
        x3 = V3.make(0, 0, 0)
        u1 = -0.1
        u2 = -0.1
        u3 = 1.0 - u1 - u2
        p = u1 * x1 + u2 * x2 + u3 * x3
        w1, w2, w3 = B.compute_barycentric_triangle(x1, x2, x3, p)
        self.assertAlmostEqual(u1, w1)
        self.assertAlmostEqual(u2, w2)
        self.assertAlmostEqual(u3, w3)

    def test_compute_barycentric_7(self):
        x1 = V3.make(1, 0, 0)
        x2 = V3.make(0, 1, 0)
        x3 = V3.make(0, 0, 1)
        x4 = V3.make(0, 0, 0)
        u1 = 0.1
        u2 = 0.1
        u3 = 0.1
        u4 = 1.0 - u1 - u2 - u3
        p = u1 * x1 + u2 * x2 + u3 * x3 + u4 * x4
        w1, w2, w3, w4 = B.compute_barycentric_tetrahedron(x1, x2, x3, x4, p)
        self.assertAlmostEqual(u1, w1)
        self.assertAlmostEqual(u2, w2)
        self.assertAlmostEqual(u3, w3)
        self.assertAlmostEqual(u4, w4)

    def test_compute_barycentric_8(self):
        x1 = V3.make(1, 0, 0)
        x2 = V3.make(0, 1, 0)
        x3 = V3.make(0, 0, 1)
        x4 = V3.make(0, 0, 0)
        u1 = -0.1
        u2 = -0.1
        u3 = -0.1
        u4 = 1.0 - u1 - u2 - u3
        p = u1 * x1 + u2 * x2 + u3 * x3 + u4 * x4
        w1, w2, w3, w4 = B.compute_barycentric_tetrahedron(x1, x2, x3, x4, p)
        self.assertAlmostEqual(u1, w1)
        self.assertAlmostEqual(u2, w2)
        self.assertAlmostEqual(u3, w3)
        self.assertAlmostEqual(u4, w4)

    def test_compute_barycentric_9(self):
        x1 = V3.make(0, 0, 0)
        x2 = V3.make(1, 0, 0)
        x3 = V3.make(0, 1, 0)
        x4 = V3.make(0, 0, 1)
        (
            grad_w1,
            grad_w2,
            grad_w3,
            grad_w4,
        ) = B.compute_barycentric_derivatives_tetrahedron(x1, x2, x3, x4)
        self.assertTrue(TEST.is_array_equal(grad_w1, V3.make(-1.0, -1.0, -1.0)))
        self.assertTrue(TEST.is_array_equal(grad_w2, V3.make(1.0, 0.0, 0.0)))
        self.assertTrue(TEST.is_array_equal(grad_w3, V3.make(0.0, 1.0, 0.0)))
        self.assertTrue(TEST.is_array_equal(grad_w4, V3.make(0.0, 0.0, 1.0)))

    def test_is_inside_edge(self):
        self.assertTrue(B.is_inside_edge(0.1, 0.9))
        self.assertFalse(B.is_inside_edge(1.1, 0.9))
        self.assertFalse(B.is_inside_edge(0.1, 1.9))
        self.assertFalse(B.is_inside_edge(-0.1, 0.9))
        self.assertFalse(B.is_inside_edge(0.1, -0.9))

    def test_is_inside_triangle(self):
        self.assertTrue(B.is_inside_triangle(0.1, 0.1, 0.8))
        self.assertFalse(B.is_inside_triangle(1.1, 0.1, 0.8))
        self.assertFalse(B.is_inside_triangle(0.1, 1.1, 0.8))
        self.assertFalse(B.is_inside_triangle(0.1, 0.1, 1.8))
        self.assertFalse(B.is_inside_triangle(-1.1, 0.1, 0.8))
        self.assertFalse(B.is_inside_triangle(0.1, -1.1, 0.8))
        self.assertFalse(B.is_inside_triangle(0.1, 0.1, -1.8))

    def test_is_inside_tetrahedron(self):
        self.assertTrue(B.is_inside_tetrahedron(0.1, 0.1, 0.1, 0.7))
        self.assertFalse(B.is_inside_tetrahedron(1.1, 0.1, 0.1, 0.7))
        self.assertFalse(B.is_inside_tetrahedron(0.1, 1.1, 0.1, 0.7))
        self.assertFalse(B.is_inside_tetrahedron(0.1, 0.1, 1.1, 0.7))
        self.assertFalse(B.is_inside_tetrahedron(0.1, 0.1, 0.1, 1.7))
        self.assertFalse(B.is_inside_tetrahedron(-1.1, 0.1, 0.1, 0.7))
        self.assertFalse(B.is_inside_tetrahedron(0.1, -1.1, 0.1, 0.7))
        self.assertFalse(B.is_inside_tetrahedron(0.1, 0.1, -1.1, 0.7))
        self.assertFalse(B.is_inside_tetrahedron(0.1, 0.1, 0.1, -1.7))










