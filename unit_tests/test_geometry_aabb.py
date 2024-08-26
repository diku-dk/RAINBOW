import unittest
import os
import sys
import numpy as np
import igl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rainbow.geometry.aabb import AABB
import rainbow.util.test_tools as TEST


class TestAABB(unittest.TestCase):
    def setUp(self):
        self.p1 = np.array([0, 0, 0], dtype=np.float64)
        self.p2 = np.array([1, 1, 1], dtype=np.float64)
        self.p3 = np.array([2, 2, 2], dtype=np.float64)
        self.p4 = np.array([3, 3, 3], dtype=np.float64)

    def test_init(self):
        aabb = AABB(self.p1, self.p2)
        self.assertTrue(TEST.is_array_equal(aabb.min_point, self.p1))
        self.assertTrue(TEST.is_array_equal(aabb.max_point, self.p2))

    def test_create_from_vertices(self):
        vertices = np.array([self.p1, self.p2, self.p3])
        aabb = AABB.create_from_vertices(vertices)
        self.assertTrue(TEST.is_array_equal(aabb.min_point, self.p1))
        self.assertTrue(TEST.is_array_equal(aabb.max_point, self.p3))

    def test_is_overlap(self):
        aabb1 = AABB(self.p1, self.p2)
        aabb2 = AABB(self.p2, self.p3)
        self.assertTrue(AABB.is_overlap(aabb1, aabb2))

        aabb3 = AABB(self.p3, self.p4)
        self.assertFalse(AABB.is_overlap(aabb1, aabb3, boundary=0.1))
        self.assertTrue(AABB.is_overlap(aabb1, aabb3, boundary=1.1))


if __name__ == "__main__":
    unittest.main()
