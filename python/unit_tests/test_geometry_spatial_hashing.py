import unittest
import os
import sys
import numpy as np
import igl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rainbow.geometry.aabb import AABB
import rainbow.geometry.spatial_hashing as SPATIAL_HASHING
import rainbow.util.test_tools as TEST


class TestHashCell(unittest.TestCase):
    def setUp(self):
        self.triangle1 = {
            "tri_idx": 0,
            "body_idx": 0,
            "aabb": AABB([0, 0, 0], [1, 1, 1])
        }

        self.triangle2 = {
            "tri_idx": 0,
            "body_idx": 1,
            "aabb": AABB([1, 1, 1], [2, 2, 2])
        }
        self.time_stamp = 0
        self.cell = SPATIAL_HASHING.HashCell()

    def test_add(self):
        self.cell.add((self.triangle1["tri_idx"], self.triangle1["body_idx"], self.triangle1["aabb"]), self.time_stamp)
        self.assertEqual(self.cell.time_stamp, self.time_stamp)
        self.assertEqual(self.cell.size, 1)

    def test_lazy_clear(self):
        self.time_stamp +=1
        self.cell.add((self.triangle1["tri_idx"], self.triangle1["body_idx"], self.triangle1["aabb"]), self.time_stamp)
        self.cell.add((self.triangle2["tri_idx"], self.triangle2["body_idx"], self.triangle2["aabb"]), self.time_stamp)
        self.assertEqual(self.cell.size, 2)
        self.assertEqual(self.cell.object_list[0][0], self.triangle1["tri_idx"])
        self.assertEqual(self.cell.object_list[0][1], self.triangle1["body_idx"])
        self.assertEqual(self.cell.object_list[1][0], self.triangle2["tri_idx"])
        self.assertEqual(self.cell.object_list[1][1], self.triangle2["body_idx"])


class TestHashGrid(unittest.TestCase):
    def setUp(self):
        self.grid = SPATIAL_HASHING.HashGird()
        self.triangle1 = {
            "tri_idx": 0,
            "body_idx": 0,
            "aabb": AABB([0, 0, 0], [1, 1, 1])
        }
        self.triangle2 = {
            "tri_idx": 0,
            "body_idx": 1,
            "aabb": AABB([1, 1, 1], [2, 2, 2])
        }
        self.time_stamp = 0

    def test_get_prefect_hash_value(self):
        self.assertIsInstance(self.grid.get_prefect_hash_value(1, 2, 3), int)

    def test_insert(self):
        overlaps = self.grid.insert(1, 1, 1, self.triangle1["tri_idx"], self.triangle1["body_idx"], self.triangle1["aabb"], self.time_stamp)
        self.assertEqual(len(overlaps), 0)

        overlaps = self.grid.insert(1, 1, 1, self.triangle2["tri_idx"], self.triangle2["body_idx"], self.triangle2["aabb"], self.time_stamp)
        self.assertTrue(len(overlaps)>0)
        self.assertEqual(overlaps[0][0], self.triangle1["tri_idx"])
        self.assertEqual(overlaps[0][1], self.triangle1["body_idx"])

    def test_compute_optial_cell_size(self):
        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        T = [[0, 1, 2], [0, 1, 3], [0, 2, 3]]
        size = SPATIAL_HASHING.HashGird.compute_optial_cell_size(V, T)
        self.assertIsInstance(size, float)


if __name__ == '__main__':
    unittest.main()