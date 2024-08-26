import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rainbow.util.USD import USD


class TestUSD(unittest.TestCase):

    def setUp(self):
        self.file_path = "test.usd"
        self.usd_instance = USD(self.file_path)
        self.sample_mesh_name = "sample_mesh"
        self.sample_vertex_positions = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.sample_triangle_faces = np.array([[0, 1, 2]])

    def test_add_mesh(self):
        """ Test if a mesh can be added to the USD instance.
        """
        self.usd_instance.add_mesh(
            self.sample_mesh_name, self.sample_vertex_positions, self.sample_triangle_faces)
        self.assertIn(self.sample_mesh_name, self.usd_instance.meshes,
                      "Mesh was not added successfully.")

    def test_set_mesh_positions(self):
        """ Test if the positions of a mesh can be set at a given time stamp.
        """
        self.usd_instance.add_mesh(
            self.sample_mesh_name, self.sample_vertex_positions, self.sample_triangle_faces)
        new_positions = np.array(
            [[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]], dtype=np.float64)
        time_stamp = 1.0
        self.usd_instance.set_mesh_positions(
            self.sample_mesh_name, new_positions, time_stamp)
        updated_positions = self.usd_instance.get_mesh_positions(self.sample_mesh_name, time_stamp)
        self.assertTrue(np.allclose(new_positions, updated_positions, rtol=1e-5, atol=1e-10))

    def test_set_mesh_positions_with_invalid_name(self):
        """ Test if the mesh does not exist, an error will be raised.
        """
        with self.assertRaises(ValueError):
            self.usd_instance.set_mesh_positions(
                "invalid_mesh_name", self.sample_vertex_positions, 1.0)

    def test_set_animation_time(self):
        """ Test if the animation time can be set successfully.
        """
        duration = 10.0
        self.usd_instance.set_animation_time(duration)
        self.assertEqual(self.usd_instance.stage.GetEndTimeCode(
        ), duration, "Animation time was not set correctly.")

if __name__ == "__main__":
    unittest.main()