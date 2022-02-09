import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../../")
import isl.simulators.prox_rigid_bodies.api as api


class TestRigidBodiesAPI(unittest.TestCase):
    """
        Hello world
    """

    def test_generate_unique_name_1(self):
        test_name = "Name"
        unique_name_1 = api.generate_unique_name(test_name)
        unique_name_2 = api.generate_unique_name(test_name)
        self.assertNotEqual(unique_name_1, unique_name_2)
