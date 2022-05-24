import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.simulators.prox_rigid_bodies.api as API


class TestRigidBodiesAPI(unittest.TestCase):

    def test_generate_unique_name_1(self):
        test_name = "Name"
        unique_name_1 = API.generate_unique_name(test_name)
        unique_name_2 = API.generate_unique_name(test_name)
        self.assertNotEqual(unique_name_1, unique_name_2)
