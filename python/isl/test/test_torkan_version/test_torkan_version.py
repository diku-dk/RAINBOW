import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../../")
import isl.tutorial_test.pdl_function as pdl

class TestPDL(unittest.TestCase):
    """
        Hello world
    """

    def test_pdl(self):
        test_name = "Name"
        unique_name_1 = pdl.add(1,2) # expected output
        expected = 3
        self.assertEqual(unique_name_1, expected)
