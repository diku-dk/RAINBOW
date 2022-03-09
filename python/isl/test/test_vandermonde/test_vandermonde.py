import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../../")

import isl.math.vandermonde as V


class TestVandermonde(unittest.TestCase):

    def test_make_poly_string(self):
        expected = ""
        actual = V.make_poly_str(-1)
        self.assertEqual(expected, actual, 'Negative degree polynomial test')

        expected = "1"
        actual = V.make_poly_str(0)
        self.assertEqual(expected, actual, 'Zeroth order polynomial test')

        expected = "1 + x^1 + y^1"
        actual = V.make_poly_str(1)
        self.assertEqual(expected, actual, 'First order polynomial test')

        expected = "1 + x^1 + y^1 + x^2 + x^1y^1 + y^2"
        actual = V.make_poly_str(2)
        self.assertEqual(expected, actual, 'Second order polynomial test')

        expected = "1 + x^1 + y^1 + x^2 + x^1y^1 + y^2 + x^3 + x^2y^1 + x^1y^2 + y^3"
        actual = V.make_poly_str(3)
        self.assertEqual(expected, actual, 'Third order polynomial test')
