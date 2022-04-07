import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../../")

import isl.math.angle as angle


class TestAngleAPI(unittest.TestCase):
    def test_radians_to_degrees_1(self):
        radian = 2
        expected = 114.591
        actual = angle.radians_to_degrees(radian)
        self.assertAlmostEqual(actual, expected, 2)

    def test_radians_to_degrees_2(self):
        radian = 3.701
        expected = 212.125
        actual = angle.radians_to_degrees(radian)
        self.assertAlmostEqual(actual, expected, 0)

    def test_degrees_to_radians_1(self):
        degree = 212.125
        expected = 3.701
        actual = angle.degrees_to_radians(degree)
        self.assertAlmostEqual(actual, expected, 0)
    
    def test_degrees_to_radians_2(self):
        degree = 114.591
        expected = 2
        actual = angle.degrees_to_radians(degree)
        self.assertAlmostEqual(actual, expected, 0)
    
    def test_angle_mix_1(self):
        degree = 114.591
        expected = degree
        actual = angle.radians_to_degrees(angle.degrees_to_radians(degree))
        self.assertAlmostEqual(actual, expected, 0)
    
    def test_angle_mix_2(self):
        radian = 4.701
        expected = radian
        actual = angle.degrees_to_radians(angle.radians_to_degrees(radian))
        self.assertAlmostEqual(actual, expected, 0)
