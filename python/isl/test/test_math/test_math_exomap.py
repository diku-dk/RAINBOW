from cmath import sin, tan
from ctypes import util
import unittest
import os
import sys
import numpy as np
import math as m

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../../")
import isl.math.expmap as exm
import isl.math.quaternion as Q
import isl.test.test_common as utils

class TestExomapAPI(unittest.TestCase):
    def test_log_exp_1(self):
        expected = Q.rand()
        v = exm.log(expected)
        actual = exm.exp(v)
        self.assertTrue(utils.array_equal(actual,expected))

    def test_log_exp_2(self):
        expected = Q.rand()
        v = exm.to_expmap(expected)
        actual = exm.to_quaternion(v)
        self.assertTrue(utils.array_equal(actual,expected))

    def test_time_derivative_1(self):
        pass
