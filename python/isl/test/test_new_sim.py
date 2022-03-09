"""
Created on Wed Mar 9

@authro: l1r
"""

import unittest
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../")
import isl.simulators.brandNewSim as bns

class TestNewSim(unittest.TestCase):
    def test_func_correct(self):
        expected = 1
        actual = bns.simulator_func(1)
        self.assertEqual(actual, expected)

    def test_func_error(self):
        expected = 2
        actual = bns.simulator_func(1)
        self.assertEqual(actual, expected)
