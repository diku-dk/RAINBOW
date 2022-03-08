# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:36:58 2022

@author: Rasmus Netterstrøm
"""

import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../../")
import isl.simulators.new_simulator.brandNewSimulator_python as bns
import isl.test.test_common as utils


class TestNewSimulator(unittest.TestCase):
    def test_func_correct(self):
        expected = 1
        actual   = bns.simulator_func(1)
        self.assertEqual(actual, expected)
    
    def test_func_array(self):
        expected = np.array([1,2,3], dtype=np.float64)
        actual   = np.array([
            bns.simulator_func(1),
            bns.simulator_func(2),
            bns.simulator_func(3)
            ])
        
        self.assertTrue(
            utils.array_equal(actual, expected)
            )
        
    def test_func_error(self):
        expected = 1
        actual   = bns.simulator_func_error(1)
        self.assertEqual(actual, expected)
        