import unittest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspth(__file__)))+  '/../../')
import isl.simulators.new_simulators.brandNewSimulator as bns
class TestNewSimulator(unittest.TestCase):
	def test_func_correct(self):
		expected = 1
		actual = bns.simulator_func(1)
		self.assertEqual(actual, expected)

	def test_func_error(self):
		expected = 1
		actual = bns.simulator_func_eoor(1)
		self.assertEqual(actual, expected)