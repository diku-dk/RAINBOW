import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../../')

import isl.tutorial_test.test_code_peidi as peidi

class TestNewSimulator(unittest.TestCase):
	def test_func_correct(self):
		expected = peidi.fabo(5) + peidi.fabo(4)
		# actual = bns.simulator_func_eoor(1)
		actual = peidi.fabo(6)
		self.assertEqual(actual, expected)
		
	# def test_func_error(self):
	# 	expected = peidi.fabo(5)
	# 	# actual = bns.simulator_func_eoor(1)
	# 	actual = 2
	# 	self.assertEqual(actual, expected)

	def test_fabo(self):
		expected = 5
		actual = peidi.fabo(4)
		self.assertEqual(actual, expected)