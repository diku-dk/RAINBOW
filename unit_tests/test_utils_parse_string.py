import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.util.parse_string as PARSE


class TestQuaternionAPI(unittest.TestCase):
    def test_parse_rotation_1(self):
        input = "rx: 1.3"
        actual = PARSE.parse_string_to_rotation_check(input)
        self.assertTrue(actual)

    def test_parse_rotation_2(self):
        input = "ry: 1.3"
        actual = PARSE.parse_string_to_rotation_check(input)
        self.assertTrue(actual)

    def test_parse_rotation_3(self):
        input = "rz: 1.3"
        actual = PARSE.parse_string_to_rotation_check(input)
        self.assertTrue(actual)

    def test_parse_rotation_4(self):
        input = "rz: -1.3"
        actual = PARSE.parse_string_to_rotation_check(input)
        self.assertTrue(actual)

    def test_parse_rotation_5(self):
        input = "rz: -1e+20"
        actual = PARSE.parse_string_to_rotation_check(input)
        self.assertTrue(actual)

    def test_parse_rotation_6(self):
        input = "rz: -1.2e+2.3"
        actual = PARSE.parse_string_to_rotation_check(input)
        self.assertTrue(actual)

    def test_parse_rotation_7(self):
        input = "rz: 1.2e-2.3"
        actual = PARSE.parse_string_to_rotation_check(input)
        self.assertTrue(actual)

    def test_parse_rotation_8(self):
        input = "rz:1.2e-2.3"
        actual = PARSE.parse_string_to_rotation_check(input)
        self.assertTrue(actual)

    def test_parse_rotation_9(self):
        input = "rz:1.2e--2.3"
        actual = PARSE.parse_string_to_rotation_check(input)
        self.assertFalse(actual)

    def test_parse_array_1(self):
        input = "[1,2,3]"
        actual = PARSE.parse_string_to_array_check(input)
        self.assertTrue(actual)

    def test_parse_array_2(self):
        input = "[1,2,3,4]"
        actual = PARSE.parse_string_to_array_check(input)
        self.assertTrue(actual)

    def test_parse_array_3(self):
        input = "[1,-2,1.3e-10,4.0]"
        actual = PARSE.parse_string_to_array_check(input)
        self.assertTrue(actual)

    def test_parse_array_4(self):
        input = "[1, -2, 1.3e-10, 4.0]"
        actual = PARSE.parse_string_to_array_check(input)
        self.assertTrue(actual)

    def test_parse_array_5(self):
        input = "[1, -02, 1.3e-10, 4.0]"
        actual = PARSE.parse_string_to_array_check(input)
        self.assertFalse(actual)
