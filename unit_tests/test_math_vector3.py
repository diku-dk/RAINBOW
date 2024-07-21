import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.math.vector3 as V3
import rainbow.util.test_tools as TEST


class TestVector3API(unittest.TestCase):
    '''
        Need to ask about function make_orthonomal_vectors
    '''
    '''
        Adding test on less
    '''

    def test_less_1(self):
        test_vec_1 = V3.make(1, 2, 3)
        test_vec_2 = V3.make(0, 0, 0)
        self.assertTrue(V3.less(test_vec_2, test_vec_1))
        self.assertFalse(V3.less(test_vec_1, test_vec_2))

    def test_less_2(self):
        test_vec_1 = V3.make(1, 2, 3)
        test_vec_2 = V3.make(1, 0, 0)
        self.assertTrue(V3.less(test_vec_2, test_vec_1))
        self.assertFalse(V3.less(test_vec_1, test_vec_2))

    def test_less_3(self):
        test_vec_1 = V3.make(1, 2, 3)
        test_vec_2 = V3.make(1, 2, 0)
        self.assertTrue(V3.less(test_vec_2, test_vec_1))
        self.assertFalse(V3.less(test_vec_1, test_vec_2))

    def test_less_4(self):
        test_vec_1 = V3.make(1, 2, 3)
        self.assertFalse(V3.less(test_vec_1, test_vec_1))

    def test_less_5(self):
        test_vec_1 = V3.make(1, 2, 3)
        test_vec_2 = V3.make(0, 2, 3)
        self.assertTrue(V3.less(test_vec_2, test_vec_1))
        self.assertFalse(V3.less(test_vec_1, test_vec_2))

    def test_less_5(self):
        test_vec_1 = V3.make(1, 2, 3)
        test_vec_2 = V3.make(0, 2, 3)
        self.assertTrue(V3.less(test_vec_2, test_vec_1))
        self.assertFalse(V3.less(test_vec_1, test_vec_2))

    def test_greater_1(self):
        test_vec_1 = V3.make(0, 0, 0)
        test_vec_2 = V3.make(1, 2, 3)
        self.assertTrue(V3.greather(test_vec_2, test_vec_1))
        self.assertFalse(V3.greather(test_vec_1, test_vec_2))

    def test_greater_2(self):
        test_vec_1 = V3.make(1, 0, 0)
        test_vec_2 = V3.make(1, 2, 3)
        self.assertTrue(V3.greather(test_vec_2, test_vec_1))
        self.assertFalse(V3.greather(test_vec_1, test_vec_2))

    def test_greater_3(self):
        test_vec_1 = V3.make(1, 2, 0)
        test_vec_2 = V3.make(1, 2, 3)
        self.assertTrue(V3.greather(test_vec_2, test_vec_1))
        self.assertFalse(V3.greather(test_vec_1, test_vec_2))

    def test_greater_4(self):
        test_vec_1 = V3.make(0, 2, 3)
        test_vec_2 = V3.make(1, 2, 3)
        self.assertTrue(V3.greather(test_vec_2, test_vec_1))
        self.assertFalse(V3.greather(test_vec_1, test_vec_2))

    def test_greater_5(self):
        test_vec_1 = V3.make(1, 2, 3)
        test_vec_2 = V3.make(1, 2, 3)
        self.assertFalse(V3.greather(test_vec_1, test_vec_2))
        self.assertFalse(V3.greather(test_vec_2, test_vec_1))

    def test_greater_than_equal(self):
        test_vec_1 = V3.make(0, 0, 0)
        test_vec_2 = V3.make(1, 2, 3)
        self.assertTrue(V3.greather_than_equal(test_vec_2, test_vec_1))

    def test_less_than_equal(self):
        test_vec_1 = V3.make(0, 0, 0)
        test_vec_2 = V3.make(1, 2, 3)
        self.assertFalse(V3.less_than_equal(test_vec_2, test_vec_1))

    def test_zeros(self):
        actual = V3.zero()
        expected = np.zeros((3,), dtype=np.float64)
        self.assertTrue(None ==
                        np.testing.assert_array_equal(
                            actual, expected
                        )
                        )

    def test_ones(self):
        actual = V3.ones()
        expected = np.ones((3,), dtype=np.float64)
        self.assertTrue(None ==
                        np.testing.assert_array_equal(
                            actual, expected
                        )
                        )

    def test_make(self):
        actual = V3.make(1, 2, 3)
        expected = np.array([1, 2, 3], dtype=np.float64)
        self.assertTrue(None ==
                        np.testing.assert_array_equal(
                            actual, expected
                        )
                        )

    def test_make_vec4(self):
        actual = V3.make_vec4(1, 2, 3, 4)
        expected = np.array([1, 2, 3, 4], dtype=np.float64)
        self.assertTrue(None ==
                        np.testing.assert_array_equal(
                            actual, expected
                        )
                        )

    def test_string_1(self):
        actual = V3.from_string("ones")
        expected = V3.ones()
        self.assertTrue(None ==
                        np.testing.assert_array_equal(
                            actual, expected
                        )
                        )

    def test_string_2(self):
        actual = V3.from_string("zero")
        expected = V3.zero()
        self.assertTrue(None ==
                        np.testing.assert_array_equal(
                            actual, expected
                        )
                        )

    def test_string_3(self):
        actual = V3.from_string("i")
        expected = V3.i()
        self.assertTrue(None ==
                        np.testing.assert_array_equal(
                            actual, expected
                        )
                        )

    def test_string_4(self):
        actual = V3.from_string("j")
        expected = V3.j()
        self.assertTrue(None ==
                        np.testing.assert_array_equal(
                            actual, expected
                        )
                        )

    def test_string_5(self):
        actual = V3.from_string("k")
        expected = V3.k()
        self.assertTrue(None ==
                        np.testing.assert_array_equal(
                            actual, expected
                        )
                        )

    def test_string_6(self):
        expected_lower, expected_higher = 1, 2
        actual = V3.from_string(f"rand:{expected_lower}:{expected_higher}")
        testing_to_bool_lower = actual <= expected_higher
        testing_to_bool_higher = actual >= expected_lower
        self.assertTrue(None == np.testing.assert_array_equal(
            testing_to_bool_lower, testing_to_bool_higher
        ))

    def test_string_7(self):
        actual = V3.from_string("[1,2,3]")
        expected = V3.make(1, 2, 3)
        self.assertTrue(None == np.testing.assert_array_equal(
            actual, expected
        ))

    def test_i(self):
        actual = V3.i()
        expected = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.assertTrue(
            None == np.testing.assert_array_equal(actual, expected)
        )

    def test_j(self):
        actual = V3.j()
        expected = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.assertTrue(
            None == np.testing.assert_array_equal(actual, expected)
        )

    def test_k(self):
        actual = V3.k()
        expected = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.assertTrue(
            None == np.testing.assert_array_equal(actual, expected)
        )

    def test_cross_1(self):
        test_arr_1 = V3.i()
        test_arr_2 = V3.j()
        actual = V3.cross(test_arr_1, test_arr_2)
        expected = V3.k()
        self.assertTrue(
            None == np.testing.assert_array_equal(actual, expected)
        )

    def test_cross_2(self):
        test_arr_1 = V3.j()
        test_arr_2 = V3.k()
        actual = V3.cross(test_arr_1, test_arr_2)
        expected = V3.i()
        self.assertTrue(
            None == np.testing.assert_array_equal(actual, expected)
        )

    def test_cross_3(self):
        test_arr_1 = V3.j()
        test_arr_2 = V3.j()
        actual = V3.cross(test_arr_1, test_arr_2)
        expected = V3.zero()
        self.assertTrue(
            None == np.testing.assert_array_equal(actual, expected)
        )

    def test_unit(self):
        non_unit_vec = np.array([1, 2, 3])
        norm_vec = np.sqrt((non_unit_vec[0] ** 2 + non_unit_vec[1] ** 2 + non_unit_vec[2] ** 2))
        expected = non_unit_vec / norm_vec
        actual = V3.unit(non_unit_vec)
        self.assertTrue(
            None == np.testing.assert_array_equal(actual, expected)
        )

    def test_max_abs_component_1(self):
        test_vec = V3.make(1, 2, 3)
        actual = V3.max_abs_component(test_vec)
        expected = 2
        self.assertEqual(actual, expected)

    def test_max_abs_component_2(self):
        test_vec = V3.make(1, 3, 2)
        actual = V3.max_abs_component(test_vec)
        expected = 1
        self.assertEqual(actual, expected)

    def test_max_abs_component_3(self):
        test_vec = V3.make(3, 2, 2)
        actual = V3.max_abs_component(test_vec)
        expected = 0
        self.assertEqual(actual, expected)

    def test_max_abs_component_4(self):
        test_vec = -1 * V3.make(1, 2, 3)
        actual = V3.max_abs_component(test_vec)
        expected = 2
        self.assertEqual(actual, expected)

    def test_make_orthonomal_vectors_1(self):
        vector = np.array([1, 2, 3], dtype=np.float64)
        norm_vector = V3.unit(vector)
        tangent, binormal, n = V3.make_orthonormal_vectors(norm_vector)
        expected = n
        actual = np.cross(tangent, binormal)
        self.assertTrue(
            TEST.is_array_equal(actual, expected)
        )

    def test_make_orthonomal_vectors_2(self):
        vector = np.array([1.3, 0.1, -3], dtype=np.float64)
        norm_vector = V3.unit(vector)
        tangent, binormal, n = V3.make_orthonormal_vectors(norm_vector)
        expected = n
        actual = np.cross(tangent, binormal)
        self.assertTrue(
            TEST.is_array_equal(actual, expected)
        )

    def test_make_orthonomal_vectors_3(self):
        vector = np.array([1.3, 0.1, -3], dtype=np.float64)
        norm_vector = V3.unit(vector)
        tangent, binormal, n = V3.make_orthonormal_vectors(norm_vector)
        expected = tangent
        actual = np.cross(binormal, n)
        self.assertTrue(
            TEST.is_array_equal(actual, expected)
        )

    def test_make_orthonomal_vectors_4(self):
        vector = np.array([1.3, 0.1, -3], dtype=np.float64)
        norm_vector = V3.unit(vector)
        tangent, binormal, n = V3.make_orthonormal_vectors(norm_vector)
        expected = tangent
        actual = np.cross(binormal, n)
        self.assertTrue(
            TEST.is_array_equal(actual, expected)
        )

    def test_make_orthonomal_vectors_5(self):
        vector = np.array([1.3, 0.1, -3], dtype=np.float64)
        norm_vector = V3.unit(vector)
        tangent, binormal, n = V3.make_orthonormal_vectors(norm_vector)
        expected = binormal
        actual = np.cross(n, tangent)
        self.assertTrue(
            TEST.is_array_equal(actual, expected)
        )

    def test_make_orthonomal_vectors_6(self):
        vector = np.array([2, 1, 3], dtype=np.float64)
        norm_vector = V3.unit(vector)
        tangent, binormal, n = V3.make_orthonormal_vectors(norm_vector)
        expected = n
        actual = np.cross(tangent, binormal)
        self.assertTrue(
            TEST.is_array_equal(actual, expected)
        )

    def test_make_orthonomal_vectors_7(self):
        vector = np.array([2, 3, 1], dtype=np.float64)
        norm_vector = V3.unit(vector)
        tangent, binormal, n = V3.make_orthonormal_vectors(norm_vector)
        expected = n
        actual = np.cross(tangent, binormal)
        self.assertTrue(
            TEST.is_array_equal(actual, expected)
        )

    def test_make_orthonomal_vectors_8(self):
        vector = np.array([2, 3, 1], dtype=np.float64)
        norm_vector = V3.unit(vector)
        tangent, binormal, n = V3.make_orthonormal_vectors(norm_vector)
        expected = n
        actual = np.cross(tangent, binormal)
        self.assertTrue(
            TEST.is_array_equal(actual, expected)
        )

    def test_make_orthonomal_vectors_9(self):
        vector = np.array([2, 3, 1], dtype=np.float64)
        norm_vector = V3.unit(vector)
        tangent, binormal, n = V3.make_orthonormal_vectors(norm_vector)
        expected = n
        actual = np.cross(binormal, tangent)
        self.assertTrue(
            TEST.is_array_not_equal(actual, expected)
        )

    def test_make_orthonomal_vectors_10(self):
        vector = np.array([4, 2, 1], dtype=np.float64)
        norm_vector = V3.unit(vector)
        tangent, binormal, n = V3.make_orthonormal_vectors(norm_vector)
        expected = n
        actual = np.cross(binormal, tangent)
        self.assertTrue(
            TEST.is_array_not_equal(actual, expected)
        )

    def test_from_string_err_1(self):
        input_ = "Hello world"
        with self.assertRaises(AssertionError):
            V3.from_string(input_)

    def test_from_string_err_2(self):
        input_ = "ones1"
        with self.assertRaises(AssertionError):
            V3.from_string(input_)

    def test_from_string_err_3(self):
        input_ = "ik"
        with self.assertRaises(AssertionError):
            V3.from_string(input_)

    def test_from_string_err_4(self):
        input_ = "rand:1:"
        with self.assertRaises(AssertionError):
            V3.from_string(input_)

    def test_from_string_err_5(self):
        input_ = "rand:1:3a"
        with self.assertRaises(AssertionError):
            V3.from_string(input_)

    def test_from_string_err_6(self):
        input_ = "[2,3,4]a"
        with self.assertRaises(AssertionError):
            V3.from_string(input_)

    def test_from_string_err_7(self):
        input_ = "a[2,3,4]"
        with self.assertRaises(AssertionError):
            V3.from_string(input_)

    def test_from_string_err_8(self):
        input_ = "[2,3]"
        with self.assertRaises(AssertionError):
            V3.from_string(input_)
