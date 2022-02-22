import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../../")
import isl.math.vector3 as vec3


class TestRigidBodiesAPI(unittest.TestCase):
    '''
        Need to ask about function make_orthonomal_vectors
    '''
    # def test_make_orthonormal_vectors(self):
        # test_vec = vec3.make(1,2,3)
        # vs, vt, vn = vec3.make_orthonormal_vectors(test_vec)
        # self.assertEqual(np.linalg.norm(vn), 1)
        # self.assertEqual(np.linalg.norm(vt), 1)
        # self.assertEqual(np.linalg.norm(vs), 1)
    '''
        Adding test on less
    ''' 
    def test_less_1(self):
        test_vec_1 = vec3.make(1,2,3)
        test_vec_2 = vec3.make(0,0,0)
        self.assertTrue(vec3.less(test_vec_2, test_vec_1))
        self.assertFalse(vec3.less(test_vec_1, test_vec_2))
    
    def test_less_2(self):
        test_vec_1 = vec3.make(1,2,3)
        test_vec_2 = vec3.make(1,0,0)
        self.assertTrue(vec3.less(test_vec_2, test_vec_1))
        self.assertFalse(vec3.less(test_vec_1, test_vec_2))
    
    def test_less_3(self):
        test_vec_1 = vec3.make(1,2,3)
        test_vec_2 = vec3.make(1,2,0)
        self.assertTrue(vec3.less(test_vec_2, test_vec_1))
        self.assertFalse(vec3.less(test_vec_1, test_vec_2))
    
    def test_less_4(self):
        test_vec_1 = vec3.make(1,2,3)
        self.assertFalse(vec3.less(test_vec_1, test_vec_1))
     
    def test_less_5(self):
        test_vec_1 = vec3.make(1,2,3)
        test_vec_2 = vec3.make(0,2,3)
        self.assertTrue(vec3.less(test_vec_2, test_vec_1))
        self.assertFalse(vec3.less(test_vec_1, test_vec_2))
    
    def test_less_5(self):
        test_vec_1 = vec3.make(1,2,3)
        test_vec_2 = vec3.make(0,2,3)
        self.assertTrue(vec3.less(test_vec_2, test_vec_1))
        self.assertFalse(vec3.less(test_vec_1, test_vec_2))
    
    def test_greater_1(self):
        test_vec_1 = vec3.make(0,0,0)
        test_vec_2 = vec3.make(1,2,3)
        self.assertTrue(vec3.greather(test_vec_2, test_vec_1))
        self.assertFalse(vec3.greather(test_vec_1, test_vec_2))
    
    def test_greater_2(self):
        test_vec_1 = vec3.make(1,0,0)
        test_vec_2 = vec3.make(1,2,3)
        self.assertTrue(vec3.greather(test_vec_2, test_vec_1))
        self.assertFalse(vec3.greather(test_vec_1, test_vec_2))

    def test_greater_3(self):
        test_vec_1 = vec3.make(1,2,0)
        test_vec_2 = vec3.make(1,2,3)
        self.assertTrue(vec3.greather(test_vec_2, test_vec_1))
        self.assertFalse(vec3.greather(test_vec_1, test_vec_2))

    def test_greater_4(self):
        test_vec_1 = vec3.make(0,2,3)
        test_vec_2 = vec3.make(1,2,3)
        self.assertTrue(vec3.greather(test_vec_2, test_vec_1))
        self.assertFalse(vec3.greather(test_vec_1, test_vec_2))
    
    def test_greater_than_equal(self):
        test_vec_1 = vec3.make(0,0,0)
        test_vec_2 = vec3.make(1,2,3)
        self.assertTrue(vec3.greather_than_equal(test_vec_2, test_vec_1))
    
    def test_less_than_equal(self):
        test_vec_1 = vec3.make(0,0,0)
        test_vec_2 = vec3.make(1,2,3)
        self.assertFalse(vec3.less_than_equal(test_vec_2, test_vec_1))
    
    def test_zeros(self):
        actual = vec3.zero()
        expected = np.zeros((3,), dtype=np.float64)
        self.assertTrue(None ==
            np.testing.assert_array_equal(
                actual, expected
                                                    )
                        )
    
    def test_ones(self):
        actual = vec3.ones()
        expected = np.ones((3,), dtype=np.float64)
        self.assertTrue(None ==
            np.testing.assert_array_equal(
                actual, expected
                                                    )
                        )
    
    def test_make(self):
        actual = vec3.make(1,2,3)
        expected = np.array([1,2,3], dtype=np.float64)
        self.assertTrue(None ==
            np.testing.assert_array_equal(
                actual, expected
                                                    )
                        )
    
    def test_make_vec4(self):
        actual = vec3.make_vec4(1,2,3,4)
        expected = np.array([1,2,3,4], dtype=np.float64)
        self.assertTrue(None ==
            np.testing.assert_array_equal(
                actual, expected
                                                    )
                        )
    
    def test_string_1(self):
        actual   = vec3.from_string("ones") 
        expected = vec3.ones()  
        self.assertTrue(None ==
            np.testing.assert_array_equal(
                actual, expected
                                                    )
                        )
    
    def test_string_2(self):
        actual   = vec3.from_string("zero") 
        expected = vec3.zero()  
        self.assertTrue(None ==
            np.testing.assert_array_equal(
                actual, expected
                                                    )
                        )

    def test_string_3(self):
        actual   = vec3.from_string("i") 
        expected = vec3.i()  
        self.assertTrue(None ==
            np.testing.assert_array_equal(
                actual, expected
                                                    )
                        )

    def test_string_4(self):
        actual   = vec3.from_string("j") 
        expected = vec3.j()  
        self.assertTrue(None ==
            np.testing.assert_array_equal(
                actual, expected
                                                    )
                        )

    def test_string_5(self):
        actual   = vec3.from_string("k") 
        expected = vec3.k()  
        self.assertTrue(None ==
            np.testing.assert_array_equal(
                actual, expected
                                                    )
                        )
    
    def test_string_6(self):
        expected_lower, expected_higher =  1,2
        actual = vec3.from_string(f"rand:{expected_lower}:{expected_higher}") 
        testing_to_bool_lower  = actual <= expected_higher
        testing_to_bool_higher = actual >= expected_lower
        self.assertTrue(None == np.testing.assert_array_equal(
            testing_to_bool_lower, testing_to_bool_higher
        ) )
    
    def test_string_7(self):
        actual   = vec3.from_string("[1,2,3]")
        expected = vec3.make(1,2,3)
        self.assertTrue(None == np.testing.assert_array_equal(
            actual, expected
        ) )

    def test_i(self):
        actual   = vec3.i()
        expected = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.assertTrue(
            None == np.testing.assert_array_equal(actual, expected)
        )


    def test_j(self):
        actual   = vec3.j()
        expected = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.assertTrue(
            None == np.testing.assert_array_equal(actual, expected)
        )

    def test_k(self):
        actual   = vec3.k()
        expected = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.assertTrue(
            None == np.testing.assert_array_equal(actual, expected)
        )

    def test_cross_1(self):
        test_arr_1 = vec3.i()
        test_arr_2 = vec3.j()
        actual     = vec3.cross(test_arr_1, test_arr_2)
        expected   = vec3.k()
        self.assertTrue(
            None == np.testing.assert_array_equal(actual, expected)
        )
    
    def test_cross_2(self):
        test_arr_1 = vec3.j()
        test_arr_2 = vec3.k()
        actual     = vec3.cross(test_arr_1, test_arr_2)
        expected   = vec3.i()
        self.assertTrue(
            None == np.testing.assert_array_equal(actual, expected)
        )
    
    def test_cross_3(self):
        test_arr_1 = vec3.j()
        test_arr_2 = vec3.j()
        actual     = vec3.cross(test_arr_1, test_arr_2)
        expected   = vec3.zero()
        self.assertTrue(
            None == np.testing.assert_array_equal(actual, expected)
        )
    
    def test_unit(self):
        non_unit_vec = np.array([1,2,3])
        norm_vec     = np.sqrt((non_unit_vec[0]**2 + non_unit_vec[1]**2 + non_unit_vec[2]**2))
        expected     = non_unit_vec / norm_vec
        actual       = vec3.unit(non_unit_vec)
        self.assertTrue(
            None == np.testing.assert_array_equal(actual, expected)
        )

    def test_max_abs_component_1(self):
        test_vec = vec3.make(1,2,3)
        actual   = vec3.max_abs_component(test_vec)
        expected = 2
        self.assertEqual(actual, expected)        

    def test_max_abs_component_2(self):
        test_vec = vec3.make(1,3,2)
        actual   = vec3.max_abs_component(test_vec)
        expected = 1
        self.assertEqual(actual, expected)

    def test_max_abs_component_3(self):
        test_vec = vec3.make(3,2,2)
        actual   = vec3.max_abs_component(test_vec)
        expected = 0
        self.assertEqual(actual, expected)
    
    def test_max_abs_component_4(self):
        test_vec = -1*vec3.make(1,2,3)
        actual   =    vec3.max_abs_component(test_vec)
        expected = 2
        self.assertEqual(actual, expected)


