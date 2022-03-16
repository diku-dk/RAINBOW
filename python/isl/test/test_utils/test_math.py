from cmath import sin, tan
from ctypes import util
import unittest
import os
import sys
import numpy as np
import math as m

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../../")
import isl.math.vector3 as vec3
import isl.math.angle as angle
import isl.test.test_common as utils
import isl.math.quaternion as quat
import isl.math.matrix3 as mat3

def Rx(theta):
    return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])



class TestRigidBodiesAPI(unittest.TestCase):
    '''
        Need to ask about function make_orthonomal_vectors
    '''
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
    
    def test_greater_5(self):
        test_vec_1 = vec3.make(1,2,3)
        test_vec_2 = vec3.make(1,2,3)
        self.assertFalse(vec3.greather(test_vec_1, test_vec_2))
        self.assertFalse(vec3.greather(test_vec_2, test_vec_1))

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
    
    def test_make_orthonomal_vectors_1(self):
        vector      = np.array([1,2,3], dtype=np.float64)
        norm_vector = vec3.unit(vector)
        tangent, binormal, n = vec3.make_orthonormal_vectors(norm_vector)
        expected = n
        actual   = np.cross(tangent, binormal)
        self.assertTrue(
            utils.array_equal(actual, expected)
        )
    
    def test_make_orthonomal_vectors_2(self):
        vector      = np.array([1.3,0.1,-3], dtype=np.float64)
        norm_vector = vec3.unit(vector)
        tangent, binormal, n = vec3.make_orthonormal_vectors(norm_vector)
        expected = n
        actual   = np.cross(tangent, binormal)
        self.assertTrue(
            utils.array_equal(actual, expected)
        )
    
    def test_make_orthonomal_vectors_3(self):
        vector      = np.array([1.3,0.1,-3], dtype=np.float64)
        norm_vector = vec3.unit(vector)
        tangent, binormal, n = vec3.make_orthonormal_vectors(norm_vector)
        expected = tangent
        actual   = np.cross(binormal, n)
        self.assertTrue(
            utils.array_equal(actual, expected)
        )

    def test_make_orthonomal_vectors_4(self):
        vector      = np.array([1.3,0.1,-3], dtype=np.float64)
        norm_vector = vec3.unit(vector)
        tangent, binormal, n = vec3.make_orthonormal_vectors(norm_vector)
        expected = tangent
        actual   = np.cross(binormal, n)
        self.assertTrue(
            utils.array_equal(actual, expected)
        )
    
    def test_make_orthonomal_vectors_5(self):
        vector      = np.array([1.3,0.1,-3], dtype=np.float64)
        norm_vector = vec3.unit(vector)
        tangent, binormal, n = vec3.make_orthonormal_vectors(norm_vector)
        expected = binormal
        actual   = np.cross(n, tangent)
        self.assertTrue(
            utils.array_equal(actual, expected)
        )
    
    def test_make_orthonomal_vectors_6(self):
        vector      = np.array([2,1,3], dtype=np.float64)
        norm_vector = vec3.unit(vector)
        tangent, binormal, n = vec3.make_orthonormal_vectors(norm_vector)
        expected = n
        actual   = np.cross(tangent, binormal)
        self.assertTrue(
            utils.array_equal(actual, expected)
        )
    
    def test_make_orthonomal_vectors_7(self):
        vector      = np.array([2,3,1], dtype=np.float64)
        norm_vector = vec3.unit(vector)
        tangent, binormal, n = vec3.make_orthonormal_vectors(norm_vector)
        expected = n
        actual   = np.cross(tangent, binormal)
        self.assertTrue(
            utils.array_equal(actual, expected)
        )
    
    def test_make_orthonomal_vectors_8(self):
        vector      = np.array([2,3,1], dtype=np.float64)
        norm_vector = vec3.unit(vector)
        tangent, binormal, n = vec3.make_orthonormal_vectors(norm_vector)
        expected = n
        actual   = np.cross(tangent, binormal)
        self.assertTrue(
            utils.array_equal(actual, expected)
        )

    def test_make_orthonomal_vectors_9(self):
        vector      = np.array([2,3,1], dtype=np.float64)
        norm_vector = vec3.unit(vector)
        tangent, binormal, n = vec3.make_orthonormal_vectors(norm_vector)
        expected = n
        actual   = np.cross(binormal, tangent)
        self.assertTrue(
            utils.array_not_equal(actual, expected)
            )
    
    def test_make_orthonomal_vectors_10(self):
        vector      = np.array([4,2,1], dtype=np.float64)
        norm_vector = vec3.unit(vector)
        tangent, binormal, n = vec3.make_orthonormal_vectors(norm_vector)
        expected = n
        actual   = np.cross(binormal, tangent)
        self.assertTrue(
            utils.array_not_equal(actual, expected)
            )

    def test_radians_to_degrees_1(self):
        radian = 2
        expected = 114.591
        actual   = angle.radians_to_degrees(radian)
        self.assertAlmostEqual(actual, expected, 2)

    def test_radians_to_degrees_2(self):
        radian = 3.701
        expected = 212.125
        actual   = angle.radians_to_degrees(radian)
        self.assertAlmostEqual(actual, expected, 0)

    def test_degrees_to_radians_1(self):
        degree = 212.125
        expected = 3.701
        actual   = angle.degrees_to_radians(degree)
        self.assertAlmostEqual(actual, expected, 0)
    
    def test_degrees_to_radians_2(self):
        degree = 114.591
        expected = 2
        actual   = angle.degrees_to_radians(degree)
        self.assertAlmostEqual(actual, expected, 0)
    
    def test_angle_mix_1(self):
        degree = 114.591
        expected = degree
        actual   = angle.radians_to_degrees(angle.degrees_to_radians(degree))
        self.assertAlmostEqual(actual, expected, 0)
    
    def test_angle_mix_2(self):
        radian   = 4.701
        expected = radian
        actual   = angle.degrees_to_radians(angle.radians_to_degrees(radian))
        self.assertAlmostEqual(actual, expected, 0)

    def test_quaterion_make_1(self):
        # First define a quaterion q:
        # [s, (x,y,z)]
        # p. 600, Physics-Based Animation, K. Erleben et al
        
        expected = np.array([1,2,3,4], dtype=np.float64)
        actual   = quat.make(1,2,3,4)
        self.assertTrue(utils.array_equal(actual,expected))

    def test_quaterion_make_2(self):
        # First define a quaterion q:
        # [s, (x,y,z)]
        # p. 600, Physics-Based Animation, K. Erleben et al
        
        expected = np.array([1,2,3,4], dtype=np.float64)
        actual   = quat.make(1,-2,3,4)
        self.assertTrue(utils.array_not_equal(actual,expected))
    
    def test_quaterion_prod_1(self):
        # First define a quaterion q:
        # [s, (x,y,z)]
        # p. 600, Physics-Based Animation, K. Erleben et al
        
        q = np.array([1,2,3,4])

        # Using definition 18.41
        to_c_v   = np.array([1,-1,-1,-1], dtype=np.float64)
        q_c      = to_c_v * q
        qq_c     = np.ones(4)
        qq_c[0]  =  q[0]*q_c[0] - np.dot(q[1:],q_c[1:])
        qq_c[1:] = q[0] * q_c[1:] + q_c[0] * q[1:] + np.cross(q[1:], q_c[1:], axis=0)
        #

        expected = qq_c
        actual   = quat.prod(q,q_c)
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_quaterion_prod_2(self):
        # First define a quaterion q:
        # [s, (x,y,z)]
        # p. 600, Physics-Based Animation, K. Erleben et al
        
        q = np.array([3,1,6,-4])

        # Using definition 18.41
        to_c_v   = np.array([1,-1,-1,-1], dtype=np.float64)
        q_c      = to_c_v * q
        qq_c     = np.ones(4)
        qq_c[0]  =  q[0]*q_c[0] - np.dot(q[1:],q_c[1:])
        qq_c[1:] = q[0] * q_c[1:] + q_c[0] * q[1:] + np.cross(q[1:], q_c[1:], axis=0)
        #
        
        expected = qq_c
        actual   = quat.prod(q,q_c)
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_quaterion_prod_3(self):
        # First define a quaterion q:
        # [s, (x,y,z)]
        # p. 600, Physics-Based Animation, K. Erleben et al
        
        q = np.array([0.3,0.10000,0.00006,-8])

        to_c_v   = np.array([1,-1,-1,-1], dtype=np.float64)
        q_c      = to_c_v * q
        qq_c     = np.ones(4)
        qq_c[0]  =  q[0]*q_c[0] - np.dot(q[1:],q_c[1:])
        qq_c[1:] = q[0] * q_c[1:] + q_c[0] * q[1:] + np.cross(q[1:], q_c[1:], axis=0)
        #
        
        expected = qq_c
        actual   = quat.prod(q,q_c)
        self.assertTrue(utils.array_equal(actual,expected))

    def test_quaterion_unit_1(self):
        # Using def 18.44
        q        = np.array([2,2,3,4], dtype=np.float64)
        q_c      = quat.conjugate(q)
        # Remember: ai * (-ai) = -aaii = a^2 
        power_two = (  q[0]*(   q_c[0]) 
                     + q[1]*(-1*q_c[1])
                     + q[2]*(-1*q_c[2])
                     + q[3]*(-1*q_c[3]))
        norm      = np.sqrt(power_two)
        expected = q/norm
        actual   = quat.unit(q) 

        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_quaterion_unit_2(self):
        # Using def 18.44
        q        = np.array([0.1,2.3,-1000,0.004], dtype=np.float64)
        q_c      = quat.conjugate(q)
        # Remember: ai * (-ai) = -aaii = a^2 
        power_two = (  q[0]*(   q_c[0]) 
                     + q[1]*(-1*q_c[1])
                     + q[2]*(-1*q_c[2])
                     + q[3]*(-1*q_c[3]))
        norm      = np.sqrt(power_two)
        expected = q/norm
        actual   = quat.unit(q) 

        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_quaterion_unit_3(self):
        # Using def 18.44
        q        = np.array([0,0,0,0], dtype=np.float64)
        expected = np.array([1,0,0,0], dtype=np.float64)
        actual   = quat.unit(q) 

        self.assertTrue(utils.array_equal(actual,expected))
        
    def test_quaterion_unit_4(self):
        # Using def 18.44
        q        = np.array([2,2,3,4], dtype=np.float64)
        q_c      = quat.conjugate(q)
        # Remember: ai * (-ai) = -aaii = a^2 
        
        norm      = np.sqrt(quat.prod(q,q_c)[0])
        expected = q/norm
        actual   = quat.unit(q) 

        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_rotate_1(self):
        n     = np.array([1,0,0], dtype=np.float64)
        theta = 2*np.pi 
        q     = quat.make(np.cos(theta), np.sin(n[0]), np.sin(n[1]), np.sin(n[2]))
        p     = np.array([0,2,2,2], dtype=np.float64)

        qr    = np.array([0, p[0], p[1], p[2]], dtype=np.float64) 

        expected = quat.prod(quat.prod(q, qr), quat.conjugate(q))
        actual   = quat.rotate(q, p)
        self.assertTrue(utils.array_equal(actual,expected[1:]))
    
    def test_rotate_2(self):
        q     = quat.unit(quat.make(1,2,3,4))
        #n     = np.array([1,0,0], dtype=np.float64)
        #theta = 2*np.pi 
        #q     = quat.make(np.cos(theta), np.sin(n[0]), np.sin(n[1]), np.sin(n[2]))
        p     = np.array([0,2,2,2], dtype=np.float64)

        qr    = np.array([0, p[0], p[1], p[2]], dtype=np.float64) 

        expected = quat.prod(quat.prod(q, qr), quat.conjugate(q))
        actual   = quat.rotate(q, p)
        self.assertTrue(utils.array_equal(actual,expected[1:]))

    def test_rotate_3(self):
        r      = np.array([1,0,0])
        n      = np.array([0,1,0])
        radian = (0.5 * np.pi) / 2
        q      = np.array([np.cos(radian),n[0]*np.sin(radian),n[1]*np.sin(radian),n[2]*np.sin(radian)], dtype=np.float64)
        expected = np.array([0,0,-1])
        actual   = quat.rotate(q, r)
        self.assertTrue(utils.array_equal(actual,expected))

    def test_rotate_4(self):
        r      = np.array([1,0,0])
        n      = np.array([0,0,1])
        radian = (0.5 * np.pi) / 2
        q      = np.array([np.cos(radian),n[0]*np.sin(radian),n[1]*np.sin(radian),n[2]*np.sin(radian)], dtype=np.float64)
        expected = np.array([0,1,0])
        actual   = quat.rotate(q, r)
        self.assertTrue(utils.array_equal(actual,expected))

    def test_rotate_5(self):
        r      = np.array([1,0,0])
        n      = np.array([1,0,0])
        radian = (0.5 * np.pi) / 2
        q      = np.array([np.cos(radian),n[0]*np.sin(radian),n[1]*np.sin(radian),n[2]*np.sin(radian)], dtype=np.float64)
        expected = np.array([1,0,0])
        actual   = quat.rotate(q, r)
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_rotate_6(self):
        r      = np.array([0,1,0])
        n      = np.array([0,0,1])
        radian = (0.5 * np.pi) / 2
        q      = np.array([np.cos(radian),n[0]*np.sin(radian),n[1]*np.sin(radian),n[2]*np.sin(radian)], dtype=np.float64)
        expected = np.array([-1,0,0])
        actual   = quat.rotate(q, r)
        self.assertTrue(utils.array_equal(actual,expected))

    def test_rotate_7(self):
        r      = np.array([0,1,0])
        n      = np.array([1,0,0])
        radian = (0.5 * np.pi) / 2
        q      = np.array([np.cos(radian),n[0]*np.sin(radian),n[1]*np.sin(radian),n[2]*np.sin(radian)], dtype=np.float64)
        expected = np.array([0,0,1])
        actual   = quat.rotate(q, r)
        self.assertTrue(utils.array_equal(actual,expected))

    def test_rotate_8(self):
        r      = np.array([0,1,0])
        n      = np.array([0,1,0])
        radian = (0.5 * np.pi) / 2
        q      = np.array([np.cos(radian),n[0]*np.sin(radian),n[1]*np.sin(radian),n[2]*np.sin(radian)], dtype=np.float64)
        expected = np.array([0,1,0])
        actual   = quat.rotate(q, r)
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_rotate_9(self):
        r      = np.array([0,0,1])
        n      = np.array([1,0,0])
        radian = (0.5 * np.pi) / 2
        q      = np.array([np.cos(radian),n[0]*np.sin(radian),n[1]*np.sin(radian),n[2]*np.sin(radian)], dtype=np.float64)
        expected = np.array([0,-1,0])
        actual   = quat.rotate(q, r)
        self.assertTrue(utils.array_equal(actual,expected))

    def test_rotate_10(self):
        r      = np.array([0,0,1])
        n      = np.array([0,1,0])
        radian = (0.5 * np.pi) / 2
        q      = np.array([np.cos(radian),n[0]*np.sin(radian),n[1]*np.sin(radian),n[2]*np.sin(radian)], dtype=np.float64)
        expected = np.array([1,0,0])
        actual   = quat.rotate(q, r)
        self.assertTrue(utils.array_equal(actual,expected))

    def test_rotate_11(self):
        r      = np.array([0,0,1])
        n      = np.array([0,0,1])
        radian = (0.5 * np.pi) / 2
        q      = np.array([np.cos(radian),n[0]*np.sin(radian),n[1]*np.sin(radian),n[2]*np.sin(radian)], dtype=np.float64)
        expected = np.array([0,0,1])
        actual   = quat.rotate(q, r)
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_from_string_1(self):
        expected = np.array([1, 0, 0, 0], dtype=np.float64)
        actual   = quat.from_string("identity")
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_from_string_2(self):
        degree          = 90
        expected_radian = (0.5*np.pi) / 2 
        expected        = np.array([np.cos(expected_radian), np.sin(expected_radian), 0 , 0], dtype=np.float64)
        actual          = quat.from_string(f"rx: {degree}")
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_from_string_3(self):
        degree          = 90
        expected_radian = (0.5*np.pi) / 2 
        expected        = np.array([np.cos(expected_radian), np.sin(expected_radian), 0 , 0], dtype=np.float64)
        actual          = quat.from_string(f"Rx: {degree}")
        self.assertTrue(utils.array_equal(actual,expected))

    def test_from_string_3b(self):
        degree          = 90
        expected_radian = (0.5*np.pi) / 2 
        expected        = np.array([np.cos(expected_radian), np.sin(expected_radian), 0 , 0], dtype=np.float64)
        actual          = quat.from_string(f"RX: {degree}")
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_from_string_3c(self):
        degree          = 90
        expected_radian = (0.5*np.pi) / 2 
        expected        = np.array([np.cos(expected_radian), np.sin(expected_radian), 0 , 0], dtype=np.float64)
        actual          = quat.from_string(f"rX: {degree}")
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_from_string_4(self):
        degree          = 90
        expected_radian = (0.5*np.pi) / 2 
        expected        = np.array([np.cos(expected_radian),0, np.sin(expected_radian) , 0], dtype=np.float64)
        actual          = quat.from_string(f"ry: {degree}")
        self.assertTrue(utils.array_equal(actual,expected))

    def test_from_string_5(self):
        degree          = 90
        expected_radian = (0.5*np.pi) / 2 
        expected        = np.array([np.cos(expected_radian),0, 0, np.sin(expected_radian)], dtype=np.float64)
        actual          = quat.from_string(f"rz: {degree}")
        self.assertTrue(utils.array_equal(actual,expected))

    def test_from_string_6(self):
        degree          = 90
        expected_radian = (0.5*np.pi) / 2 
        axis            = "[1,0,0]"
        expected        = np.array([np.cos(expected_radian), np.sin(expected_radian), 0, 0,], dtype=np.float64)
        actual          = quat.from_string(f"ru: {degree}:{axis}")
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_from_string_7(self):
        degree          = 90
        expected_radian = (0.5*np.pi) / 2 
        axis            = "[0,1,0]"
        expected        = np.array([np.cos(expected_radian),0, np.sin(expected_radian), 0,], dtype=np.float64)
        actual          = quat.from_string(f"ru: {degree}:{axis}")
        self.assertTrue(utils.array_equal(actual,expected))

    def test_from_string_8(self):
        degree          = 90
        expected_radian = (0.5*np.pi) / 2 
        axis            = "[0,0,1]"
        expected        = np.array([np.cos(expected_radian),0, 0, np.sin(expected_radian)], dtype=np.float64)
        actual          = quat.from_string(f"ru: {degree}:{axis}")
        self.assertTrue(utils.array_equal(actual,expected))

    def test_from_string_9(self):
        expected        = np.array([1,2,3,4], dtype=np.float64)
        actual          = quat.from_string(f"[1,2,3,4,5,6,7]")
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_from_string_10(self):
        with self.assertRaises(ValueError):
            quat.from_string(f"[1,2,3]")

    def test_from_string_11(self):
        with self.assertRaises(ValueError):
            quat.from_string(f"rx: [1,2,3]")

    def test_from_vector3_1(self):
        expected = np.array([0, 1,2,3], dtype=np.float64)
        actual   = quat.from_vector3(np.array([1,2,3]))
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_from_vector3_2(self):
        expected = np.array([0, 1, 2, 3], dtype=np.float64)
        actual   = quat.from_vector3(np.array([1,2,3,4]))
        self.assertTrue(utils.array_equal(actual,expected))

    def test_from_vector3_3(self):
        with self.assertRaises(IndexError):
            quat.from_vector3(np.array([1,2]))

    def test_to_matrix_1(self):
        radian   = (0.5 * np.pi)/2
        expected = quat.make(np.cos(radian),np.sin(radian),0,0)
        actual   = quat.from_matrix(quat.to_matrix(expected))
        self.assertTrue(utils.array_equal(actual,expected))

    def test_to_matrix_2(self):
        radian    = (0.5 * np.pi)/2
        axis      = np.array([1,1,0])
        axis_unit = axis / np.linalg.norm(axis)
        expected  = quat.make(np.cos(radian),axis_unit[0] * np.sin(radian),axis_unit[1] * np.sin(radian),axis_unit[2] * np.sin(radian))
        actual    = quat.from_matrix(quat.to_matrix(expected))
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_to_matrix_3(self):
        '''
            p. 602, theorem 18.44
        '''
        radian    = (0.5 * np.pi)/2
        q         = np.array([np.cos(radian), np.sin(radian), 0, 0]) 
        expected  = np.array([
            [1,                               0,                                 0],
            [0,       (1-2*(np.sin(radian)**2)),  -2*np.cos(radian)*np.sin(radian)],
            [0, 2*np.cos(radian)*np.sin(radian), 1-2*(np.sin(radian)**2)          ]
        ])

        actual    = quat.to_matrix(q)
        self.assertTrue(utils.array_equal(actual,expected))

    def test_from_matrix_1(self):
        '''
         Use Theorem 18.44, however use reverted indexing to support
         numpy arrays
        '''
        radian    = (0.5*np.pi)/2
        M  = np.array([
            [1,                               0,                                 0],
            [0,       (1-2*(np.sin(radian)**2)),  -2*np.cos(radian)*np.sin(radian)],
            [0, 2*np.cos(radian)*np.sin(radian), 1-2*(np.sin(radian)**2)          ]
        ])
        s = 0.5 * np.sqrt(M[0,0] + M[1,1] + M[2,2] + 1)
        x = (M[2,1]-M[1,2])/(4*s)
        y = (M[2,0]-M[0,2])/(4*s)
        z = (M[0,1]-M[1,0])/(4*s)
        expected = np.array([s,x,y,z], dtype=np.float64)
        actual   = quat.from_matrix(M)
        self.assertTrue(utils.array_equal(actual,expected))

    def test_from_matrix_1(self):
        '''
         Use Theorem 18.44, however use reverted indexing to support
         numpy arrays
        '''
        radian    = (0.5*np.pi)/2
        M  = np.array([
            [1,                               0,                                 0],
            [0,       (1-2*(np.sin(radian)**2)),  -2*np.cos(radian)*np.sin(radian)],
            [0, 2*np.cos(radian)*np.sin(radian), 1-2*(np.sin(radian)**2)          ]
        ])
        s = 0.5 * np.sqrt(M[0,0] + M[1,1] + M[2,2] + 1)
        x = (M[2,1]-M[1,2])/(4*s)
        y = (M[2,0]-M[0,2])/(4*s)
        z = (M[0,1]-M[1,0])/(4*s)
        expected = np.array([s,x,y,z], dtype=np.float64)
        actual   = quat.from_matrix(M)
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_from_matrix_2(self):
        radian = np.pi / 2
        q      = np.array([np.cos(radian), np.sin(radian), 0,0], dtype=np.float64)
        M      = quat.to_matrix(q)
        x      = (np.sqrt(M[0,0] - M[1,1] - M[2,2] + 1.0))/(2)
        y      = (M[0,1]+M[1,0])/(4*x)
        z      = (M[2,0]+M[0,2])/(4*x)
        s      = (M[1,2]-M[2,1])/(4*x)
        expected = np.array([s,x,y,z], dtype=np.float64)
        actual   = quat.from_matrix(M)
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_from_matrix_3(self):
        radian = np.pi / 2
        axis   = [0,1,0]
        q      = np.array([np.cos(radian), axis[0]*np.sin(radian), axis[1]*np.sin(radian),axis[2]*np.sin(radian)], dtype=np.float64)
        M      = quat.to_matrix(q)
        expected = q
        actual   = quat.from_matrix(M)
        self.assertTrue(utils.array_equal(actual,expected))
    
    def test_from_matrix_4(self):
        radian = np.pi / 2
        axis   = [0,0,1]
        q      = np.array([np.cos(radian), axis[0]*np.sin(radian), axis[1]*np.sin(radian),axis[2]*np.sin(radian)], dtype=np.float64)
        M      = quat.to_matrix(q)
        expected = q
        actual   = quat.from_matrix(M)
        self.assertTrue(utils.array_equal(actual,expected))

    def test_rotate_array_1(self):
        radian    = (0.5 * np.pi)/2
        axis      = np.array([0,1,0])
        axis_unit = axis / np.linalg.norm(axis)
        q  = quat.make(np.cos(radian),axis_unit[0] * np.sin(radian),axis_unit[1] * np.sin(radian),axis_unit[2] * np.sin(radian))
        rs = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ], dtype=np.float64)
        expected = np.array([
            [0,0,-1],
            [0,1, 0],
            [1,0, 0]
        ], dtype=np.float64)
        actual = quat.rotate_array(q, rs)
        self.assertTrue(utils.array_equal(actual,expected))

    def test_prod_array_1(self):
        radian    = (0.5 * np.pi)/2
        axis      = np.array([0,1,0])
        axis_unit = axis / np.linalg.norm(axis)
        rs = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ], dtype=np.float64)
        
        with self.assertRaises(NotImplementedError):
            quat.prod_array(rs, rs)

    def test_prod_array_3(self):
        wrong_input = np.array([])
        with self.assertRaises(ValueError):
            quat.prod_array(wrong_input, wrong_input)

    def test_angle_axis_1(self):
        radian    = (0.5 * np.pi)/2
        axis      = np.array([0,1,0])
        axis_unit = axis / np.linalg.norm(axis)
        q  = quat.make(np.cos(radian),axis_unit[0] * np.sin(radian),axis_unit[1] * np.sin(radian),axis_unit[2] * np.sin(radian))
        expected_theta = radian  * 2
        expected_axis  = axis_unit 
        actual_theta, actual_axis   = quat.to_angle_axis(q)
        self.assertTrue(utils.array_equal(actual_theta,expected_theta))
        self.assertTrue(utils.array_equal(actual_axis,expected_axis))

    def test_angle_axis_2(self):
        radian    = (0.75 * np.pi)/2
        axis      = np.array([1,1,0])
        axis_unit = axis / np.linalg.norm(axis)
        q  = quat.make(np.cos(radian),axis_unit[0] * np.sin(radian),axis_unit[1] * np.sin(radian),axis_unit[2] * np.sin(radian))
        expected_theta = radian  * 2
        expected_axis  = axis_unit 
        actual_theta, actual_axis   = quat.to_angle_axis(q)
        self.assertTrue(utils.array_equal(actual_theta,expected_theta))
        self.assertTrue(utils.array_equal(actual_axis,expected_axis))

    def test_angle_axis_3(self):
        radian    = (0.00001 * np.pi)/2
        axis      = np.array([0.001,0,0])
        axis_unit = axis / np.linalg.norm(axis)
        q  = quat.make(np.cos(radian),axis_unit[0] * np.sin(radian),axis_unit[1] * np.sin(radian),axis_unit[2] * np.sin(radian))
        expected_theta = radian  * 2
        expected_axis  = axis_unit 
        actual_theta, actual_axis   = quat.to_angle_axis(q)
        self.assertTrue(utils.array_equal(actual_theta,expected_theta))
        self.assertTrue(utils.array_equal(actual_axis,expected_axis))

    def test_angle_axis_4(self):
        radian    = np.pi/2
        axis      = np.array([0,0,0])
        axis_unit = axis # axis / np.linalg.norm(axis)
        q  = quat.make(np.cos(radian),axis_unit[0] * np.sin(radian),axis_unit[1] * np.sin(radian),axis_unit[2] * np.sin(radian))
        expected_theta = [0.0,radian  * 4]
        expected_axis  = axis_unit 
        actual_theta, actual_axis   = quat.to_angle_axis(q)
        self.assertTrue(actual_theta in expected_theta)
        self.assertTrue(utils.array_equal(actual_axis,expected_axis))
    
    def test_to_angle_1(self):
        radian    = (0.5 * np.pi)/2
        axis      = np.array([0,1,0])
        axis_unit = axis / np.linalg.norm(axis)
        q  = quat.make(np.cos(radian),axis_unit[0] * np.sin(radian),axis_unit[1] * np.sin(radian),axis_unit[2] * np.sin(radian))
        expected = radian  * 2
        actual   = quat.to_angle(q, axis_unit)
        self.assertTrue(utils.array_equal(actual, expected))
    
    def test_to_angle_2(self):
        radian    = (1.5 * np.pi)/2
        axis      = np.array([0,1,0])
        axis_unit = axis / np.linalg.norm(axis)
        q  = quat.make(np.cos(radian),axis_unit[0] * np.sin(radian),axis_unit[1] * np.sin(radian),axis_unit[2] * np.sin(radian))
        expected = np.pi-(radian  * 2)
        actual   = quat.to_angle(q, axis_unit)
        self.assertTrue(utils.array_equal(actual, expected))

    def test_hat_1(self):
        radian    = (0.5 * np.pi)/2
        axis      = np.array([0,1,0])
        axis_unit = axis / np.linalg.norm(axis)
        q  = quat.make(np.cos(radian),axis_unit[0] * np.sin(radian),axis_unit[1] * np.sin(radian),axis_unit[2] * np.sin(radian))
        expected = 0.5*np.pi 
        actual , _   = quat.to_angle_axis(quat.hat(q)) 
        self.assertTrue(utils.array_equal(actual / 2, expected))

    def test_hat_2(self):
        radian    = (0.25 * np.pi)/2
        axis      = np.array([0,1,0])
        axis_unit = axis / np.linalg.norm(axis)
        q  = quat.make(np.cos(radian),axis_unit[0] * np.sin(radian),axis_unit[1] * np.sin(radian),axis_unit[2] * np.sin(radian))
        expected = 0.5*np.pi 
        actual , _   = quat.to_angle_axis(quat.hat(q))
        self.assertTrue(utils.array_equal(actual / 2, expected))


    def test_lerp_1(self):
        radian    = (0.5 * np.pi)/2
        axis_1      = np.array([0,1,0])
        axis_2      = np.array([0,1,0])
        axis_unit_1 = axis_1 / np.linalg.norm(axis_1)
        axis_unit_2 = axis_2 / np.linalg.norm(axis_2)
        q_1  = quat.make(np.cos(radian),axis_unit_1[0] * np.sin(radian),axis_unit_1[1] * np.sin(radian),axis_unit_1[2] * np.sin(radian))
        q_2  = quat.make(np.cos(radian),axis_unit_2[0] * np.sin(radian),axis_unit_2[1] * np.sin(radian),axis_unit_2[2] * np.sin(radian))
        h    = 0.5
        expected = q_1*(1-h)+q_2*h
        actual   = quat.lerp(q_1,q_2,h)
        self.assertTrue(utils.array_equal(actual, expected))
    
    def test_lerp_2(self):
        radian    = (0.5 * np.pi)/2
        axis_1      = np.array([0,1,0])
        axis_2      = np.array([1,0,0])
        axis_unit_1 = axis_1 / np.linalg.norm(axis_1)
        axis_unit_2 = axis_2 / np.linalg.norm(axis_2)
        q_1  = quat.make(np.cos(radian),axis_unit_1[0] * np.sin(radian),axis_unit_1[1] * np.sin(radian),axis_unit_1[2] * np.sin(radian))
        q_2  = quat.make(np.cos(radian),axis_unit_2[0] * np.sin(radian),axis_unit_2[1] * np.sin(radian),axis_unit_2[2] * np.sin(radian))
        h    = 0.7
        expected = q_1*(1-h)+q_2*h
        actual   = quat.lerp(q_1,q_2,h)
        self.assertTrue(utils.array_equal(actual, expected))
    
    def test_slerp_1(self):
        radian    = (0.5 * np.pi)/2
        axis_1      = np.array([0,1,0])
        axis_2      = np.array([1,0,0])
        axis_unit_1 = axis_1 / np.linalg.norm(axis_1)
        axis_unit_2 = axis_2 / np.linalg.norm(axis_2)
        q_1  = quat.make(np.cos(radian),axis_unit_1[0] * np.sin(radian),axis_unit_1[1] * np.sin(radian),axis_unit_1[2] * np.sin(radian))
        q_2  = quat.make(np.cos(radian),axis_unit_2[0] * np.sin(radian),axis_unit_2[1] * np.sin(radian),axis_unit_2[2] * np.sin(radian))
        h    = 0.0
        expected = q_1
        actual   = quat.slerp(q_1,q_2,h)
        self.assertTrue(utils.array_equal(actual, expected))

    def test_slerp_2(self):
        radian    = (0.5 * np.pi)/2
        axis_1      = np.array([0,1,0])
        axis_2      = np.array([1,0,0])
        axis_unit_1 = axis_1 / np.linalg.norm(axis_1)
        axis_unit_2 = axis_2 / np.linalg.norm(axis_2)
        q_1  = quat.make(np.cos(radian),axis_unit_1[0] * np.sin(radian),axis_unit_1[1] * np.sin(radian),axis_unit_1[2] * np.sin(radian))
        q_2  = quat.make(np.cos(radian),axis_unit_2[0] * np.sin(radian),axis_unit_2[1] * np.sin(radian),axis_unit_2[2] * np.sin(radian))
        h    = 1.0
        expected = q_2
        actual   = quat.slerp(q_1,q_2,h)
        self.assertTrue(utils.array_equal(actual, expected))

    def test_slerp_3(self):
        radian    = (0.5 * np.pi)/2
        axis_1      = np.array([0,1,0])
        axis_2      = np.array([0,1,0])
        axis_unit_1 = axis_1 / np.linalg.norm(axis_1)
        axis_unit_2 = axis_2 / np.linalg.norm(axis_2)
        q_1  = quat.make(np.cos(radian),axis_unit_1[0] * np.sin(radian),-axis_unit_1[1] * np.sin(radian),axis_unit_1[2] * np.sin(radian))
        q_2  = quat.make(-np.cos(radian),axis_unit_2[0] * np.sin(radian),axis_unit_2[1] * np.sin(radian),axis_unit_2[2] * np.sin(radian))
        h    = 1.0
        expected = q_2
        actual   = quat.slerp(q_1,q_2,h)
        self.assertTrue(utils.array_equal(actual, expected))

    def test_diag_from_array_1(self):
        v = np.array([1,2,3])
        expected = np.diag(v)
        actual   = mat3.diag_from_array(v)
        self.assertTrue(utils.array_equal(actual, expected))
    
    def test_diag_1(self):
        a, b, c = 1,2,3
        expected = np.diag(np.array([a,b,c]))
        actual   = mat3.diag(a,b,c)
        self.assertTrue(utils.array_equal(actual, expected))

    def test_diag_2(self):
        a, b, c = 1,2,"three"
        with self.assertRaises(ValueError):
            mat3.diag(a,b,c)
    
    def test_matrix3_make_1(self):
        a,b,c,d,e,f,h,i,j = 1,2,3,4,5,6,7,8,9
        expected = np.array(
            [
                [1,2,3],
                [4,5,6],
                [7,8,9]
            ]
        )
        actual   = mat3.make(a,b,c,d,e,f,h,i,j)
        self.assertTrue(utils.array_equal(actual, expected))

    def test_make_from_row_1(self):
        a,d,h = (np.array([1,2,3]),
                 np.array([4,5,6]),
                 np.array([7,8,9])
                 )
        expected = np.array(
            [
                [1,2,3],
                [4,5,6],
                [7,8,9]
            ]
        )
        actual = mat3.make_from_rows(a,d,h)
        self.assertTrue(utils.array_equal(actual, expected))
    
    def test_make_from_col_1(self):
        a,d,h = (np.array([1,4,7]),
                 np.array([2,5,8]),
                 np.array([3,6,9])
                 )
        expected = np.array(
            [
                [1,2,3],
                [4,5,6],
                [7,8,9]
            ]
        )
        actual = mat3.make_from_cols(a,d,h)
        self.assertTrue(utils.array_equal(actual, expected))
    
    def test_star_1(self):
        rand_array = utils.psu_rand_array_gen((2,3))
        v1, v2 = rand_array[0], rand_array[1]
        expected = np.cross(v1,v2)
        actual   = np.dot(mat3.star(v1), v2)
        self.assertTrue(utils.array_equal(actual, expected))
    
    def test_orthonomalize_1(self):
        M    = np.random.rand(3,3)
        M_orthonomalized = mat3.ortonormalize(M)
        expected = mat3.identity()
        actual   = np.dot(np.transpose(M_orthonomalized), M_orthonomalized)
        expected_false_actual = np.dot(np.transpose(M), M)
        self.assertTrue(utils.array_equal(actual, expected))
        self.assertTrue(utils.array_not_equal(expected_false_actual, expected))

    def test_matrix3_rotation_1(self):
       v = np.array([1,0,0])
       radian_x = 0.5 * np.pi
       radian_y = 0.0
       radian_z = 0.0
       rotation_matrix = np.dot(mat3.Rx(radian_x), np.dot(mat3.Ry(radian_y), mat3.Rz(radian_z)))
       expected        = v
       actual          = np.dot(rotation_matrix, v)
       self.assertTrue(utils.array_equal(actual, expected))
    
    def test_matrix3_rotation_2(self):
       v = np.array([1,0,0])
       radian_x = 0.0 
       radian_y = 0.0
       radian_z = 0.5 * np.pi
       rotation_matrix = np.dot(mat3.Rx(radian_x), np.dot(mat3.Ry(radian_y), mat3.Rz(radian_z)))
       expected        = np.array([0,1,0])
       actual          = np.dot(rotation_matrix, v)
       self.assertTrue(utils.array_equal(actual, expected))
    
    def test_matrix3_rotation_3(self):
       v = np.array([1,0,0])
       radian_x = 0.0 
       radian_y = 0.5 * np.pi
       radian_z = 0.0 
       rotation_matrix = np.dot(mat3.Rx(radian_x), np.dot(mat3.Ry(radian_y), mat3.Rz(radian_z)))
       expected        = np.array([0,0,-1])
       actual          = np.dot(rotation_matrix, v)
       self.assertTrue(utils.array_equal(actual, expected))
    
    def test_matrix3_rotation_4(self):
       v = np.array([0,1,0])
       radian_x = 0.0 
       radian_y = 0.5 * np.pi
       radian_z = 0.0 
       rotation_matrix = np.dot(mat3.Rx(radian_x), np.dot(mat3.Ry(radian_y), mat3.Rz(radian_z)))
       expected        = v 
       actual          = np.dot(rotation_matrix, v)
       self.assertTrue(utils.array_equal(actual, expected))
    
    def test_matrix3_rotation_5(self):
       v = np.array([0,1,0])
       radian_x = 0.5 * np.pi 
       radian_y = 0.0 
       radian_z = 0.0 
       rotation_matrix = np.dot(mat3.Rx(radian_x), np.dot(mat3.Ry(radian_y), mat3.Rz(radian_z)))
       expected        = np.array([0,0,1]) 
       actual          = np.dot(rotation_matrix, v)
       self.assertTrue(utils.array_equal(actual, expected))
    
    def test_matrix3_rotation_6(self):
       v = np.array([0,1,0])
       radian_x = 0.0 
       radian_y = 0.0 
       radian_z = 0.5 * np.pi 
       rotation_matrix = np.dot(mat3.Rx(radian_x), np.dot(mat3.Ry(radian_y), mat3.Rz(radian_z)))
       expected        = np.array([-1,0,0]) 
       actual          = np.dot(rotation_matrix, v)
       self.assertTrue(utils.array_equal(actual, expected))
    
    def test_matrix3_rotation_7(self):
       v = np.array([0,0,1])
       radian_x = 0.0 
       radian_y = 0.0 
       radian_z = 0.5 * np.pi 
       rotation_matrix = np.dot(mat3.Rx(radian_x), np.dot(mat3.Ry(radian_y), mat3.Rz(radian_z)))
       expected        = v 
       actual          = np.dot(rotation_matrix, v)
       self.assertTrue(utils.array_equal(actual, expected))
    
    def test_matrix3_rotation_8(self):
       v = np.array([0,0,1])
       radian_x = 0.0 
       radian_y = 0.5 * np.pi 
       radian_z = 0.0 
       rotation_matrix = np.dot(mat3.Rx(radian_x), np.dot(mat3.Ry(radian_y), mat3.Rz(radian_z)))
       expected        = np.array([1,0,0]) 
       actual          = np.dot(rotation_matrix, v)
       self.assertTrue(utils.array_equal(actual, expected))   
    
    def test_matrix3_rotation_9(self):
       v = np.array([0,0,1])
       radian_x = 0.5 * np.pi 
       radian_y = 0.0 
       radian_z = 0.0 
       rotation_matrix = np.dot(mat3.Rx(radian_x), np.dot(mat3.Ry(radian_y), mat3.Rz(radian_z)))
       expected        = np.array([0,-1,0]) 
       actual          = np.dot(rotation_matrix, v)
       self.assertTrue(utils.array_equal(actual, expected))
    
    def test_polar_decompostion_1(self):
        M = np.random.rand(3,3)
        R, S = mat3.polar_decomposition(M)

        actual_1 = S
        actual_2 = np.dot(R, np.transpose(R)) 
        actual_3 = M

        expected_1 = np.transpose(S)
        expected_2 = mat3.identity()
        expected_3 = np.dot(R,S)
        
        self.assertTrue(utils.array_equal(actual_1, expected_1))
        self.assertTrue(utils.array_equal(actual_2, expected_2))
        self.assertTrue(utils.array_equal(actual_3, expected_3))

    def test_polar_decompostion_2(self):
        M = mat3.identity()
        M[0,0] = 0
        M[1,1] = -2
        M[2,2] = -1

        with self.assertRaises(ValueError):
            mat3.polar_decomposition(M)
    
    def test_ru_1(self):
        radians = 0.5 * np.pi
        axis     = np.array([1,0,0]) 
        radian_x = 0.5 * np.pi 
        radian_y = 0.0 
        radian_z = 0.0 
        rotation_matrix = np.dot(mat3.Rx(radian_x), np.dot(mat3.Ry(radian_y), mat3.Rz(radian_z)))
        expected        = rotation_matrix 
        actual          = mat3.Ru(radians, axis)
        self.assertTrue(utils.array_equal(actual, expected))

    def test_ru_2(self):
        radians = 0.5 * np.pi
        axis     = np.array([0,1,0]) 
        radian_x = 0.0 
        radian_y = 0.5 * np.pi 
        radian_z = 0.0 
        rotation_matrix = np.dot(mat3.Rx(radian_x), np.dot(mat3.Ry(radian_y), mat3.Rz(radian_z)))
        expected        = rotation_matrix 
        actual          = mat3.Ru(radians, axis)
        self.assertTrue(utils.array_equal(actual, expected))
    
    def test_ru_3(self):
        radians = 0.5 * np.pi
        axis     = np.array([0,0,1]) 
        radian_x = 0.0 
        radian_y = 0.0 
        radian_z = 0.5 * np.pi 
        rotation_matrix = np.dot(mat3.Rx(radian_x), np.dot(mat3.Ry(radian_y), mat3.Rz(radian_z)))
        expected        = rotation_matrix 
        actual          = mat3.Ru(radians, axis)
        self.assertTrue(utils.array_equal(actual, expected))

    def test_ru_3(self):
        n      = np.array([1,1,0])
        axis      = n / np.linalg.norm(n)
        radians = (0.5 * np.pi) / 2
        q      = np.array([np.cos(radians),axis[0]*np.sin(radians),axis[1]*np.sin(radians),axis[2]*np.sin(radians)], dtype=np.float64)
        expected        = quat.to_matrix(q)
        actual          = mat3.Ru(radians*2, axis)
        self.assertTrue(utils.array_equal(actual, expected))

    def test_ru_4(self):
        n      = np.array([1,1,1])
        axis      = n / np.linalg.norm(n)
        radians = (0.75 * np.pi) / 2
        q      = np.array([np.cos(radians),axis[0]*np.sin(radians),axis[1]*np.sin(radians),axis[2]*np.sin(radians)], dtype=np.float64)
        expected        = quat.to_matrix(q)
        actual          = mat3.Ru(radians*2, axis)
        self.assertTrue(utils.array_equal(actual, expected))

    def test_ru_5(self):
        n      = np.array([1,0.56,1])
        axis      = n / np.linalg.norm(n)
        radians = (1.75 * np.pi) / 2
        q      = np.array([np.cos(radians),axis[0]*np.sin(radians),axis[1]*np.sin(radians),axis[2]*np.sin(radians)], dtype=np.float64)
        expected        = quat.to_matrix(q)
        actual          = mat3.Ru(radians*2, axis)
        self.assertTrue(utils.array_equal(actual, expected))

    def test_polar_decompostion_array_1(self):
        number_of_M = 4
        shape = (number_of_M,3,3)
        Ms = np.random.rand(shape[0], shape[1], shape[2])
        Rs, Ss = mat3.polar_decomposition_array(Ms)

        actual_1 = Ss
        actual_2 = np.array([np.dot(r, np.transpose(r)) for r in Rs])
        actual_3 = Ms
        
        expected_1 = np.array([np.transpose(s) for s in Ss])
        expected_2 = np.array([mat3.identity() for _ in range(number_of_M)])
        expected_3 = np.array([np.dot(r,s) for r,s in zip(Rs,Ss)])

        self.assertTrue(utils.array_equal(actual_1, expected_1))
        self.assertTrue(utils.array_equal(actual_2, expected_2))
        self.assertTrue(utils.array_equal(actual_3, expected_3))

    def test_polar_decompostion_array_2(self):
        number_of_M = 4
        shape = (number_of_M,3,3)
        Ms = np.zeros(shape)
        Ms[:,0,0] = 1

        with self.assertRaises(ValueError):
            mat3.polar_decomposition_array(Ms)
