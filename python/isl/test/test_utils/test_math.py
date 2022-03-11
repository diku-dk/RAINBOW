from cmath import sin, tan
from ctypes import util
import unittest
import os
import sys
import numpy as np
import math as m
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../../")
import isl.math.vector3 as vec3
import isl.math.angle as angle
import isl.test.test_common as utils
import isl.math.quaternion as quat

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