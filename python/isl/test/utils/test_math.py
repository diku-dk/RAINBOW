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
    '''
        Just a copy
    '''
    def test_less_5(self):
        test_vec_1 = vec3.make(1,2,3)
        test_vec_2 = vec3.make(0,2,3)
        self.assertTrue(vec3.less(test_vec_2, test_vec_1))
        self.assertFalse(vec3.less(test_vec_1, test_vec_2))
