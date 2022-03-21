import re
import unittest
import os
import sys
import numpy as np
import math as m

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../../")
import isl.math.functions as func
import isl.test.test_common as utils


def dxxf(x):
    return -(((x**2-2)*np.sin(x)+2*x*np.cos(x))/(x**3))

def dxf(x):
    return (x*np.cos(x) - np.sin(x)) / (x**2)

def f(x):
    return np.sin(x) / x

def taylor_expansion(x):
    a     = 0.166666666666666666667
    term1 = f(a)
    term2 = dxf(a)*(x-a)
    term3 = (dxxf(a) * (x-a)**2)/m.factorial(2)
    return term1+term2+term3
    


class TestFunctionAPI(unittest.TestCase):
    def test_func1(self):
        x = 1.5e-5
        expected = taylor_expansion(x)
        actual   = func.sinc(x)
        self.assertAlmostEqual(actual, expected, 4)

    # chech issue 53
    # https://github.com/diku-dk/libRAINBOW/issues/53
    # def test_func2(self):
    #     x = np.array([x for x in range(10)])
    #     x[x % 2 == 0] = -1*x[x % 2 == 0]
    #     x = x * 1e-10
    #     expected = taylor_expansion(x)
    #     actual   = func.sinc(x)
    #     self.assertTrue(utils.array_equal(expected, expected))

    # Why not use np.clip instead
    def test_clamp_1(self):
        value = np.random.randint(0,10)
        lower = np.random.randint(0,5)
        upper = np.random.randint(lower+1, 10)
        expected = np.clip(value, lower, upper)
        actual   = func.clamp(value, lower, upper)
        self.assertEqual(actual, expected) 
    
    def test_pca_1(self):
        mu, sigma = 1, 0.1 # mean and standard deviation
        M = 200
        points_3d = np.random.normal(mu, sigma, size=(M, 3))
        means = np.mean(points_3d, axis=0)
        expected = means[0]
        actual_mean, _ ,_ = func.PCA(points_3d)

        self.assertAlmostEqual(actual_mean[0], expected, 3)
        
    def test_pca_2(self):
        mu = 0
        M = 10
        points_1 = np.random.normal(mu, 0, size=(M,3))
        points_2 = np.random.normal(mu, 0, size=(M,3))
        points_3 = np.random.normal(mu, 0, size=(M,3))
        
        points   = np.vstack([points_1,
                              points_2,
                              points_3])

        mean, values, vectors = func.PCA(points)

        self.assertTrue(utils.array_equal(mean,np.array([mu,mu,mu])))
        self.assertTrue(utils.array_equal(values,np.array([0,0,0])))
        self.assertTrue(utils.array_equal(vectors,np.array([[1,0,0], [0,1,0],[0,0,1]])))
