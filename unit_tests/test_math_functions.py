import unittest
import os
import sys
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.math.functions as FUNC
import rainbow.util.test_tools as TEST


def dxxf(x):
    return -(((x ** 2 - 2) * np.sin(x) + 2 * x * np.cos(x)) / (x ** 3))


def dxf(x):
    return (x * np.cos(x) - np.sin(x)) / (x ** 2)


def f(x):
    return np.sin(x) / x


def taylor_expansion(x):
    a = 0.166666666666666666667
    term1 = f(a)
    term2 = dxf(a) * (x - a)
    term3 = (dxxf(a) * (x - a) ** 2) / math.factorial(2)
    return term1 + term2 + term3


class TestFunctionAPI(unittest.TestCase):
    def test_func1(self):
        x = 1.5e-5
        expected = taylor_expansion(x)
        actual = FUNC.sinc(x)
        self.assertAlmostEqual(actual, expected, 4)

    def test_pca_1(self):
        amount = 12
        length = 2 * np.pi
        step_size = length / amount
        input = np.arange(0, length, step_size)
        x = np.cos(input)
        y = np.sin(input)
        z = np.zeros(len(x), dtype=np.float64)

        points = np.moveaxis(np.array([x, y, z]), 0, 1)

        _, values, _ = FUNC.PCA(points)

        self.assertTrue(values[0] > 0)
        self.assertTrue(values[1] > 0)
        self.assertFalse(values[2] > 0)

    def test_pca_2(self):
        amount = 12
        x = np.zeros(amount, dtype=np.float64)
        y = np.zeros(amount, dtype=np.float64)
        z = np.arange(amount, dtype=np.float64)

        points = np.moveaxis(np.array([x, y, z]), 0, 1)

        _, values, _ = FUNC.PCA(points)

        self.assertFalse(values[0] > 0)
        self.assertFalse(values[1] > 0)
        self.assertTrue(values[2] > 0)

    def test_pca_3(self):
        amount = 12
        length = 2 * np.pi
        step_size = length / amount
        input = np.arange(0, length, step_size)
        x = np.cos(input)
        y = np.sin(input)
        z = np.zeros(len(x), dtype=np.float64)

        points = np.moveaxis(np.array([x, y, z]), 0, 1)

        mean, _, _ = FUNC.PCA(points)

        self.assertAlmostEqual(mean[0], 0)
        self.assertAlmostEqual(mean[1], 0)
        self.assertAlmostEqual(mean[2], 0)

    def test_pca_4(self):
        amount = 12
        x = np.zeros(amount, dtype=np.float64)
        y = np.zeros(amount, dtype=np.float64)
        z = np.arange(amount, dtype=np.float64)

        points = np.moveaxis(np.array([x, y, z]), 0, 1)

        _, values, vector = FUNC.PCA(points)

        pc1 = vector[np.argmax(values)]

        self.assertTrue(pc1[2] > 0)

    def test_direction_of_most_variance(self):
        amount = 12
        x = np.zeros(amount, dtype=np.float64)
        y = np.zeros(amount, dtype=np.float64)
        z = np.arange(amount, dtype=np.float64)

        points = np.moveaxis(np.array([x, y, z]), 0, 1)

        _, values, vector = FUNC.PCA(points)

        expected = vector[np.argmax(values)]
        actual = FUNC.direction_of_most_variance(points)

        self.assertTrue(TEST.is_array_equal(actual, expected))
