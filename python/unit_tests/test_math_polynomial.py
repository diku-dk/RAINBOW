import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.math.polynomial as POLY


class TestPolynomial(unittest.TestCase):

    def test_to_str(self):
        # Test if negative order results in exception
        self.assertRaises(ValueError, POLY.Poly, 2, -1)

        P0 = POLY.Poly(2, 0)
        expected = "1"
        self.assertEqual(expected, POLY.to_str(P0), 'Zeroth order polynomial test')
        self.assertEqual(expected, str(P0), 'Zeroth order polynomial test')

        P1 = POLY.Poly(2, 1)
        expected = "1 + x + y"
        self.assertEqual(expected, POLY.to_str(P1), 'First order polynomial test')
        self.assertEqual(expected, str(P1), 'First order polynomial test')

        P2 = POLY.Poly(2, 2)
        expected = "1 + x + y + x^2 + x y + y^2"
        self.assertEqual(expected, POLY.to_str(P2), 'Second order polynomial test')
        self.assertEqual(expected, str(P2), 'Second order polynomial test')

        P3 = POLY.Poly(2, 3)
        expected = "1 + x + y + x^2 + x y + y^2 + x^3 + x^2 y + x y^2 + y^3"
        self.assertEqual(expected, POLY.to_str(P3), 'Third order polynomial test')
        self.assertEqual(expected, str(P3), 'Third order polynomial test')

    def test_evaluate(self):
        x = np.array([1, 1])
        P1 = POLY.Poly(2, 1)
        self.assertEqual(P1(x), 3)
        self.assertEqual(POLY.evaluate(P1, x), 3)
        P2 = POLY.Poly(2, 2)
        self.assertEqual(P2(x), 6)
        self.assertEqual(POLY.evaluate(P2, x), 6)
        P3 = POLY.Poly(2, 3)
        self.assertEqual(P3(x), 10)
        self.assertEqual(POLY.evaluate(P3, x), 10)
        P4 = POLY.Poly(2, 4)
        self.assertEqual(P4(x), 15)
        self.assertEqual(POLY.evaluate(P4, x), 15)

    def test_derivative(self):
        # 2022-03-14 Kenny TODO: The current test does not test for correct order of derivative polynomial
        #                   nor for correct power-pattern.
        P = POLY.Poly(2, 4)
        dPdx, dPdy = POLY.derivative(P)
        dx_expected = "1 + 2 x + y + 3 x^2 + 2 x y + y^2 + 4 x^3 + 3 x^2 y + 2 x y^2 + y^3"
        self.assertEqual(dx_expected, POLY.to_str(dPdx), 'x-derivative was wrong')
        expected = "1 + x + 2 y + x^2 + 2 x y + 3 y^2 + x^3 + 2 x^2 y + 3 x y^2 + 4 y^3"
        self.assertEqual(expected, POLY.to_str(dPdy), 'y-derivative was wrong')

    def test_vandermonde_matrix(self):
        # 2022-03-14 Kenny TODO: Current tests do not test for wrong input parameters.
        x = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
        order = 1
        V1_expected = np.array(
            [[1., 1., 1.],
             [1., 2., 2.],
             [1., 3., 3.]]
        )
        V1 = POLY.Helpers.make_vandermonde_matrix(order, x)
        np.testing.assert_allclose(V1, V1_expected)

        order = 2
        V2_expected = np.array(
            [[1.,  1.,  1.,  1.,  1.,  1.],
             [1.,  2.,  2.,  4.,  4.,  4.],
             [1.,  3.,  3.,  9.,  9.,  9.],
             [1.,  4.,  4., 16., 16., 16.],
             [1.,  5.,  5., 25., 25., 25.],
             [1.,  6.,  6., 36., 36., 36.]]
        )
        V2 = POLY.Helpers.make_vandermonde_matrix(order, x)
        np.testing.assert_allclose(V2, V2_expected)

        x = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
        order = 2
        W2 = POLY.Helpers.make_vandermonde_matrix(order, x)
        W2_expected = np.array(
            [[1.,  1.,  1.,  1.,  1.,  1.],
             [1.,  1.,  2.,  1.,  2.,  4.],
             [1.,  1.,  3.,  1.,  3.,  9.],
             [1.,  1.,  4.,  1.,  4., 16.],
             [1.,  1.,  5.,  1.,  5., 25.],
             [1.,  1.,  6.,  1.,  6., 36.]]
        )
        np.testing.assert_allclose(W2, W2_expected)

    def test_interpolate(self):
        # 2022-03-14 Kenny TODO: Current tests do not test for wrong input parameters.
        x = np.array([[1., 0.], [0., 1.], [0., 0.]])
        v = np.array(
            [[1., 1.],
             [1., 2.],
             [1., 3.],
             ])
        order = 1
        P0, P1 = POLY.interpolate(order, x, v)
        self.assertAlmostEqual(P0(x[0, :]), v[0, 0])
        self.assertAlmostEqual(P0(x[1, :]), v[1, 0])
        self.assertAlmostEqual(P0(x[2, :]), v[2, 0])
        self.assertAlmostEqual(P1(x[0, :]), v[0, 1])
        self.assertAlmostEqual(P1(x[1, :]), v[1, 1])
        self.assertAlmostEqual(P1(x[2, :]), v[2, 1])

        x = np.array([[2., 0.], [1., 0.],  [0., 2.], [0., 1.], [1., 1.], [0., 0.]])
        v = np.array(
            [[1., 1.],
             [1., 2.],
             [1., 3.],
             [1., 4.],
             [1., 5.],
             [1., 6.],
             ])
        order = 2
        P2, P3 = POLY.interpolate(order, x, v)
        self.assertAlmostEqual(P2(x[0, :]), v[0, 0])
        self.assertAlmostEqual(P2(x[1, :]), v[1, 0])
        self.assertAlmostEqual(P2(x[2, :]), v[2, 0])
        self.assertAlmostEqual(P2(x[3, :]), v[3, 0])
        self.assertAlmostEqual(P2(x[4, :]), v[4, 0])
        self.assertAlmostEqual(P2(x[5, :]), v[5, 0])
        self.assertAlmostEqual(P3(x[0, :]), v[0, 1])
        self.assertAlmostEqual(P3(x[1, :]), v[1, 1])
        self.assertAlmostEqual(P3(x[2, :]), v[2, 1])
        self.assertAlmostEqual(P3(x[3, :]), v[3, 1])
        self.assertAlmostEqual(P3(x[4, :]), v[4, 1])
        self.assertAlmostEqual(P3(x[5, :]), v[5, 1])

        x = np.random.rand(10, 2)*5
        v = np.random.rand(10, 1).ravel()
        order = 3
        [P4] = POLY.interpolate(order, x, v)
        self.assertAlmostEqual(P4(x[0, :]), v[0])
        self.assertAlmostEqual(P4(x[1, :]), v[1])
        self.assertAlmostEqual(P4(x[2, :]), v[2])
        self.assertAlmostEqual(P4(x[3, :]), v[3])
        self.assertAlmostEqual(P4(x[4, :]), v[4])
        self.assertAlmostEqual(P4(x[5, :]), v[5])
        self.assertAlmostEqual(P4(x[6, :]), v[6])
        self.assertAlmostEqual(P4(x[7, :]), v[7])
        self.assertAlmostEqual(P4(x[8, :]), v[8])
        self.assertAlmostEqual(P4(x[9, :]), v[9])

        x = np.random.rand(15, 2)*5
        v = np.random.rand(15, 1).ravel()
        order = 4
        [P5] = POLY.interpolate(order, x, v)
        self.assertAlmostEqual(P5(x[0, :]), v[0])
        self.assertAlmostEqual(P5(x[1, :]), v[1])
        self.assertAlmostEqual(P5(x[2, :]), v[2])
        self.assertAlmostEqual(P5(x[3, :]), v[3])
        self.assertAlmostEqual(P5(x[4, :]), v[4])
        self.assertAlmostEqual(P5(x[5, :]), v[5])
        self.assertAlmostEqual(P5(x[6, :]), v[6])
        self.assertAlmostEqual(P5(x[7, :]), v[7])
        self.assertAlmostEqual(P5(x[8, :]), v[8])
        self.assertAlmostEqual(P5(x[9, :]), v[9])
        self.assertAlmostEqual(P5(x[10, :]), v[10])
        self.assertAlmostEqual(P5(x[11, :]), v[11])
        self.assertAlmostEqual(P5(x[12, :]), v[12])
        self.assertAlmostEqual(P5(x[13, :]), v[13])
        self.assertAlmostEqual(P5(x[14, :]), v[14])
