import unittest
import os
import sys
import numpy as np
from numba import cuda

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.util.test_tools as TEST
import rainbow.cuda.unit_tests.test_math_matrix_kernel as TEST_MK


@unittest.skipIf(not cuda.is_available(), "CUDA not available")
class TestMatrixCuda(unittest.TestCase):
    
    def test_mat33_T(self):
        m_host = np.array([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]]], dtype=np.float64)
        expected = np.array([[[1.0, 4.0, 7.0],
                             [2.0, 5.0, 8.0],
                             [3.0, 6.0, 9.0]]], dtype=np.float64)
        result_host = np.zeros((1, 3, 3), dtype=np.float64)
        m_device = cuda.to_device(m_host)
        result_device = cuda.device_array_like(result_host)
        TEST_MK.mat33_T_kernel[1, 1](m_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))

    def test_mat33_zero(self):
        expected = np.array([[[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]]], dtype=np.float64)
        result_host = np.zeros((1, 3, 3), dtype=np.float64)
        result_device = cuda.device_array_like(result_host)
        TEST_MK.mat33_zero_kernel[1, 1](result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))

    def test_mat33_determinant(self):
        m_host = np.array([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]]], dtype=np.float64)
        expected = np.array([-0.0], dtype=np.float64)
        result_host = np.zeros((1), dtype=np.float64)
        m_device = cuda.to_device(m_host)
        result_device = cuda.device_array_like(result_host)
        TEST_MK.mat33_determinant_kernel[1, 1](m_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))

    def test_mat33_make(self):
        expected = np.array([[[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]]], dtype=np.float64)
        result_host = np.zeros((1, 3, 3), dtype=np.float64)
        result_device = cuda.device_array_like(result_host)
        TEST_MK.mat33_make_kernel[1, 1](np.array([1.0]), np.array([2.0]), np.array([3.0]),
                                        np.array([4.0]), np.array(
                                            [5.0]), np.array([6.0]),
                                        np.array([7.0]), np.array(
                                            [8.0]), np.array([9.0]),
                                        result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))

    def test_mat33_make_from_cols(self):
        col0_host = np.array([[1.0, 4.0, 7.0]], dtype=np.float64)
        col1_host = np.array([[2.0, 5.0, 8.0]], dtype=np.float64)
        col2_host = np.array([[3.0, 6.0, 9.0]], dtype=np.float64)
        expected = np.array([[[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]]], dtype=np.float64)
        result_host = np.zeros((1, 3, 3), dtype=np.float64)
        col0_device = cuda.to_device(col0_host)
        col1_device = cuda.to_device(col1_host)
        col2_device = cuda.to_device(col2_host)
        result_device = cuda.device_array_like(result_host)
        TEST_MK.mat33_make_from_cols_kernel[1, 1](
            col0_device, col1_device, col2_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))

    def test_mat33_dot_vec3(self):
        m_host = np.array([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]]], dtype=np.float64)
        vec3_host = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        expected = np.array([[14.0, 32.0, 50.0]], dtype=np.float64)
        result_host = np.zeros((1, 3), dtype=np.float64)
        m_device = cuda.to_device(m_host)
        vec3_device = cuda.to_device(vec3_host)
        result_device = cuda.device_array_like(result_host)
        TEST_MK.mat33_dot_vec3_kernel[1, 1](
            m_device, vec3_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))

    def test_mat43_T(self):
        m_host = np.array([[[1.0, 5.0, 9.0],
                            [2.0, 6.0, 10.0],
                            [3.0, 7.0, 11.0],
                            [4.0, 8.0, 12.0]]], dtype=np.float64)
        expected = np.array([[[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0],
                              [9.0, 10.0, 11.0, 12.0]]], dtype=np.float64)
        result_host = np.zeros((1, 3, 4), dtype=np.float64)
        m_device = cuda.to_device(m_host)
        result_device = cuda.device_array_like(result_host)
        TEST_MK.mat43_T_kernel[1, 1](m_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))

    def test_mat34_dot_vec4(self):
        m_host = np.array([[[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 10.0, 11.0, 12.0]]], dtype=np.float64)
        vec4_host = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
        expected = np.array([[30.0, 70.0, 110.0]], dtype=np.float64)
        result_host = np.zeros((1, 3), dtype=np.float64)
        m_device = cuda.to_device(m_host)
        vec4_device = cuda.to_device(vec4_host)
        result_device = cuda.device_array_like(result_host)
        TEST_MK.mat34_dot_vec4_kernel[1, 1](
            m_device, vec4_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))


if __name__ == '__main__':
    unittest.main()
