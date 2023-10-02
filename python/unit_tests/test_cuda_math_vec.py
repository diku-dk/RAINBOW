import unittest
import os
import sys
import numpy as np
from numba import cuda

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.cuda.unit_tests.test_math_vec_kernel as TEST_VK
import rainbow.util.test_tools as TEST


@unittest.skipIf(not cuda.is_available(), "CUDA not available")
class TestVec3Cuda(unittest.TestCase):
    
    def test_vec3_add(self):
        v1_host = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        v2_host = np.array([[4.0, 5.0, 6.0]], dtype=np.float64)
        expected = np.array([[5.0, 7.0, 9.0]], dtype=np.float64)
        result_host = np.zeros((1, 3), dtype=np.float64)
        v1_device = cuda.to_device(v1_host)
        v2_device = cuda.to_device(v2_host)
        result_device = cuda.device_array_like(result_host)
        TEST_VK.vec3_add_kernel[1, 1](v1_device, v2_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))
    
    def test_vec3_sub(self):
        v1_host = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        v2_host = np.array([[4.0, 5.0, 6.0]], dtype=np.float64)
        expected = np.array([[-3.0, -3.0, -3.0]], dtype=np.float64)
        result_host = np.zeros((1, 3), dtype=np.float64)
        v1_device = cuda.to_device(v1_host)
        v2_device = cuda.to_device(v2_host)
        result_device = cuda.device_array_like(result_host)
        TEST_VK.vec3_sub_kernel[1, 1](v1_device, v2_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))
    
    def test_vec3_norm(self):
        v_host = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        expected = np.array([1.0], dtype=np.float64)
        result_host = np.zeros((1), dtype=np.float64)
        v_device = cuda.to_device(v_host)
        result_device = cuda.device_array_like(result_host)
        TEST_VK.vec3_norm_kernel[1, 1](v_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))
    
    def test_vec3_dot(self):
        v1_host = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        v2_host = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        expected = np.array([0.0], dtype=np.float64)
        result_host = np.zeros((1), dtype=np.float64)
        v1_device = cuda.to_device(v1_host)
        v2_device = cuda.to_device(v2_host)
        result_device = cuda.device_array_like(result_host)
        TEST_VK.vec3_dot_kernel[1, 1](v1_device, v2_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))
    
    def test_vec3_cross(self):
        v1_host = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        v2_host = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        expected = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        result_host = np.zeros((1, 3), dtype=np.float64)
        v1_device = cuda.to_device(v1_host)
        v2_device = cuda.to_device(v2_host)
        result_device = cuda.device_array_like(result_host)
        TEST_VK.vec3_cross_kernel[1, 1](v1_device, v2_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))
    
    def test_vec3_mul(self):
        v_host = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        expected = np.array([[2.0, 4.0, 6.0]], dtype=np.float64)
        result_host = np.zeros((1, 3), dtype=np.float64)
        v_device = cuda.to_device(v_host)
        result_device = cuda.device_array_like(result_host)
        TEST_VK.vec3_mul_kernel[1, 1](v_device, 2.0, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))

    def test_vec_argmin(self):
        v_host = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        s_host = np.array([3])
        expected = np.array([0], dtype=np.float64)
        result_host = np.zeros((1), dtype=np.float64)
        v_device = cuda.to_device(v_host)
        s_device = cuda.to_device(s_host)
        result_device = cuda.device_array_like(result_host)
        TEST_VK.vec_argmin_kernel[1, 1](v_device, s_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))


if __name__ == '__main__':
    unittest.main()