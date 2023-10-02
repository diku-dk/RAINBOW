import unittest
import os
import sys
import numpy as np
from numba import cuda

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.util.test_tools as TEST
import rainbow.geometry.grid3 as GRID
import rainbow.cuda.unit_tests.test_geometry_barycentric_kernel as TEST_BCK


class TestGrid3Cuda(unittest.TestCase):
    
    def test_compute_barycentric_tetrahedron(self):
        x1_host = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float64)
        x2_host = np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float64)
        x3_host = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float64)
        x4_host = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
        u1 = np.array([0.1, -0.1])
        u2 = np.array([0.1, -0.1])
        u3 = np.array([0.1, -0.1])
        p_host = []
        expected = []
        for i, _ in enumerate(u1):
            u4 = 1 - u1[i] - u2[i] - u3[i]
            p_host.append(u1[i]*x1_host[i] + u2[i]*x2_host[i] + u3[i]*x3_host[i] + u4*x4_host[i])
            expected.append(np.array([u1[i], u2[i], u3[i], u4]))
        p_host = np.array(p_host, dtype=np.float64)
        expected = np.array(expected, dtype=np.float64)

        x1_device = cuda.to_device(x1_host)
        x2_device = cuda.to_device(x2_host)
        x3_device = cuda.to_device(x3_host)
        x4_device = cuda.to_device(x4_host)
        p_device = cuda.to_device(p_host)

        result_host = np.zeros((p_host.shape[0], 4), dtype=np.float64)
        result_device = cuda.device_array_like(result_host)
        TEST_BCK.compute_barycentric_tetrahedron_kernel[1, p_host.shape[0]](x1_device, x2_device, x3_device, x4_device, p_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))


if __name__ == '__main__':
    unittest.main()
