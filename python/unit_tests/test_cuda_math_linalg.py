import unittest
import os
import sys
import numpy as np
from numba import cuda

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.cuda.unit_tests.test_math_linalg_kernel as TEST_LinAlgK
import rainbow.util.test_tools as TEST


@unittest.skipIf(not cuda.is_available(), "CUDA not available")
class TestLinAlgCuda(unittest.TestCase):
    
    def test_cramer_solver(self):
        A_host = np.array([[[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0]]], dtype=np.float64)
        b_host = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        expected = np.array([[-0.0, 0.0, 0.0]], dtype=np.float64)
        result_host = np.zeros((1, 3), dtype=np.float64)
        A_device = cuda.to_device(A_host)
        b_device = cuda.to_device(b_host)
        result_device = cuda.device_array_like(result_host)
        TEST_LinAlgK.cramer_solver_kernel[1, 1](
            A_device, b_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))


if __name__ == '__main__':
    unittest.main()
