import unittest
import os
import sys
import numpy as np
from numba import cuda

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.util.test_tools as TEST
import rainbow.geometry.grid3 as GRID
import rainbow.cuda.unit_tests.test_geometry_grid3_kernel as TEST_GRID3K


def simpelfunc(coord):
    _, _, z = coord[0], coord[1], coord[2]
    return z


@unittest.skipIf(not cuda.is_available(), "CUDA not available")
class TestGrid3Cuda(unittest.TestCase):

    def setUp(self):
        min_coord = np.array([-1, -1, -1])
        max_coord = np.array([1, 1, 1])
        I = 4
        J = 4
        K = 4

        self.grid = GRID.Grid(min_coord, max_coord, I, J, K)
    
    def test_get_enclosing_cell_idx(self):
        p_host = np.array([ [0, 0, 0],
                            [-0.5, -0.5, -0.5],
                            [0.5, 0.5, 0.5],
                            [-1, -1, -1],
                            [1, 1, 1],
                            [-2, -2, -2],
                            [2, 2, 2]], dtype=np.float64)
        expected = np.array([[1, 1, 1],
                                  [0, 0, 0],
                                  [2, 2, 2],
                                  [0, 0, 0],
                                  [2, 2, 2],
                                  [0, 0, 0],
                                  [2, 2, 2]], dtype=np.int32)
        min_coord_host = np.tile(self.grid.min_coord, p_host.shape)
        spacing_host = np.tile(self.grid.spacing, p_host.shape)
        I_host = np.repeat(self.grid.I, p_host.shape[0])
        J_host = np.repeat(self.grid.J, p_host.shape[0])
        K_host = np.repeat(self.grid.K, p_host.shape[0])
        
        p_device = cuda.to_device(p_host)
        min_coord_device = cuda.to_device(min_coord_host)
        spacing_device = cuda.to_device(spacing_host)
        I_device = cuda.to_device(I_host)
        J_device = cuda.to_device(J_host)
        K_device = cuda.to_device(K_host)

        result_host = np.zeros(p_host.shape, dtype=np.int32)
        result_device = cuda.device_array_like(result_host)
        TEST_GRID3K.get_enclosing_cell_idx_kernel[1,  p_host.shape[0]](p_device, min_coord_device, spacing_device, I_device, J_device, K_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))

    def test_get_node_value(self):
        min_coord = np.array([-1, -1, -1])
        max_coord = np.array([1, 1, 1])
        I = 5
        J = 5
        K = 5
        grid = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(grid, simpelfunc)

        i_host = np.array([0, 0])
        j_host = np.array([0, 0])
        k_host = np.array([4, 0])
        values_host = np.tile(grid.values.reshape(1, -1), (i_host.shape[0], 1))
        I_host = np.repeat(grid.I, i_host.shape[0])
        J_host = np.repeat(grid.J, i_host.shape[0])
        expected = np.array([1, -1], dtype=np.float64)

        i_device = cuda.to_device(i_host)
        j_device = cuda.to_device(j_host)
        k_device = cuda.to_device(k_host)
        values_device = cuda.to_device(values_host)
        I_device = cuda.to_device(I_host)
        J_device = cuda.to_device(J_host)

        result_host = np.zeros((i_host.shape[0]), dtype=np.float64)
        result_device = cuda.device_array_like(result_host)
        TEST_GRID3K.get_node_value_kernel[1,  i_host.shape[0]](i_device, j_device, k_device, values_device, I_device, J_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))

    def test_get_value(self):

        def func(p):
            return p[0]
        
        min_coord = np.array([-1, -1, -1])
        max_coord = np.array([1, 1, 1])
        I = 5
        J = 5
        K = 5
        grid = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(grid, func)

        p_host = np.array([ [0, 0, 0],
                            [-0.5, -0.5, -0.5],
                            [0.5, 0.5, 0.5],
                            [-1, -1, -1],
                            [1, 1, 1],
                            [-2, -2, -2],
                            [2, 2, 2]], dtype=np.float64)
        expected = np.array([func(p_host[0]),
                                  func(p_host[1]),
                                  func(p_host[2]),
                                  func(p_host[3]),
                                  func(p_host[4]),
                                  func(p_host[5]),
                                  func(p_host[6])], dtype=np.float64)
        min_coord_host = np.tile(grid.min_coord, p_host.shape)
        spacing_host = np.tile(grid.spacing, p_host.shape)
        I_host = np.repeat(grid.I, p_host.shape[0])
        J_host = np.repeat(grid.J, p_host.shape[0])
        K_host = np.repeat(grid.K, p_host.shape[0])
        values_host = np.tile(grid.values.reshape(1, -1), (p_host.shape[0], 1))
        p_device = cuda.to_device(p_host)
        min_coord_device = cuda.to_device(min_coord_host)
        spacing_device = cuda.to_device(spacing_host)
        I_device = cuda.to_device(I_host)
        J_device = cuda.to_device(J_host)
        K_device = cuda.to_device(K_host)
        values_device = cuda.to_device(values_host)

        result_host = np.zeros(p_host.shape[0], dtype=np.float64)
        result_device = cuda.device_array_like(result_host)
        TEST_GRID3K.get_value_kernel[1,  p_host.shape[0]](p_device, min_coord_device, spacing_device, I_device, J_device, K_device, values_device, result_device)
        result_device.copy_to_host(result_host)
        for res, exp in zip(result_host, expected):
            self.assertAlmostEqual(res, exp)

    def test_is_inside(self):
        p_host = np.array([ [-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1],
                            [-1.1, -1.1, -1.1],
                            [1.1, 1.1, 1.1],
                            [-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1],
                            [-1.1, -1.1, -1.1],
                            [1.1, 1.1, 1.1]], dtype=np.float64)
        boundary_host = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        min_coord_host = np.tile(self.grid.min_coord, p_host.shape)
        max_coord_host = np.tile(self.grid.max_coord, p_host.shape)
        expected = np.array([True, True, True, False, False, False, True, False, False, False], dtype=bool)

        p_device = cuda.to_device(p_host)
        boundary_device = cuda.to_device(boundary_host)
        min_coord_device = cuda.to_device(min_coord_host)
        max_coord_device = cuda.to_device(max_coord_host)

        result_host = np.zeros(p_host.shape[0], dtype=bool)
        result_device = cuda.device_array_like(result_host)
        TEST_GRID3K.is_inside_kernel[1,  p_host.shape[0]](p_device, min_coord_device, max_coord_device, boundary_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))
    
    def test_get_gradient(self):
        def func(p):
            return p[0]

        def grad(p):
            return np.array([1, 0, 0])
        
        min_coord = np.array([-1, -1, -1])
        max_coord = np.array([1, 1, 1])
        I = 5
        J = 5
        K = 5
        grid = GRID.Grid(min_coord, max_coord, I, J, K)
        GRID.eval_on_grid(grid, func)

        p_host = np.array([ [0, 0, 0],
                            [-0.5, -0.5, -0.5],
                            [0.5, 0.5, 0.5],
                            [-1, -1, -1],
                            [1, 1, 1],
                            [-2, -2, -2],
                            [2, 2, 2]], dtype=np.float64)
        expected = np.array([ grad(p_host[0]),
                              grad(p_host[1]),
                              grad(p_host[2]),
                              grad(p_host[3]),
                              grad(p_host[4]),
                              grad(p_host[5]),
                              grad(p_host[6])], dtype=np.float64)
        min_coord_host = np.tile(grid.min_coord, p_host.shape)
        spacing_host = np.tile(grid.spacing, p_host.shape)
        I_host = np.repeat(grid.I, p_host.shape[0])
        J_host = np.repeat(grid.J, p_host.shape[0])
        K_host = np.repeat(grid.K, p_host.shape[0])
        values_host = np.tile(grid.values.reshape(1, -1), (p_host.shape[0], 1))

        p_device = cuda.to_device(p_host)
        min_coord_device = cuda.to_device(min_coord_host)
        spacing_device = cuda.to_device(spacing_host)
        I_device = cuda.to_device(I_host)
        J_device = cuda.to_device(J_host)
        K_device = cuda.to_device(K_host)
        values_device = cuda.to_device(values_host)

        result_host = np.zeros(p_host.shape, dtype=np.float64)
        result_device = cuda.device_array_like(result_host)
        TEST_GRID3K.get_gradient_kernel[1,  p_host.shape[0]](p_device, min_coord_device, spacing_device, I_device, J_device, K_device, values_device, result_device)
        result_device.copy_to_host(result_host)
        self.assertTrue(TEST.is_array_equal(result_host, expected))


if __name__ == '__main__':
    unittest.main()
