import unittest
import os
import sys
import numpy as np
import igl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.cuda.geometry.grid3 as CUDAGRID
import rainbow.geometry.grid3 as GRID
import rainbow.simulators.prox_soft_bodies.api as API
import rainbow.geometry.volume_mesh as VM
from numba import cuda

class TestGrid(unittest.TestCase):
    def test_get_enclosing_cell_idx(self):
        grid = GRID.Grid([0,0,0], [9,9,9], 10, 10, 10)
        points = np.random.rand(1000, 3) * 10
        results_gpu = cuda.device_array((points.shape[0], 3), dtype=np.int32)
        
        threads_per_block = 256
        blocks_per_grid = (points.shape[0] + threads_per_block - 1) // threads_per_block

        CUDAGRID.get_enclosing_cell_idx_kernel[blocks_per_grid, threads_per_block](points, grid.min_coord, grid.spacing, grid.I, grid.J, grid.K, results_gpu)

        results_cpu = np.zeros((points.shape[0], 3), dtype=np.int32)
        for i in range(points.shape[0]):
            p = points[i]
            results_cpu[i] = grid.get_enclosing_cell_idx(p)

        self.assertTrue(np.all(results_cpu == results_gpu.copy_to_host()))
    
    def test_get_gradient(self):
        engine = API.create_engine()
        V_beam, T_beam = VM.create_beam(10, 10, 10, 100.0, 100.0, 100.0)
        API.create_soft_body(engine, 'beam', V_beam, T_beam)
        body = engine.bodies['beam']
        grid = body.grid
 
        points = np.random.rand(1000, 3) * 10
        results_gpu = cuda.device_array((points.shape[0], 3), dtype=np.int32)
        
        threads_per_block = 256
        blocks_per_grid = (points.shape[0] + threads_per_block - 1) // threads_per_block

        CUDAGRID.get_gradient_kernel[blocks_per_grid, threads_per_block](points, grid.min_coord, grid.spacing, grid.I, grid.J, grid.K, grid.values, results_gpu)

        results_cpu = np.zeros((points.shape[0], 3), dtype=np.int32)
        for i in range(points.shape[0]):
            p = points[i]
            results_cpu[i] = GRID.get_gradient(grid, p)

        self.assertTrue(np.all(results_cpu == results_gpu.copy_to_host()))

if __name__ == '__main__':
    unittest.main()