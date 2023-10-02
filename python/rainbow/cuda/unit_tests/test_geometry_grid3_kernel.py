import numpy as np
from numba import cuda, float64, int32, boolean
import rainbow.cuda.geometry.grid3 as GRID3


@cuda.jit
def get_enclosing_cell_idx_kernel(p: float64[:, :], min_coord: float64[:, :], spacing: float64[:, :], I: int32[:], J: int32[:], K: int32[:], result: int32[:, :]):
    """ Unit test kernel function for get_enclosing_cell_idx_device

    Args:
        p (float64[:, :]): The 3D spatial point.
        min_coord (float64[:, :]): The minimum corner of the grid.
        spacing (float64[:, :]): The grid spacing.
        I (int32[:]): The number of cells in the x-direction of the grid.
        J (int32[:]): The number of cells in the y-direction of the grid.
        K (int32[:]): The number of cells in the z-direction of the grid.
        result (float64[:, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    GRID3.get_enclosing_cell_idx_device(p[tid, :], min_coord[tid, :], spacing[tid, :], I[tid], J[tid], K[tid], result[tid, :])


@cuda.jit
def get_value_kernel(p: float64[:, :], min_coord: float64[:, :], spacing: float64[:, :], I: int32[:], J: int32[:], K: int32[:], values: float64[:, :], result: float64[:]):
    """ Unit test kernel function for get_value_device

    Args:
        p (float64[:, :]): The 3D spatial point.
        min_coord (float64[:, :]): The minimum corner of the grid.
        spacing (float64[:, :]): The grid spacing.
        I (int32[:]): The number of cells in the x-direction of the grid.
        J (int32[:]): The number of cells in the y-direction of the grid.
        K (int32[:]): The number of cells in the z-direction of the grid.
        values (float64[:, :]): It is an array storing signed distances of grid points to a given mesh.
        result (float64[:]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    result[tid] = GRID3.get_value_device(p[tid, :], min_coord[tid, :], spacing[tid, :], I[tid], J[tid], K[tid], values[tid, :])


@cuda.jit
def get_node_value_kernel(i: int32[:], j: int32[:], k: int32[:], values: float64[:, :], I: int32[:], J: int32[:], result: float64[:]):
    """ Unit test kernel function for get_node_value_device

    Args:
        i (int32[:]): The i-coordinate of the grid.
        j (int32[:]): The j-coordinate of the grid.
        k (int32[:]): The k-coordinate of the grid.
        values (float64[:, :]): It is an array storing signed distances of grid points to a given mesh.
        I (int32[:]): The number of cells in the x-direction of the grid.
        J (int32[:]): The number of cells in the y-direction of the grid.
        result (float64[:]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    result[tid] = GRID3.get_node_value_device(i[tid], j[tid], k[tid], values[tid, :], I[tid], J[tid], )


@cuda.jit
def is_inside_kernel(p: float64[:, :], min_coord: float64[:, :], max_coord: float64[:, :], boundary: float64[:], result: boolean[:]):
    """ Unit test kernel function for is_inside_device

    Args:
        p (float64[:, :]): The 3D spatial point.
        min_coord (float64[:, :]): The minimum corner of the grid.
        max_coord (float64[:, :]): The maximum corner of the grid.
        boundary (float64[:]): The boundary of the grid.
        result (boolean[:]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    result[tid] = GRID3.is_inside_device(p[tid, :], min_coord[tid, :], max_coord[tid, :], boundary[tid])


@cuda.jit
def get_gradient_kernel(p: float64[:, :], min_coord: float64[:, :], spacing: float64[:, :], I: int32[:], J: int32[:], K: int32[:], values: float64[:, :], result: float64[:, :]):
    """ Unit test kernel function for get_gradient_device

    Args:
        p (float64[:, :]): The 3D spatial point.
        min_coord (float64[:, :]): The minimum corner of the grid.
        spacing (float64[:, :]): The grid spacing.
        I (int32[:]): The number of cells in the x-direction of the grid.
        J (int32[:]): The number of cells in the y-direction of the grid.
        K (int32[:]): The number of cells in the z-direction of the grid.
        values (float64[:, :]): It is an array storing signed distances of grid points to a given mesh.
        result (float64[:, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    GRID3.get_gradient_device(p[tid, :], min_coord[tid, :], spacing[tid, :], I[tid], J[tid], K[tid], values[tid, :], result[tid, :])