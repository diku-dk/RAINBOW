import numpy as np
from numba import cuda, float64, int32, boolean
import rainbow.cuda.geometry.barycentric as BC


@cuda.jit
def compute_barycentric_tetrahedron_kernel(x1s: float64[:, :], x2s: float64[:, :], x3s: float64[:, :], x4s: float64[:, :], ps: float64[:, :], result: float64[:, :]):
    """ Unit test kernel function for compute_barycentric_tetrahedron_device
    
    Args:
        x1s (float64[:, :]): The list of the first point of the tetrahedron.
        x2s (float64[:, :]): The list of the second point of the tetrahedron.
        x3s (float64[:, :]): The list of the third point of the tetrahedron.
        x4s (float64[:, :]): The list of the fourth point of the tetrahedron.
        ps (float64[:, :]): The list of the points.
        result (float64[:, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    BC.compute_barycentric_tetrahedron_device(x1s[tid, :], x2s[tid, :], x3s[tid, :], x4s[tid, :], ps[tid, :], result[tid, :])