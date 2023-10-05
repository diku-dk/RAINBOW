import numpy as np
from numba import cuda, float64, int32
import rainbow.cuda.math.linalg as LinAlg


@cuda.jit
def cramer_solver_kernel(A: float64[:, :, :], b: float64[:, :], result: float64[:, :]):
    """ Unit test kernel function for cramer_solver_device

    Args:
        A (float[:, :, :]): it is a list of matrix
        b (float[:, ;]): It is a list of vector
        result (float64[:, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    LinAlg.cramer_solver(A[tid, :, :], b[tid, :], result[tid, :])

