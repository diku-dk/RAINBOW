import numpy as np
from numba import cuda, float64, int32
import rainbow.cuda.math.matrix as Matrix


@cuda.jit
def mat33_T_kernel(M: float64[:, :, :], result: float64[:, :, :]):
    """ Unit test kernel function for mat33_T_device

    Args:
        M (float64[:, :, :]): it is a list of matrix
        result (float64[:, :, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    Matrix.mat33_T(M[tid, :, :], result[tid, :, :])


@cuda.jit
def mat33_zero_kernel(result: float64[:, :, :]):
    """ Unit test kernel function for mat33_zero_device

    Args:
        result (float64[:, :, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    Matrix.mat33_zero(result[tid, :, :])


@cuda.jit
def mat33_determinant_kernel(M: float64[:, :, :], result: float64[:]):
    """ Unit test kernel function for mat33_determinant_device

    Args:
        M (float64[:, :, :]): it is a list of matrix
        result (float64[:]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    result[tid] = Matrix.mat33_determinant(M[tid, :, :])


@cuda.jit
def mat33_make_kernel(M00: float64[:], M01: float64[:], M02: float64[:], M10: float64[:], M11: float64[:], M12: float64[:], M20: float64[:], M21: float64[:], M22: float64[:], result: float64[:, :, :]):
    """ Unit test kernel function for mat33_make_device

    Args:
        M00 (float64[:]): The 00 element of the matrix.
        M01 (float64[:]): The 01 element of the matrix.
        M02 (float64[:]): The 02 element of the matrix.
        M10 (float64[:]): The 10 element of the matrix.
        M11 (float64[:]): The 11 element of the matrix.
        M12 (float64[:]): The 12 element of the matrix.
        M20 (float64[:]): The 20 element of the matrix.
        M21 (float64[:]): The 21 element of the matrix.
        M22 (float64[:]): The 22 element of the matrix.
        result (float64[:, :, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    Matrix.mat33_make(M00[tid], M01[tid], M02[tid], 
                      M10[tid], M11[tid], M12[tid], 
                      M20[tid], M21[tid], M22[tid], 
                      result[tid, :, :])


@cuda.jit
def mat33_make_from_cols_kernel(col0: float64[:, :], col1: float64[:, :], col2: float64[:, :], result: float64[:, :, :]):
    """ Unit test kernel function for mat33_make_from_cols_device

    Args:
        col0 (float64[:, :]): The 0-th column of the matrix.
        col1 (float64[:, :]): The 1-th column of the matrix.
        col2 (float64[:, :]): The 2-th column of the matrix.
        result (float64[:, :, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    Matrix.mat33_make_from_cols(col0[tid, :], col1[tid, :], col2[tid, :], result[tid, :, :])


@cuda.jit
def mat33_dot_vec3_kernel(M: float64[:, :, :], V3: float64[:, :], result: float64[:, :]):
    """ Unit test kernel function for mat33_dot_vec3_device

    Args:
        M (float64[:, :, :]): it is a list of matrix
        V3 (float64[:, :]): it is a list of vector
        result (float64[:, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    Matrix.mat33_dot_vec3(M[tid, :, :], V3[tid, :], result[tid, :])


@cuda.jit
def mat43_T_kernel(M: float64[:, :, :], result: float64[:, :, :]):
    """ Unit test kernel function for mat43_T_device

    Args:
        M (float64[:, :, :]): it is a list of matrix
        result (float64[:, :, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    Matrix.mat43_T(M[tid, :, :], result[tid, :, :])


@cuda.jit
def mat34_dot_vec4_kernel(M: float64[:, :, :], vec4: float64[:, :], result: float64[:, :]):
    """ Unit test kernel function for mat34_dot_vec4_device

    Args:
        M (float64[:, :, :]): it is a list of matrix
        vec4 (float64[:, :]): it is a list of vector
        result (float64[:, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    Matrix.mat34_dot_vec4(M[tid, :, :], vec4[tid, :], result[tid, :])