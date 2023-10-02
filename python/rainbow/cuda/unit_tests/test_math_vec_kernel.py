import numpy as np
from numba import cuda, float64, int32
import rainbow.cuda.math.vec as Vec


@cuda.jit
def vec3_add_kernel(v1: float64[:, :], v2: float64[:, :], result: float64[:, :]):
    """ Unit test kernel function for vec3_add_device

    Args:
        v1 (float64[:, :]): it is a list of vector
        v2 (float64[:, :]): It is a list of vector
        result (float64[:, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    Vec.vec3_add(v1[tid, :], v2[tid, :], result[tid, :])


@cuda.jit
def vec3_sub_kernel(v1: float64[:, :], v2: float64[:, :], result: float64[:, :]):
    """ Unit test kernel function for vec3_sub_device

    Args:
        v1 (float64[:, :]): it is a list of vector
        v2 (float64[:, :]): It is a list of vector
        result (float64[:, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    Vec.vec3_sub(v1[tid, :], v2[tid, :], result[tid, :])


@cuda.jit
def vec3_norm_kernel(v: float64[:, :], result: float64[:]):
    """ Unit test kernel function for vec3_norm_device

    Args:
        v (float64[:, :]): it is a list of vector
        result (float64[:]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    result[tid] = Vec.vec3_norm(v[tid, :])


@cuda.jit
def vec3_unit_kernel(v: float64[:, :], result: float64[:, :]):
    """ Unit test kernel function for vec3_unit_device

    Args:
        v (float64[:, :]): it is a list of vector
        result (float64[:, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    Vec.vec3_unit(v[tid, :], result[tid, :])


@cuda.jit
def vec3_zero_kernel(result: float64[:, :]):
    """ Unit test kernel function for vec3_zero_device

    Args:
        result (float64[:, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    Vec.vec3_zero(result[tid, :])


@cuda.jit
def vec3_cross_kernel(v1: float64[:, :], v2: float64[:, :], result: float64[:, :]):
    """ Unit test kernel function for vec3_cross_device

    Args:
        v1 (float64[:, :]): it is a list of vector
        v2 (float64[:, :]): It is a list of vector
        result (float64[:, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    Vec.vec3_cross(v1[tid, :], v2[tid, :], result[tid, :])


@cuda.jit
def vec3_dot_kernel(v1: float64[:, :], v2: float64[:, :], result: float64[:]):
    """ Unit test kernel function for vec3_dot_device

    Args:
        v1 (float64[:, :]): it is a list of vector
        v2 (float64[:, :]): It is a list of vector
        result (float64[:]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    result[tid] = Vec.vec3_dot(v1[tid, :], v2[tid, :])


@cuda.jit
def vec3_mul_kernel(v: float64[:, :], s: float64, result: float64[:, :]):
    """ Unit test kernel function for vec3_mul_device

    Args:
        v (float64[:, :]): it is a list of vector
        s (float64): It is a scalar
        result (float64[:, :]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    Vec.vec3_mut_scalar(v[tid, :], s, result[tid, :])


@cuda.jit
def vec_argmin_kernel(v: float64[:, :], s: int32[:], result: int32[:]):
    """ Unit test kernel function for vec_argmin_device

    Args:
        v (float64[:, :]): it is a list of vector.
        s (int32[:]): It is a list of vector size or the number of elements to consider.
        result (int32[:]): this is the result array from host.
    """
    tid = cuda.threadIdx.x
    result[tid] = Vec.argmin(v[tid, :], s[tid])