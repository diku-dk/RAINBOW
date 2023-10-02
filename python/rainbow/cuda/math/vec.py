import math
from numba import cuda, float64, int32


@cuda.jit(device=True)
def vec3_zero(result: float64[:]):
    """ Return a zero vector.

    Args:
        result (float64[:]): the vector data
    """
    result[0] = 0.0
    result[1] = 0.0
    result[2] = 0.0


@cuda.jit(device=True)
def vec3_norm(v: float64[:]):
    """ Return the norm of a vector.

    Args:
        v (float64[:]): a vector
    """
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])    


@cuda.jit(device=True)
def vec3_unit(v: float64[:], result: float64[:]):
    """ Return the unit vector

    Args:
        v (float64[:]): a vector
        result (float64[:]): the unit vector data
    """
    norm = vec3_norm(v)

    result[0] = v[0] / norm
    result[1] = v[1] / norm
    result[2] = v[2] / norm


@cuda.jit(device=True)
def vec3_add(v1: float64[:], v2: float64[:], result: float64[:]):
    """ The addition of two vectors

    Args:
        v1 (float64[:]): a vector
        v2 (float64[:]): a vector
        result (float64[:]): the result of the addition
    """
    result[0] = v1[0] + v2[0]
    result[1] = v1[1] + v2[1]
    result[2] = v1[2] + v2[2]


@cuda.jit(device=True)
def vec3_sub(v1: float64[:], v2: float64[:], result: float64[:]):
    """ The subtraction of two vectors

    Args:
        v1 (float64[:]): a vector
        v2 (float64[:]): a vector
        result (float64[:]): the result of the subtraction
    """
    result[0] = v1[0] - v2[0]
    result[1] = v1[1] - v2[1]
    result[2] = v1[2] - v2[2]


@cuda.jit(device=True)
def vec3_dot(v1: float64[:], v2: float64[:]) -> float64:
    """ The dot product of two vectors

    Args:
        v1 (float64[:]): a vector
        v2 (float64[:]): a vector
        result (float64[:]): the result of the dot product
    """
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


@cuda.jit(device=True)
def vec3_cross(v1: float64[:], v2: float64[:], result: float64[:]):
    """ The cross product of two vectors

    Args:
        v1 (float64[:]): a vector
        v2 (float64[:]): a vector
        result (float64[:]): the result of the cross product
    """
    result[0] = v1[1] * v2[2] - v1[2] * v2[1]
    result[1] = v1[2] * v2[0] - v1[0] * v2[2]
    result[2] = v1[0] * v2[1] - v1[1] * v2[0]


@cuda.jit(device=True)
def vec3_mut_scalar(v1: float64[:], n: float64, result: float64[:]):
    """ The multiplication of a vector by a scalar

    Args:
        v1 (float64[:]): a vector
        v2 (float64[:]): a scalar
        result (float64[:]): the result of the multiplication
    """
    result[0] = v1[0] * n
    result[1] = v1[1] * n
    result[2] = v1[2] * n


@cuda.jit(device=True)
def argmin(a: float64[:], size: int32) -> int32:
    """ Return the index of the minimum value in a vector

    Args:
        a (float64[:]): a vector
        size (int32): the size of the vector or the number of elements to consider
    """
    min_val = a[0]
    min_idx = 0

    size = min(size, a.shape[0])

    for i in range(size):
        if a[i] < min_val:
            min_val = a[i]
            min_idx = i

    return min_idx