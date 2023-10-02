import math
from numba import cuda, types, float64, int32


@cuda.jit(device=True)
def mat33_T(M: float64[:, :], result: float64[:, :]):
    """ The transpose of a matrix

    Args:
        M (float64[:, :]): a matrix 3*3
        result (float64[:, :]): the result of the transpose
    """
    for i in range(3):
        for j in range(3):
            result[i, j] = M[j, i]
    return result


@cuda.jit(device=True)
def mat33_zero(result: float64[:, :]):
    """ Return a zero matrix.

    Args:
        result (float64[:, :]): the matrix data
    """
    for i in range(3):
        for j in range(3):
            result[i, j] = 0.0


@cuda.jit(device=True)
def mat33_determinant(M: float64[:, :]) -> float64:
    """ Return the determinant of a matrix

    Args:
        M (float64[:, :]): a matrix 3*3
    """
    return M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) - \
              M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0]) + \
                M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])


@cuda.jit(device=True)
def mat33_make(M00: float64,
               M01: float64,
               M02: float64,
               M10: float64,
               M11: float64,
               M12: float64,
               M20: float64,
               M21: float64,
               M22: float64,
               result: float64[:, :]):
    """ Make a 3*3 matrix

    Args:
        Mij (float64): the element of the matrix (i and j are from 0 to 2)
        result (float64[:, :]): the 3*3 matrix 
    """
    result[0, 0] = M00
    result[0, 1] = M01
    result[0, 2] = M02
    result[1, 0] = M10
    result[1, 1] = M11
    result[1, 2] = M12
    result[2, 0] = M20
    result[2, 1] = M21
    result[2, 2] = M22


@cuda.jit(device=True)
def mat33_make_from_cols(col0: float64[:], col1: float64[:], col2: float64[:], result: float64[:, :]):
    """ Make a 3*3 matrix from three column vectors

    Args:
        col0 (float64[:]): the first column vector
        col1 (float64[:]): the second column vector
        col2 (float64[:]): the third column vector
        result (float64[:, :]): the 3*3 matrix
    """
    mat33_make(
        col0[0], col1[0], col2[0],
        col0[1], col1[1], col2[1],
        col0[2], col1[2], col2[2],
        result
    )


@cuda.jit(device=True)
def mat33_dot_vec3(M: float64[:, :], V3: float64[:], result: float64[:]):
    """ Dot product of a matrix and a vector

    Args:
        M (float64[:, :]): a matrix 3*3
        V3 (float64[:]): a vector 3
        result (float64[:]): the result of the dot product
    """
    for i in range(3):
        sum= 0.0
        for j in range(3):
            sum += M[i, j] * V3[j]
        result[i] = sum


@cuda.jit(device=True)
def mat43_T(M: float64[:, :], result: float64[:, :]):
    """ The transpose of a matrix

    Args:
        M (float64[:, :]): a matrix 4*3
        result (float64[:, :]): the result of the transpose
    """
    for j in range(3):
        for i in range(4):
            result[j, i] = M[i, j]


@cuda.jit(device=True)
def mat34_dot_vec4(M: float64[:, :], vec4: float64[:], result: float64[:]):
    """ Dot product of a matrix and a vector

    Args:
        M (float64[:, :]): a matrix 3*4
        vec4 (float64[:]): a vector 4
        result (float64[:]): the result of the dot product
    """
    for i in range(3):
        sum= 0.0
        for j in range(4):
            sum += M[i, j] * vec4[j]
        result[i] = sum