import math
from numba import cuda, types, float64, int32
from rainbow.cuda.math.vec import vec3_zero
from rainbow.cuda.math.matrix import  mat33_determinant


@cuda.jit(device=True)
def cramer_solver(A: float64[:, :], b: float64[:], result: float64[:]):
    """ Solve a linear system Ax = b using Cramer's rule,
        refer to https://en.wikipedia.org/wiki/Cramer%27s_rule,

    Note: Currently, we use the Cramer's rule to solve the linear system, which is a fast method but not stable when det(A) = 0. With Numba cuda, we do not find a stable and fast library to solve the linear system.

    Args:
        A (float64[:, :]): a matrix 3*3
        b (float64[:]): a vector
        result (float64[:]): the result of the linear system
    """
    detA = mat33_determinant(A)
    if detA == 0.0:
        vec3_zero(result)
    else:
        Ax = cuda.local.array((3, 3), dtype=float64)
        Ay = cuda.local.array((3, 3), dtype=float64)
        Az = cuda.local.array((3, 3), dtype=float64)

        for i in range(3):
            for j in range(3):
                Ax[i, j] = A[i, j]
                Ay[i, j] = A[i, j]
                Az[i, j] = A[i, j]

        for i in range(3):
            Ax[i, 0] = b[i]
            Ay[i, 1] = b[i]
            Az[i, 2] = b[i]

        x = mat33_determinant(Ax) / detA
        y = mat33_determinant(Ay) / detA
        z = mat33_determinant(Az) / detA

        result[0] = x
        result[1] = y
        result[2] = z