import numpy as np
import scipy
import rainbow.math.vector3 as V3
from math import cos, sin


def identity():
    return np.identity(3, dtype=np.float64)


def diag_from_array(v):
    return np.array([[v[0], 0, 0], [0, v[1], 0], [0, 0, v[2]]], dtype=np.float64)


def diag(a, b, c):
    return np.array([[a, 0, 0], [0, b, 0], [0, 0, c]], dtype=np.float64)


def zero():                                      #pragma: no cover
    return np.zeros((3, 3), dtype=np.float64)    #pragma: no cover


def ones():                                      #pragma: no cover
    return np.ones((3, 3), dtype=np.float64)     #pragma: no cover


def make(A00, A01, A02, A10, A11, A12, A20, A21, A22):
    return np.array(
        [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]], dtype=np.float64
    )


def make_from_rows(row0, row1, row2):
    return np.array([row0, row1, row2], dtype=np.float64)


def make_from_cols(col0, col1, col2):
    return make(
        col0[0], col1[0], col2[0], col0[1], col1[1], col2[1], col0[2], col1[2], col2[2]
    )


def star(v):
    """
    Changes a vector cross-product into a matrix multiplication, a x b = (a*)b = b(a*)

    :param v:
    :return:
    """
    return np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64
    )


def ortonormalize(A):
    row0 = A[0, :]
    row1 = A[1, :]
    l0 = np.linalg.norm(row0)
    if l0 > 0.0:
        row0 /= l0
    row1 -= np.dot(row0, row1) * row0
    l1 = np.linalg.norm(row1)
    if l1 > 0.0:
        row1 /= l1
    row2 = np.cross(row0, row1, axis=0)
    return np.vstack((row0, row1, row2))


def Ru(radians, axis):
    c = cos(radians)
    s = sin(radians)
    u = axis / np.linalg.norm(axis)
    R = zero()
    # Foley p. 227(5.76)
    R[0, 0] = u[0] * u[0] + c * (1.0 - u[0] * u[0])
    R[0, 1] = u[0] * u[1] * (1.0 - c) - s * u[2]
    R[0, 2] = u[0] * u[2] * (1.0 - c) + s * u[1]
    R[1, 0] = u[0] * u[1] * (1.0 - c) + s * u[2]
    R[1, 1] = u[1] * u[1] + c * (1.0 - u[1] * u[1])
    R[1, 2] = u[1] * u[2] * (1.0 - c) - s * u[0]
    R[2, 0] = u[0] * u[2] * (1.0 - c) - s * u[1]
    R[2, 1] = u[1] * u[2] * (1.0 - c) + s * u[0]
    R[2, 2] = u[2] * u[2] + c * (1.0 - u[2] * u[2])
    return R


def Rx(radians):
    c = cos(radians)
    s = sin(radians)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def Ry(radians):
    c = cos(radians)
    s = sin(radians)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def Rz(radians):
    c = cos(radians)
    s = sin(radians)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def polar_decomposition(A):
    """
    Polar Decomposition of matrix A (as described by Etzmuss et. al in ``A Fast Finite Solution for Cloth Modelling'')
    The decomposed rotation matrix has the properties R*R^T = I and det(R)=1.

    Parameters:
    A (matrix3): The matrix to perform polar decomposition on

    Returns:
    R (matrix3): The orthogonal rotation matrix describing the rotation of the deformation gradient.
    S (matrix3): The symmetric matrix describing the deformation
    """
    S2 = np.matmul(A.T, A)
    # eigh takes advantage of the matrix being symmetric, and guarantees a result of real elements.
    eigvals, eigvecs = np.linalg.eigh(S2)

    # Check that all eigenvalues are positive
    if not (eigvals > 0).all():
        raise ValueError("polar_decomposition(): Not all eigenvalues are positive")

    v0 = V3.make(eigvecs[0, 0], eigvecs[1, 0], eigvecs[2, 0])
    v1 = V3.make(eigvecs[0, 1], eigvecs[1, 1], eigvecs[2, 1])
    v2 = V3.make(eigvecs[0, 2], eigvecs[1, 2], eigvecs[2, 2])

    S = (
        np.outer(v0, v0) * np.sqrt(eigvals[0])
        + np.outer(v1, v1) * np.sqrt(eigvals[1])
        + np.outer(v2, v2) * np.sqrt(eigvals[2])
    )
    R = np.matmul(A, np.linalg.inv(S))
    return R, S


def polar_decomposition_array(As):
    """
    Polar Decomposition of matrix A (as described by Etzmuss et. al in ``A Fast Finite Solution for Cloth Modelling'')
    The decomposed rotation matrix has the properties R*R^T = I and det(R)=1.

    Parameters:
    A (matrix3): The matrix to perform polar decomposition on

    Returns:
    R (matrix3): The orthogonal rotation matrix describing the rotation of the deformation gradient.
    S (matrix3): The symmetric matrix describing the deformation
    """
    S2 = np.matmul(As.transpose(0, 2, 1), As)
    # eigh takes advantage of the matrix being symmetric, and guarantees a result of real elements.
    eigvals, eigvecs = np.linalg.eigh(S2)

    # Check that all eigenvalues are positive
    if not (eigvals > 0).all():
        raise ValueError("polar_decomposition(): Not all eigenvalues are positive")

    v0 = eigvecs[:, :, 0]
    v1 = eigvecs[:, :, 1]
    v2 = eigvecs[:, :, 2]

    first_term = v0[..., None] * v0[:, None] * np.sqrt(eigvals[:, 0])[:, None, None]
    second_term = v1[..., None] * v1[:, None] * np.sqrt(eigvals[:, 1])[:, None, None]
    third_term = v2[..., None] * v2[:, None] * np.sqrt(eigvals[:, 2])[:, None, None]
    S = first_term + second_term + third_term

    R = np.matmul(As, np.linalg.inv(S))
    return R, S
