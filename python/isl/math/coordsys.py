import numpy as np
import isl.math.quaternion as Q
import isl.math.vector3 as V3


class CoordSys:
    def __init__(self):
        self.q = Q.identity()  # Orientation stored as a quaternion
        self.r = V3.zero()  # Position of Origin


def make(r, q):
    C = CoordSys()
    C.r = r
    C.q = q
    return C


def xform_point(X, p):
    return Q.rotate(X.q, p) + X.r


def xform_vector(X, v):
    return Q.rotate(X.q, p)


def xform_matrix(X, M):
    R = Q.to_matrix(X.q)
    return np.dot(R, M)


def concat(B, A):
    """
    Concate transformations.. First A then B, Hence, C = B*A
    :param B:
    :param A:
    :return:
    """
    C = CoordSys()
    C.q = Q.unit(prod(B.q, A.q))
    C.r = Q.rotate(B.q, A.r) + B.r
    return C


def inverse(X):
    C = CoordSys()
    C.q = Q.conjugate(X.q)
    C.r = Q.rotate(C.q, -X.r)
    return C


def make_coordsys_from_to(A, B):
    """
    Assumes that 'A' maps from bf_1 to wcs, and 'B' maps from bf_2 to wcs. Now compute the transform that maps from bf_1 to bf_2

    :param A:
    :param B:
    :return:
    """
    A2B = CoordSys()
    A2B.q = Q.unit(Q.prod(Q.conjugate(B.q), A.q))
    A2B.r = Q.rotate(Q.conjugate(B.q), A.r - B.r)
    return A2B
