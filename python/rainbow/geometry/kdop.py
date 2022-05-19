import rainbow.math.vector3 as V3
import numpy as np

# TODO 2020-05-07 Kenny: Complete implementation.
#  https://github.com/erleben/matchstick/blob/master/PROX/FOUNDATION/GEOMETRY/GEOMETRY/include/types/geometry_dop.h
#  alternative we can cleanup c++ code and use pybind11?


class Interval:
    def __init__(self):
        self.lower = np.inf
        self.upper = -np.inf


class KDOP:
    def __init__(self, K):
        self.K = K
        self.slabs = [Interval() for _ in range(K)]


def overlap(A, B):
    """
    This function tests for overlap between two KDOPs. The overlap is tested
    by testing for overlap between the slabs of the KDOPs.

    :param A: the first KDOP
    :param B: the second KDOP
    :return: True if there exists an overlap of the KDOPs
    """
    if A.K != B.K:
        raise RuntimeError("overlap(): kdops A and B do not share a common K")

    for k in range(A.K):
        # Check for overlap of intervals
        if A.slabs[k].upper < B.slabs[k].lower or B.slabs[k].upper < A.slabs[k].lower:
            return False

    return True


def get_direction_table(K):
    """
    This function returns a direction table of K directions for the K slabs of a
    KDOP. The direction table is used across all KDOPs such that overlap can
    be easily tested
    """
    if K == 3:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if K == 4:
        return np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]])


def make(K, V, envelope=0.0):
    """
    This function makes a KDOP bounding a set of vertices tightly.

    :param K:            The amount of intervals in the KDOP
    :param V:            The vertices to bound
    :param envelope:     The size of the envelope used to enlarge the volume. If a tight fit is wanted
                         one should use the value zero.
    :return:             A KDOP data structure
    """
    kdop = KDOP(K)
    direction_table = get_direction_table(K)
    N = len(V)
    projections = np.matmul(direction_table, V.T)
    # TODO 2020-09-07 Kenny review: Possible think about vectorizing this?
    for k in range(K):
        for j in range(N):
            projection = projections[k, j]
            kdop.slabs[k].lower = min(kdop.slabs[k].lower, projection)
            kdop.slabs[k].upper = max(kdop.slabs[k].upper, projection)
        kdop.slabs[k].lower -= envelope
        kdop.slabs[k].upper += envelope
    return kdop


def union(A, B):
    """
    This function combines two KDOPs into a larger KDOP encompassing them both

    :param A:   The first KDOP
    :param B:   The second KDOP
    :return:    A KDOP encompassing both the smaller KDOPs
    """
    if A.K != B.K:
        raise RuntimeError("union(): kdops A and B do not share a common K")
    K = A.K
    kdop = KDOP(K)
    # TODO 2020-09-07 Kenny review: can this be vectorized?
    for k in range(K):
        kdop.slabs[k].lower = min(A.slabs[k].lower, B.slabs[k].lower)
        kdop.slabs[k].upper = max(A.slabs[k].upper, B.slabs[k].upper)
    return kdop
