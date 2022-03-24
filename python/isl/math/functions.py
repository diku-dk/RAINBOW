from math import fabs, sin
import numpy as np


def sinc(x):
    """
    Compute Sinc Function.
    The implementation of this method was greatly inspired by the
    one in Open Dynamics Engine v. 0.039

    This method returns sin(x)/x. this has a singularity at 0 so special
    handling is needed for small arguments.

    :param x:    The input argument
    :return:     The value of sin(x)/x
    """
    """
    if |x| < 1e-4 then use a taylor series expansion. this two term expansion
    is actually accurate to one LS bit within this range if double precision
    is being used - so don't worry!
    """
    tiny = 1.0e-4
    factor = 0.166666666666666666667
    if fabs(x) < tiny:
        return 1.0 - x * x * factor
    return sin(x) / x  #pragma: no cover   


def PCA(P):
    """
    Perform PCA analysis on 3D data points stored in P array.
    :param P: An M-by-3 numpy array of floats, stores M 3D points.
    :return:  mean point, eigen and eigenvalues of the covariance matrix
    """
    mean = np.mean(P.T, axis=1)
    C = P - mean
    V = np.cov(C.T)
    values, vectors = np.linalg.eig(V)
    return mean, values, vectors


def direction_of_most_variance(P):
    """

    :param P:   An M-by-3 numpy array of floats, stores M 3D points.
    :return:    The direction of most variance of the points
    """
    _, values, vectors = PCA(P)            # Returns eigenvalue not sorted
    largest_eigenv_idx = np.argmax(values)
    return vectors[largest_eigenv_idx]
