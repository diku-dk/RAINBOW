import rainbow.math.matrix3 as M3
import numpy as np


def compute_barycentric_edge(x1, x2, p):
    """
    Compute Barycentric Coordinates.
    This method computes the barycentric coordinates for a point p of an edge given by the points x1 and x2.

   The barycentric coordinates w1 and w2 are defined such that p' = w1*x1 + w2*x2, is the point on the line closest to p.

    if 0 <= w1,w2 <= 1 then the point lies inside or on the perimeter of the triangle.

    Warning: this method uses a geometric approach to compute the barycentric coordinates.

    :param x1: The first point of the edge.
    :param x2: The second point of the edge.
    :param p:  The point for which the barycentric coordinates should be computed.
    :return:   A tuple containing the first and second barycentric coordinates w1 and w2.
    """
    u = x2 - x1
    uu = np.dot(u, u)
    assert np.isfinite(uu)
    assert uu > 0.0
    # Project p onto edge running from x1 to x2
    q = (np.dot(u, p - x1) / uu) * u + x1
    a = q - x2
    b = q - x1
    aa = np.dot(a, a)
    bb = np.dot(b, b)
    assert np.isfinite(aa)
    assert np.isfinite(bb)
    w1 = np.sqrt(aa / uu) if np.dot(u, a) <= 0.0 else -np.sqrt(aa / uu)
    w2 = np.sqrt(bb / uu) if np.dot(u, b) >= 0.0 else -np.sqrt(bb / uu)
    assert np.isfinite(w1)
    assert np.isfinite(w2)
    return w1, w2


def compute_barycentric_triangle(x1, x2, x3, p):
    """
    This method computes the barycentric coordinates for a point p of a triangle given by the points x1, x2, and x3
    (in counter clockwise order).

    The barycentric coordinates w1, w2, and w3 are defined such that p' = w1*x1 + w2*x2 + w3*x3, is the point in
    plane of the triangle closest to p.

    if 0 <= w1,w2,w3 <= 1 then the point lies inside or on the perimeter of the triangle.

    Warning  This method uses a geometric approach to compute the barycentric coordinates.

    :param x1: The first point of the triangle.
    :param x2: The second point of the triangle.
    :param x3: The third point of the triangle.
    :param p:  The point for which the barycentric coordinates should be computed.
    :return:   A triplet containing the first, second, and third barycentric coordinates w1, w2 and w3.
    """
    # Before computing the barycentric coordinates, we 'normalize' the triangle to ensure it is not a sliver
    b1 = x2 - x1
    b2 = x3 - x1
    b3 = np.cross(b1 / np.linalg.norm(b1), b2 / np.linalg.norm(b2))
    basis = M3.make_from_cols(b1, b2, b3)
    # q is the point p transformed to the isoparametric system
    q = np.linalg.solve(basis, p - x1)
    w1 = 1.0 - q[0] - q[1]
    w2 = q[0]
    w3 = q[1]
    assert np.isfinite(w1)
    assert np.isfinite(w2)
    assert np.isfinite(w3)
    return w1, w2, w3


def compute_barycentric_tetrahedron(x1, x2, x3, x4, p):
    """
    This method computes the barycentric coordinates for a point p of a tetrahedron
    given by the points x1,x2,x3, x4 (in right-hand order).

    :param x1: The first point of the tetrahedron.
    :param x2: The second point of the tetrahedron.
    :param x3: The third point of the tetrahedron.
    :param x4: The fourth point of the tetrahedron.
    :param p:  The point for which the barycentric coordinates should be computed.
    :return:   A quadruplet containing the first, second, third  abd fourth barycentric coordinates w1, w2, w3 and w4.
    """
    b1 = x2 - x1
    b2 = x3 - x1
    b3 = x4 - x1
    w1, w2, w3, w4 = np.Inf, np.Inf, np.Inf, np.Inf
    # check if right handed basis if not we exchange two base vectors
    if np.dot(b3, np.cross(b1, b2)) > 0:
        basis = M3.make_from_cols(b1, b2, b3)
        # q is the point p transformed to the iso-parametric system
        q = np.linalg.solve(basis, p - x1)
        w1 = 1.0 - q[0] - q[1] - q[2]
        w2 = q[0]
        w3 = q[1]
        w4 = q[2]
    else:
        basis = M3.make_from_cols(b1, b3, b2)
        # q is the point p transformed to the iso-parametric system
        q = np.linalg.solve(basis, p - x1)
        w1 = 1.0 - q[0] - q[1] - q[2]
        w2 = q[0]
        w3 = q[
            2
        ]  # We alter the sequence of barycentric coordinates to match altered basis
        w4 = q[1]
    assert np.isfinite(w1)
    assert np.isfinite(w2)
    assert np.isfinite(w3)
    assert np.isfinite(w4)
    return w1, w2, w3, w4


def compute_barycentric_derivatives_tetrahedron(x1, x2, x3, x4):
    """
    Computes the spatial derivative of the barycentric coordinates of a tetrahedron.

    :param x1: The first point of the tetrahedron.
    :param x2: The second point of the tetrahedron.
    :param x3: The third point of the tetrahedron.
    :param x4: The fourth point of the tetrahedron.
    """
    vol6 = np.dot(x4 - x1, np.cross(x2 - x1, x3 - x1))
    assert vol6 > 0.0
    grad_w1 = np.cross(x4 - x2, x3 - x2) / vol6
    grad_w2 = np.cross(x3 - x1, x4 - x1) / vol6
    grad_w3 = np.cross(x4 - x1, x2 - x1) / vol6
    grad_w4 = np.cross(x2 - x1, x3 - x1) / vol6
    return grad_w1, grad_w2, grad_w3, grad_w4


def is_inside_edge(w1, w2) -> bool:
    """
    Test of barycentric coordinates of an edge corresponds to a point inside the edge.

    :param w1: First barycentric coordinate.
    :param w2: Second barycentric coordinate.
    :return: True if inside otherwise false.
    """
    assert np.isfinite(w1)
    assert np.isfinite(w2)
    if w1 < 0.0:
        return False
    if w2 < 0.0:
        return False
    if w1 > 1.0:
        return False
    if w2 > 1.0:
        return False
    if w1 + w2 > 1.0:
        return False
    if w1 + w2 < 0.0:
        return False
    return True


def is_inside_triangle(w1, w2, w3) -> bool:
    """
    Test of barycentric coordinates of an triangle corresponds to a point inside the triangle.

    :param w1: First barycentric coordinate.
    :param w2: Second barycentric coordinate.
    :param w3: Third barycentric coordinate.
    :return: True if inside otherwise false.
    """
    assert np.isfinite(w1)
    assert np.isfinite(w2)
    assert np.isfinite(w3)
    if w1 < 0.0:
        return False
    if w2 < 0.0:
        return False
    if w3 < 0.0:
        return False
    if w1 > 1.0:
        return False
    if w2 > 1.0:
        return False
    if w3 > 1.0:
        return False
    if w1 + w2 + w3 > 1.0:
        return False
    if w1 + w2 + w3 < 0.0:
        return False
    return True


def is_inside_tetrahedron(w1, w2, w3, w4) -> bool:
    """
    Test of barycentric coordinates of an tetrahedron corresponds to a point inside the tetrahedron.

    :param w1: First barycentric coordinate.
    :param w2: Second barycentric coordinate.
    :param w3: Third barycentric coordinate.
    :param w4: Fourth barycentric coordinate.
    :return: True if inside otherwise false.
    """
    assert np.isfinite(w1)
    assert np.isfinite(w2)
    assert np.isfinite(w3)
    assert np.isfinite(w4)
    if w1 < 0.0:
        return False
    if w2 < 0.0:
        return False
    if w3 < 0.0:
        return False
    if w4 < 0.0:
        return False
    if w1 > 1.0:
        return False
    if w2 > 1.0:
        return False
    if w3 > 1.0:
        return False
    if w4 > 1.0:
        return False
    if w1 + w2 + w3 + w4 > 1.0:
        return False
    if w1 + w2 + w3 + w4 < 0.0:
        return False
    return True
