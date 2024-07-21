import numpy as np
import rainbow.math.vector3 as V3
import rainbow.math.matrix3 as M3


def compute_signed_volume(pi, pj, pk, pm):
    """
    This function computes the signed volume of the tetrahedron given
    by the four points.

    :param pi:  Coordinates of first point.
    :param pj:  Coordinates of second point.
    :param pk:  Coordinates of third point.
    :param pm:  Coordinates of fourth point.
    :return:    The signed volume of the tetrahedron.
    """
    return np.dot((pm - pi), V3.cross((pj - pi), (pk - pi))) / 6.0


def is_left_orientation(pi, pj, pk, pm):
    """
    This function tests if the tetrahedron given by the four points has left orientation.

    :param pi:  Coordinates of first point.
    :param pj:  Coordinates of second point.
    :param pk:  Coordinates of third point.
    :param pm:  Coordinates of fourth point.
    :return:     True if tetrahedron was left oriented.
    """
    return compute_signed_volume(pi, pj, pk, pm) < 0.0


def compute_circumscribed_sphere(p0, p1, p2, p3):
    """
    Compute the circle (in 3D) containing p0, p1, and p2.  The center in
    barycentric coordinates is

        C = u0*p0 + u1*p1 + u2*p2 + U3*P3

    where u0+u1+u2+U3=1.

    The center is equidistant from the FOUR points, so

        |C-p0| = |C-p1| = |C-p2| = |C-p3| = R

    where R is the radius of the circle.

    From these conditions,

        C-p0 = u0*A + u1*B + u2*C - A
        C-p1 = u0*A + u1*B + u2*C - B
        C-p2 = u0*A + u1*B + u2*C - C
        C-p3 = u0*A + u1*B + u2*C

     where A = p0 - p2, B = p1 - p2, and C = p2 - p3 which leads to

        r^2 = |u0*A + u1*B + u2*C|^2 - 2*dot(A,u0*A + u1*B + u2*C) + |A|^2
        r^2 = |u0*A + u1*B + u2*C|^2 - 2*dot(B,u0*A + u1*B + u2*C) + |B|^2
        r^2 = |u0*A + u1*B + u2*C|^2 - 2*dot(B,u0*A + u1*B + u2*C) + |C|^2
        r^2 = |u0*A + u1*B + u2*C|^2

    Subtracting the last equation from the first two and writing
    the equations as a linear system,

        +-                          -++   -+       +-        -+
        | Dot(A,A) Dot(A,B) Dot(A,C) || u0 | = 0.5 | Dot(A,A) |
        | Dot(B,A) Dot(B,B) Dot(B,C) || u1 |       | Dot(B,B) |
        | Dot(C,A) Dot(C,B) Dot(C,C) || u2 |       | Dot(C,C) |
        +-                          -++   -+       +-        -+

                      M                 u                q

     The following code solves this system for u0, u1 and u2, then
     evaluates the third equation in r^2 to obtain r.

    :param p0:
    :param p1:
    :param p2:
    :param p3:
    :return:
    """
    A = p0 - p3
    B = p1 - p3
    C = p2 - p3

    M00 = np.dot(A, A)
    M01 = np.dot(A, B)
    M02 = np.dot(A, C)
    M10 = np.dot(B, A)
    M11 = np.dot(B, B)
    M12 = np.dot(B, C)
    M20 = np.dot(C, A)
    M21 = np.dot(C, B)
    M22 = np.dot(C, C)

    M = M3.make(M00, M01, M02, M10, M11, M12, M20, M21, M22)
    q = 0.5 * V3.make(M00, M11, M22)

    u = np.linalg.lstsq(M, q, rcond=None)[0]

    u0 = u[0]
    u1 = u[1]
    u2 = u[2]
    u3 = 1.0 - u0 - u1 - u2

    radius = V3.norm(u0 * A + u1 * B + u2 * C)
    center = u0 * p0 + u1 * p1 + u2 * p2 + u3 * p3

    return center, radius


def compute_inscribed_sphere(pi, pj, pk, pm):
    """
    The center of the inscreibed sphere is equidistant to each of the face
    planes of the tetrahedron, and the distance is the radius of the inscribed
    sphere. Let the center and radius of the inscreibed sphere be denoted by c and r.
    Let the plane consisting of points j,k, and m be denoted by i then

        $$n_i * c - w_i = r$$

    where

        $$n_i = \frac{(p_m-p_j)\times(p_k-p_j)}{\norm{(p_m-p_j)\times(p_k-p_j)}}$$
        $$w_i = n_i \cdot p_j$$

    Similar we can setup three equations for the planes j,k and m (by permuting).
    This results in the system of linear equations:

        $$n_{m,x}  c_x + n_{m,y}  c_y + n_{m,z}  c_z - w_m = r$$
        $$n_{i,x}  c_x + n_{i,y}  c_y + n_{i,z}  c_z - w_i = r$$
        $$n_{j,x}  c_x + n_{j,y}  c_y + n_{j,z}  c_z - w_j = r$$
        $$n_{k,x}  c_x + n_{k,y}  c_y + n_{k,z}  c_z - w_k = r$$

    Or in as a matrix equation

        $$| n_{m,x}   n_{m,y}   n_{m,z}  -1 | | c_x | = | w_m |$$
        $$| n_{i,x}   n_{i,y}   n_{i,z}  -1 | | c_y | = | w_i |$$
        $$| n_{j,x}   n_{j,y}   n_{j,z}  -1 | | c_z | = | w_j |$$
        $$| n_{k,x}   n_{k,y}   n_{k,z}  -1 | |  r  | = | w_k |$$

          $$        A                           x     = b$$

    This is four equations with four unknowns and can be solved by inversion of the A-matrix.

    :param pi:
    :param pj:
    :param pk:
    :param pm:
    :return:
    """

    # triangle face normals
    nm = V3.unit(np.cross((pj - pi), (pk - pi)))
    ni = V3.unit(np.cross((pm - pj), (pk - pj)))
    nj = V3.unit(np.cross((pm - pk), (pi - pk)))
    nk = V3.unit(np.cross((pm - pi), (pj - pi)))

    # triangle plane offsets
    wm = np.dot(nm, pi)
    wi = np.dot(ni, pj)
    wj = np.dot(nj, pk)
    wk = np.dot(nk, pi)

    M00 = nm[0]
    M01 = nm[1]
    M02 = nm[2]
    M03 = -1.0

    M10 = ni[0]
    M11 = ni[1]
    M12 = ni[2]
    M13 = -1.0

    M20 = nj[0]
    M21 = nj[1]
    M22 = nj[2]
    M23 = -1.0

    M30 = nk[0]
    M31 = nk[1]
    M32 = nk[2]
    M33 = -1.0

    A = np.array(
        [
            [M00, M01, M02, M03],
            [M10, M11, M12, M13],
            [M20, M21, M22, M23],
            [M30, M31, M32, M33],
        ],
        dtype=np.float64,
    )
    b = np.array([wm, wi, wj, wk], dtype=np.float64)

    x = np.linalg.lstsq(A, b, rcond=None)[0]

    center = V3.make(x[0], x[1], x[2])
    radius = x[3]

    return center, radius


def generate_sample_points(N):
    """
    This function uses the "pattern" of the Pascal pyramid
    to generate sample points for a tetrahedron. The parameter N
    is not the number of sample points, but the number of layers/planes
    used in the Pascal pyramid. When using 2 planes then the pyramid is
    a tetrahedron. The function returns the sample points in barycentric
    coordinates.

    :param N: Number of layers in Pascal pyramid
    :return: Sample points in bary-centric coordinates of the
             iso-parametric tetrahedron.
    """
    S = []
    if N < 2:
        raise ValueError("N must be larger than or equal to 2")
    for m in range(N):
        for k in range(N - m):
            for j in range(N - k - m):
                i = N - j - k - m - 1
                wi = i / (N - 1)
                wj = j / (N - 1)
                wk = k / (N - 1)
                wm = m / (N - 1)
                S.append([wi, wj, wk, wm])
    return np.array(S, dtype=np.float64)


def generate_interior_sample_points(N):
    """
    This function works in the same way as the generate_sample_points(N) function.
    The only difference is that all sample points that lie on the surface
    of the tetrahedron is filtered away. Hence, only strictly interior sample
    points are kept.

    :param N: Number of layers in Pascal pyramid
    :return: Interior sample points in bary-centric coordinates of the
             iso-parametric tetrahedron.
    """
    S = generate_sample_points(N)
    return S[np.count_nonzero(S, axis=1) == 3]
