import numpy as np
import rainbow.math.quaternion as Q
import rainbow.math.vector3 as V3


class Mesh:
    """
    This class is a convenience wrapper. It simply bundles a vertex and face
    array into one data structure.
    """

    def __init__(self, V, T):
        self.V = np.copy(V)  # Vertices of the mesh
        self.T = np.copy(T)  # Triangles of the mesh


def translate(mesh, r):
    mesh.V[:, :] = mesh.V[:, :] + r
    return mesh


def rotate(mesh, q):
    mesh.V = Q.rotate_array(q, mesh.V)
    return mesh


def scale(mesh, a, b, c):
    s = np.array((a, b, c), dtype=np.float64)
    for i in range(len(mesh.V)):
        mesh.V[i, :] = np.multiply(mesh.V[i, :], s)
    return mesh


def aabb(mesh):
    lower_corner = np.amin(mesh.V, axis=0)
    upper_corner = np.amax(mesh.V, axis=0)
    return lower_corner, upper_corner


def scale_to_unit(mesh):
    """
    This specialized scaling function uniformly scales the mesh around the center point of its enclosing
    axis aligned bounding box such that the shape will be inside a unit-box.

    :param mesh:
    :return:
    """
    (l, u) = aabb(mesh)
    center = (l + u) / 2.0
    translate(mesh, -center)
    s = 1.0 / np.max(u - l)
    scale(mesh, s, s, s)
    translate(mesh, center)


def axis_sort(mesh, axis):
    """
    Sort the vertices of the specified mesh based on vertex location in relation to the given axis direction.

    Note: The implementation also work on a volume mesh.

    :param mesh:       The mesh that should be re-sorted
    :param axis:       The axis direction along which to do the sort.
    :return:
    """
    n = V3.unit(axis)
    d = np.zeros(len(mesh.V))
    for i in range(len(mesh.V)):
        d[i] = np.dot(mesh.V[i, :], n)
    new2old = sorted(range(len(mesh.V)), key=lambda k: d[k])
    mesh.V = mesh.V[new2old]

    # Second update the references in T
    old2new = sorted(range(len(d)), key=lambda k: new2old[k])
    for t in range(len(mesh.T)):
        for v in range(len(mesh.T[t])):
            mesh.T[t, v] = old2new[mesh.T[t, v]]

    # Third sort the triangles based on the vertices it consists of
    triangle_mean = np.mean(mesh.T, axis=1)
    new2old = sorted(range(len(mesh.T)), key=lambda k: triangle_mean[k])
    mesh.T = mesh.T[new2old]


def profile_sweep(profile, slices):
    N = len(profile)
    J = slices

    if N <= 2:
        raise RuntimeError("profile_sweep(): Profile must have at least 3 points")

    if J <= 2:
        raise RuntimeError(
            "profile_sweep(): Sweep must have at least 3 slices to be a proper volume."
        )

    K = (N - 2) * J + 2  # Total number of vertices
    bottom = K - 1  # Index to bottom vertex
    top = K - 2  # Index to top vertex
    H = N - 2  # Number of latitude circles
    F = 2 * J * (N - 2)  # Total number of triangle faces

    V = np.zeros((K, 3), dtype=np.float64)
    T = np.zeros((F, 3), dtype=int)

    # Make a 2D grid of vertices by sweeping profile around y-axis
    dtheta = 2.0 * np.pi / J  # The angle of each slice

    v = 0
    for j in range(J):
        theta = j * dtheta
        R = Q.Ru(theta, V3.j())
        for i in range(H):
            V[v, :] = Q.rotate(R, profile[i + 1])
            v = v + 1

    # Now fill in top and bottom vertices
    V[v, :] = profile[N - 1]
    v = v + 1

    V[v, :] = profile[0]

    # Make faces for bottom-ring
    f = 0
    for j in range(J):
        #
        #  V  = {c1} {c2} ... {cJ}  b t
        #
        # b c1.0 c2.0 | b c2.0 c3.0| ... | b cJ.0 c1.0
        # b 0 (N-2) | b (N-2) 2*(N-2)
        #
        left = j
        right = (j + 1) % J
        vi = bottom
        vj = H * right
        vk = H * left
        T[f, :] = (vi, vj, vk)
        f = f + 1

    # Make faces for middle-rings
    for i in range(H - 1):  # ring number
        for j in range(J):  # slice number
            left = j
            right = (j + 1) % J
            up = i + 1
            down = i

            vi = left * H + down
            vj = right * H + down
            vk = right * H + up
            vm = left * H + up

            T[f, :] = (vi, vj, vk)
            f = f + 1
            T[f, :] = (vi, vk, vm)
            f = f + 1

    # Make faces for top - ring
    for j in range(J):
        offset = N - 3
        left = j
        right = (j + 1) % J

        vi = top
        vj = left * H + offset
        vk = right * H + offset

        T[f, :] = (vi, vj, vk)
        f = f + 1

    return V, T


def create_cylinder(radius, height, slices=12, segments=1):
    profile = [V3.zero() for _ in range(segments + 3)]
    profile[0] = V3.make(0.0, -height / 2.0, 0.0)
    profile[1] = V3.make(radius, -height / 2.0, 0.0)
    dh = height / segments
    for i in range(segments):
        h = dh * (i + 1) - height / 2.0
        profile[i + 2] = V3.make(radius, h, 0.0)
    profile[-1] = V3.make(0.0, height / 2.0, 0.0)
    # profile = [
    #    V3.make(0.0, -height/2.0, 0.0),
    #    V3.make(radius, -height/2.0, 0.0),
    #    V3.make(radius, height/2.0, 0.0),
    #    V3.make(0.0, height/2.0, 0.0)
    # ]
    return profile_sweep(profile, slices)


def create_cone(radius, height, slices=12):
    profile = [
        V3.make(0.0, 0.0, 0.0),
        V3.make(radius, 0.0, 0.0),
        V3.make(0.0, height, 0.0),
    ]
    return profile_sweep(profile, slices)


def create_conical(bottom_radius, top_radius, height, slices=12):
    profile = [
        V3.make(0.0, 0.0, 0.0),
        V3.make(bottom_radius, 0.0, 0.0),
        V3.make(top_radius, height, 0.0),
        V3.make(0.0, height, 0.0),
    ]
    return profile_sweep(profile, slices)


def create_sphere(radius, slices=12, segments=12):
    profile = [V3.zero() for _ in range(segments)]
    dtheta = np.pi / (segments - 1)
    for i in range(segments):
        theta = dtheta * i
        R = Q.Ru(theta, V3.k())
        profile[i] = Q.rotate(R, V3.make(0.0, -radius, 0.0))
    return profile_sweep(profile, slices)


def create_ellipsoid(a, b, c, slices=12, segments=12):
    V, T = create_sphere(1.0, slices, segments)
    s = np.array((a, b, c), dtype=np.float64)
    for i in range(len(V)):
        V[i, :] = np.multiply(V[i, :], s)
    return V, T


def create_capsule(radius, height, slices=12, segments=12):
    profile = [V3.zero() for _ in range(segments)]
    dtheta = np.pi / (segments - 1)
    for i in range(segments):
        theta = dtheta * i
        R = Q.Ru(theta, V3.k())
        dh = -height / 2.0 if i < (segments / 2) else height / 2.0
        profile[i] = Q.rotate(R, V3.make(0.0, -radius, 0.0)) + V3.make(0.0, dh, 0.0)
    return profile_sweep(profile, slices)


def create_cuboid(p0, p1, p2, p3, p4, p5, p6, p7):
    """
    Creates a cuboid like object. The ccw-order front face is given by [p0 p1 p2 p3], the ccw-order back-face is given
     by [p4, p7, p6, p5]

    :param p0:
    :param p1:
    :param p2:
    :param p3:
    :param p4:
    :param p5:
    :param p6:
    :param p7:
    :return:
    """
    V = np.zeros((8, 3), dtype=np.float64)
    T = np.zeros((12, 3), dtype=int)

    V[0, :] = p0
    V[1, :] = p1
    V[2, :] = p2
    V[3, :] = p3
    V[4, :] = p4
    V[5, :] = p5
    V[6, :] = p6
    V[7, :] = p7

    quads = [
        [0, 1, 2, 3],  # front face
        [4, 7, 6, 5],  # back face
        [4, 0, 3, 7],  # left face
        [1, 5, 6, 2],  # right face
        [7, 3, 2, 6],  # top face
        [5, 1, 0, 4],  # bottom face
    ]

    for i in range(6):
        vi = quads[i][0]
        vj = quads[i][1]
        vk = quads[i][2]
        vm = quads[i][3]
        T[i * 2 + 0, :] = (vi, vj, vk)
        T[i * 2 + 1, :] = (vi, vk, vm)
    return V, T


def create_box(width, height, depth):
    V = np.zeros((8, 3), dtype=np.float64)
    T = np.zeros((12, 3), dtype=int)

    V[0, :] = V3.make(-width, -height, -depth) * 0.5
    V[1, :] = V3.make(width, -height, -depth) * 0.5
    V[3, :] = V3.make(width, height, -depth) * 0.5
    V[2, :] = V3.make(-width, height, -depth) * 0.5

    V[4, :] = V3.make(-width, -height, depth) * 0.5
    V[5, :] = V3.make(width, -height, depth) * 0.5
    V[7, :] = V3.make(width, height, depth) * 0.5
    V[6, :] = V3.make(-width, height, depth) * 0.5
    #
    #          6                        7
    #
    #    4                        5
    #
    #
    #
    #
    #          2                        3
    #
    #    0                        1
    #
    # Bottom
    T[0, :] = (0, 2, 3)
    T[1, :] = (0, 3, 1)
    # Top
    T[2, :] = (4, 5, 7)
    T[3, :] = (4, 7, 6)
    # Right
    T[4, :] = (1, 3, 7)
    T[5, :] = (1, 7, 5)
    # Left
    T[6, :] = (0, 4, 6)
    T[7, :] = (0, 6, 2)
    # front
    T[8, :] = (0, 1, 5)
    T[9, :] = (0, 5, 4)
    # back
    T[10, :] = (3, 2, 6)
    T[11, :] = (3, 6, 7)
    return V, T


def create_convex_hull(points):
    """

    :param points:
    :return:
    """
    # TODO 2021-05-08 Kenny: Replace with libigl functionality when available in python.
    from pyhull.convex_hull import ConvexHull

    H = ConvexHull(points)
    N = len(H.points)  # Number of vertices
    K = len(H.vertices)  # Number of triangles
    V = np.zeros((N, 3), dtype=np.float64)
    T = np.zeros((K, 3), dtype=int)
    for idx, p in enumerate(H.points):
        V[idx, :] = p
    for idx, v in enumerate(H.vertices):
        vi = v[0]
        vj = v[1]
        vk = v[2]
        T[idx, :] = (vi, vk, vj)
    return V, T
