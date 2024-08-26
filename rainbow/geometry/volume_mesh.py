import numpy as np
import rainbow.geometry.grid3 as GRID


class T4Mesh:
    """
    This class bundles together arrays of related vertices and tetrahedra.
    """

    def __init__(self, V, T):
        self.V = np.copy(V)  # Vertices of the mesh
        self.T = np.copy(T)  # Tetrahedra of the mesh


def create_tetrahedron_array_from_cube_mesh(I, J, K):
    """
    Generates a tetrahedron array  that fills a cubic cell block structure with I,J,K nodes. The fill pattern is
    generated to minimize bias by flipping the fill pattern.

    :param I: Number of nodes along the x-axis
    :param J: Number of nodes along the y-axis
    :param K: Number of nodes along the z-axis
    :return: A 5(I-1)(J-1)(K-1)-by-4 array. Each row corresponds to one tetrahedon and columns correspond to nodal
             indices of that tetrahedron.
    """
    N = (I - 1) * (J - 1) * (K - 1) * 5
    T = np.zeros((N, 4), dtype=np.int32)
    n = 0
    for k in range(K - 1):
        for j in range(J - 1):
            for i in range(I - 1):
                i000 = (k * J + j) * I + i
                i001 = (k * J + j) * I + (i + 1)
                i010 = (k * J + (j + 1)) * I + i
                i011 = (k * J + (j + 1)) * I + (i + 1)
                i100 = ((k + 1) * J + j) * I + i
                i101 = ((k + 1) * J + j) * I + (i + 1)
                i110 = ((k + 1) * J + (j + 1)) * I + i
                i111 = ((k + 1) * J + (j + 1)) * I + (i + 1)
                #
                #       i110 *----------------------* i111
                #           /|                     /|
                #          / |                    / |
                #         /  |                   /  |
                #        /   |                  /   |
                #       /    |                 /    |
                # i100 *----------------------* i101|
                #      |i010 *----------------|-----*  i011
                #      |    /                 |     /
                #      |   /                  |    /
                #      |  /                   |   /
                #      | /                    |  /
                #      |/                     | /
                #      *----------------------*
                #    i000                   i001
                #
                flip = (i + j + k) % 2 == 1
                if flip:
                    T[n, :] = (i000, i001, i010, i100)
                    T[n + 1, :] = (i010, i001, i011, i111)
                    T[n + 2, :] = (i100, i110, i111, i010)
                    T[n + 3, :] = (i100, i111, i101, i001)
                    T[n + 4, :] = (i010, i111, i100, i001)
                else:
                    T[n, :] = (i000, i001, i011, i101)
                    T[n + 1, :] = (i000, i011, i010, i110)
                    T[n + 2, :] = (i100, i110, i101, i000)
                    T[n + 3, :] = (i101, i110, i111, i011)
                    T[n + 4, :] = (i000, i011, i110, i101)
                n += 5
    return T


def create_beam(I, J, K, width, height, depth):
    """
    This function creates a tetrahedral mesh that resembles a beam structure.

    :param I:       Number of nodes along the x-axis
    :param J:       Number of nodes along the y-axis
    :param K:       Number of nodes along the z-axis
    :param width:   The width of the beam
    :param height:  The height of the beam
    :param depth:   The depth of the beam
    :return:        The resulting tetrahedral mesh as a V-array and a T-array.
    """
    x0 = -width / 2.0
    y0 = -height / 2.0
    z0 = -depth / 2.0

    dx = width / float(I - 1)
    dy = height / float(J - 1)
    dz = depth / float(K - 1)

    V = np.zeros((I * J * K, 3), dtype=np.float64)
    for k in range(K):
        for j in range(J):
            for i in range(I):
                m = i + j * I + k * I * J
                V[m, 0] = x0 + i * dx
                V[m, 1] = y0 + j * dy
                V[m, 2] = z0 + k * dz

    T = create_tetrahedron_array_from_cube_mesh(I, J, K)

    return V, T


def from_grid3d(grid):
    """
    This function converts a grid into an equivalent tetrahedral mesh. The
    vertices of the mesh will be collocated with the grid nodes and the grid
    values will be returned as a vertex attribute array.

    :param grid:   The grid that should be converted to a tetrahedral mesh.
    :return:       The corresponding tetrahedral mesh as a V-array and T-array, together with attribute array A.
    """
    V = np.copy(GRID.get_nodes_array(grid))
    T = create_tetrahedron_array_from_cube_mesh(grid.I, grid.J, grid.K)
    A = np.copy(grid.values)
    return V, T, A


def to_triangles(T):
    """
    Transforms the tetrahedron mesh to a triangle mesh

    :param T:     The tetrahedrons from the tetrahedral mesh as a T-array.
    :return:      An array of face-triangles of each tetrahedron.
    """
    # TODO 2021-03-25 kenny: This function exists only as a quick fix for the renderer. If a solution for rendering
    #  tetrahedrons comes up, this should be deleted.
    F = np.zeros((len(T) * 4, 3), dtype=int)
    for i in range(len(T)):
        t = T[i]
        F[i * 4 + 0] = (t[1], t[0], t[2])
        F[i * 4 + 1] = (t[0], t[1], t[3])
        F[i * 4 + 2] = (t[1], t[2], t[3])
        F[i * 4 + 3] = (t[2], t[0], t[3])
    return F


def to_tetrahedrons(Vin, T, scale=0.9):
    """
    Converts a tetra mesh to a collection of surface tetrahedron meshes. This is useful for visualizing
    tetra mesh with "air" between the tetrahedrons.

    :param Vin:    3D input nodal points of tetra mesh.
    :param T:      Tetrahedrons array of tetra mesh
    :param scale:  A down-scale parameter that is ued to make the individual tetrahedronss smaller.
    :return:       A surface mesh V and F arrays that represent the tetra mesh.
    """
    F = np.zeros((len(T) * 4, 3), dtype=int)
    Vout = np.zeros((len(T) * 4, 3), dtype=np.float64)
    for idx in range(len(T)):
        pi = Vin[T[idx, 0]]
        pj = Vin[T[idx, 1]]
        pk = Vin[T[idx, 2]]
        pm = Vin[T[idx, 3]]
        c = (pi + pj + pk + pm) / 4.0
        i = idx * 4 + 0
        j = idx * 4 + 1
        k = idx * 4 + 2
        m = idx * 4 + 3
        Vout[i] = c + (pi - c) * scale
        Vout[j] = c + (pj - c) * scale
        Vout[k] = c + (pk - c) * scale
        Vout[m] = c + (pm - c) * scale
        F[i] = (j, i, k)
        F[j] = (i, j, m)
        F[k] = (j, k, m)
        F[m] = (k, i, m)
    return Vout, F


def create_loop_subdivision(V, T):
    """
    This function implements a Loop inspired sub-division of the tetrahedral mesh. The
     subdivision is based on splitting all edges into two halves.

    :param V:   The input vertex coordinates of the tetrahedral mesh.
    :param T:   The input tetrahedrons.
    :return:    The output is the sub-divived mesh as a V-array and T-array.
    """

    def create_midpoint(_i, _j, _pi, _pj, _lut, _offset):
        key = (_i, _j) if _i < _j else (_j, _i)
        if key in _lut:
            idx, p = _lut[key]
        else:
            p = (_pi + _pj) / 2
            idx = len(_lut)
            _lut[key] = (idx, p)
        return idx + _offset, p

    offset = len(V)
    lut = {}
    K = len(T)
    T = np.concatenate((T, np.zeros((7 * K, 4), dtype=np.int64)), axis=0)
    last = K
    for e in range(K):
        i, j, k, m = T[e]
        pi = V[i]
        pj = V[j]
        pk = V[k]
        pm = V[m]
        ij, pij = create_midpoint(i, j, pi, pj, lut, offset)
        ik, pik = create_midpoint(i, k, pi, pk, lut, offset)
        im, pim = create_midpoint(i, m, pi, pm, lut, offset)
        jk, pjk = create_midpoint(j, k, pj, pk, lut, offset)
        jm, pjm = create_midpoint(j, m, pj, pm, lut, offset)
        km, pkm = create_midpoint(k, m, pk, pm, lut, offset)
        T[e] = [im, jm, km, m]
        T[last] = [ij, ik, im, i]
        T[last + 1] = [ij, jk, jm, j]
        T[last + 2] = [ik, km, jk, k]
        T[last + 3] = [ik, im, ij, jm]
        T[last + 4] = [ik, ij, jk, jm]
        T[last + 5] = [ik, jk, km, jm]
        T[last + 6] = [ik, km, im, jm]
        last = last + 7
    C = len(lut)
    P = np.zeros((C, 3), dtype=np.float64)
    for _, value in lut.items():
        P[value[0]] = value[1]
    V = np.concatenate((V, P), axis=0)
    return V, T


def compute_volumes(V, T):
    """
    Computes the signed volumes of a given set of tetrahedra with given coordinates.

    :param V:    The vertex coordinates to use for the volume computation.
    :param T:    The tetrahedron array.
    :return:     A array of tetrahedron signed volumues.
    """
    vol = np.zeros(len(T))
    for e in range(len(T)):
        r = V[T[e]]
        u_ji = r[1] - r[0]
        u_mi = r[3] - r[0]
        u_ki = r[2] - r[0]
        vol[e] = np.dot(u_mi, np.cross(u_ji, u_ki)) / 6
    return vol


def compute_neighbors(T):
    """
    This function computes adjacent tetrahedral information. The result is returned in a Neihbor lookup array, N.

    The N-array works such that if n = N[e,v] then tetrahedron n is adjacent to tetrahedron e. The tetrahedron
    n is sharing the face opposite node v of tetrahedron e.

    Concretely if T[e,:] = [i,j,k,m] and say v = 3 then it will be the triangle face i,j,k that are
    shared between tetrahedra e and n.

    If n is -1 then it means that there are no neighboring tetrahedra on the other side
    of face i,j,k. This means that face i,j,k is a surface boundary face of the tetrahedral mesh.

    :param T:    The tetrahedra of the mesh
    :return:     The neighbor tetrahedra lookup array
    """
    N = -np.ones(T.shape, dtype=np.int32)
    face_info = np.zeros((4 * len(T), 5), dtype=np.int32)
    for e in range(len(T)):
        i, j, k, m = T[e]
        face_info[e * 4 + 0, :] = sorted([j, k, m]) + [e, 0]
        face_info[e * 4 + 1, :] = sorted([i, k, m]) + [e, 1]
        face_info[e * 4 + 2, :] = sorted([i, j, m]) + [e, 2]
        face_info[e * 4 + 3, :] = sorted([i, j, k]) + [e, 3]
    face_info = face_info[
        np.lexsort((face_info[:, 0], face_info[:, 1], face_info[:, 2]))
    ]
    for i in range(4 * len(T) - 1):
        if np.array_equal(face_info[i, 0:3], face_info[i + 1, 0:3]):
            # We have found a shared face in tetrahedral mesh
            elem, elem_opp_node = face_info[i, 3:5]
            twin, twin_opp_node = face_info[i + 1, 3:5]
            N[elem, elem_opp_node] = twin
            N[twin, twin_opp_node] = elem
    return N


def compute_surface(T, N):
    """
    This function extract surface boundary information from a tetrahedral mesh.

    The function returns A S-array that holds a face-index array of the boundary
    surface triangles and an O-array that holds the corresponding index of the
    tetrahedra that owns the surface triangles as well as the local node-index
    that is opposite the tetrahedra triangle. If we have K triangles then S-array
    is dimension K-by-3 and O-array is K-by-2.

    Example, if we have a single tetrahedron then we have 4 surface triangles. Let
    i,j,k = S[s] for the s'th surface triangle. Then if e = O[s,0] this means face
    s is a triangle of tetrahedron e. Now let v = O[s,1] then it means that triangle s
    is oppsite vertex v of tetrahedron e.

    :param T:    The tetrahedral elements of the mesh.
    :param N:    The neighbor tetrahedral lookup array.
    :return:     A S-array of the boundary surface triangles and an O-array of the
                 tetrahedral that owns the surface triangles and opposing vertex indices.
    """
    K = np.count_nonzero(N < 0)  # Number of surface triangles
    S = -np.ones((K, 3), dtype=np.int32)  # Face indexed array of surface triangles
    O = -np.ones(
        (K, 2), dtype=np.int32
    )  # Indexes of the tetrahedral element where a surface triangles comes from.
    s = 0
    for e in range(len(N)):
        i, j, k, m = T[e]
        # We assume that tetrahedron (i,j,k,m) has positive orientation such that we generate faces with
        # positive orientation as well (counter-clock-wise order will give correct outward normals)
        if N[e, 0] == -1:
            # Face (j,k,m) is a surface triangle of e, and is opposite 0th node
            S[s] = (j, k, m)
            O[s] = e, 0
            s += 1
        if N[e, 1] == -1:
            # Face (i,k,m) is a surface triangle of e, and is opposite 1st node
            S[s] = (i, m, k)
            O[s] = e, 1
            s += 1
        if N[e, 2] == -1:
            # Face (i,j,m) is a surface triangle of e, and is opposite 2nd node
            S[s] = (j, m, i)
            O[s] = e, 2
            s += 1
        if N[e, 3] == -1:
            # Face (i,j,k) is a surface triangle of e, and is opposite 3rd node
            S[s] = (j, i, k)
            O[s] = e, 3
            s += 1
    return S, O


class VertexTetrahedronCirculator:
    """
    """

    def __init__(self, V, T):
        """

        :param self:
        :param V:
        :param T:
        :return:
        """
        self.V2T = np.full((4 * len(T), 2), -1, dtype=np.int32)
        for e in range(len(T)):
            self.V2T[4 * e + 0] = (T[e, 0], e)
            self.V2T[4 * e + 1] = (T[e, 1], e)
            self.V2T[4 * e + 2] = (T[e, 2], e)
            self.V2T[4 * e + 3] = (T[e, 3], e)
        self.V2T = self.V2T[np.lexsort((self.V2T[:, 1], self.V2T[:, 0]))]
        self.vertex_start = np.full((len(V),), -1, dtype=np.int32)
        self.vertex_end = np.full((len(V),), -1, dtype=np.int32)
        for idx in range(len(self.V2T)):
            j, f = self.V2T[idx]
            i, e = self.V2T[idx - 1] if idx > 0 else (-1, -1)
            k, g = self.V2T[idx + 1] if idx < len(self.V2T) - 1 else (-1, -1)
            if i != j:
                self.vertex_start[j] = idx
            if j != k:
                self.vertex_end[j] = idx + 1

    def tetrahedra(self, vertex_idx):
        """

        :param self:
        :param vertex_idx:
        :return:
        """
        offset = self.vertex_start[vertex_idx]
        end = self.vertex_end[vertex_idx]
        return self.V2T[offset:end][:, 1]
