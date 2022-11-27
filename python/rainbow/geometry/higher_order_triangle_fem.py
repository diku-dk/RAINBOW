import numpy as np


class TriangleLayout:
    """
    This class defines the nodal layout of a triangle element. We use a Pascal triangle to define the layout.
    """

    @staticmethod
    def binomial(n: int, k: int) -> int:
        """
        This function gives the value of the binomial coefficient B(n,k)

        :param n:   The first integer of the binomial coefficient.
        :param k:   The second integer of the binomial coefficient.
        :return:    The value of B(n,k)
        """
        import math
        if n < 0:
            return 0
        return math.comb(n, k)

    @staticmethod
    def number_of_total_nodes(P: int) -> int:
        return TriangleLayout.binomial(P + 2, 2)

    @staticmethod
    def number_of_corner_nodes(P: int) -> int:
        return 1 if P < 1 else 3

    @staticmethod
    def number_of_edge_nodes(P: int) -> int:
        return TriangleLayout.binomial(P - 1, 1)

    @staticmethod
    def number_of_cell_nodes(P: int) -> int:
        return TriangleLayout.binomial(P - 1, 2)

    @staticmethod
    def compute_ijk_format(P: int) -> np.ndarray:
        """
        This function generates the ijk-format for all the nodes of a P'th order triangle element.

        :param P:  The order of the triangle element.

        :return: An array of IJK formats for all the nodes of the Pascal triangle.
        """
        # First we generate the IJK patterns of all nodes of the Pascal triangle.
        # We do not really consider the ordering in how we store the patterns. We will
        # fix the ordering afterwards.
        patterns = []
        for j in range(P + 1):
            for i in range(P - j + 1):
                k = P - i - j
                patterns.append([i, j, k])
        ijk_format = np.array(patterns, dtype=int)
        # Next we create a mask to help us identify which patterns correspond to a vertex,
        # edge or interior cell node. First we make a binary mask of the IJK format to know
        # which indices are non-zero or zero.
        mask = np.zeros_like(ijk_format)
        mask[ijk_format > 0] = 1
        # Next we count the number of non-zeros in this mask. If there is one non-zero then
        # we know it is a vertex, two means an edge, and 3 means we have an interior cell node.
        # Hence, this code can be used to filter out the IJK patterns based on where the node
        # is located.
        code = np.sum(mask, 1)

        def edge_filter(i: int, j: int, k: int) -> np.ndarray:
            """
            This function helps extract the IJK patterns specific for the edge going from i to j node.
            """
            return np.logical_and(np.logical_and(ijk_format[:, i], ijk_format[:, j]), ijk_format[:, k] == 0)

        def edge_reorder(i: int, j: int, ij_edges: np.ndarray) -> np.ndarray:
            """
            This function helps reorder the edge IJK patterns of the edge
            going from i to j, so the patterns also go from i to j.
            """
            return ij_edges[np.lexsort((ij_edges[:, i], ij_edges[:, j]))]

        def vertex_reorder(vertices: np.ndarray) -> np.ndarray:
            """
            This function reorder IJK patterns of vertices, to have i-vertex first, j second and k third.
            """
            return vertices[np.lexsort((vertices[:, 0], vertices[:, 1], vertices[:, 2]))]

        def cell_reorder(vertices: np.ndarray) -> np.ndarray:
            """
            This function lexigraphic sorts IJK patterns of interior cell
            nodes. First on j-index, then i and finally k-index.
            """
            return vertices[np.lexsort((vertices[:, 2], vertices[:, 0], vertices[:, 1]))]

        # Now we are ready to extract patterns from specific features of the triangle and
        # reorder these patterns in the desired order we wish to have.
        ijk_vertices = vertex_reorder(ijk_format[code == 1])
        ijk_cell = cell_reorder(ijk_format[code == 3])
        ij_edges = edge_reorder(0, 1, ijk_format[edge_filter(0, 1, 2)])
        jk_edges = edge_reorder(1, 2, ijk_format[edge_filter(1, 2, 0)])
        ki_edges = edge_reorder(2, 0, ijk_format[edge_filter(2, 0, 1)])
        # We now have all the individual patterns that we want and can combine them all
        # into one long IJK pattern array, so they are stored in a consecutive manner.
        ijk_format = np.concatenate((ijk_vertices, ijk_cell, ij_edges, jk_edges, ki_edges), axis=0)
        return ijk_format

    @staticmethod
    def get_global_indices(encoding: list[int], P: int) -> list[int]:
        """
        All nodes from a P-order triangle elements can be specified and located using only 10 numbers, always.

        The trick is to assume that interior edge and cell nodes are stored with increasing global index
        values. Hence, one only need to remember the offset index to where cell and edge indices start. One
        also need to remember in which order the nodes should be visited. We turn this orientation, and this
        can be +1 (meaning increasing) or -1 (meaning reverse order).

        Hence, when storing mesh information we often just store a triangular finite element as 10 integer
        values as a single row in a matrix. The encoding format is as follows:

        endocing = [
          i-idx, j-idx, k-idx,
          cell-offset,
          ij-offset, jk-offset, ki-offset, ij-orientation, jk-orientation, ki-orientation
          ]

        :param encoding:   An array/list of 10 integer values encoding the nodes of the triangle element.
        :param P:          The order of the triangle.

        :return:           An list of global node indices corresponding to the triangle element given by the encoding.
        """
        if P < 0:
            raise ValueError("P must be non-negative.")
        indices = []
        # First we add the global vertex indices
        if encoding[0] >= 0:
            indices.append(encoding[0])
        if encoding[1] >= 0:
            indices.append(encoding[1])
        if encoding[2] >= 0:
            indices.append(encoding[2])
        # Next we collect the global indices of all the cell (aka interior) nodes
        if encoding[3] >= 0:
            indices.extend([encoding[3] + i for i in range(TriangleLayout.number_of_cell_nodes(P))])
        # Next we collect the global indices for edge running from node i to node j
        if encoding[4] >= 0:
            ij_edge = [encoding[4] + i for i in range(TriangleLayout.number_of_edge_nodes(P))]
            # We need to make sure that the order of edge-nodes are correct.
            if encoding[7] == -1:
                ij_edge.reverse()
            indices.extend(ij_edge)
        # Next we collect the global indices for edge running from node j to node k
        if encoding[5] >= 0:
            jk_edge = [encoding[5] + i for i in range(TriangleLayout.number_of_edge_nodes(P))]
            # We need to make sure that the order of edge-nodes are correct.
            if encoding[8] == -1:
                jk_edge.reverse()
            indices.extend(jk_edge)
        # Next we collect the global indices for edge running from node k to node i
        if encoding[6] >= 0:
            ki_edge = [encoding[6] + i for i in range(TriangleLayout.number_of_edge_nodes(P))]
            # We need to make sure that the order of edge-nodes are correct.
            if encoding[9] == -1:
                ki_edge.reverse()
            indices.extend(ki_edge)
        return indices


class TriangleShapeFunction:
    """
    This class implement a Lagrangian shape function for a triangular element.

    The magic behind this implementation is that the shape function is evaluated using
    barycentric coordinates. Hence, in 2D we have 3-barycentric coordinates.
    """

    def __init__(self, IJK):
        """
        Initializes the shape function for the node of the triangular element represented by the given IJK format.

        :param IJK:  The IJK format that identifies the node corresponding to this
                     Lagrange shape function. The IJK format are an array of 3 integers. The integers
                     indentify the location of the shape function inside a triangle.
        """
        if IJK[0] < 0:
            raise ValueError("I must be non-negative.")
        if IJK[1] < 0:
            raise ValueError("J must be non-negative.")
        if IJK[2] < 0:
            raise ValueError("K must be non-negative.")
        self.I = IJK[0]
        self.J = IJK[1]
        self.K = IJK[2]
        self.P = self.I + self.J + self.K  # The order of the shape function.

    @staticmethod
    def __L(P: int, N: int, w: np.ndarray) -> float:
        """
        This is an auxiliary function that allow us to implement the
        evaluation of the Lagrange shape function in a simple way.

        :param P:   The order of the triangle.
        :param N:   The order corresponding to the barycentric coordinate that we evaluate wrt to.
        :param w:   The barycentric coordinate at which the Lagrangian shape function is evaluated.
        :return:    The value of the Lagrange shape function of order M at the bary-centric coordinate w.
        """
        value = 1.0
        for n in range(N):
            value *= (P * w - n) / (n + 1.0)
        return value

    def value(self, w) -> float:
        """
        Evaluate the value of the shape function at the given barycentric coordinate.

        :param w:    Barycentric coordinate of the point we wish to evaluate the shape function for.
        :return:     The value of the shape function of IJK node at barycentric position w.
        """
        return TriangleShapeFunction.__L(self.P, self.I, w[0]) * \
               TriangleShapeFunction.__L(self.P, self.J, w[1]) * \
               TriangleShapeFunction.__L(self.P, self.K, w[2])

    def __call__(self, w: np.ndarray) -> float:
        return self.value(w)


class TriangleElement:
    """
    This class creates lookup data structures for an P'th order Pascal triangle. The
    purpose is mostly to make bookkeeping more simple in finite element method codes.
    """

    def __init__(self, P):
        """
        Create an instance of an P'th order ReferenceTriangle.

        :param P: The order of the ReferenceTriangle.
        """
        if P < 1:
            raise ValueError("P must be of positive")
        self.P = P
        self.ijk_format = TriangleLayout.compute_ijk_format(P)
        self.shape_functions = [TriangleShapeFunction(self.ijk_format[i]) for i in range(len(self.ijk_format))]
        self.barycentric = self.ijk_format.astype(dtype=np.float64) / self.P


class TriangleMesh:
    """
    A P'th order triangle mesh.

    A triangle mesh consists of an element type. The element type tells us something about the nodal layout of
    each triangle and what shape function to use.

    Further, the triangle mesh contains the coordinates of all the nodes. This is called the vertices. Individual
    triangles can be representing in one of two ways. Either as a list of the global indices that makes up the
    triangle, or using a compact run-length encoding of the index lists.

    The index representation takes up more storage but is faster, whereas the encoding uses minimal
    storage but takes a few flops to expand when one needs to find the global indices that defines a triangle.
    """

    def __init__(self, P: int):
        """
        Create a 2D FEM mesh of order P.

        :param P:                The order of the triangle mesh.
        """
        if P < 0:
            raise ValueError("P must be of first order or higher.")
        self.P = P  # The order of the triangle.
        self.element = TriangleElement(self.P)  # The triangle element type.
        self.vertices = None  # V-by-3 array of vertex coordinates, assuming V total nodes.
        self.encodings = None  # T-by-10 array of triangle encodings, assuming T triangles.
        self.indices = None  # T-by-N array of global node indices, assuming N nodes per triangle.


def interpolate(U, node_indices, shape_functions, w):
    """
    Interpolate field at barycentric position.

    :param U:                 The field to be interpolated. Essential has a value of every node in the mesh.
    :param node_indices:      Global node indices of the triangle nodes.
    :param shape_functions:   Shape functions of the triangle nodes.
    :param w:                 Barycentric coordinate of the evaluation point.
    :return:                  The value of the U-field at the point w.
    """
    values = np.array([U[k] for k in node_indices], dtype=np.float64)
    weights = np.array([phi.value(w) for phi in shape_functions], dtype=np.float64)
    return np.dot(values.T, weights)


class TriangleMeshFactory:
    """
    This class wraps the functionality needed to create a higher order Triangular finite element mesh.
    """

    @staticmethod
    def make(V, T, P: int, keep_indices: bool):
        """
        The input to this factory function is a linear triangle element mesh, which are defined by its vertex-array V
        and its triangle array T. The output will then be a higher order triangle mesh.

        :param V:                The vertices off a 1st order triangle mesh.
        :param T:                The triangle index array of a 1st order triangle mesh.
        :param P:                The order of the Triangle elements in the mesh.
        :param keep_indices:     A boolean flag telling whether to store global indices.
        :return:                 A P'th order triangle mesh.
        """
        mesh = TriangleMesh(P)
        linear_element = TriangleElement(1)

        lut = {}
        vertices = []
        encodings = []
        global_indices = []

        for idx in range(len(V)):
            coordinates = V[idx]
            vertices.append(coordinates)

        for e in range(len(T)):
            i, j, k = T[e]  # Get global corner indices of input triangle

            cell_offset = TriangleMeshFactory.__create_cell_vertices(vertices, P)

            ij_offset, ij_orientation = TriangleMeshFactory.__create_edge_vertices(i, j, lut, vertices, P)
            jk_offset, jk_orientation = TriangleMeshFactory.__create_edge_vertices(j, k, lut, vertices, P)
            ki_offset, ki_orientation = TriangleMeshFactory.__create_edge_vertices(k, i, lut, vertices, P)

            encoding = [
                i, j, k,
                cell_offset, ij_offset, jk_offset, ki_offset,
                ij_orientation, jk_orientation, ki_orientation
            ]

            encodings.append(encoding)

            indices = TriangleLayout.get_global_indices(encoding, P)
            if keep_indices:
                global_indices.append(indices)

            for local_idx, global_idx in enumerate(global_indices):
                # linear_encoding = [i, j, k, -1, -1, -1, -1, 0, 0, 0]
                # linear_indices = TriangleLayout.get_global_indices(linear_encoding, 1)
                linear_indices = [i, j, k]
                vertices[global_idx] = interpolate(V,
                                                   linear_indices,
                                                   linear_element.shape_functions,
                                                   mesh.element.barycentric[local_idx]
                                                   )

        mesh.vertices = np.array(vertices, dtype=np.float64)
        mesh.encodings = np.array(encodings, dtype=int)
        if keep_indices:
            mesh.indices = np.array(global_indices, dtype=int)

        return mesh

    @staticmethod
    def __key(src_idx: int, dst_idx: int) -> tuple[int, int]:
        """

        :param src_idx:
        :param dst_idx:
        :return:
        """
        return (src_idx, dst_idx) if src_idx < dst_idx else (dst_idx, src_idx)

    @staticmethod
    def __create_edge_vertices(src_idx: int, dst_idx: int, lut, vertices, P: int) -> tuple[int, int]:
        """

        :param src_idx:
        :param dst_idx:
        :param lut:
        :param vertices:
        :param P:
        :return:
        """
        if TriangleLayout.number_of_edge_nodes(P) > 0:
            key = TriangleMeshFactory.__key(src_idx, dst_idx)
            if key in lut:
                order, offset = lut[key]
                orientation = 1 if order == key else -1
            else:
                offset = len(vertices)
                orientation = 1
                for s in range(TriangleLayout.number_of_edge_nodes(P)):
                    coordinates = [0, 0]
                    vertices.append(coordinates)
                lut[key] = ((src_idx, dst_idx), offset)
        else:
            offset = -1
            orientation = 0
        return offset, orientation

    @staticmethod
    def __create_cell_vertices(vertices, P: int) -> int:
        """

        :param vertices:
        :param P:
        :return:
        """
        offset = len(vertices) if TriangleLayout.number_of_cell_nodes(P) > 0 else -1
        for c in range(TriangleLayout.number_of_cell_nodes(P)):
            coordinates = [0, 0]
            vertices.append(coordinates)
        return offset






def make_mesh(width, height, I, J):
    """
    Creates a square test mesh centered around the origin.

    :param width:          The length of the mesh (x-axis).
    :param height:         The height of the mesh (y-axis).
    :param I:              Number of cells along the x-axis.
    :param J:              Number of cells along the y-axis.
    :return:               A tuple of a V- and T-array. V is N-by-2 vertex coordinates,
                           and T is K-by-3 triangle index array.
    """
    x0 = -width / 2.0
    y0 = -height / 2.0
    dx = width / float(I)
    dy = height / float(J)
    V = np.zeros(((I + 1) * (J + 1), 2), dtype=np.float64)
    for j in range(J + 1):
        for i in range(I + 1):
            k = i + j * (I + 1)
            V[k, 0] = x0 + i * dx
            V[k, 1] = y0 + j * dy
    T = np.zeros((2 * I * J, 3), dtype=np.int32)
    for j in range(J):
        for i in range(I):
            k00 = i + j * (I + 1)
            k01 = (i + 1) + j * (I + 1)
            k10 = i + (j + 1) * (I + 1)
            k11 = (i + 1) + (j + 1) * (I + 1)
            e = 2 * (i + j * I)
            if (i + j + 1) % 2:
                T[e, :] = (k00, k01, k11)
                T[e + 1, :] = (k00, k11, k10)
            else:
                T[e, :] = (k10, k00, k01)
                T[e + 1, :] = (k10, k01, k11)
    return V, T


class TestPascal:

    def __init__(self):
        print('Testing started')
        # self.test_ijk_format(1)
        # self.test_ijk_format(2)
        # self.test_ijk_format(3)
        # self.test_ijk_format(4)
        # self.test_shape_function_properties(3)
        # self.test_shape_function_interpolation(2, 0, 0)
        # self.test_shape_function_interpolation(1, 1, 2)
        # self.test_shape_function_interpolation(3, 0, 0)
        # self.test_shape_function_interpolation(1, 1, 1)
        # self.test_shape_function_interpolation(1, 2, 0)
        # self.test_shape_function_interpolation(2, 1, 0)
        # self.test_shape_function_interpolation(0, 2, 1)
        # self.test_shape_function_interpolation(0, 1, 2)
        # self.test_shape_function_interpolation(2, 0, 1)
        # self.test_shape_function_interpolation(1, 0, 2)
        # self.test_shape_function_interpolation(2, 2, 2)
        # self.test_build_fem_mesh(1)
        # self.test_build_fem_mesh(2)
        # self.test_build_fem_mesh(3)
        # self.test_pascal_triangles(6)
        # self.test_triangle_interpolation(3)
        self.test_mesh_interpolation(3)
        print('Done testing')

    def test_ijk_format(self, P):
        triangle = ReferenceTriangle(P)
        print('--------------------------------------------------')
        print(triangle.P, '-th order Pascal ReferenceTriangle Info:')
        print('\tTotal number of nodes         :', number_of_total_nodes(P))
        print('\tNumber of corner nodes        :', number_of_corner_nodes(P))
        print('\tNumber of nodes on edges      :', number_of_edge_nodes(P))
        print('\tNumber of interior cell nodes :', number_of_cell_nodes(P))
        print('IJK format of nodes:\n', triangle.ijk_format)
        print('Barycentric coordinates:\n', triangle.barycentric)
        print('--------------------------------------------------')

    def test_shape_function_properties(self, P):
        import math
        print('--------------------------------------------------')
        triangle = ReferenceTriangle(P)
        for n in range(number_of_total_nodes(P)):
            phi = LagrangeShapeFunction(triangle.ijk_format[n])
            for m in range(number_of_total_nodes(P)):
                x = triangle.barycentric[m]
                value = phi.value(x)
                if n == m and math.fabs(1.0 - value) > 0.0001:
                    print('failure')
                if n != m and math.fabs(value) > 0.00001:
                    print('failure')
        print('--------------------------------------------------')

    def test_shape_function_interpolation(self, I, J, K):
        print('--------------------------------------------------')
        X = []
        Y = []
        Z = []
        phi = LagrangeShapeFunction([I, J, K])
        triangle = ReferenceTriangle(30)
        for m in range(number_of_total_nodes(triangle.P)):
            x = triangle.barycentric[m]
            value = phi.value(x)
            X.append(x[0])
            Y.append(x[1])
            Z.append(value)
        import matplotlib.pyplot as plt
        latex_shape_func_name = '$N_{' + str(I) + str(J) + str(K) + '}$'
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_title('Lagrangian shape function ' + latex_shape_func_name)
        ax.set_xlabel('$x$', labelpad=20)
        ax.set_ylabel('$y$', labelpad=20)
        ax.set_zlabel(latex_shape_func_name, labelpad=20)
        ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True)
        plt.show()
        file_name = 'N_' + str(I) + str(J) + str(K) + '.pdf'
        # plt.savefig(file_name, format='pdf')
        print('--------------------------------------------------')

    def test_build_fem_mesh(self, P):
        print('--------------------------------------------------')
        #
        #     2    3    5
        #     *    *    *
        #
        #     *    *    *
        #     0    1    4
        #
        V = np.array([[0, 0],
                      [1, 0],
                      [0, 1],
                      [1, 1],
                      [2, 0],
                      [2, 1]], dtype=np.float64)
        T = np.array([[0, 1, 2],
                      [1, 3, 2],
                      [1, 5, 3],
                      [1, 4, 5]], dtype=int)
        mesh = FiniteElementMesh(V, T, P)
        print('FEM Nodes:\n', mesh.V)
        print('FEM Elements:\n', mesh.E)
        print('--------------------------------------------------')

    def test_pascal_triangles(self, P):
        print('--------------------------------------------------')
        X = []
        Y = []
        Z = []
        for i in range(1, P):
            samples = ReferenceTriangle(i).barycentric  # Reference sampling points
            for s in samples:
                X.append(s[0])
                Y.append(s[1])
                # Z.append(s[2])
                Z.append(i)
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_title('Pascal triangle nodes')
        ax.set_xlabel('$x$', labelpad=20)
        ax.set_ylabel('$y$', labelpad=20)
        ax.set_zlabel('Level', labelpad=20)
        ax.scatter(X, Y, Z)
        plt.show()
        # file_name = 'mesh_interpolation.pdf'
        # plt.savefig(file_name, format='pdf')
        print('--------------------------------------------------')

    def test_triangle_interpolation(self, P):
        print('--------------------------------------------------')
        ref = ReferenceTriangle(P)
        indices = np.arange(len(ref.barycentric))
        V = ref.barycentric
        U = V[:, 0] ** 2  # Create scalar field that lives on new mesh nodes.
        samples = ReferenceTriangle(60).barycentric  # Reference sampling points
        X = []
        Y = []
        Z = []
        for s in samples:
            value = interpolate(U, indices, ref.shape_func, s)
            x = interpolate(V, indices, ref.shape_func, s)
            X.append(x[0])
            Y.append(x[1])
            Z.append(value)
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_title('Interpolation test on ' + str(P) + '-order mesh')
        ax.set_xlabel('$x$', labelpad=20)
        ax.set_ylabel('$y$', labelpad=20)
        ax.set_zlabel('$x^2$', labelpad=20)
        ax.scatter(X, Y, Z)
        plt.show()
        # file_name = 'mesh_interpolation.pdf'
        # plt.savefig(file_name, format='pdf')
        print('--------------------------------------------------')

    def test_mesh_interpolation(self, P):
        print('--------------------------------------------------')
        ref = ReferenceTriangle(P)
        V, T = make_mesh(8.0, 8.0, 4, 4)
        mesh = FiniteElementMesh(V, T, P)
        PX = mesh.V[:, 0]
        PY = mesh.V[:, 1]
        U = 3 * (1 - PX) ** 2. * np.exp(-(PX ** 2) - (PY + 1) ** 2) \
            - 10 * (PX / 5 - PX ** 3 - PY ** 5) * np.exp(-PX ** 2 - PY ** 2) \
            - 1 / 3 * np.exp(-(PX + 1) ** 2 - PY ** 2)
        samples = ReferenceTriangle(20).barycentric  # Reference sampling points
        X = []
        Y = []
        Z = []
        for encoding in mesh.E:
            indices = global_indices(encoding, P)
            for s in samples:
                value = interpolate(U, indices, ref.shape_func, s)
                x = interpolate(mesh.V, indices, ref.shape_func, s)
                X.append(x[0])
                Y.append(x[1])
                Z.append(value)
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_title('Interpolation test on ' + str(P) + '-order mesh')
        ax.set_xlabel('$x$', labelpad=20)
        ax.set_ylabel('$y$', labelpad=20)
        ax.set_zlabel('Peaks', labelpad=20)
        ax.scatter(X, Y, Z, c=Z)
        plt.show()
        file_name = 'mesh_interpolation_order_' + str(P) + '.pdf'
        plt.savefig(file_name, format='pdf')
        print('--------------------------------------------------')


if __name__ == '__main__':
    test = TestPascal()
