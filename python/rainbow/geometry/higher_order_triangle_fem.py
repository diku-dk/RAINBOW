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
        All nodes from a P-order triangle element can be specified and located using only 10 numbers, always.

        The trick is to assume that interior edge and cell nodes are stored with increasing global index
        values. Hence, one only need to remember the offset index to where cell and edge indices start. One
        also need to remember in which order the nodes should be visited. We name this the orientation of the
        edges. It can be +1 (meaning increasing) or -1 (meaning reverse order).

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

    def __init__(self, P):
        """
        Create an instance of an P'th order Pascal Triangle Layout.

        :param P: The order of the Triangle.
        """
        if P < 1:
            raise ValueError("P must be of positive")
        self.P = P
        self.ijk_format = TriangleLayout.compute_ijk_format(P)
        self.barycentric = self.ijk_format.astype(dtype=np.float64) / self.P


class TriangleShapeFunction:
    """
    This class implement a recursively defined Lagrangian shape function for a triangular element.

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
    This class creates lookup data structures for shape functions defined on a Pascal triangle. The
    purpose is mostly to make bookkeeping of shape functions more simple in finite element
    method codes.
    """

    def __init__(self, layout: TriangleLayout):
        """
        Create an instance of a triangle element from a triangle layout template. This combines the layout
        with the choice of the function space to use for the element.

        :param layout: The layout of the triangle element.
        """
        # TODO 2022-12-15 Kenny review: Add code to test for valid input layout argument.
        self.shape_functions = [TriangleShapeFunction(layout.ijk_format[i]) for i in range(len(layout.ijk_format))]


class TriangleMesh:
    """
    A P'th order triangle mesh.

    A triangle mesh consists of an element type. The element type tells us something about the shape function space
    used. The element is defined from a specific triangle layout saying how many nodes each element has and where
    they are placed in iso-parametric space (barycentric coordinates).

    Further, the triangle mesh contains the coordinates of all the nodes. This is called the vertices. Individual
    triangles can be representing in one of two ways. Either as a list of the global indices that makes up the
    triangle, or using a compact run-length encoding of the index lists. The conversion from encoding to global
    indices are defined by the triangle layout.

    The index representation takes up more storage but is faster, whereas the encoding uses minimal
    storage but takes a few flops to expand when one needs to find the global indices that defines a triangle.
    """

    def __init__(self, P: int):
        """
        Create a 2D triangular FEM mesh of order P.

        :param P:  The order of the triangle mesh.
        """
        self.layout = TriangleLayout(P)
        # TODO 2022-12-15 Kenny review: The recursive Lagrangian simplicial element type allow us to reuse the
        #  shape-functions for all elements in our mesh. This is because we do not need to compute polynomial
        #  coefficients for each individual shape-function. This is space efficient as all elements of the mesh
        #  share the exact same IJK format, barycentric coordinates and shape-functions.
        #  If we changed to Lagrangian element that uses a generalized Vandemonde approach for computing
        #  polynomial coefficient then the coefficients would be unique for each shape-function for each
        #  node in the mesh. Hence, we can not share shape functions across elements in this case. The current
        #  code is hard-wired to the recursive shape functions.
        self.element = TriangleElement(self.layout)  # The triangle element type.
        self.vertices = None  # V-by-3 array of vertex coordinates, assuming V total nodes.
        self.encodings = None  # T-by-10 array of triangle encodings, assuming T triangles.
        self.indices = None  # T-by-N array of global node indices, assuming N nodes per triangle.


class Field:
    """
    This class represents a field that is sampled onto the nodes of a mesh, and the mesh element type is
    used for retrieving the field values.
    """

    class IsoParametric:
        """
        This class encapsulates functionality that is specific to only work in the iso-parametric space of the element.
        """

        @staticmethod
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

    def __init__(self, mesh: TriangleMesh, shape: tuple[int, ...]):
        """
        Initialize the field.
        The initializer will not allocate the values of the field. We supply a series of factory
        functions that allocates and initialize the actual field values.

        :param mesh:   The mesh that the field is associated with. The filed lives on the nodes of the mesh.
        :param shape:  A tuple that indicates the shape (aka dimension) of the field.
        """
        # TODO 2022-12-15 Kenny review: Add code to test for valid input values of mesh and shape arguments.
        self.mesh = mesh
        self.shape = (len(mesh.vertices),) + shape
        self.values = None

    def get_value(self, idx, w):
        """
        This method retrieves the field value at the desired location.

        :param idx:  The index of the triangle element from within the field value is wanted.
        :param w:    Barycentric value at which the field value should be retrieved.
        :return:     The field value at location w in triangle with index idx.
        """
        e = self.mesh.encodings[idx]
        # TODO 2022-12-15 Kenny review: Here we could reuses indices if the mesh keeps them at creation time.
        #  Currently the code is hardwired to assume that indices are not stored. Hence, they must be recomputed.
        indices = TriangleLayout.get_global_indices(encoding=e, P=self.mesh.layout.P)
        return Field.IsoParametric.interpolate(self.values, indices, self.mesh.element.shape_functions, w)


class _TriangleMeshFactory:
    """
    This class wraps the functionality needed to create a higher order Triangular finite element mesh.
    """

    @staticmethod
    def make(V, T, P: int, keep_indices=False) -> TriangleMesh:
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
        linear_element = TriangleElement(TriangleLayout(1))

        lut = {}
        vertices = []
        encodings = []
        global_indices = []

        for idx in range(len(V)):
            coordinates = V[idx]
            vertices.append(coordinates)

        for e in range(len(T)):
            i, j, k = T[e]  # Get global corner indices of input triangle

            cell_offset = _TriangleMeshFactory.create_cell_vertices(vertices, P)

            ij_offset, ij_orientation = _TriangleMeshFactory.create_edge_vertices(i, j, lut, vertices, P)
            jk_offset, jk_orientation = _TriangleMeshFactory.create_edge_vertices(j, k, lut, vertices, P)
            ki_offset, ki_orientation = _TriangleMeshFactory.create_edge_vertices(k, i, lut, vertices, P)

            encoding = [
                i, j, k,
                cell_offset, ij_offset, jk_offset, ki_offset,
                ij_orientation, jk_orientation, ki_orientation
            ]

            encodings.append(encoding)

            indices = TriangleLayout.get_global_indices(encoding, P)
            if keep_indices:
                global_indices.append(indices)

            for local_idx, global_idx in enumerate(indices):
                linear_indices = [i, j, k]
                vertices[global_idx] = Field.IsoParametric.interpolate(V,
                                                                       linear_indices,
                                                                       linear_element.shape_functions,
                                                                       mesh.layout.barycentric[local_idx]
                                                                       )

        mesh.vertices = np.array(vertices, dtype=np.float64)
        mesh.encodings = np.array(encodings, dtype=int)
        if keep_indices:
            mesh.indices = np.array(global_indices, dtype=int)

        return mesh

    @staticmethod
    def key(src_idx: int, dst_idx: int) -> tuple[int, int]:
        """

        :param src_idx:
        :param dst_idx:
        :return:
        """
        return (src_idx, dst_idx) if src_idx < dst_idx else (dst_idx, src_idx)

    @staticmethod
    def create_edge_vertices(src_idx: int, dst_idx: int, lut, vertices, P: int) -> tuple[int, int]:
        """

        :param src_idx:
        :param dst_idx:
        :param lut:
        :param vertices:
        :param P:
        :return:
        """
        if TriangleLayout.number_of_edge_nodes(P) > 0:
            key = _TriangleMeshFactory.key(src_idx, dst_idx)
            if key in lut:
                order, offset = lut[key]
                orientation = 1 if order == (src_idx, dst_idx) else -1
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
    def create_cell_vertices(vertices, P: int) -> int:
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


def make_mesh(V, T, P: int, keep_indices=False) -> TriangleMesh:
    """
    This is just a convenience function to make user-interface less verbose.

    :param V:                The vertices off a 1st order triangle mesh.
    :param T:                The triangle index array of a 1st order triangle mesh.
    :param P:                The order of the Triangle elements in the mesh.
    :param keep_indices:     A boolean flag telling whether to store global indices.
    :return:                 A P'th order triangle mesh.
    """
    return _TriangleMeshFactory.make(V, T, P, keep_indices)


def make_zero_field(mesh: TriangleMesh, shape: tuple[int, ...]) -> Field:
    """
    This function creates a field associated to a mesh and initialize all
    field values to be zero.

    :param mesh:     The mesh the field lives on.
    :param shape:    The shape (aka dimension) of the field.
    :return:         A new field value instance representing the new field.
    """
    field = Field(mesh, shape)
    field.values = np.zeros(field.shape, dtype=float)
    return field


def make_field_from_array(mesh: TriangleMesh, values: np.ndarray, copy=True) -> Field:
    """
    This function creates a field from an input array and associates it with the given
    mesh. It is important that the number of mesh vertices (aka nodes) are equal to the
    first dimension of the input array.
    
    :param mesh:       The mesh for which the field should be created.
    :param values:     An array of values to use to initialize the field values with.
    :param copy:       A boolean flag that tells whether a new copy of the input array is
                       created or whether the field just keeps a reference to the input array. Default
                       behavior is to create a copy.
    :return:           A new field value instance representing the new field.
    """
    if values.shape[0] != len(mesh.vertices):
        raise ValueError("Array did not have correct number of elements. Must be the same as number of nodes in mesh")

    field = Field(mesh, values.shape[1:])
    if copy:
        field.values = np.copy(values)
    else:
        field.values = values
    return field
