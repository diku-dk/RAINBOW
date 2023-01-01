import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.geometry.higher_order_triangle_fem as FEM
import rainbow.util.test_tools as TEST
from rainbow.math.functions import peaks


def make_test_mesh(width, height, I, J):
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


def _plot_shape_function(I: int, J: int, K: int) -> None:
    """
    Plot a shape function.

    :param I:   The i-index of the corresponding node of the shape function.
    :param J:   The j-index of the corresponding node of the shape function.
    :param K:   The k-index of the corresponding node of the shape function.
    :return:    None.
    """
    X = []
    Y = []
    Z = []
    phi = FEM.TriangleShapeFunction([I, J, K])
    triangle = FEM.TriangleLayout(30)
    for m in range(FEM.TriangleLayout.number_of_total_nodes(30)):
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


def poly_surf(X, Y):
    """
    This is just some polynomial surface used for testing interpolation properties of the fem mesh.

    Implementation supports both scalars and arrays as input/output.

    :param X:  The x-coordinate value.
    :param Y:  The y-coordinate value.
    :return:   The resulting z value.
    """
    Z = 3 * (1 - X) ** 2. * ((X ** 2) + (Y + 1) ** 2) \
        - 10 * (X / 5 - X ** 2 - Y ** 3) * (X ** 2 + Y ** 2) \
        - 1 / 3 * ((X + 1) ** 2 + Y ** 2)
    return Z/30000


class TestHigherOrderTriangleFEM(unittest.TestCase):

    def test_ijk_format_order_1(self):
        P = 1
        self.assertEqual(3, FEM.TriangleLayout.number_of_total_nodes(P))
        self.assertEqual(3, FEM.TriangleLayout.number_of_corner_nodes(P))
        self.assertEqual(0, FEM.TriangleLayout.number_of_edge_nodes(P))
        self.assertEqual(0, FEM.TriangleLayout.number_of_cell_nodes(P))
        triangle = FEM.TriangleLayout(P)
        self.assertEqual(3, triangle.ijk_format.shape[0])
        self.assertEqual(3, triangle.ijk_format.shape[1])
        self.assertTrue(TEST.is_array_equal([1, 0, 0], triangle.ijk_format[0]))
        self.assertTrue(TEST.is_array_equal([0, 1, 0], triangle.ijk_format[1]))
        self.assertTrue(TEST.is_array_equal([0, 0, 1], triangle.ijk_format[2]))
        self.assertEqual(3, triangle.barycentric.shape[0])
        self.assertEqual(3, triangle.barycentric.shape[1])
        self.assertTrue(TEST.is_array_equal([1., 0., 0.], triangle.barycentric[0]))
        self.assertTrue(TEST.is_array_equal([0., 1., 0.], triangle.barycentric[1]))
        self.assertTrue(TEST.is_array_equal([0., 0., 1.], triangle.barycentric[2]))

    def test_ijk_format_order_2(self):
        P = 2
        self.assertEqual(6, FEM.TriangleLayout.number_of_total_nodes(P))
        self.assertEqual(3, FEM.TriangleLayout.number_of_corner_nodes(P))
        self.assertEqual(1, FEM.TriangleLayout.number_of_edge_nodes(P))
        self.assertEqual(0, FEM.TriangleLayout.number_of_cell_nodes(P))
        triangle = FEM.TriangleLayout(P)
        self.assertEqual(6, triangle.ijk_format.shape[0])
        self.assertEqual(3, triangle.ijk_format.shape[1])
        self.assertTrue(TEST.is_array_equal([2, 0, 0], triangle.ijk_format[0]))
        self.assertTrue(TEST.is_array_equal([0, 2, 0], triangle.ijk_format[1]))
        self.assertTrue(TEST.is_array_equal([0, 0, 2], triangle.ijk_format[2]))
        self.assertTrue(TEST.is_array_equal([1, 1, 0], triangle.ijk_format[3]))
        self.assertTrue(TEST.is_array_equal([0, 1, 1], triangle.ijk_format[4]))
        self.assertTrue(TEST.is_array_equal([1, 0, 1], triangle.ijk_format[5]))
        self.assertEqual(6, triangle.barycentric.shape[0])
        self.assertEqual(3, triangle.barycentric.shape[1])
        self.assertTrue(TEST.is_array_equal([1., 0., 0.], triangle.barycentric[0]))
        self.assertTrue(TEST.is_array_equal([0., 1., 0.], triangle.barycentric[1]))
        self.assertTrue(TEST.is_array_equal([0., 0., 1.], triangle.barycentric[2]))
        self.assertTrue(TEST.is_array_equal([0.5, 0.5, 0.], triangle.barycentric[3]))
        self.assertTrue(TEST.is_array_equal([0., 0.5, 0.5], triangle.barycentric[4]))
        self.assertTrue(TEST.is_array_equal([0.5, 0., 0.5], triangle.barycentric[5]))

    def test_ijk_format_order_3(self):
        P = 3
        self.assertEqual(10, FEM.TriangleLayout.number_of_total_nodes(P))
        self.assertEqual(3, FEM.TriangleLayout.number_of_corner_nodes(P))
        self.assertEqual(2, FEM.TriangleLayout.number_of_edge_nodes(P))
        self.assertEqual(1, FEM.TriangleLayout.number_of_cell_nodes(P))
        triangle = FEM.TriangleLayout(P)
        self.assertEqual(10, triangle.ijk_format.shape[0])
        self.assertEqual(3, triangle.ijk_format.shape[1])
        self.assertTrue(TEST.is_array_equal([3, 0, 0], triangle.ijk_format[0]))
        self.assertTrue(TEST.is_array_equal([0, 3, 0], triangle.ijk_format[1]))
        self.assertTrue(TEST.is_array_equal([0, 0, 3], triangle.ijk_format[2]))
        self.assertTrue(TEST.is_array_equal([1, 1, 1], triangle.ijk_format[3]))
        self.assertTrue(TEST.is_array_equal([2, 1, 0], triangle.ijk_format[4]))
        self.assertTrue(TEST.is_array_equal([1, 2, 0], triangle.ijk_format[5]))
        self.assertTrue(TEST.is_array_equal([0, 2, 1], triangle.ijk_format[6]))
        self.assertTrue(TEST.is_array_equal([0, 1, 2], triangle.ijk_format[7]))
        self.assertTrue(TEST.is_array_equal([1, 0, 2], triangle.ijk_format[8]))
        self.assertTrue(TEST.is_array_equal([2, 0, 1], triangle.ijk_format[9]))
        self.assertEqual(10, triangle.barycentric.shape[0])
        self.assertEqual(3, triangle.barycentric.shape[1])
        self.assertTrue(TEST.is_array_equal([1., 0., 0.], triangle.barycentric[0]))
        self.assertTrue(TEST.is_array_equal([0., 1., 0.], triangle.barycentric[1]))
        self.assertTrue(TEST.is_array_equal([0., 0., 1.], triangle.barycentric[2]))
        self.assertTrue(TEST.is_array_equal([1/3, 1/3, 1/3], triangle.barycentric[3]))
        self.assertTrue(TEST.is_array_equal([2/3., 1/3, 0.], triangle.barycentric[4]))
        self.assertTrue(TEST.is_array_equal([1/3, 2/3, 0.], triangle.barycentric[5]))
        self.assertTrue(TEST.is_array_equal([0., 2/3, 1/3], triangle.barycentric[6]))
        self.assertTrue(TEST.is_array_equal([0, 1/3, 2/3], triangle.barycentric[7]))
        self.assertTrue(TEST.is_array_equal([1/3., 0., 2/3], triangle.barycentric[8]))
        self.assertTrue(TEST.is_array_equal([2/3, 0, 1/3], triangle.barycentric[9]))

    def test_ijk_format_order_4(self):
        P = 4
        self.assertEqual(15, FEM.TriangleLayout.number_of_total_nodes(P))
        self.assertEqual(3, FEM.TriangleLayout.number_of_corner_nodes(P))
        self.assertEqual(3, FEM.TriangleLayout.number_of_edge_nodes(P))
        self.assertEqual(3, FEM.TriangleLayout.number_of_cell_nodes(P))
        triangle = FEM.TriangleLayout(P)
        self.assertEqual(15, triangle.ijk_format.shape[0])
        self.assertEqual(3, triangle.ijk_format.shape[1])
        self.assertTrue(TEST.is_array_equal([4, 0, 0], triangle.ijk_format[0]))
        self.assertTrue(TEST.is_array_equal([0, 4, 0], triangle.ijk_format[1]))
        self.assertTrue(TEST.is_array_equal([0, 0, 4], triangle.ijk_format[2]))
        self.assertTrue(TEST.is_array_equal([1, 1, 2], triangle.ijk_format[3]))
        self.assertTrue(TEST.is_array_equal([2, 1, 1], triangle.ijk_format[4]))
        self.assertTrue(TEST.is_array_equal([1, 2, 1], triangle.ijk_format[5]))
        self.assertTrue(TEST.is_array_equal([3, 1, 0], triangle.ijk_format[6]))
        self.assertTrue(TEST.is_array_equal([2, 2, 0], triangle.ijk_format[7]))
        self.assertTrue(TEST.is_array_equal([1, 3, 0], triangle.ijk_format[8]))
        self.assertTrue(TEST.is_array_equal([0, 3, 1], triangle.ijk_format[9]))
        self.assertTrue(TEST.is_array_equal([0, 2, 2], triangle.ijk_format[10]))
        self.assertTrue(TEST.is_array_equal([0, 1, 3], triangle.ijk_format[11]))
        self.assertTrue(TEST.is_array_equal([1, 0, 3], triangle.ijk_format[12]))
        self.assertTrue(TEST.is_array_equal([2, 0, 2], triangle.ijk_format[13]))
        self.assertTrue(TEST.is_array_equal([3, 0, 1], triangle.ijk_format[14]))
        self.assertEqual(15, triangle.barycentric.shape[0])
        self.assertEqual(3, triangle.barycentric.shape[1])
        self.assertTrue(TEST.is_array_equal([1., 0., 0.], triangle.barycentric[0]))
        self.assertTrue(TEST.is_array_equal([0., 1., 0.], triangle.barycentric[1]))
        self.assertTrue(TEST.is_array_equal([0., 0., 1.], triangle.barycentric[2]))
        self.assertTrue(TEST.is_array_equal([1/4, 1/4, 2/4], triangle.barycentric[3]))
        self.assertTrue(TEST.is_array_equal([2/4, 1/4, 1/4], triangle.barycentric[4]))
        self.assertTrue(TEST.is_array_equal([1/4, 2/4, 1/4], triangle.barycentric[5]))
        self.assertTrue(TEST.is_array_equal([3/4, 1/4, 0.], triangle.barycentric[6]))
        self.assertTrue(TEST.is_array_equal([2/4, 2/4, 0.], triangle.barycentric[7]))
        self.assertTrue(TEST.is_array_equal([1/4, 3/4, 0.], triangle.barycentric[8]))
        self.assertTrue(TEST.is_array_equal([0., 3/4, 1/4], triangle.barycentric[9]))
        self.assertTrue(TEST.is_array_equal([0., 2/4, 2/4], triangle.barycentric[10]))
        self.assertTrue(TEST.is_array_equal([0., 1/4, 3/4], triangle.barycentric[11]))
        self.assertTrue(TEST.is_array_equal([1/4, 0., 3/4], triangle.barycentric[12]))
        self.assertTrue(TEST.is_array_equal([2/4, 0., 2/4], triangle.barycentric[13]))
        self.assertTrue(TEST.is_array_equal([3/4, 0., 1/4], triangle.barycentric[14]))

    def test_shape_function_properties(self):
        # This unit-test verifies the shape interpolation properties of a shape function. That
        # is if evaluated at the location of the node that the shape function belongs to the value
        # should ne 1 and at all other nodes the value should be zero.
        for P in range(1, 5):
            triangle = FEM.TriangleLayout(P)
            for n in range(FEM.TriangleLayout.number_of_total_nodes(P)):
                phi = FEM.TriangleShapeFunction(triangle.ijk_format[n])
                for m in range(FEM.TriangleLayout.number_of_total_nodes(P)):
                    w = triangle.barycentric[m]
                    value = phi.value(w)
                    if n == m:
                        self.assertAlmostEqual(value, 1.)
                    else:
                        self.assertAlmostEqual(value, 0.)

    def test_shape_function(self):
        """
        This is really not a test, this is just plotting stuff.

        :return:
        """
        _plot_shape_function(2, 0, 0)
        _plot_shape_function(1, 1, 2)
        _plot_shape_function(3, 0, 0)
        _plot_shape_function(1, 1, 1)
        _plot_shape_function(1, 2, 0)
        _plot_shape_function(2, 1, 0)
        _plot_shape_function(0, 2, 1)
        _plot_shape_function(0, 1, 2)
        _plot_shape_function(2, 0, 1)
        _plot_shape_function(1, 0, 2)
        _plot_shape_function(2, 1, 0)
        _plot_shape_function(2, 2, 2)

    def test_make_mesh(self):
        # This is the input mesh
        #
        #     2    3    5
        #     +----+----+
        #     |\   |   /|
        #     | \  |  / |
        #     |  \ | /  |
        #     |   \|/   |
        #     +----+----+
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
        # Now we create a first order FEM mesh. This is basically the same as the input. So the interesting
        # thing to see is if the encodings are done correctly.
        mesh = FEM.make_mesh(V, T, P=1, keep_indices=False)
        self.assertEqual(mesh.vertices.shape[0], 6)
        self.assertEqual(mesh.vertices.shape[1], 2)
        self.assertTrue(TEST.is_array_equal([0., 0.], mesh.vertices[0]))
        self.assertTrue(TEST.is_array_equal([1., 0.], mesh.vertices[1]))
        self.assertTrue(TEST.is_array_equal([0., 1.], mesh.vertices[2]))
        self.assertTrue(TEST.is_array_equal([1., 1.], mesh.vertices[3]))
        self.assertTrue(TEST.is_array_equal([2., 0.], mesh.vertices[4]))
        self.assertTrue(TEST.is_array_equal([2., 1.], mesh.vertices[5]))
        self.assertEqual(mesh.encodings.shape[0], 4)
        self.assertEqual(mesh.encodings.shape[1], 10)
        self.assertTrue(TEST.is_array_equal([0,  1,  2, -1, -1, -1, -1,  0,  0,  0], mesh.encodings[0]))
        self.assertTrue(TEST.is_array_equal([1,  3,  2, -1, -1, -1, -1,  0,  0,  0], mesh.encodings[1]))
        self.assertTrue(TEST.is_array_equal([1,  5,  3, -1, -1, -1, -1,  0,  0,  0], mesh.encodings[2]))
        self.assertTrue(TEST.is_array_equal([1,  4,  5, -1, -1, -1, -1,  0,  0,  0], mesh.encodings[3]))
        # Now we got for a second order mesh. The second order element will have an extra node
        # on the edges, but no interior cell nodes.
        mesh = FEM.make_mesh(V, T, P=2, keep_indices=False)
        self.assertEqual(mesh.vertices.shape[0], 15)
        self.assertEqual(mesh.vertices.shape[1], 2)
        self.assertTrue(TEST.is_array_equal([0., 0.], mesh.vertices[0]))
        self.assertTrue(TEST.is_array_equal([1., 0.], mesh.vertices[1]))
        self.assertTrue(TEST.is_array_equal([0., 1.], mesh.vertices[2]))
        self.assertTrue(TEST.is_array_equal([1., 1.], mesh.vertices[3]))
        self.assertTrue(TEST.is_array_equal([2., 0.], mesh.vertices[4]))
        self.assertTrue(TEST.is_array_equal([2., 1.], mesh.vertices[5]))
        self.assertTrue(TEST.is_array_equal([.5, 0.], mesh.vertices[6]))
        self.assertTrue(TEST.is_array_equal([.5, .5], mesh.vertices[7]))
        self.assertTrue(TEST.is_array_equal([0., .5], mesh.vertices[8]))
        self.assertTrue(TEST.is_array_equal([1., .5], mesh.vertices[9]))
        self.assertTrue(TEST.is_array_equal([0.5, 1.], mesh.vertices[10]))
        self.assertTrue(TEST.is_array_equal([1.5, .5], mesh.vertices[11]))
        self.assertTrue(TEST.is_array_equal([1.5, 1.], mesh.vertices[12]))
        self.assertTrue(TEST.is_array_equal([1.5, 0.], mesh.vertices[13]))
        self.assertTrue(TEST.is_array_equal([2., .5], mesh.vertices[14]))
        self.assertEqual(mesh.encodings.shape[0], 4)
        self.assertEqual(mesh.encodings.shape[1], 10)
        self.assertTrue(TEST.is_array_equal([0,  1,  2, -1, 6, 7, 8,  1,  1,  1], mesh.encodings[0]))
        self.assertTrue(TEST.is_array_equal([1,  3,  2, -1, 9, 10, 7,  1,  1,  -1], mesh.encodings[1]))
        self.assertTrue(TEST.is_array_equal([1,  5,  3, -1, 11, 12, 9,  1,  1,  -1], mesh.encodings[2]))
        self.assertTrue(TEST.is_array_equal([1,  4,  5, -1, 13, 14, 11,  1,  1,  -1], mesh.encodings[3]))
        # Now we will try order 3 mesh. This time there will be exactly one interior node in each mesh, and edges
        # will have to nodes, so orientation is really important.
        mesh = FEM.make_mesh(V, T, P=3, keep_indices=False)
        self.assertEqual(mesh.vertices.shape[0], 28)
        self.assertEqual(mesh.vertices.shape[1], 2)
        self.assertTrue(TEST.is_array_equal([0., 0.], mesh.vertices[0]))
        self.assertTrue(TEST.is_array_equal([1., 0.], mesh.vertices[1]))
        self.assertTrue(TEST.is_array_equal([0., 1.], mesh.vertices[2]))
        self.assertTrue(TEST.is_array_equal([1., 1.], mesh.vertices[3]))
        self.assertTrue(TEST.is_array_equal([2., 0.], mesh.vertices[4]))
        self.assertTrue(TEST.is_array_equal([2., 1.], mesh.vertices[5]))
        self.assertTrue(TEST.is_array_equal([0.33333333, 0.33333333], mesh.vertices[6]))
        self.assertTrue(TEST.is_array_equal([0.33333333, 0.], mesh.vertices[7]))
        self.assertTrue(TEST.is_array_equal([0.66666667, 0.], mesh.vertices[8]))
        self.assertTrue(TEST.is_array_equal([0.66666667, 0.33333333], mesh.vertices[9]))
        self.assertTrue(TEST.is_array_equal([0.33333333, 0.66666667], mesh.vertices[10]))
        self.assertTrue(TEST.is_array_equal([0.,         0.66666667], mesh.vertices[11]))
        self.assertTrue(TEST.is_array_equal([0.,         0.33333333], mesh.vertices[12]))
        self.assertTrue(TEST.is_array_equal([0.66666667, 0.66666667], mesh.vertices[13]))
        self.assertTrue(TEST.is_array_equal([1.,         0.33333333], mesh.vertices[14]))
        self.assertTrue(TEST.is_array_equal([1.,         0.66666667], mesh.vertices[15]))
        self.assertTrue(TEST.is_array_equal([0.66666667, 1.], mesh.vertices[16]))
        self.assertTrue(TEST.is_array_equal([0.33333333, 1.], mesh.vertices[17]))
        self.assertTrue(TEST.is_array_equal([1.33333333, 0.66666667], mesh.vertices[18]))
        self.assertTrue(TEST.is_array_equal([1.33333333, 0.33333333], mesh.vertices[19]))
        self.assertTrue(TEST.is_array_equal([1.66666667, 0.66666667], mesh.vertices[20]))
        self.assertTrue(TEST.is_array_equal([1.66666667, 1.], mesh.vertices[21]))
        self.assertTrue(TEST.is_array_equal([1.33333333, 1.], mesh.vertices[22]))
        self.assertTrue(TEST.is_array_equal([1.66666667, 0.33333333], mesh.vertices[23]))
        self.assertTrue(TEST.is_array_equal([1.33333333, 0.], mesh.vertices[24]))
        self.assertTrue(TEST.is_array_equal([1.66666667, 0.], mesh.vertices[25]))
        self.assertTrue(TEST.is_array_equal([2.,        0.33333333], mesh.vertices[26]))
        self.assertTrue(TEST.is_array_equal([2.,        0.66666667], mesh.vertices[27]))
        self.assertEqual(mesh.encodings.shape[0], 4)
        self.assertEqual(mesh.encodings.shape[1], 10)
        self.assertTrue(TEST.is_array_equal([0,  1,  2, 6, 7, 9, 11,  1,  1,  1], mesh.encodings[0]))
        self.assertTrue(TEST.is_array_equal([1,  3,  2, 13, 14, 16, 9,  1,  1,  -1], mesh.encodings[1]))
        self.assertTrue(TEST.is_array_equal([1,  5,  3, 18, 19, 21, 14,  1,  1,  -1], mesh.encodings[2]))
        self.assertTrue(TEST.is_array_equal([1,  4,  5, 23, 24, 26, 19,  1,  1,  -1], mesh.encodings[3]))

    def test_pascal_triangles(self):
        # This text function is really just showing the nodes of the Pascal triangle for increasing order. This is
        # merely just a visual inspection and not a unit-test.
        P = 5    # Maximum order of Pascal triangle visualized
        X = []
        Y = []
        Z = []
        for i in range(1, P+1):
            samples = FEM.TriangleLayout(i).barycentric  # Reference sampling points
            for s in samples:
                X.append(s[0])
                Y.append(s[1])
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

    def test_triangle_interpolation(self):
        P = 3   # The order of the triangle we are testing.
        layout = FEM.TriangleLayout(P)
        element = FEM.TriangleElement(layout)
        indices = np.arange(len(layout.barycentric))
        V = layout.barycentric
        U = V[:, 0] ** 2 + V[:, 1] ** 2 # Create scalar field that lives on the ref triangle.
        samples = FEM.TriangleLayout(60).barycentric  # Sampling Test Points
        X = []
        Y = []
        Z = []
        for s in samples:
            value = FEM.Field.IsoParametric.interpolate_value(U, indices, element.shape_functions, s)
            x = FEM.Field.IsoParametric.interpolate_value(V, indices, element.shape_functions, s)
            X.append(x[0])
            Y.append(x[1])
            Z.append(value)
            self.assertAlmostEqual(value, x[0]**2 + x[1]**2)
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_title('Interpolation test on ' + str(P) + '-order triangle element')
        ax.set_xlabel('$x$', labelpad=20)
        ax.set_ylabel('$y$', labelpad=20)
        ax.set_zlabel('$x^2+Y^2$', labelpad=20)
        ax.scatter(X, Y, Z)
        plt.show()
        # file_name = 'mesh_interpolation.pdf'
        # plt.savefig(file_name, format='pdf')

    def test_mesh_interpolation(self):
        P = 5  # The order of the elements.
        layout = FEM.TriangleLayout(P)
        element = FEM.TriangleElement(layout)
        V, T = make_test_mesh(8.0, 8.0, 4, 4)
        mesh = FEM.make_mesh(V, T, P)
        PX = mesh.vertices[:, 0]
        PY = mesh.vertices[:, 1]
        U = poly_surf(PX, PY)
        samples = FEM.TriangleLayout(10).barycentric  # Test sampling points
        X = []
        Y = []
        Z = []
        for encoding in mesh.encodings:
            indices = FEM.TriangleLayout.get_global_indices(encoding, P)
            for s in samples:
                value = FEM.Field.IsoParametric.interpolate_value(U, indices, element.shape_functions, s)
                p = FEM.Field.IsoParametric.interpolate_value(mesh.vertices, indices, element.shape_functions, s)
                X.append(p[0])
                Y.append(p[1])
                Z.append(value)
                self.assertAlmostEqual(value, poly_surf(p[0], p[1]))
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_title('Interpolation test on ' + str(P) + '-order mesh')
        ax.set_xlabel('$x$', labelpad=20)
        ax.set_ylabel('$y$', labelpad=20)
        ax.set_zlabel('Peaks', labelpad=20)
        ax.scatter(X, Y, Z, c=Z)
        plt.show()
        # file_name = 'mesh_interpolation_order_' + str(P) + '.pdf'
        # plt.savefig(file_name, format='pdf')

    def test_make_field(self):
        P = 3
        V, T = make_test_mesh(8.0, 8.0, 4, 4)
        mesh = FEM.make_mesh(V, T, P)
        scalar_field = FEM.make_zero_field(mesh, shape=(1,))
        vector_field = FEM.make_zero_field(mesh, shape=(3,))
        tensor_field = FEM.make_zero_field(mesh, shape=(2, 2))
        self.assertEqual(scalar_field.values.shape, (169, 1))
        self.assertEqual(vector_field.values.shape, (169, 3))
        self.assertEqual(tensor_field.values.shape, (169, 2, 2))
        # Now we will test if we get exceptions if we try to create a
        # field using a wrong sized array.
        self.assertRaises(ValueError, FEM.make_field_from_array, mesh, V, True)
        self.assertRaises(ValueError, FEM.make_field_from_array, mesh, V, False)
        # Now we will test if the copy argument works, so we either get a copy or reference into
        # the array used to create the field.
        array_field = FEM.make_field_from_array(mesh, mesh.vertices, False)
        self.assertEqual(array_field.values.shape, (169, 2))
        mesh.vertices[1, 1] = 100
        self.assertEqual(array_field.values[1, 1], 100)
        array_field = FEM.make_field_from_array(mesh, mesh.vertices, True)
        self.assertEqual(array_field.values.shape, (169, 2))
        mesh.vertices[1, 1] = -100
        self.assertEqual(array_field.values[1, 1], 100)

    def test_field_interpolation(self):
        P = 5  # The order of the elements.
        V, T = make_test_mesh(8.0, 8.0, 4, 4)

        mesh = FEM.make_mesh(V, T, P)
        U = FEM.make_field_from_array(mesh, poly_surf(mesh.vertices[:, 0], mesh.vertices[:, 1]))
        X0 = FEM.make_field_from_array(mesh, mesh.vertices, False)

        samples = FEM.TriangleLayout(10).barycentric  # Test sampling points
        X = []
        Y = []
        Z = []
        for k in range(len(mesh.encodings)):
            for s in samples:
                value = U.get_value(k, s)
                p = X0.get_value(k, s)
                X.append(p[0])
                Y.append(p[1])
                Z.append(value)
                self.assertAlmostEqual(value, poly_surf(p[0], p[1]))
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_title('Interpolation test on ' + str(P) + '-order field')
        ax.set_xlabel('$x$', labelpad=20)
        ax.set_ylabel('$y$', labelpad=20)
        ax.set_zlabel('Peaks', labelpad=20)
        ax.scatter(X, Y, Z, c=Z)
        plt.show()
        # file_name = 'field_interpolation_order_' + str(P) + '.pdf'
        # plt.savefig(file_name, format='pdf')

    def test_shape_function_gradients(self) -> None:
        """
        In this test function we make sure that we can compute the derivative
        of the shape functions correctly.

        We assume that our code for evaluating the value of a shape function is correct. This allows us to
        create a symbolic expression of the shape function and then take the symbolic derivative. This way
        we can symbolically generate a closed form solution for evaluating the value of our shape function
        gradient.

        :return: None
        """
        import sympy as sym

        def test_order(P) -> None:
            """
            Local test routine that makes it easier to add more test cases of higher order.

            :param P:  The order of the shape function to test.
            :return:   None
            """
            samples = FEM.TriangleLayout(10).barycentric  # Test sampling points
            element = FEM.TriangleElement(FEM.TriangleLayout(P))
            for phi in element.shape_functions:
                # First we create a symbolic expression for the shape function that we are testing
                w0 = sym.Symbol('w0')
                w1 = sym.Symbol('w1')
                w2 = sym.Symbol('w2')
                phi_expr = sym.expand(phi.value([w0, w1, w2]))
                # Then we compute a symbolic derivative of that shape function, observe we keep it as partial derivatives.
                dphi_d0_expr = sym.diff(phi_expr, w0)
                dphi_d1_expr = sym.diff(phi_expr, w1)
                dphi_d2_expr = sym.diff(phi_expr, w2)
                for w in samples:
                    # From our expressions of the symbolic partical derivatives we can now substitute the barycentric
                    # coordinates to evaluate the exact value of the partial derivatives.
                    d0 = dphi_d0_expr.subs([(w0, w[0]), (w1, w[1]), (w2, w[2])])
                    d1 = dphi_d1_expr.subs([(w0, w[0]), (w1, w[1]), (w2, w[2])])
                    d2 = dphi_d2_expr.subs([(w0, w[0]), (w1, w[1]), (w2, w[2])])
                    # We now use our recursive "fast" way of evaluation the gradient value.
                    gradient = phi.gradient(w)
                    # Finally, we can compute the symbolic values to the actual computed values
                    self.assertTrue(TEST.is_array_equal([d0, d1, d2], gradient))
        # Now we can just run our test scenarios with increasing orders of the shape function.
        test_order(P=1)
        test_order(P=2)
        test_order(P=3)
        test_order(P=4)
        test_order(P=5)

    #def test_field_gradients(self):
    #    P = 5  # The order of the elements.
    #    V, T = make_test_mesh(1.0, 1.0, 1, 1) # Unit cube made of two triangles
    #    mesh = FEM.make_mesh(V, T, P)
    #    U = FEM.make_field_from_array(mesh, poly_surf(mesh.vertices[:, 0], mesh.vertices[:, 1]))
    #    X0 = FEM.make_field_from_array(mesh, mesh.vertices, False)
    #
    #    samples = FEM.TriangleLayout(3).barycentric  # Test sampling points
    #    for k in range(len(mesh.encodings)):
    #        for w in samples:
    #            g = U.get_gradient(k, w)
    #            p = X0.get_value(k, w)
    #            print(p, g)
    #            #self.assertAlmostEqual(value, poly_surf(p[0], p[1]))

