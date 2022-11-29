import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.geometry.higher_order_triangle_fem as FEM
import rainbow.util.test_tools as TEST


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



class TestHigherOrderTriangleFEM(unittest.TestCase):

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
        triangle = FEM.TriangleElemenet(P)
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
