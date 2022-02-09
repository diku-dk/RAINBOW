import numpy as np
import igl


class Grid:
    """
    A 3D regular grid data structure.
    This data structure only defines functionality for handling indices, coordinates and retrieving nodal values.
    The internal representation of the grid is a flat array. The convention is used that i is the running index.
    Hence, for cache friendly access end users should access nodal values
    in lexicographic increasing order of (i,j,k). No internal testing is done for proper access all methods should
    be considered unsafe in the sense that there are no bounds testing on input values and no testing if the value
    array has been created.
    """

    def __init__(self, min_coord, max_coord, I, J, K):
        """
        This constructor creates an empty grid instance. It will initialize all internal member values necessary
        for supporting index and coordinate methods. However, the actual value array is defined to None.

        :param min_coord: The spatial 3D coordinates of the lower-left-back node. The one with smallest lexicographic
                          index value.
        :param max_coord: The spatial 3D coordinates of the upper-right-front node. The one with largest lexicographic
                          index value.
        :param I: The total number of nodes along the x-axis.
        :param J: The total number of nodes along the y-axis.
        :param K: The total number of nodes along the z-axis.
        """
        if np.any(min_coord > max_coord):
            raise ValueError()
        if I < 2:
            raise ValueError()
        if J < 2:
            raise ValueError()
        if K < 2:
            raise ValueError()
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.I = I  # Number of nodes along x axis
        self.J = J  # Number of nodes along y axis
        self.K = K  # Number of nodes along z axis
        res = np.array([I - 1, J - 1, K - 1], dtype=np.float64)
        dims = self.max_coord - self.min_coord
        self.spacing = np.divide(dims, res)
        self.values = None

    def is_valid_node_index(self, i, j, k):
        """
        This method can be used to test if a node index corresponds to a node that exists in the grid.

        :param i:  The node index along the x-axis.
        :param j:  The node index along the y-axis.
        :param k:  The node index along the z-axis.
        :return: True if node index (i,j,k) are a valid index that is if the node exists in the grid.
        """
        if np.any(np.array([i, j, k]) < 0):
            return False
        if np.any(np.array([i, j, k]) >= np.array([self.I, self.J, self.K])):
            return False
        return True

    def get_linear_index(self, i, j, k):
        """
        This method converts from a 3D node index space (i,j,k) to a linear node idx space.

        :param i:  The node index along the x-axis.
        :param j:  The node index along the y-axis.
        :param k:  The node index along the z-axis.
        :return:   The corresponding linear index of the 3D node index (i,j,k)
        """
        return i + self.I * (j + self.J * k)

    def get_node_value(self, i, j, k):
        """
        This mehthod retrieves the grid value stored at the node with indices (i,j,k). The method
        does not check if (i,j,k) are valid values. Hence if values are given outside the grid node
        range then one should expect and access exception. Similar not test is done for whether the
        internal value array has been allocated and filled with meaningful values.

        :param i:  The node index along the x-axis.
        :param j:  The node index along the y-axis.
        :param k:  The node index along the z-axis.
        :return:  The value stored at grid node with index (i,j,k). If grid has not been created properly
                  or (i,j,k) does not exist then the behavior is undefined.
        """
        return self.values[self.get_linear_index(i, j, k)]

    def get_node_coord(self, i, j, k):
        """
        This method computes the 3D spatial coordinates of the grid node with 3D indices (i,j,k). The method
        does not do any bounds testing. Hence if (i,j,k) are outside the valid range then one will get the
        coordinates of a "virtual" node that does not exist in the grid.

        :param i:  The node index along the x-axis.
        :param j:  The node index along the y-axis.
        :param k:  The node index along the z-axis.
        :return: The 3D spatial coordinates of the node with index (i,j,k).
        """
        # z = self.min_coord[2] + self.spacing[2] * k
        # y = self.min_coord[1] + self.spacing[1] * j
        # x = self.min_coord[0] + self.spacing[0] * i
        # return V3.make(x, y, z)
        return self.min_coord + np.multiply(
            self.spacing, np.array([i, j, k], dtype=np.float64)
        )

    def get_enclosing_cell_idx(self, p):
        """
        This method computes the 3D cell index (i,j,k) of the grid cell that contains
        the point p. If the cell index is out-of-bounds that is if p is outside the
        grid then we project the cell index onto the closest cell index that is valid. This
        is done by a simple projection to valid range of cell indices.

        :param p: The 3D spatial point.
        :return: The 3-tuple (i,j,k) that identifies the enclosing cell of the point p. Or the closest cell if p is
                 outside the grid.
        """
        idx = np.floor((p - self.min_coord) / self.spacing)
        # if index is outside the cells then project onto closest cell
        i = np.clip(int(idx[0]), 0, self.I - 2)
        j = np.clip(int(idx[1]), 0, self.J - 2)
        k = np.clip(int(idx[2]), 0, self.K - 2)
        return i, j, k


def get_value(grid, p):
    """
    This function compute the value of the scalar field that is sampled on the
    supplied regular 3D grid. It is assumed that linear shape (basis) functions are
    used for reconstruction of the scalar field. That is linear interpolation is the
    building block. If the point p is outside the grid bounding box
    then the closest grid cell is used to interpolate the value that should be used
    at p. This is a kind of Neumann condition that assumes the scalar field has constant
    slope outside the grid bounding box. If one wish to implement other boundary conditions
    the one can do so by first testing if the specified point is outside the grid bounding
    box and then apply the desired boundary condition. If the point p is inside then one can
    simply use the interpolated value of this function.

    :param grid: The 3D grid that stores a regular sampling of the scalar field
    :param p: The point where the value is evaluated
    :return: The value of the scalar field at position p
    """
    i, j, k = grid.get_enclosing_cell_idx(p)

    d000 = grid.get_node_value(i, j, k)
    d001 = grid.get_node_value(i, j, k + 1)
    d010 = grid.get_node_value(i, j + 1, k)
    d011 = grid.get_node_value(i, j + 1, k + 1)

    d100 = grid.get_node_value(i + 1, j, k)
    d101 = grid.get_node_value(i + 1, j, k + 1)
    d110 = grid.get_node_value(i + 1, j + 1, k)
    d111 = grid.get_node_value(i + 1, j + 1, k + 1)

    s = (p[0] - ((i * grid.spacing[0]) + grid.min_coord[0])) / grid.spacing[0]
    t = (p[1] - ((j * grid.spacing[1]) + grid.min_coord[1])) / grid.spacing[1]
    u = (p[2] - ((k * grid.spacing[2]) + grid.min_coord[2])) / grid.spacing[2]
    #
    #             011
    #            *------x11----* 111
    #           /|            /|
    #          / |           / |
    #     001 *------x01----* 101
    #         |  |          |  |
    #         |  *-----x10--|--* 110
    #         | /  010      | /
    #         |/            |/
    #     000 *------x00----* 100
    #
    x00 = (d100 - d000) * s + d000
    x01 = (d101 - d001) * s + d001
    x10 = (d110 - d010) * s + d010
    x11 = (d111 - d011) * s + d011
    y0 = (x10 - x00) * t + x00
    y1 = (x11 - x01) * t + x01
    z = (y1 - y0) * u + y0
    return z


def get_gradient(grid, p):
    """
    This function compute the gradient of the scalar field that is sampled on the
    supplied regular 3D grid. It is assumed that linear shape (basis) functions are
    used for reconstruction of the scalar field. That is linear interpolation is the
    building block. The gradient is computed by taking the spatial derivative of
    that interpolation equation. If the point p is outside the bounding box
    then the closest grid cell is used to interpolate the value that should be used
    at p. This is a kind of Neumann condition that assumes the scalar field has constant
    slope outside the grid bounding box. If one wish to implement other boundary conditions
    the one can do so by first testing if the specified point is outside the grid bounding
    box and then apply the desired boundary condition. If the point p is inside then one can
    simply use the interpolated value of this function.

    :param grid: The 3D grid that stores a regular sampling of the scalar field
    :param p: The point where the gradient is evaluated
    :return: The gradient value of the scalar field at position p
    """

    i, j, k = grid.get_enclosing_cell_idx(p)

    d000 = grid.get_node_value(i, j, k)
    d001 = grid.get_node_value(i, j, k + 1)
    d010 = grid.get_node_value(i, j + 1, k)
    d011 = grid.get_node_value(i, j + 1, k + 1)
    d100 = grid.get_node_value(i + 1, j, k)
    d101 = grid.get_node_value(i + 1, j, k + 1)
    d110 = grid.get_node_value(i + 1, j + 1, k)
    d111 = grid.get_node_value(i + 1, j + 1, k + 1)
    s = (p[0] - (i * grid.spacing[0] + grid.min_coord[0])) / grid.min_coord[0]
    t = (p[1] - (j * grid.spacing[1] + grid.min_coord[1])) / grid.min_coord[1]
    u = (p[2] - (k * grid.spacing[2] + grid.min_coord[2])) / grid.min_coord[2]

    #
    #             011
    #            *------x11----* 111
    #           /|            /|
    #          / |           / |
    #     001 *------x01----* 101
    #         |  |          |  |
    #         |  *-----x10--|--* 110
    #         | /  010      | /
    #         |/            |/
    #     000 *------x00----* 100
    #
    x00 = (d100 - d000) * s + d000
    x01 = (d101 - d001) * s + d001
    x10 = (d110 - d010) * s + d010
    x11 = (d111 - d011) * s + d011
    y0 = (x10 - x00) * t + x00
    y1 = (x11 - x01) * t + x01
    dx00_ds = d100 - d000
    dx01_ds = d101 - d001
    dx10_ds = d110 - d010
    dx11_ds = d111 - d011
    dy0_ds = (dx10_ds - dx00_ds) * t + dx00_ds
    dy1_ds = (dx11_ds - dx01_ds) * t + dx01_ds
    dy0_dt = x10 - x00
    dy1_dt = x11 - x01

    dp_ds = (dy1_ds - dy0_ds) * u + dy0_ds
    dp_dt = (dy1_dt - dy0_dt) * u + dy0_dt
    dp_du = y1 - y0

    ds_dx = 1.0 / grid.spacing[0]
    dt_dy = 1.0 / grid.spacing[1]
    du_dz = 1.0 / grid.spacing[2]

    dp_dx = dp_ds * ds_dx
    dp_dy = dp_dt * dt_dy
    dp_dz = dp_du * du_dz

    grad = np.array([dp_dx, dp_dy, dp_dz], dtype=np.float64)

    return grad.flatten()


def eval_on_grid(grid, func):
    """
    This function creates a value array for the supplied grid dat structure and then it evaluates the given
    function on each nodal coordinate position and store the value in the corresponding grid node.

    :param grid: The grid data structure to create a value array storing values of the given function.
    :param func: The given function that should be evaluated on each grid node. It is assumed to be a
                 scalar vector function mapping a 3D spatial location to a single scalar value.
    """
    grid.values = np.zeros(((grid.I * grid.J * grid.K), 1), dtype=np.float64)
    for k in range(grid.K):
        for j in range(grid.J):
            for i in range(grid.I):
                coord = grid.get_node_coord(i, j, k)
                row_idx = grid.get_linear_index(i, j, k)
                grid.values[row_idx] = func(coord)


def is_inside(grid, p, boundary=0.5):
    """
    This function test if the given spatial 3D point is inside the given grid and the boundary value.

    :param grid: The grid.
    :param p: The 3D spatial point.
    :param boundary: The boundary width. This specifies how far inside the grid bounding box the point has to be to be
                    considered inside. Using a value of zero means one is testing on the bounding box itself.
    :return: True if the given point is inside the given grid.
    """
    if boundary < 0:
        raise ValueError()
    if p[0] > (grid.max_coord[0] - boundary):
        return False
    if p[1] > (grid.max_coord[1] - boundary):
        return False
    if p[2] > (grid.max_coord[2] - boundary):
        return False

    if p[0] < (grid.min_coord[0] + boundary):
        return False
    if p[1] < (grid.min_coord[1] + boundary):
        return False
    if p[2] < (grid.min_coord[2] + boundary):
        return False
    return True


def write_matlab_file(filename, grid):
    """
    This function writes the grid data into a binary mMtlab mat file. This is convenient for getting data into
    Matlab and doing post-processing there.

    :param filename:  The path and filename of the file to make.
    :param grid: The grid instance that should be written to the given file
    """
    data = {
        "I": grid.I,
        "J": grid.J,
        "K": grid.K,
        "min_coord": grid.min_coord,
        "max_coord": grid.max_coord,
        "spacing": grid.spacing,
        "values": grid.values,
    }
    from scipy.io import savemat

    savemat(file_name=filename, mdict=data, appendmat=False)


def read_matlab_file(filename):
    """
    This function reads grid data from a boary Matlab mat file.

    :param filename:  The filename to read data from.
    :return: A new Grid instance with the data that was read.
    """
    from scipy.io import loadmat

    data = {}
    loadmat(file_name=filename, mdict=data, appendmat=False)
    I = int(data["I"])
    J = int(data["J"])
    K = int(data["K"])
    min_coord = data["min_coord"]
    max_coord = data["max_coord"]
    grid = Grid(min_coord=min_coord, max_coord=max_coord, I=I, J=J, K=K)
    grid.values = data["values"]
    return grid


def get_nodes_array(grid):
    """
    This function creates a flat array with the 3D spatial coordinates of all the nodes in the mesh.

    :param grid: The given grid from which one wants to know all the nodal coordinates.
    :return: A (I*J*K) by 3 array. First column holds x-values, second y-values and last column holds z-values.
    """
    nodes = np.zeros(((grid.I * grid.J * grid.K), 3), dtype=np.float64)
    for k in range(grid.K):
        for j in range(grid.J):
            for i in range(grid.I):
                coord = grid.get_node_coord(i, j, k)
                row_idx = grid.get_linear_index(i, j, k)
                nodes[row_idx, :] = coord
    return nodes


def create_signed_distance(V, F, I, J, K, boundary=0.5):
    """
    This function creates a grid instance that contains a signed distance field of the given mesh data.

    :param V:    Input vertex array of the mesh.
    :param F:    Input face array of the mesh.
    :param I:    The number of nodes along the x-axis in the resulting grid.
    :param J:    The number of nodes along the y-axis in the resulting grid.
    :param K:    The number of nodes along the z-axis in the resulting grid.
    :param boundary:  If boundary is zero then the grid bounding box is a tight fit around the mesh. Increasing this value enlarges the bounding box by the same distance.
    :return:          The new grid that contains the signed distance field.
    """
    if boundary < 0:
        raise ValueError()
    min_coord = V.min(axis=0) - boundary
    max_coord = V.max(axis=0) + boundary
    grid = Grid(min_coord, max_coord, I, J, K)
    nodes = get_nodes_array(grid)
    grid.values, _, _ = igl.signed_distance(nodes, V, F)
    return grid


def show_layer(grid, k):
    import matplotlib.pyplot as plt

    layer = grid.I * grid.J
    img = grid.values[k * layer : (k + 1) * layer].reshape((grid.I, grid.J))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Values")
    plt.imshow(img)
    ax.set_aspect("equal")
    # cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    # cax.get_xaxis().set_visible(False)
    # cax.get_yaxis().set_visible(False)
    # cax.patch.set_alpha(0)
    # cax.set_frame_on(False)
    plt.colorbar(orientation="vertical")
    plt.show()


def save_layer(grid, k, filename):
    import matplotlib.pyplot as plt

    layer = grid.I * grid.J
    img = grid.values[k * layer : (k + 1) * layer].reshape((grid.I, grid.J))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Values")
    plt.imshow(img)
    ax.set_aspect("equal")
    plt.colorbar(orientation="vertical")
    plt.savefig(filename, format="png", bbox_inches="tight")


class TestGrid:
    def __init__(self):
        self.test_signed_distance_field()
        self.test_valid_node_idx()
        self.test_get_enclosing_cell_idx()
        self.test_get_value(self.test_linear_func)
        self.test_get_gradient(self.test_linear_func, self.test_linear_grad)
        self.test_is_inside()

    def test_get_enclosing_cell_idx(self):
        import isl.math.vector3 as V3

        print(
            "---- test_get_enclosing_cell_idx ---------------------------------------"
        )
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 4
        J = 4
        K = 4

        A = Grid(min_coord, max_coord, I, J, K)

        p = V3.make(0, 0, 0)
        idx = A.get_enclosing_cell_idx(p)
        print(p, idx)

        p = V3.make(-0.5, -0.5, -0.5)
        idx = A.get_enclosing_cell_idx(p)
        print(p, idx)

        p = V3.make(0.5, 0.5, 0.5)
        idx = A.get_enclosing_cell_idx(p)
        print(p, idx)

        p = V3.make(-1, -1, -1)
        idx = A.get_enclosing_cell_idx(p)
        print(p, idx)

        p = V3.make(1, 1, 1)
        idx = A.get_enclosing_cell_idx(p)
        print(p, idx)

        p = V3.make(-2, -2, -2)
        idx = A.get_enclosing_cell_idx(p)
        print(p, idx)

        p = V3.make(2, 2, 2)
        idx = A.get_enclosing_cell_idx(p)
        print(p, idx)
        print("")
        print("-------------------------------------------")

    def test_valid_node_idx(self):
        import isl.math.vector3 as V3

        print("--- test_valid_node_idx ----------------------------------------")
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 4
        J = 4
        K = 4
        A = Grid(min_coord, max_coord, I, J, K)
        print(A.is_valid_node_index(-1, 0, 0), "= FALSE")
        A = Grid(min_coord, max_coord, I, J, K)
        print(A.is_valid_node_index(0, 0, 0), "= TRUE")
        A = Grid(min_coord, max_coord, I, J, K)
        print(A.is_valid_node_index(0, 0, 3), "= TRUE")
        A = Grid(min_coord, max_coord, I, J, K)
        print(A.is_valid_node_index(0, 4, 3), "= FALSE")
        print("")
        print("-------------------------------------------")

    def test_linear_func(self, p):
        return p[0]

    def test_linear_grad(self, p):
        return np.array([1, 0, 0])

    def test_get_value(self, func):
        import isl.math.vector3 as V3

        print("----- test_get_value --------------------------------------")
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = Grid(min_coord, max_coord, I, J, K)
        eval_on_grid(A, func)

        p = V3.make(0, 0, 0)
        val = get_value(A, p)
        print(func(p), val)

        p = V3.make(-0.5, -0.5, -0.5)
        val = get_value(A, p)
        print(func(p), val)

        p = V3.make(0.5, 0.5, 0.5)
        val = get_value(A, p)
        print(func(p), val)

        p = V3.make(-1, -1, -1)
        val = get_value(A, p)
        print(func(p), val)

        p = V3.make(1, 1, 1)
        val = get_value(A, p)
        print(func(p), val)

        p = V3.make(-2, -2, -2)
        val = get_value(A, p)
        print(func(p), val)

        p = V3.make(2, 2, 2)
        val = get_value(A, p)
        print(func(p), val)
        print("")
        print("-------------------------------------------")

    def test_get_gradient(self, func, grad):
        import isl.math.vector3 as V3

        print("------ test_get_gradient -------------------------------------")
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 5
        J = 5
        K = 5
        A = Grid(min_coord, max_coord, I, J, K)
        eval_on_grid(A, func)

        p = V3.make(0, 0, 0)
        g = get_gradient(A, p)
        print(grad(p), g)

        p = V3.make(-0.5, -0.5, -0.5)
        g = get_gradient(A, p)
        print(grad(p), g)

        p = V3.make(0.5, 0.5, 0.5)
        g = get_gradient(A, p)
        print(grad(p), g)

        p = V3.make(-1, -1, -1)
        g = get_gradient(A, p)
        print(grad(p), g)

        p = V3.make(1, 1, 1)
        g = get_gradient(A, p)
        print(grad(p), g)

        p = V3.make(-2, -2, -2)
        g = get_gradient(A, p)
        print(grad(p), g)

        p = V3.make(2, 2, 2)
        g = get_gradient(A, p)
        print(grad(p), g)
        print("")
        print("-------------------------------------------")

    def test_is_inside(self):
        import isl.math.vector3 as V3

        print("--- test_valid_node_idx ----------------------------------------")
        min_coord = V3.make(-1, -1, -1)
        max_coord = V3.make(1, 1, 1)
        I = 4
        J = 4
        K = 4
        A = Grid(min_coord, max_coord, I, J, K)

        boundary = 0
        p = V3.make(-1, -1, -1)
        print(is_inside(A, p, boundary), "= True")

        boundary = 0
        p = V3.make(0, 0, 0)
        print(is_inside(A, p, boundary), "= True")

        boundary = 0
        p = V3.make(1, 1, 1)
        print(is_inside(A, p, boundary), "= True")

        boundary = 0
        p = V3.make(-1.1, -1.1, -1.1)
        print(is_inside(A, p, boundary), "= False")

        boundary = 0
        p = V3.make(1.1, 1.1, 1.1)
        print(is_inside(A, p, boundary), "= False")

        boundary = 0.5
        p = V3.make(-1, -1, -1)
        print(is_inside(A, p, boundary), "= False")

        boundary = 0.5
        p = V3.make(0, 0, 0)
        print(is_inside(A, p, boundary), "= True")

        boundary = 0.5
        p = V3.make(1, 1, 1)
        print(is_inside(A, p, boundary), "= False")

        boundary = 0.5
        p = V3.make(-1.1, -1.1, -1.1)
        print(is_inside(A, p, boundary), "= False")

        boundary = 0.5
        p = V3.make(1.1, 1.1, 1.1)
        print(is_inside(A, p, boundary), "= False")

        print("")
        print("-------------------------------------------")

    def test_signed_distance_field(self):
        import isl.math.vector3 as V3

        print("--- test_signed_distance_field ----------------------------------------")
        V, F = igl.read_triangle_mesh("../../data/armadillo.obj")
        A = create_signed_distance(V, F, 98, 98, 98, 0.5)
        write_matlab_file("../../data/test.mat", A)
        B = read_matlab_file("../../data/test.mat")

        print(A.I, B.I)
        print(A.J, B.J)
        print(A.K, B.K)
        print(A.min_coord, B.min_coord)
        print(A.max_coord, B.max_coord)
        print(A.spacing, B.spacing)
        print(A.values, B.values)
        show_layer(A, 48)
        print("")
        print("-------------------------------------------")


if __name__ == "__main__":
    test = TestGrid()
