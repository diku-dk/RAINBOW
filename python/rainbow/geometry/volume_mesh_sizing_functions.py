import numpy as np
import igl
import rainbow.geometry.grid3 as grid3


def create_linear_field_around_mesh(V, I, J, K, min_val, max_val, ax=0, boundary=0.5):
    """
    This function creates a grid instance that contains an increase sizing function of the given mesh data.

    :param V:        Input vertex array of the mesh.
    :param I:        The number of nodes along the x-axis in the resulting grid.
    :param J:        The number of nodes along the y-axis in the resulting grid.
    :param K:        The number of nodes along the z-axis in the resulting grid.
    :param min_val:  The minimum value of the sizing function
    :param max_val:  The maximum value of the sizing function
    :param ax:       The axis of the increase.
    :param boundary: If boundary is zero then the grid bounding box is a tight fit around the mesh. Increasing this
                     value enlarges the bounding box by the same distance.
    :return:         The new grid that contains the sizing function.
    """
    if boundary < 0:
        raise ValueError()
    if ax < 0 or ax > 2:
        raise ValueError()
    min_coord = V.min(axis=0) - boundary
    max_coord = V.max(axis=0) + boundary
    grid = grid3.Grid(min_coord, max_coord, I, J, K)
    V = grid3.get_nodes_array(grid)
    ax_vals = [I, J, K]

    return_list = np.zeros((len(V), 1))
    indexes = np.arange(0, len(V))
    Vidx = np.c_[V, indexes]  # Adding index before we sort
    order = Vidx[Vidx[:, ax].argsort()]  # Sorting after the assigned -axis

    increase = (max_val - min_val) / (ax_vals[ax] - 1)
    return_list[0] = min_val

    for i in range(1, len(V)):
        if order[i][ax] == order[i - 1][ax]:
            return_list[i] = return_list[i - 1]
        else:
            return_list[i] = return_list[i - 1] + increase

    Aidx = np.c_[
        return_list, order[:, 3]
    ]  # We need to sort the list back to the original order
    return_list = Aidx[Aidx[:, 1].argsort()][:, 0]  # Remove the index

    grid.values = return_list
    return grid


def create_signed_distance_2_mesh(V, V2, F2, I, J, K, boundary=0.5):
    """
    This function creates a grid instance that contains a signed distance field of the given mesh data.

    :param V:    Input vertex array of the mesh.
    :param V2:   Input vertex array of the second mesh.
    :param F2:   Input face array of the second mesh.
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
    grid = grid3.Grid(min_coord, max_coord, I, J, K)
    nodes = grid3.get_nodes_array(grid)
    grid.values, _, _ = igl.signed_distance(nodes, V2, F2)
    return grid


def const(c):
    """
    :c: The constant value to returned
    :returns: A constant function
    """
    return lambda p: c


def signed_distance_field(V=None, F=None, grid=None):
    """
    :V:    N x 3 array of vertices
    :F:    M x 3 array of faces
    :grid: grid3 object containing signed distance field
    :returns: A function that given a point p returns the distance to some surface
    :remarks: Function can be called with either a mesh (V, F) or a grid where
              the signed distance function of some surface is sampled at the grid 
              nodes and then interpolated at p here.
    """
    if grid is None:
        return lambda p: igl.signed_distance(np.array([p]), V, F)[0]
    else:
        return lambda p: GRID3.get_value(grid, p)


def curvature(phi, dx=0.5, dy=0.5, dz=0.5):
    """
    :phi: A 3-dimensional signed distance function of some mesh
    :dx:  The step size in the x-direction for the finite difference approximation
    :dy:  The step size in the y-direction for the finite difference approximation
    :dz:  The step size in the z-direction for the finite difference approximation
    :returns: A function that given a point p returns the mean curvature at p
    """
    inv_2dx = 1.0 / (2.0 * dx)
    inv_2dy = 1.0 / (2.0 * dy)
    inv_2dz = 1.0 / (2.0 * dz)
    inv_dxdx = 1.0 / (dx * dx)
    inv_dydy = 1.0 / (dy * dy)
    inv_dzdz = 1.0 / (dz * dz)
    inv_4dxdy = 1.0 / (4.0 * dx * dy)
    inv_4dxdz = 1.0 / (4.0 * dx * dz)
    inv_4dydz = 1.0 / (4.0 * dy * dz)
    kappa_max = 1 / (max(dz, max(dx, dy)))
    kappa_min = -kappa_max

    def curvature_at_point(p):
        i = p[0]
        j = p[1]
        k = p[2]
        d000 = 2 * phi(np.array([i, j, k], dtype=np.float64))
        dp00 = phi(np.array([i + dx, j, k], dtype=np.float64))
        dm00 = phi(np.array([i - dx, j, k], dtype=np.float64))
        d0p0 = phi(np.array([i, j + dy, k], dtype=np.float64))
        d0m0 = phi(np.array([i, j - dy, k], dtype=np.float64))
        d00p = phi(np.array([i, j, k + dz], dtype=np.float64))
        d00m = phi(np.array([i, j, k - dz], dtype=np.float64))
        dpp0 = phi(np.array([i + dx, j + dy, k], dtype=np.float64))
        dmp0 = phi(np.array([i - dx, j + dy, k], dtype=np.float64))
        dpm0 = phi(np.array([i + dx, j - dy, k], dtype=np.float64))
        dmm0 = phi(np.array([i - dx, j - dy, k], dtype=np.float64))
        dp0p = phi(np.array([i + dx, j, k + dz], dtype=np.float64))
        dm0p = phi(np.array([i - dx, j, k + dz], dtype=np.float64))
        dp0m = phi(np.array([i + dx, j, k - dz], dtype=np.float64))
        dm0m = phi(np.array([i - dx, j, k - dz], dtype=np.float64))
        d0pp = phi(np.array([i, j + dy, k + dz], dtype=np.float64))
        d0pm = phi(np.array([i, j + dy, k - dz], dtype=np.float64))
        d0mp = phi(np.array([i, j - dy, k + dz], dtype=np.float64))
        d0mm = phi(np.array([i, j - dy, k - dz], dtype=np.float64))

        Dx = (dp00 - dm00) * inv_2dx
        Dy = (d0p0 - d0m0) * inv_2dy
        Dz = (d00p - d00m) * inv_2dz
        Dxx = (dp00 + dm00 - d000) * inv_dxdx
        Dyy = (d0p0 + d0m0 - d000) * inv_dydy
        Dzz = (d00p + d00m - d000) * inv_dzdz
        Dxy = (dpp0 - dmp0 - dpm0 + dmm0) * inv_4dxdy
        Dxz = (dp0p - dm0p - dp0m + dm0m) * inv_4dxdz
        Dyz = (d0pp - d0pm - d0mp + d0mm) * inv_4dydz

        g = np.array([Dx, Dy, Dz])
        l = np.linalg.norm(g)
        if l < 0.5:
            l = 1.0

        num = (
            Dxx * (Dy ** 2 + Dz ** 2)
            + Dyy * (Dx ** 2 + Dz ** 2)
            + Dzz * (Dx ** 2 + Dy ** 2)
            - 2 * (Dxy * Dx * Dy + Dxz * Dx * Dz + Dyz * Dy * Dz)
        )

        denom = l ** 3

        kappa = np.clip(num / denom, kappa_min, kappa_max)
        return kappa

    return curvature_at_point


def scale(SF1, SF2):
    """
    :SF1: A sizing field function
    :SF2: A sizing field function
    :returns: A function that given p returns the multiplication
              of the given sizing functions at p
    """
    return lambda p: SF1(p) * SF2(p)


def absolute(SF):
    """
    :SF: A sizing field function
    :returns: The sizing field function in composition
              with the absolute operator
    """
    return lambda p: np.absolute(SF(p))


def clamp(SF, min_SF, max_SF):
    """
    :SF:     Sizing field function
    :min_SF: Sizing field function
    :max_SF: Sizing field function
    :returns: A sizing field function where each value returned by SF 
              is clamped between the corresponding values returned by min_SF and max_SF
    """
    return lambda p: np.clip(SF(p), min_SF(p), max_SF(p))


def add(SF1, SF2):
    """
    :SF1: A sizing field function
    :SF2: A sizing field function
    :returns: A function that given p returns the addition of SF1(p) and SF2(p)
    """
    return lambda p: SF1(p) + SF2(p)


def less(SF1, SF2):
    """
    :SF1: A sizing field function
    :SF2: A sizing field function
    :returns: True if SF1(p) is less than SF2(p)
    """
    return lambda p: SF1(p) < SF2(p)


def greater(SF1, SF2):
    """
    :SF1: A sizing field function
    :SF2: A sizing field function
    :returns: True if SF1(p) is greater than SF2(p)
    """
    return lambda p: SF1(p) > SF2(p)


def if_statement(cond, SF1, SF2):
    """
    :cond: Some condition that given x evaluates to True or False
    :SF1:  Some function to be performed if cond is True
    :SF2:  Some function to be performed if cond is False
    :returns: A function that given an input x performs SF1(p) if cond(p)
              evaluates to true, otherwise SF2(p) is performed
    """
    return lambda p: SF1(p) if cond(p) else SF2(p)


def sample_vertices(V, SF):
    """
    :V:  N x 3 array of vertices
    :SF: Some sizing function
    :returns: N x 1 array where each entry, i, is SF(V[i]) 
    """
    A = np.zeros(len(V))
    for i, v in enumerate(V):
        A[i] = SF(v)
    return A
