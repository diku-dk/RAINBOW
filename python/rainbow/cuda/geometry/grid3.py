import numpy as np
from numba import cuda, float64, int32, boolean
import math


@cuda.jit(device=True)
def get_enclosing_cell_idx_device(p: float64[:], min_coord: float64[:], spacing: float64[:], I: int32, J: int32, K: int32, result: int32[:]):
    """ This method computes the 3D cell index (i,j,k) of the grid cell that contains
        the point p. If the cell index is out-of-bounds that is if p is outside the
        grid then we project the cell index onto the closest cell index that is valid. This
        is done by a simple projection to valid range of cell indices.

    Args:
        p (float64[:]): The 3D spatial point.
        min_coord (float64[:]): The minimum corner of the grid.
        spacing (float64[:]): The grid spacing.
        I (int32): The number of cells in the x-direction of the grid.
        J (int32): The number of cells in the y-direction of the grid.
        K (int32): The number of cells in the z-direction of the grid.
        result (float64[:]): The 3-tuple (i,j,k) that identifies the enclosing cell of the point p. Or the closest cell if p is outside the grid.
    """
    idx_x = math.floor((p[0] - min_coord[0]) / spacing[0])
    idx_y = math.floor((p[1] - min_coord[1]) / spacing[1])
    idx_z = math.floor((p[2] - min_coord[2]) / spacing[2])

    i = int32(min(max(idx_x, 0), I - 2))
    j = int32(min(max(idx_y, 0), J - 2))
    k = int32(min(max(idx_z, 0), K - 2))

    result[0] = i
    result[1] = j
    result[2] = k


@cuda.jit(device=True)
def get_linear_index_device(i: int32, j: int32, k: int32, I: int32, J: int32) -> int32:
    """ This method converts from a 3D node index space (i,j,k) to a linear node idx space.

    Args:
        i (int32): The node index along the x-axis.
        j (int32): The node index along the y-axis.
        k (int32): The node index along the z-axis.
        I (int32): The number of cells in the x-direction of the grid.
        J (int32): The number of cells in the y-direction of the grid.
        :return:   
    Returns:
        int32: The corresponding linear index of the 3D node index (i,j,k)
    """
    return int32(i + I * (j + J * k))


@cuda.jit(device=True)
def get_node_value_device(i: int32, j: int32, k: int32, values: float64[:], I: int32, J: int32) -> float64:
    """ This method retrieves the grid value stored at the node with indices (i,j,k). The method
        does not check if (i,j,k) are valid values. Hence if values are given outside the grid node
        range then one should expect and access exception. Similar not test is done for whether the
        internal value array has been allocated and filled with meaningful values.

    Args:
        i (int32): The node index along the x-axis.
        j (int32): The node index along the y-axis.
        k (int32): The node index along the z-axis.
        values (float64[:]): It is an array storing signed distances of grid points to a given mesh.
        I (int32): The number of cells in the x-direction of the grid.
        J (int32): The number of cells in the y-direction of the grid.
        result (float64[:]): The value stored at grid node with index (i,j,k). If grid has not been created properly
                            or (i,j,k) does not exist then the behavior is undefined.
    """
    idx = get_linear_index_device(i, j, k, I, J)
    return values[idx]


@cuda.jit(device=True)
def get_gradient_device(p: float64[:], min_coord: float64[:], spacing: float64[:], I: int32, J: int32, K: int32, values: float64[:], result: float64[:]):
    """ This function compute the gradient of the scalar field that is sampled on the
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

    Args:
        p (float64[:]): The 3D spatial point.
        min_coord (float64[:]): The minimum corner of the grid.
        spacing (float64[:]): The grid spacing.
        I (int32): The number of cells in the x-direction of the grid.
        J (int32): The number of cells in the y-direction of the grid.
        K (int32): The number of cells in the z-direction of the grid.
        values (float64[:]): It is an array storing signed distances of grid points to a given mesh.
        result (float64[:]): The gradient value of the scalar field at position p
    """
    arr = cuda.local.array(3, dtype=float64)
    get_enclosing_cell_idx_device(p, min_coord, spacing, I, J, K, arr)
    i = arr[0]
    j = arr[1]
    k = arr[2]

    d000 = get_node_value_device(i, j, k, values, I, J)
    d001 = get_node_value_device(i, j, k + 1, values, I, J)
    d010 = get_node_value_device(i, j + 1, k, values, I, J)
    d011 = get_node_value_device(i, j + 1, k + 1, values, I, J)
    d100 = get_node_value_device(i + 1, j, k, values, I, J)
    d101 = get_node_value_device(i + 1, j, k + 1, values, I, J)
    d110 = get_node_value_device(i + 1, j + 1, k, values, I, J)
    d111 = get_node_value_device(i + 1, j + 1, k + 1, values, I, J)

    s = (p[0] - (i * spacing[0] + min_coord[0])) / min_coord[0]
    t = (p[1] - (j * spacing[1] + min_coord[1])) / min_coord[1]
    u = (p[2] - (k * spacing[2] + min_coord[2])) / min_coord[2]

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

    ds_dx = 1.0 / spacing[0]
    dt_dy = 1.0 / spacing[1]
    du_dz = 1.0 / spacing[2]

    dp_dx = dp_ds * ds_dx
    dp_dy = dp_dt * dt_dy
    dp_dz = dp_du * du_dz

    result[0] = dp_dx
    result[1] = dp_dy
    result[2] = dp_dz


@cuda.jit(device=True)
def is_inside_device(p: float64[:], min_coord: float64[:], max_coord: float64[:], boundary: float64) -> boolean:
    """ This function test if the given spatial 3D point is inside the given grid and the boundary value.


    Args:
        p (float64[:]): The 3D spatial point.
        min_coord (float64[:]): The minimum corner of the grid.
        max_coord (float64[:]): The maximum corner of the grid.
        boundary (float, optional): The boundary width. This specifies how far inside the grid bounding box the     point has to be to be considered inside. Using a value of zero means one is testing on the bounding box itself.. Defaults to 0.5.

    Returns:
        bool: True if the given point is inside the given grid.
    """
    if p[0] > (max_coord[0] - boundary):
        return False
    if p[1] > (max_coord[1] - boundary):
        return False
    if p[2] > (max_coord[2] - boundary):
        return False

    if p[0] < (min_coord[0] + boundary):
        return False
    if p[1] < (min_coord[1] + boundary):
        return False
    if p[2] < (min_coord[2] + boundary):
        return False
    return True


@cuda.jit(device=True)
def get_value_device(p: float64[:], min_coord: float64[:], spacing: float64[:], I: int32, J: int32, K: int32, values: float64[:]) -> float64:
    """ This function compute the value of the scalar field that is sampled on the
        supplied regular 3D grid. It is assumed that linear shape (basis) functions are
        used for reconstruction of the scalar field. That is linear interpolation is the
        building block. If the point p is outside the grid bounding box
        then the closest grid cell is used to interpolate the value that should be used
        at p. This is a kind of Neumann condition that assumes the scalar field has constant
        slope outside the grid bounding box. If one wish to implement other boundary conditions
        the one can do so by first testing if the specified point is outside the grid bounding
        box and then apply the desired boundary condition. If the point p is inside then one can
        simply use the interpolated value of this function.

    Args:
        p (float64[:]): The 3D spatial point.
        min_coord (float64[:]): The minimum corner of the grid.
        spacing (float64[:]): The grid spacing.
        I (int32): The number of cells in the x-direction of the grid.
        J (int32): The number of cells in the y-direction of the grid.
        K (int32): The number of cells in the z-direction of the grid.
        values (float64[:]): It is an array storing signed distances of grid points to a given mesh.

    Returns:
        float64: The value of the scalar field at position p
    """
    temp_res = cuda.local.array(3, dtype=int32)
    get_enclosing_cell_idx_device(p, min_coord, spacing, I, J, K, temp_res)
    i = temp_res[0]
    j = temp_res[1]
    k = temp_res[2]
    
    d000 = get_node_value_device(i, j, k, values, I, J)
    d001 = get_node_value_device(i, j, k + 1, values, I, J)
    d010 = get_node_value_device(i, j + 1, k, values, I, J)
    d011 = get_node_value_device(i, j + 1, k + 1, values, I, J)

    d100 = get_node_value_device(i + 1, j, k, values, I, J)
    d101 = get_node_value_device(i + 1, j, k + 1, values, I, J)
    d110 = get_node_value_device(i + 1, j + 1, k, values, I, J)
    d111 = get_node_value_device(i + 1, j + 1, k + 1, values, I, J)

    s = (p[0] - ((i * spacing[0]) + min_coord[0])) / spacing[0]
    t = (p[1] - ((j * spacing[1]) + min_coord[1])) / spacing[1]
    u = (p[2] - ((k * spacing[2]) + min_coord[2])) / spacing[2]

    x00 = (d100 - d000) * s + d000
    x01 = (d101 - d001) * s + d001
    x10 = (d110 - d010) * s + d010
    x11 = (d111 - d011) * s + d011
    y0 = (x10 - x00) * t + x00
    y1 = (x11 - x01) * t + x01
    z = (y1 - y0) * u + y0
    return z