import numpy as np
from numba import cuda, float64, int32
import math

## transform functions of Grid class to GPU
## device functions information can be found in python/rainbow/geometry/grid3.py 

@cuda.jit(device=True)
def get_enclosing_cell_idx_device(p, min_coord, spacing, I, J, K, result):
    idx_x = math.floor((p[0] - min_coord[0]) / spacing[0])
    idx_y = math.floor((p[1] - min_coord[1]) / spacing[1])
    idx_z = math.floor((p[2] - min_coord[2]) / spacing[2])

    i = int(min(max(int(idx_x), 0), I - 2))
    j = int(min(max(int(idx_y), 0), J - 2))
    k = int(min(max(int(idx_z), 0), K - 2))

    result[0] = i
    result[1] = j
    result[2] = k


@cuda.jit(device=True)
def get_linear_index_device(i, j, k, I, J):
    return i + I * (j + J * k)

@cuda.jit(device=True)
def get_node_value_device(i, j, k, values, I, J):
    # values are the SDF 
    return values[get_linear_index_device(i, j, k, I, J)]

@cuda.jit(device=True)
def get_gradient_device(p, min_coord, spacing, I, J, K, values, result):
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
def is_inside_device(p, min_coord, max_coord, boundary = 0.5):
    boudnary = 0.5
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
def get_linear_index_device(i, j, k, I, J):
    return int(i + I * (j + J * k))

@cuda.jit(device=True)
def get_node_value_device(i, j, k, values, I, J):
    return values[get_linear_index_device(i, j, k, I, J)]

@cuda.jit(device=True)
def get_value_device(p, min_coord, spacing, I, J, K, values):
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



## GPU kernel functions for parallel computing
@cuda.jit
def get_enclosing_cell_idx_kernel(points, min_coord, spacing, I, J, K, res):
    gid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if gid < points.shape[0]:
        p = points[gid]

        res[gid] = get_enclosing_cell_idx_device(p, min_coord, spacing, I, J, K)

@cuda.jit
def get_gradient_kernel(points, min_coord, spacing, I, J, K, values, res):
    gid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if gid < points.shape[0]:
        p = points[gid]
        i, j, k = get_enclosing_cell_idx_device(p, min_coord, spacing, I, J, K)

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

        res[gid, 0] = dp_dx
        res[gid, 1] = dp_dy
        res[gid, 2] = dp_dz









