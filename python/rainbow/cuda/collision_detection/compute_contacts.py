import numpy as np
from numba import cuda, float64

import rainbow.cuda.math.vec as Vector
import rainbow.cuda.math.matrix as Matrix
import rainbow.cuda.math.linalg as LinAlg
import rainbow.cuda.geometry.barycentric as BC
import rainbow.cuda.geometry.grid3 as Grid3


@cuda.jit(device=True)
def xform_triangle_to_model_space_device(P, X, X0, P0):
    """ Converts a world space triangle into the material space of a tetrahedron.

    Args:
        P (float64[:]): The input triangle points in world space
        X (float64[:, :]): The input tetrahedron corners in world space.
        X0 (float64[:, :]): The input tetrahedron corners in material space.
        P0 (float64[:]): The values are the triangle corner points in tetrahedron B's material coordinate space.
    """
    Matrix.mat33_zero(P0)

    w0 = cuda.local.array(4, dtype=float64)
    w1 = cuda.local.array(4, dtype=float64)
    w2 = cuda.local.array(4, dtype=float64)

    BC.compute_barycentric_tetrahedron_device(X[0], X[1], X[2], X[3], P[0], w0)
    BC.compute_barycentric_tetrahedron_device(X[0], X[1], X[2], X[3], P[1], w1)
    BC.compute_barycentric_tetrahedron_device(X[0], X[1], X[2], X[3], P[2], w2)

    t = cuda.local.array((3, 4), dtype=float64)
    Matrix.mat43_T(X0, t)

    Matrix.mat34_dot_vec4(t, w0, P0[0])
    Matrix.mat34_dot_vec4(t, w1, P0[1])
    Matrix.mat34_dot_vec4(t, w2, P0[2])


@cuda.jit(device=True)
def compute_omega_device(p, X, omega):
    """ Compute the barycentric coordinates for a point p of a tetrahedron

    Args:
        p (float64[:]): The contact point in the model space of a body 
        X (float64[:, :]): The corner points of a tetrahedron in world coordinates.
        omega (float64[:]): the barycentric coordinates  w1, w2, w3 and w4.
    """
    BC.compute_barycentric_tetrahedron_device(X[0], X[1], X[2], X[3], p, omega)


@cuda.jit(device=True)
def compute_p_device(p, X, omega, result):
    """ Compute the contact point in the world space of a body

    Args:
        p (float64[:]): The contact point in the model space of a body 
        X (float64[:, :]): The corner points of a tetrahedron in world coordinates.
        omega (float64[:]): the barycentric coordinates  w1, w2, w3 and w4.
        result (float64[:]): The contact point in the world space of a body
    """
    t = cuda.local.array((3, 4), dtype=float64)
    Matrix.mat43_T(X, t)
    Matrix.mat34_dot_vec4(t, omega, result)


@cuda.jit(device=True)
def compute_n_device(n, X, X0, result):
    """ Compute the contact normal in the world space of a body

    Args:
        n (float64[:]): The contact normal in the model space of a body
        X (flaot64[:, :]): The corner points of a tetrahedron in world coordinates.
        X0 (flaot64[:, :]): The corner points of a tetrahedron in material coordinates.
        result (float64[:]): The contact normal in the world space of a body
    """
    D = cuda.local.array((3,3), dtype=float64)
    D0 = cuda.local.array((3, 3), dtype=float64)
    for i in range(3):
        for j in range(3):
            D[i, j] = X[i, j] - X[3, j]
            D0[i, j] = X0[i, j] - X0[3, j]
    solve_res = cuda.local.array(3, dtype=float64)
    LinAlg.cramer_solver(D0, n, solve_res)
    Matrix.mat33_dot_vec3(D, solve_res, result)


@cuda.jit(device=True)
def compute_contacts_device( idx_triA, idx_triB, A_owners, B_owners,
                             A_x, B_x, B_x0, A_surface, A_T, B_T,  
                             B_grid_min_coord, B_grid_max_coord, B_grid_spacing, 
                             B_grid_I, B_grid_J, B_grid_K, B_grid_values, 
                             max_iteration, optimization_tolerance, envelope, boundary, result):
    """ Compute the contact points between a triangle and a tetrahedron.

    Args:
        idx_triA (int32): The index of the triangle in the triangle array of body A.
        idx_triB (int32): The index of the triangle in the triangle array of body B.
        A_owners (dict): The dict of the tetrahedra of body A that own the triangle.
        B_owners (dict): The dict of the tetrahedra of body B that own the triangle.
        A_x (float64[:, :]): The vertices in deformed coordinates of body A.
        B_x (float64[:, :]): The vertices in deformed coordinates of body B.
        B_x0 (float64[:, :]): The vertices in material coordinates of body B.
        A_surface (int32[:, :]): The indices of the vertices of the triangle of body A.
        A_T (int32[:, :]): The indices of the vertices of the tetrahedra of body A.
        B_T (int32[:, :]): The indices of the vertices of the tetrahedra of body B.
        B_grid_min_coord (float64): The minimum coordinate of the grid of body B.
        B_grid_max_coord (float64): The maximum coordinate of the grid of body B.
        B_grid_spacing (float64[:]): The spacing of the grid of body B.
        B_grid_I (int32): The number of cells in the x-direction of the grid.
        B_grid_J (int32): The number of cells in the y-direction of the grid.
        B_grid_K (int32): The number of cells in the z-direction of the grid.
        B_grid_values (float64[:]): It is an array storing signed distances of grid points to a given mesh.
        max_iteration (int32): Maximum number of Gauss-Seidel iterations.
        optimization_tolerance (float64): The tolerance for the frank wolfe collision detection algorithm.
        envelope (float64): Any geometry within this distance generates a contact point.
        boundary (float64):  The boundary width. This specifies how far inside the grid bounding box the     point has to be to be considered inside.
        result (dcit): The data of the contact point computation for creating a ContactPoint instance on CPU.
    """
    idx_tetA = A_owners[idx_triA][0]
    idx_tetB = B_owners[idx_triB][0]

    P = cuda.local.array((3, 3), dtype=float64)
    XB = cuda.local.array((4, 3), dtype=float64)
    X0B = cuda.local.array((4, 3), dtype=float64)

    surface_indices = A_surface[idx_triA] 
    for j in range(3):
        for k in range(3): 
            P[j, k] = A_x[surface_indices[j], k]
        
    tet_indices = B_T[idx_tetB] 
    for j in range(4):
        for k in range(3): 
            XB[j, k] = B_x[tet_indices[j], k]
            X0B[j, k] = B_x0[tet_indices[j], k]

    P0 = cuda.local.array((3, 3), dtype=float64)
    xform_triangle_to_model_space_device(P, XB, X0B, P0) 
        
    gradient_norms = cuda.local.array(3, dtype=float64)
    gradient0 = cuda.local.array(3, float64)
    gradient1 = cuda.local.array(3, float64)
    gradient2 = cuda.local.array(3, float64)

    Grid3.get_gradient_device(P0[0], B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, gradient0)
    Grid3.get_gradient_device(P0[1], B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, gradient1)
    Grid3.get_gradient_device(P0[2], B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, gradient2)

    gradient_norms[0] = Vector.vec3_norm(gradient0)
    gradient_norms[1] = Vector.vec3_norm(gradient1)
    gradient_norms[2] = Vector.vec3_norm(gradient2)

    x_i = cuda.local.array(3, dtype=float64)
    for i in range(3):
        x_i[i] = P0[Vector.argmin(gradient_norms, 3)][i]

    for i in range(max_iteration):
        t = cuda.local.array(3, dtype=float64)
        Grid3.get_gradient_device(x_i, B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, t)

        objectives = cuda.local.array(3, dtype=float64)
        for j in range(3):
            objectives[j] = Vector.vec3_dot(P0[j], t)
            
        vertex = Vector.argmin(objectives, 3)
        s_i = P0[vertex]
        alpha = 2 / (i + 2)

        sx_sub = cuda.local.array(3, dtype=float64)
        Vector.vec3_sub(s_i, x_i, sx_sub)

        sx_sub_alpha = cuda.local.array(3, dtype=float64)
        Vector.vec3_mut_scalar(sx_sub, alpha, sx_sub_alpha)

        x_i_new = cuda.local.array(3, dtype=float64)
        Vector.vec3_add(x_i, sx_sub_alpha, x_i_new)

        for j in range(3):
            x_i[j] = x_i_new[j]

        if objectives[vertex] > optimization_tolerance:
            break
        
        # contact point generation
    if Grid3.is_inside_device(x_i, B_grid_min_coord, B_grid_max_coord, 0.5):
        phi = Grid3.get_value_device(x_i, B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values)
        if phi < envelope:
            gap = phi
            n_res = cuda.local.array(3, dtype=float64)
            Grid3.get_gradient_device(x_i, B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, n_res)
            if Vector.vec3_norm(n_res) > 0:
                XA = cuda.local.array((4, 3), dtype=float64)
                indices = A_T[idx_tetA]
                for i in range(4):
                    for j in range(3):
                        XA[i][j] = A_x[indices[i], j]
                    
                omegaB = cuda.local.array(4, dtype=float64)
                compute_omega_device(x_i, X0B, omegaB)
                p = cuda.local.array(3, dtype=float64)
                compute_p_device(x_i, XB, omegaB, p)
                omegaA = cuda.local.array(4, dtype=float64)
                compute_omega_device(p, XA, omegaA)
                n_res_new = cuda.local.array(3, dtype=float64)
                compute_n_device(n_res, XB, X0B, n_res_new)
                unit_n = cuda.local.array(3, dtype=float64)
                Vector.vec3_unit(n_res_new, unit_n)

                result['idx_tetB'] = idx_tetB
                result['idx_tetA'] = idx_tetA
                result['omegaB'] = omegaB
                result['omegaA'] = omegaA
                result['p'] = p
                result['unit_n'] = unit_n
                result['gap'] = gap

            else:
                result['idx_tetB'] = -1
                result['idx_tetA'] = -1
        else:
            result['idx_tetB'] = -1
            result['idx_tetA'] = -1
    else:
        result['idx_tetB'] = -1
        result['idx_tetA'] = -1


@cuda.jit(lineinfo=True)
def contact_points_computing_kernel( d_bodyA_idxs, d_bodyB_idxs, d_overlaps, 
                                    d_B_values, d_A_owners, d_B_owners, d_A_xs, 
                                    d_B_xs, d_B_x0s, d_A_surfaces, d_A_Ts, d_B_Ts,
                                    d_B_grid_min_coords, d_B_grid_max_coords, d_B_grid_spacings,
                                    d_B_grid_Is, d_B_grid_Js, d_B_grid_Ks,
                                    max_iterations, tolerance, envelope, boundary, result_gpu):
    """ This a kernel function for computing contact points between a triangle and a tetrahedron.

    Args:
        d_B_values (float64[:, :]): It is an array storing signed distances of grid points to a given mesh.
        d_A_owners (list(dict)): The dict of the tetrahedra of body A that own the triangle.
        d_B_owners (list(dict)): The dict of the tetrahedra of body B that own the triangle.
        d_A_xs (float64[:, :, :]): The vertices in deformed coordinates of body A.
        d_B_xs (float64[:, :, :]): The vertices in deformed coordinates of body B.
        d_B_x0s (float64[:, :, :]): The vertices in material coordinates of body B.
        d_A_surfaces (float64[:]): The indices of the vertices of the triangle of body A.
        d_A_Ts (int32[:, :]): The indices of the vertices of the tetrahedra of body A.
        d_B_Ts (int32[:, :]): The indices of the vertices of the tetrahedra of body B.
        d_B_grid_min_coords (_type_): The minimum coordinate of the grid of body B.
        d_B_grid_max_coords (float64[:]): The maximum coordinate of the grid of body B.
        d_B_grid_spacings (float64[:]):  The spacing of the grid of body B.
        d_B_grid_Is (int32[:]): The number of cells in the x-direction of the grid.
        d_B_grid_Js (int32[:]): The number of cells in the y-direction of the grid.
        d_B_grid_Ks (int32[:]): The number of cells in the z-direction of the grid.
        max_iterations (int32): Maximum number of Gauss-Seidel iterations.
        tolerance (float64): Maximum number of Gauss-Seidel iterations.
        envelope (float64): Any geometry within this distance generates a contact point.
        boundary (float64): The boundary width. This specifies how far inside the grid bounding box the     point has to be to be considered inside.
        result_gpu (list(dict)): The data of the contact point computation for creating a ContactPoint instance on CPU.
    """
    gid = cuda.grid(1)

    if gid < d_overlaps.shape[0]:
        bodyA_idx = d_bodyA_idxs[gid]
        bodyB_idx = d_bodyB_idxs[gid]
        idx_triA, idx_triB = d_overlaps[gid]
        B_grid_values = d_B_values[gid]
        A_owners = d_A_owners[gid]
        B_owners = d_B_owners[gid]
        A_x = d_A_xs[gid]
        B_x = d_B_xs[gid]
        B_x0 = d_B_x0s[gid]
        A_surface = d_A_surfaces[gid]
        A_T = d_A_Ts[gid]
        B_T = d_B_Ts[gid]
        B_grid_min_coord = d_B_grid_min_coords[gid]
        B_grid_max_coord = d_B_grid_max_coords[gid]
        B_grid_spacing = d_B_grid_spacings[gid]
        B_grid_I = d_B_grid_Is[gid]
        B_grid_J = d_B_grid_Js[gid]
        B_grid_K = d_B_grid_Ks[gid]

        compute_contacts_device(idx_triA, idx_triB, A_owners, B_owners, A_x, B_x, B_x0, A_surface, A_T, B_T,  B_grid_min_coord, B_grid_max_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, max_iterations, tolerance, envelope, boundary, result_gpu[gid])