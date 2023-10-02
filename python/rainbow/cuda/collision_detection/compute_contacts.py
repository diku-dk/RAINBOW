import numpy as np
from numba import cuda, float64

import rainbow.cuda.math.vec as Vector
import rainbow.cuda.math.matrix as Matrix
import rainbow.cuda.math.linalg as LinAlg
import rainbow.cuda.geometry.barycentric as BC
import rainbow.cuda.geometry.grid3 as Grid3


@cuda.jit(device=True)
def xform_triangle_to_model_space_device(P, X, X0, P0):
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
def compute_omegaB_device(p, X0B, omegaB):
    BC.compute_barycentric_tetrahedron_device(X0B[0], X0B[1], X0B[2], X0B[3], p, omegaB)


@cuda.jit(device=True)
def compute_omegaA_device(p, XA, omegaA):
    BC.compute_barycentric_tetrahedron_device(XA[0], XA[1], XA[2], XA[3], p, omegaA)


@cuda.jit(device=True)
def compute_p_device(p, XB, omegaB, result):
    t = cuda.local.array((3, 4), dtype=float64)
    Matrix.mat43_T(XB, t)
    Matrix.mat34_dot_vec4(t, omegaB, result)


@cuda.jit(device=True)
def compute_n_device(n, XB, X0B, result):
    D = cuda.local.array((3,3), dtype=float64)
    D0 = cuda.local.array((3, 3), dtype=float64)
    for i in range(3):
        for j in range(3):
            D[i, j] = XB[i, j] - XB[3, j]
            D0[i, j] = X0B[i, j] - X0B[3, j]
    solve_res = cuda.local.array(3, dtype=float64)
    LinAlg.cramer_solver(D0, n, solve_res)
    Matrix.mat33_dot_vec3(D, solve_res, result)


@cuda.jit(device=True)
def compute_contacts_device(idx_triA, idx_triB, A_owners, B_owners, A_x, B_x, B_x0, A_surface, A_T, B_T,  B_grid_min_coord, B_grid_max_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, max_iteration, optimization_tolerance, envelope, boundary, result):
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
                    compute_omegaB_device(x_i, X0B, omegaB)
                    p = cuda.local.array(3, dtype=float64)
                    compute_p_device(x_i, XB, omegaB, p)
                    omegaA = cuda.local.array(4, dtype=float64)
                    compute_omegaA_device(p, XA, omegaA)
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
                    result['idx_tetB'] = -3
                    result['idx_tetA'] = -3
            else:
                result['idx_tetB'] = -2
                result['idx_tetA'] = -2
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
        

        ## Those data maybe need to tansform to Shared Memory, but I just test two small bodies in scene, those data size total is over 30KB except the 'grid_values', and the 'grid_values' is over 4MB. Those data size is too large to put in Shared Memory. Maybe we need to consider other ways to optimize this.
        

        compute_contacts_device(idx_triA, idx_triB, A_owners, B_owners, A_x, B_x, B_x0, A_surface, A_T, B_T,  B_grid_min_coord, B_grid_max_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, max_iterations, tolerance, envelope, boundary, result_gpu[gid])

