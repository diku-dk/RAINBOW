import numpy as np
from numba import cuda, float64

import rainbow.cuda.math.tensor as CUDA_TANSOR
import rainbow.cuda.geometry.barycentric as CUDA_BC
import rainbow.cuda.geometry.grid3 as CUDA_GRID


####   Device functions section for computing closest contact points.   ####

@cuda.jit(device=True)
def xform_triangle_to_model_space_device(P, X, X0, P0):
    CUDA_TANSOR.mat3_zero(P0)

    w0 = cuda.local.array(4, dtype=float64)
    w1 = cuda.local.array(4, dtype=float64)
    w2 = cuda.local.array(4, dtype=float64)

    CUDA_BC.compute_barycentric_tetrahedron_device(X[0], X[1], X[2], X[3], P[0], w0)
    CUDA_BC.compute_barycentric_tetrahedron_device(X[0], X[1], X[2], X[3], P[1], w1)
    CUDA_BC.compute_barycentric_tetrahedron_device(X[0], X[1], X[2], X[3], P[2], w2)

    t = cuda.local.array((3, 4), dtype=float64)
    CUDA_TANSOR.mat43_T(X0, t)

    CUDA_TANSOR.mat34_v4_dot(t, w0, P0[0])
    CUDA_TANSOR.mat34_v4_dot(t, w1, P0[1])
    CUDA_TANSOR.mat34_v4_dot(t, w2, P0[2])

@cuda.jit(device=True)
def compute_omegaB_device(p, X0B, omegaB):
    CUDA_BC.compute_barycentric_tetrahedron_device(X0B[0], X0B[1], X0B[2], X0B[3], p, omegaB)

@cuda.jit(device=True)
def compute_p_device(p, XB, omegaB, result):
    t = cuda.local.array((3, 4), dtype=float64)
    CUDA_TANSOR.mat43_T(XB, t)
    CUDA_TANSOR.mat34_v4_dot(t, omegaB, result)

@cuda.jit(device=True)
def compute_omegaA_device(p, XA, omegaA):
    CUDA_BC.compute_barycentric_tetrahedron_device(XA[0], XA[1], XA[2], XA[3], p, omegaA)

@cuda.jit(device=True)
def compute_n_device(n, XB, X0B, result):
    D = cuda.local.array((3,3), dtype=float64)
    D0 = cuda.local.array((3, 3), dtype=float64)
    for i in range(3):
        for j in range(3):
            D[i, j] = XB[i, j] - XB[3, j]
            D0[i, j] = X0B[i, j] - X0B[3, j]
    solve_res = cuda.local.array(3, dtype=float64)
    CUDA_TANSOR.mat3_solve(D0, n, solve_res)
    CUDA_TANSOR.mat3_dot(D, solve_res, result)


# ## compute_contacts kernel function
# @cuda.jit
# def compute_contacts_kernel(overlaps, A_onwers, B_onwers, A_x, B_x, B_x0, A_surface, A_T, B_T,  B_grid_min_coord, B_grid_max_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, max_iteration, optimization_tolerance, envelope, boundary, results):
#     gid = cuda.grid(1)
#     if gid < overlaps.shape[0]:
#         idx_triA, idx_triB = overlaps[gid]
#         idx_tetA = A_onwers[idx_triA][0]
#         idx_tetB = B_onwers[idx_triB][0]

#         # Transform to local coordinates
#         P = cuda.local.array((3, 3), dtype=float64)
#         XB = cuda.local.array((4, 3), dtype=float64)
#         X0B = cuda.local.array((4, 3), dtype=float64)

#         surface_indices = A_surface[idx_triA] 
#         for j in range(3):
#             for k in range(3):  # bodyA.x的第二维度大小为3
#                 P[j, k] = A_x[surface_indices[j], k]
        
#         tet_indices = B_T[idx_tetB] 
#         for j in range(4):
#             for k in range(3):  # bodyB.x和bodyB.x0的第二维度大小为3
#                 XB[j, k] = B_x[tet_indices[j], k]
#                 X0B[j, k] = B_x0[tet_indices[j], k]


#         # P = A_x[A_surface[idx_triA], :]  
#         # XB = B_x[B_T[idx_tetB], :]  
#         # X0B = B_x0[B_T[idx_tetB], :]

#         P0 = cuda.local.array((3, 3), dtype=float64)
#         xform_triangle_to_model_space_device(P, XB, X0B, P0) 
#         # P0 = P0_list[gid]

#         gradient_norms = cuda.local.array(3, dtype=float64)
#         # for p_b in P0:
#         #     gradient = CUDA_GRID.get_gradient_device(p_b, B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values)
#         gradient0 = cuda.local.array(3, float64)
#         gradient1 = cuda.local.array(3, float64)
#         gradient2 = cuda.local.array(3, float64)

#         CUDA_GRID.get_gradient_device(P0[0], B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, gradient0)
#         CUDA_GRID.get_gradient_device(P0[1], B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, gradient1)
#         CUDA_GRID.get_gradient_device(P0[2], B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, gradient2)

#         gradient_norms[0] = CUDA_TANSOR.vec3_norm(gradient0)
#         gradient_norms[1] = CUDA_TANSOR.vec3_norm(gradient1)
#         gradient_norms[2] = CUDA_TANSOR.vec3_norm(gradient2)

#         x_i = cuda.local.array(3, dtype=float64)
#         for i in range(3):
#             x_i[i] = P0[CUDA_TANSOR.argmin(gradient_norms, 3)][i]

        

#         for i in range(max_iteration):
#             t = cuda.local.array(3, dtype=float64)
#             CUDA_GRID.get_gradient_device(x_i, B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, t)

#             # objectives = [CUDA_TANSOR.vec3_dot(s_i, t) for s_i in P0]
#             objectives = cuda.local.array(3, dtype=float64)
#             for j in range(3):
#                 objectives[j] = CUDA_TANSOR.vec3_dot(P0[j], t)
            
#             vertex = CUDA_TANSOR.argmin(objectives, 3)
#             s_i = P0[vertex]
#             alpha = 2 / (i + 2)
#             # x_i = x_i + alpha * (s_i - x_i) ; x_i and s_i are 3d vectors
#             #(s_i - x_i)
#             sx_sub = cuda.local.array(3, dtype=float64)
#             CUDA_TANSOR.vec3_sub(s_i, x_i, sx_sub)
#             # alpha * (s_i - x_i)
#             sx_sub_alpha = cuda.local.array(3, dtype=float64)
#             CUDA_TANSOR.vec3_mut(sx_sub, alpha, sx_sub_alpha)
#             # x_i + alpha * (s_i - x_i)
#             x_i_new = cuda.local.array(3, dtype=float64)
#             CUDA_TANSOR.vec3_add(x_i, sx_sub_alpha, x_i_new)
#             # x_i = x_i_new
#             for j in range(3):
#                 x_i[j] = x_i_new[j]

#             if objectives[vertex] > optimization_tolerance:
#                 break
        
#         # contact point generation
#         if CUDA_GRID.is_inside_device(x_i, B_grid_min_coord, B_grid_max_coord, 0.5):
#             phi = CUDA_GRID.get_value_device(x_i, B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values)
#             if phi < envelope:
#                 gap = phi
#                 n_res = cuda.local.array(3, dtype=float64)
#                 CUDA_GRID.get_gradient_device(x_i, B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, n_res)
#                 if CUDA_TANSOR.vec3_norm(n_res) > 0:
#                     XA = cuda.local.array((4, 3), dtype=float64)
#                     indices = A_T[idx_tetA]
#                     for i in range(4):
#                         for j in range(3):
#                             XA[i][j] = A_x[indices[i], j]
#                     # XA = A_x[A_T[idx_tetA], :]
#                     omegaB = cuda.local.array(4, dtype=float64)
#                     compute_omegaB_device(x_i, X0B, omegaB)
#                     p = cuda.local.array(3, dtype=float64)
#                     compute_p_device(x_i, XB, omegaB, p)
#                     omegaA = cuda.local.array(4, dtype=float64)
#                     compute_omegaA_device(p, XA, omegaA)
#                     n_res_new = cuda.local.array(3, dtype=float64)
#                     compute_n_device(n_res, XB, X0B, n_res_new)


        
#                     unit_n = cuda.local.array(3, dtype=float64)
#                     CUDA_TANSOR.vec3_unit(n_res_new, unit_n)

#                     results[gid]['idx_tetB'] = idx_tetB
#                     results[gid]['idx_tetA'] = idx_tetA
#                     results[gid]['omegaB'] = omegaB
#                     results[gid]['omegaA'] = omegaA
#                     results[gid]['p'] = p
#                     results[gid]['unit_n'] = unit_n


#                     # for i in range(3):
#                     #     results[gid]['vec3_unit_n'][i] = vec_unit_n[i]

#                     results[gid]['gap'] = gap
#                 else:
#                     results[gid]['gap'] =  -3
#                     results[gid]['idx_tetB'] = -3
#                     results[gid]['idx_tetA'] = -3
#             else:
#                 results[gid]['gap'] =  -2
#                 results[gid]['idx_tetB'] = -2
#                 results[gid]['idx_tetA'] = -2
#         else:
#             unit_n = cuda.local.array(3, dtype=float64)
#             unit_n[0] = -1
#             unit_n[1] = -1
#             unit_n[2] = -1
#             results[gid]['unit_n'] = unit_n
#             results[gid]['idx_tetB'] = -1
#             results[gid]['idx_tetA'] = -1

## This is device function for computing closest contact points.
## The CPU version of this function is in rainbow/collision_detection/collision_detection.py,
## function name is called "_compute_contacts".
@cuda.jit(device=True)
def compute_contacts_device(idx_triA, idx_triB, A_owners, B_owners, A_x, B_x, B_x0, A_surface, A_T, B_T,  B_grid_min_coord, B_grid_max_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, max_iteration, optimization_tolerance, envelope, boundary, result):
        idx_tetA = A_owners[idx_triA][0]
        idx_tetB = B_owners[idx_triB][0]

        # Transform to local coordinates
        P = cuda.local.array((3, 3), dtype=float64)
        XB = cuda.local.array((4, 3), dtype=float64)
        X0B = cuda.local.array((4, 3), dtype=float64)

        # P = A_x[A_surface[idx_triA], :]  
        surface_indices = A_surface[idx_triA] 
        for j in range(3):
            for k in range(3):  # bodyA.x的第二维度大小为3
                P[j, k] = A_x[surface_indices[j], k]
        
        # XB = B_x[B_T[idx_tetB], :]  
        # X0B = B_x0[B_T[idx_tetB], :]
        tet_indices = B_T[idx_tetB] 
        for j in range(4):
            for k in range(3):  # bodyB.x和bodyB.x0的第二维度大小为3
                XB[j, k] = B_x[tet_indices[j], k]
                X0B[j, k] = B_x0[tet_indices[j], k]

        # P0 = P0_list[gid]
        P0 = cuda.local.array((3, 3), dtype=float64)
        xform_triangle_to_model_space_device(P, XB, X0B, P0) 
        
        # for p_b in P0:
        #     gradient = CUDA_GRID.get_gradient_device(p_b, B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values)
        gradient_norms = cuda.local.array(3, dtype=float64)
        gradient0 = cuda.local.array(3, float64)
        gradient1 = cuda.local.array(3, float64)
        gradient2 = cuda.local.array(3, float64)

        CUDA_GRID.get_gradient_device(P0[0], B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, gradient0)
        CUDA_GRID.get_gradient_device(P0[1], B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, gradient1)
        CUDA_GRID.get_gradient_device(P0[2], B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, gradient2)

        gradient_norms[0] = CUDA_TANSOR.vec3_norm(gradient0)
        gradient_norms[1] = CUDA_TANSOR.vec3_norm(gradient1)
        gradient_norms[2] = CUDA_TANSOR.vec3_norm(gradient2)

        x_i = cuda.local.array(3, dtype=float64)
        for i in range(3):
            x_i[i] = P0[CUDA_TANSOR.argmin(gradient_norms, 3)][i]

        for i in range(max_iteration):
            t = cuda.local.array(3, dtype=float64)
            CUDA_GRID.get_gradient_device(x_i, B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, t)

            # objectives = [CUDA_TANSOR.vec3_dot(s_i, t) for s_i in P0]
            objectives = cuda.local.array(3, dtype=float64)
            for j in range(3):
                objectives[j] = CUDA_TANSOR.vec3_dot(P0[j], t)
            
            vertex = CUDA_TANSOR.argmin(objectives, 3)
            s_i = P0[vertex]
            alpha = 2 / (i + 2)
            # x_i = x_i + alpha * (s_i - x_i) ; x_i and s_i are 3d vectors
            #(s_i - x_i)
            sx_sub = cuda.local.array(3, dtype=float64)
            CUDA_TANSOR.vec3_sub(s_i, x_i, sx_sub)
            # alpha * (s_i - x_i)
            sx_sub_alpha = cuda.local.array(3, dtype=float64)
            CUDA_TANSOR.vec3_mut(sx_sub, alpha, sx_sub_alpha)
            # x_i + alpha * (s_i - x_i)
            x_i_new = cuda.local.array(3, dtype=float64)
            CUDA_TANSOR.vec3_add(x_i, sx_sub_alpha, x_i_new)
            # x_i = x_i_new
            for j in range(3):
                x_i[j] = x_i_new[j]

            if objectives[vertex] > optimization_tolerance:
                break
        
        # contact point generation
        if CUDA_GRID.is_inside_device(x_i, B_grid_min_coord, B_grid_max_coord, 0.5):
            phi = CUDA_GRID.get_value_device(x_i, B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values)
            if phi < envelope:
                gap = phi
                n_res = cuda.local.array(3, dtype=float64)
                CUDA_GRID.get_gradient_device(x_i, B_grid_min_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, n_res)
                if CUDA_TANSOR.vec3_norm(n_res) > 0:

                    # XA = A_x[A_T[idx_tetA], :]
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
                    CUDA_TANSOR.vec3_unit(n_res_new, unit_n)

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


####   Kernel function section for computing closest contact points.   ####
@cuda.jit(lineinfo=True)
def contact_points_computing_kernel(d_bodyA_idxs, d_bodyB_idxs, d_overlaps, d_ownerss, d_xs, d_x0s, d_surfaces, d_Ts, d_grid_min_coords, d_grid_max_coords, d_grid_spacings, d_grid_Is, d_grid_Js, d_grid_Ks, d_grid_valuess, max_iterations, tolerance, envelope, boundary, result_gpu):
    gid = cuda.grid(1)
    lid = cuda.threadIdx.x


    if gid < d_overlaps.shape[0]:
        bodyA_idx = d_bodyA_idxs[gid]
        bodyB_idx = d_bodyB_idxs[gid]
        idx_triA, idx_triB = d_overlaps[gid]

        ## Those data maybe need to tansform to Shared Memory, but I just test two small bodies in scene, those data size total is over 30KB except the 'grid_values', and the 'grid_values' is over 4MB. Those data size is too large to put in Shared Memory. Maybe we need to consider other ways to optimize this.
        A_owners = d_ownerss[bodyA_idx]
        B_owners = d_ownerss[bodyB_idx]
        A_x = d_xs[bodyA_idx]
        B_x = d_xs[bodyB_idx]
        B_x0 = d_x0s[bodyB_idx]
        A_surface = d_surfaces[bodyA_idx]
        A_T = d_Ts[bodyA_idx]
        B_T = d_Ts[bodyB_idx]
        B_grid_min_coord = d_grid_min_coords[bodyB_idx]
        B_grid_max_coord = d_grid_max_coords[bodyB_idx]
        B_grid_spacing = d_grid_spacings[bodyB_idx]
        B_grid_I = d_grid_Is[bodyB_idx]
        B_grid_J = d_grid_Js[bodyB_idx]
        B_grid_K = d_grid_Ks[bodyB_idx]
        B_grid_values = d_grid_valuess[bodyB_idx]

        compute_contacts_device(idx_triA, idx_triB, A_owners, B_owners, A_x, B_x, B_x0, A_surface, A_T, B_T,  B_grid_min_coord, B_grid_max_coord, B_grid_spacing, B_grid_I, B_grid_J, B_grid_K, B_grid_values, max_iterations, tolerance, envelope, boundary, result_gpu[gid])


