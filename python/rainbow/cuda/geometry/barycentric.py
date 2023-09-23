from numba import cuda, float64
import rainbow.cuda.math.tensor as CUDA_TANSOR


@cuda.jit(device=True)
def compute_barycentric_tetrahedron_device(x1, x2, x3, x4, p, result):
    b1 = cuda.local.array(3, dtype=float64)
    b2 = cuda.local.array(3, dtype=float64)
    b3 = cuda.local.array(3, dtype=float64)

    CUDA_TANSOR.vec3_sub(x2, x1, b1)
    CUDA_TANSOR.vec3_sub(x3, x1, b2)
    CUDA_TANSOR.vec3_sub(x4, x1, b3)

    cross_result = cuda.local.array(3, dtype=float64)
    CUDA_TANSOR.vec3_cross(b2, b3, cross_result)

    if CUDA_TANSOR.vec3_dot(b1, cross_result) > 0:
        basic = cuda.local.array((3, 3), dtype=float64)
        CUDA_TANSOR.mat3_make_from_cols(b1, b2, b3, basic)
        p_sub_x1 = cuda.local.array(3, dtype=float64)
        CUDA_TANSOR.vec3_sub(p, x1, p_sub_x1)
        q = cuda.local.array(3, dtype=float64)
        ## Currently, we use the Cramer's rule to solve the linear system, also tried the LU decomposition, but it is not faster than Cramer's rule and sometimes is unstable.
        # Cramer's rule :  https://en.wikipedia.org/wiki/Cramer%27s_rule
        ## Cramer's rule is also not stable, but it is faster than LU decomposition,
        ## With Numba cuda, we do not find a stable and fast library to solve the linear system.
        ## Maybe we can import CPP + cuSOLVER to solve the linear system in the future.
        CUDA_TANSOR.mat3_solve(basic, p_sub_x1, q)
        w1 = 1.0 - q[0] - q[1] - q[2]
        w2 = q[0]
        w3 = q[1]
        w4 = q[2]
    else:
        basic = cuda.local.array((3, 3), dtype=float64)
        CUDA_TANSOR.mat3_make_from_cols(b1, b3, b2, basic)
        p_sub_x1 = cuda.local.array(3, dtype=float64)
        CUDA_TANSOR.vec3_sub(p, x1, p_sub_x1)
        q = cuda.local.array(3, dtype=float64)
        CUDA_TANSOR.mat3_solve(basic, p_sub_x1, q)
        
        w1 = 1.0 - q[0] - q[1] - q[2]
        w2 = q[0]
        w3 = q[2]
        w4 = q[1]
    
    result[0] = w1
    result[1] = w2
    result[2] = w3
    result[3] = w4
        
