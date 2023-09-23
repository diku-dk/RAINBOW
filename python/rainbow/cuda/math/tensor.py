from numba import cuda, float64, int32
import math

############################# device functions for vec_3 ######################
@cuda.jit(device=True)
def vec3_zero(result):
    result[0] = 0.0
    result[1] = 0.0
    result[2] = 0.0

@cuda.jit(device=True)
def vec3_unit(a, result):
    norm = vec3_norm(a)
    
    result[0] = a[0] / norm
    result[1] = a[1] / norm
    result[2] = a[2] / norm

@cuda.jit(device=True)
def vec3_add(a, b, result):
    result[0] = a[0] + b[0]
    result[1] = a[1] + b[1]
    result[2] = a[2] + b[2]

@cuda.jit(device=True)
def vec3_sub(a, b, result):
    result[0] = a[0] - b[0]
    result[1] = a[1] - b[1]
    result[2] = a[2] - b[2]

@cuda.jit(device=True)
def vec3_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@cuda.jit(device=True)
def vec3_cross(a, b, result):
    result[0] = a[1] * b[2] - a[2] * b[1]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - a[1] * b[0]

@cuda.jit(device=True)
def vec3_norm(a):
    return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)

@cuda.jit(device=True)
def vec3_mut(V, a, result):
    result[0] = V[0] * a
    result[1] = V[1] * a
    result[2] = V[2] * a

############################# device functions for mat_3_3 ######################
@cuda.jit(device=True)
def mat3_T(M):
    result = cuda.local.array((3, 3), dtype=float64)
    for i in range(3):
        for j in range(3):
            result[i, j] = M[j, i]
    return result

@cuda.jit(device=True)
def mat3_zero(result):
    for i in range(3):
        for j in range(3):
            result[i, j] = 0.0
    return result

@cuda.jit(device=True)
def mat3_determinant(M):
    return M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) - \
              M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0]) + \
                M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])

## Cramer's rule for solve linear eqations, refer to https://en.wikipedia.org/wiki/Cramer%27s_rule
## Currently, we use the Cramer's rule to solve the linear system, also tried the LU decomposition, but it is not faster than Cramer's rule and sometimes is unstable.
## Cramer's rule is also not stable, but it is faster than LU decomposition,
## With Numba cuda, we do not find a stable and fast library to solve the linear system.
## Maybe we can import CPP + cuSOLVER to solve the linear system in the future.
@cuda.jit(device=True)
def mat3_solve(A, b, result):
    detA = mat3_determinant(A)
    if detA == 0.0:
        vec3_zero(result)
    else:
        Ax = cuda.local.array((3, 3), dtype=float64)
        Ay = cuda.local.array((3, 3), dtype=float64)
        Az = cuda.local.array((3, 3), dtype=float64)

        for i in range(3):
            for j in range(3):
                Ax[i, j] = A[i, j]
                Ay[i, j] = A[i, j]
                Az[i, j] = A[i, j]

        for i in range(3):
            Ax[i, 0] = b[i]
            Ay[i, 1] = b[i]
            Az[i, 2] = b[i]
        
        x = mat3_determinant(Ax) / detA
        y = mat3_determinant(Ay) / detA
        z = mat3_determinant(Az) / detA

        result[0] = x
        result[1] = y
        result[2] = z

@cuda.jit(device=True)
def mat3_make(M00, M01, M02, M10, M11, M12, M20, M21, M22, result):
    result[0, 0] = M00
    result[0, 1] = M01
    result[0, 2] = M02
    result[1, 0] = M10
    result[1, 1] = M11
    result[1, 2] = M12
    result[2, 0] = M20
    result[2, 1] = M21
    result[2, 2] = M22

@cuda.jit(device=True)
def mat3_make_from_cols(col0, col1, col2, result):
    mat3_make(
        col0[0], col1[0], col2[0],
        col0[1], col1[1], col2[1],
        col0[2], col1[2], col2[2],
        result
    )
    
@cuda.jit(device=True)
def mat3_dot(M, V3, result):
    for i in range(3):
        sum= 0.0
        for j in range(3):
            sum += M[i, j] * V3[j]
        result[i] = sum


############################# device functions for mat_4_3 ######################
@cuda.jit(device=True)
def mat43_T(M, result):
    for i in range(3):
        for j in range(4):
            result[i, j] = M[j, i]


@cuda.jit(device=True)
def mat34_v4_dot(M, v4, result):
    
    for i in range(3):
        sum = 0.0
        for j in range(4):
            sum += M[i, j] * v4[j]
        result[i] = sum

@cuda.jit(device=True)
def argmin(a, size):
    min_val = a[0]
    min_idx = 0

    size = min(size, a.shape[0])

    for i in range(size):
        if a[i] < min_val:
            min_val = a[i]
            min_idx = i
            
    return min_idx