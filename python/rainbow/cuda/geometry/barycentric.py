from numba import cuda, float64
import rainbow.cuda.math.vec as Vector
import rainbow.cuda.math.linalg as LinAlg
import rainbow.cuda.math.matrix as Matrix


@cuda.jit(device=True)
def compute_barycentric_tetrahedron_device(x1: float64[:], x2: float64[:], x3: float64[:], x4: float64[:], p: float64[:], result: float64[:]):
    """ This method computes the barycentric coordinates for a point p of a tetrahedron
        given by the points x1,x2,x3, x4 (in right-hand order).

    Args:
        x1 (float64[:]): The first point of the tetrahedron.
        x2 (float64[:]): The second point of the tetrahedron.
        x3 (float64[:]): The third point of the tetrahedron.
        x4 (float64[:]): The fourth point of the tetrahedron.
        p (float64[:]): The point for which the barycentric coordinates should be computed.
        result (float64[:]): A quadruplet containing the first, second, third  abd fourth 
                        barycentric coordinates  w1, w2, w3 and w4.
    """
    b1 = cuda.local.array(3, dtype=float64)
    b2 = cuda.local.array(3, dtype=float64)
    b3 = cuda.local.array(3, dtype=float64)

    Vector.vec3_sub(x2, x1, b1)
    Vector.vec3_sub(x3, x1, b2)
    Vector.vec3_sub(x4, x1, b3)

    cross_result = cuda.local.array(3, dtype=float64)
    Vector.vec3_cross(b2, b3, cross_result)

    if Vector.vec3_dot(b1, cross_result) > 0:
        basic = cuda.local.array((3, 3), dtype=float64)
        Matrix.mat33_make_from_cols(b1, b2, b3, basic)
        p_sub_x1 = cuda.local.array(3, dtype=float64)
        Vector.vec3_sub(p, x1, p_sub_x1)
        q = cuda.local.array(3, dtype=float64)
        LinAlg.cramer_solver(basic, p_sub_x1, q)
        w1 = 1.0 - q[0] - q[1] - q[2]
        w2 = q[0]
        w3 = q[1]
        w4 = q[2]
    else:
        basic = cuda.local.array((3, 3), dtype=float64)
        Matrix.mat33_make_from_cols(b1, b3, b2, basic)
        p_sub_x1 = cuda.local.array(3, dtype=float64)
        Vector.vec3_sub(p, x1, p_sub_x1)
        q = cuda.local.array(3, dtype=float64)
        LinAlg.cramer_solver(basic, p_sub_x1, q)
        
        w1 = 1.0 - q[0] - q[1] - q[2]
        w2 = q[0]
        w3 = q[2]
        w4 = q[1]
    
    result[0] = w1
    result[1] = w2
    result[2] = w3
    result[3] = w4
        