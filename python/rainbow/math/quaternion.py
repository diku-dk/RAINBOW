import numpy as np
import rainbow.math.matrix3 as M3
import rainbow.math.vector3 as V3
import rainbow.math.angle as ANGLE
import rainbow.util.parse_string as parse
from math import cos, sin, sqrt, pi, atan2, acos

def make(qs, qx, qy, qz):
    return np.array([qs, qx, qy, qz], dtype=np.float64)

def from_array(data):
    return np.array(data[0:4], dtype=np.float64)

def check_string(lower_value):
    """
        Parsing a string. The function returns true if the given
        input string follows one of the above templates

        :param: A text string, that consist of valid oprations.
        :return: A boolean telling if the string is valid.
    """
    if   parse.parse_string_to_rotation_check(lower_value):
        return True
    elif parse.parse_string_to_array_check(lower_value):
        return True
    elif parse.parse_string_to_ru_check(lower_value):
        return True
    return parse.parse_string_to_keywords_check(lower_value)

def from_string(value):

    value_lower = value.lower()

    assert check_string(value_lower)

    if value_lower == "identity":
        return identity()

    if value_lower.startswith("rx:"):
        degrees = float(value_lower.strip("rx:"))
        radians = ANGLE.degrees_to_radians(degrees)
        return Rx(radians)

    if value_lower.startswith("ry:"):
        degrees = float(value_lower.strip("ry:"))
        radians = ANGLE.degrees_to_radians(degrees)
        return Ry(radians)

    if value_lower.startswith("rz:"):
        degrees = float(value_lower.strip("rz:"))
        radians = ANGLE.degrees_to_radians(degrees)
        return Rz(radians)

    if value_lower.startswith("ru:"):
        (degrees_str, axis_str) = value_lower.strip("ru:").split(":")
        degrees = float(degrees_str)
        radians = ANGLE.degrees_to_radians(degrees)
        axis = V3.from_string(axis_str)
        return Ru(radians, axis)

    string_2_array =  np.fromstring(value_lower.strip("[]"), dtype=np.float64, sep=",")

    assert len(string_2_array) >= 4

    return string_2_array[:4]


def from_vector3(w):
    return np.array([0, w[0], w[1], w[2]], dtype=np.float64)


def identity():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def unit(q):
    l = np.linalg.norm(q)
    if l > 0.0:
        return from_array(q / l)
    return identity()


def Ru(radians, axis):
    c = cos(radians / 2.0)
    s = sin(radians / 2.0)
    n = axis / np.linalg.norm(axis)
    return from_array([c, s * n[0], s * n[1], s * n[2]])


def Rx(radians):
    c = cos(radians / 2.0)
    s = sin(radians / 2.0)
    return from_array([c, s, 0.0, 0.0])


def Ry(radians):
    c = cos(radians / 2.0)
    s = sin(radians / 2.0)
    return from_array([c, 0.0, s, 0.0])


def Rz(radians):
    c = cos(radians / 2.0)
    s = sin(radians / 2.0)
    return from_array([c, 0.0, 0.0, s])


def to_matrix(Q):

    R = M3.zero()

    q = unit(Q)

    qs = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    R[0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    R[1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    R[2, 2] = 1.0 - 2.0 * (qy * qy + qx * qx)
    R[1, 0] = 2.0 * (qx * qy + qs * qz)
    R[0, 1] = 2.0 * (qx * qy - qs * qz)
    R[2, 0] = 2.0 * (-qs * qy + qx * qz)
    R[0, 2] = 2.0 * (qs * qy + qx * qz)
    R[2, 1] = 2.0 * (qz * qy + qs * qx)
    R[1, 2] = 2.0 * (qz * qy - qs * qx)

    return R


def from_matrix(M):
    M00 = M[0, 0]
    M01 = M[0, 1]
    M02 = M[0, 2]
    M10 = M[1, 0]
    M11 = M[1, 1]
    M12 = M[1, 2]
    M20 = M[2, 0]
    M21 = M[2, 1]
    M22 = M[2, 2]
    tr = M00 + M11 + M22
    if tr < 0.0:
        i = 0
        if M11 > M00:
            i = 1
        if M22 > M[i, i]:
            i = 2
        if i == 0:
            r = sqrt((M00 - (M11 + M22)) + 1.0)
            qx = 0.5 * r
            r = 0.5 / r
            qy = (M01 + M10) * r
            qz = (M20 + M02) * r
            qs = (M21 - M12) * r
            return from_array([qs, qx, qy, qz])
        if i == 1:
            r = sqrt((M11 - (M22 + M00)) + 1.0)
            qy = 0.5 * r
            r = 0.5 / r
            qz = (M12 + M21) * r
            qx = (M01 + M10) * r
            qs = (M02 - M20) * r
            return from_array([qs, qx, qy, qz])
        if i == 2:
            r = sqrt((M22 - (M00 + M11)) + 1.0)
            qz = 0.5 * r
            r = 0.5 / r
            qx = (M20 + M02) * r
            qy = (M12 + M21) * r
            qs = (M10 - M01) * r
            return from_array([qs, qx, qy, qz])
    r = sqrt(tr + 1.0)
    qs = 0.5 * r
    r = 0.5 / r
    qx = (M21 - M12) * r
    qy = (M02 - M20) * r
    qz = (M10 - M01) * r
    return from_array([qs, qx, qy, qz])


def conjugate(Q):
    return from_array([Q[0], -Q[1], -Q[2], -Q[3]])


def prod(Qa, Qb):
    a = Qa[0]
    A = Qa[1:]
    b = Qb[0]
    B = Qb[1:]
    qs = a * b - np.dot(A, B)
    qv = a * B + A * b + np.cross(A, B, axis=0)
    return from_array([qs, qv[0], qv[1], qv[2]])


def rotate(q, r):
    qr = from_array([0.0, r[0], r[1], r[2]])
    return prod(prod(q, qr), conjugate(q))[1:]


def prod_array(Qa, Qb):
    """
    prod_array takes two quaternial inputs, of which at least one must be a quaternion array, and returns the element wise quaternion multiplication.
    If one of the arguments is just a single quaternion, then that quaternion is multiplied with every element of the other argument array.
    """
    if Qa.ndim == 1 and Qb.ndim == 2:
        N = len(Qb)
        first = (
            Qa[0] * Qb[:, 0] - Qa[1] * Qb[:, 1] - Qa[2] * Qb[:, 2] - Qa[3] * Qb[:, 3]
        ).reshape(N, 1)
        second = (
            Qa[0] * Qb[:, 1] + Qa[1] * Qb[:, 0] + Qa[2] * Qb[:, 3] - Qa[3] * Qb[:, 2]
        ).reshape(N, 1)
        third = (
            Qa[0] * Qb[:, 2] - Qa[1] * Qb[:, 3] + Qa[2] * Qb[:, 0] + Qa[3] * Qb[:, 1]
        ).reshape(N, 1)
        fourth = (
            Qa[0] * Qb[:, 3] + Qa[1] * Qb[:, 2] - Qa[2] * Qb[:, 1] + Qa[3] * Qb[:, 0]
        ).reshape(N, 1)
        return np.concatenate([first, second, third, fourth], axis=1)
    if Qa.ndim == 2 and Qb.ndim == 1:
        N = len(Qa)
        first = (
            Qa[:, 0] * Qb[0] - Qa[:, 1] * Qb[1] - Qa[:, 2] * Qb[2] - Qa[:, 3] * Qb[3]
        ).reshape(N, 1)
        second = (
            Qa[:, 0] * Qb[1] + Qa[:, 1] * Qb[0] + Qa[:, 2] * Qb[3] - Qa[:, 3] * Qb[2]
        ).reshape(N, 1)
        third = (
            Qa[:, 0] * Qb[2] - Qa[:, 1] * Qb[3] + Qa[:, 2] * Qb[0] + Qa[:, 3] * Qb[1]
        ).reshape(N, 1)
        fourth = (
            Qa[:, 0] * Qb[3] + Qa[:, 1] * Qb[2] - Qa[:, 2] * Qb[1] + Qa[:, 3] * Qb[0]
        ).reshape(N, 1)
        return np.concatenate([first, second, third, fourth], axis=1)
    if Qa.ndim == 2 and Qb.ndim == 2:
        raise NotImplementedError(
            "Quaternion multiplication between two arrays of Quaternions has not yet been implemented"
        )

    raise ValueError("Either one or both of the arguments has to be a quaternion array")


def rotate_array(q, rs):
    conjugated = conjugate(q)
    qrs = np.concatenate((np.zeros((len(rs), 1)), rs), axis=1)
    inner = prod_array(q, qrs)
    outer = prod_array(inner, conjugated)

    return outer[:, 1:]


def to_angle_axis(Q):
    ct2 = Q[0]  # cos(theta / 2)
    st2 = np.linalg.norm(Q[1:])  # | sin(theta / 2) |

    theta = 2.0 * atan2(st2, ct2)

    if st2 > 0.0:
        return theta, Q[1:] / st2
    
    #Happens if the rotation is 2pi, 0 or if the point is 
    #already on the rotating axis.
    return theta, V3.zero()


def to_angle(Q, axis):
    '''
        If axis and and the rotation axis of the quaternion is zero 
        then the function does not work as intended.
    '''
    ct2 = Q[0]  # cos(theta / 2)
    st2 = np.linalg.norm(Q[1:])  # | sin(theta / 2) |

    if np.dot(Q[1:], axis) >= 0.0:
        theta = 2.0 * atan2(st2, ct2)
    else:                                          #pragma: no cover
        # This case will never happen unless we    #pragma: no cover
        # work with negative axis.                 #pragma: no cover  
        theta = 2.0 * atan2(st2, -ct2)             #pragma: no cover

    if theta > pi:
        theta -= 2.0 * pi

    return theta


def lerp(a, b, t):
    """
    Linear Interpolation of Quaternions.

    :param a:    Quaternion input
    :param b:    Quaternion input
    :param t:    Interpolation parameter, 0 <= t <= 1
    :return:     Returns q(t) = (1-t)*a + t*b
    """
    t = np.clip(t, 0.0, 1.0)
    return a + (b - a) * t


def slerp(a, b, t):
    """
    Spherical Linear Interpolation of Quaternions.

    :param a:    Quaternion input
    :param b:    Quaternion input
    :param t:    Interpolation parameter, 0 <= t <= 1
    :return:     Returns q(t) = (sin((1-t)Omega)/sin(Omega))* a + (sin(Omega*t)/sin(Omega))*b
    """
    t = np.clip(t, 0.0, 1.0)

    dot = np.clip(np.dot(a, b), -1.0, 1.0)

    flip = False
    if dot < 0.0:
        dot = -dot
        flip = True

    small_angle_threshold = 0.9999999

    if dot > small_angle_threshold:
        return unit(lerp(a, b, t))

    omega = acos(dot)
    sin_omega = sin(omega)
    t_omega = t * omega
    x = sin(omega - t_omega) / sin_omega
    z = sin(t_omega) / sin_omega
    y = -z if flip else z
    return x * a + y * b


def rand():                                      #pragma: no cover  
    return unit(np.random.uniform(-1.0, 1.0, 4)) #pragma: no cover 


def hat(q):
    """
    Orthogonal Quaternion.
    This method sets this Quaternion to an orthogonal Quaternion
    of the specified Quaternion. In other words the resulting
    angle between the specified Quaternion and this Quaternion
    is pi/2.

    :param q:
    :return:
    """
    return make(q[3], -q[2], q[1], -q[0])
