from rainbow.math.quaternion import to_matrix
from math import atan2, asin, pi


class EulerXYZ:
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha  # Rotation angle around x-axis in radians
        self.beta = beta  # Rotation angle around y-axis in radians
        self.gamma = gamma  # Rotation angle around z-axis in radians

# Actually R = R_z(theta_z)R_y(theta_y)R_x(theta_x)
def make_euler_xyz_from_matrix(R):
    r00 = R[0, 0]
    r01 = R[0, 1]
    r02 = R[0, 2]
    r10 = R[1, 0]
    r20 = R[2, 0]
    r21 = R[2, 1]
    r22 = R[2, 2]

    if r20 >= 1.0:
        rz = atan2(-r01, -r02)
        ry = -pi / 2.0
        rx = 0.0
    elif r20 <= -1.0:
        rz = atan2(-r01, r02)
        ry = pi / 2.0
        rx = 0.0
    else:
        rz = atan2(r10, r00)
        ry = asin(-r20)
        rx = atan2(r21, r22)

    return EulerXYZ(rx, ry, rz)


def make_euler_xyz_from_quaternion(Q):
    return make_euler_xyz_from_matrix(to_matrix(Q))
