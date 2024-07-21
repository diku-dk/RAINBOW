import rainbow.math.matrix3 as M3
import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
from math import cos, sin, pi, acos, fmod, fabs
from numpy import tan


class Constants:
    @staticmethod
    def min_angle():
        """
        Crossover point to Taylor Series approximation. Figuring 16 decimal digits
        of precision for doubles, the Taylor approximation should be indistinguishable (to
        machine precision) for angles of magnitude less than 1e-4.To be conservative, we add
        on three additional order of magnitude.

        :return:
        """
        return 1e-6

    @staticmethod
    def cutoff_angle():
        """
        Angle beyond which we perform dynamic reparameterization of a 3DOF expoential map
        """
        return pi


def reparameterization(v):
    """
    To escape the vanishing derivatives at shells of 2PI rotations, we reparameterize
    to a rotation of (2PI - theta) about the opposite axis when we get too close to 2PI

    :param v:
    :return:
    """
    theta = V3.norm(v)
    # First get theta into range 0..2PI
    if theta > 2.0 * pi:
        psi = fmod(theta, 2 * pi)
        v = v * (psi / theta)
        theta = V3.norm(v)
    
    if theta > Constants.cutoff_angle():
        # Observe this flips the direction of v! Because
        # for all values of pi < theta < 2 pi we have
        # that (1 - 2 pi)/theta < 0
        v = v * (1.0 - (2.0 * pi / theta))
    return v


def exp(v, reparm=True):
    """
    Exponent
    Sets the Quaternion equal to the exponent of the specified Quaternion.

    :param v:         A 3D expoential map vector
    :param reparm:    A boolean flag that controls if v will be reparameterized.
    :return:          The corresponding quaternion
    """
    if reparm:
        v = reparameterization(v)
    theta = V3.norm(v)
    c = cos(theta / 2.0)
    s = sin(theta / 2.0)
    # Use Taylor Series for sinc if theta is small enough
    w = (
        (0.5 - (theta ** 2) / 48.0) * v
        if theta < Constants.min_angle()
        else v * s / theta
    )
    return Q.make(c, w[0], w[1], w[2])


def log(q):
    """
    Natural Logarithm

    :param q:    A reference to an unit quaternion.
    :return:     Returns the Quaternion equal to the natural logarithm of the specified Quaternion.
    """
    if fabs(q[0]) > 0:
        # Given
        #
        #    q = [s,w] = [cos(theta/2], sin(theta/2) n ]
        #
        # Compute
        #
        #    v = theta * n
        #
        # Let
        #
        #    t = theta/2  = acos(s)
        #    n = w / sin(t)
        #    v = 2*t*n = 2*t*w / sin(t)
        #
        # If q is the identity quaternion the log(q) should map to zero vector. Recall that
        # identity is given by q = [1 0 0 0] the above works for this.
        t = acos(q[0])
        s = sin(t)
        f = 2 * t / s
        return V3.make(f * q[1], f * q[2], f * q[3])
    return V3.zero()


def to_quaternion(v, reparm=True):
    return exp(v, reparm)


def to_expmap(q):
    return log(q)


def to_matrix(v):
    return Q.to_matrix(to_quaternion(v))


def dvdt(omega, v):
    """
    Compute the time derivative of v necessary for dynamic simulation as a function
    of the current exponential map  orientation 'v' and the current angular velocity
    'omega'.  The results are undefined when 'v' represents a rotation
    of 2*n*PI about any axis.

    :param v:
    :param omega:
    :return:
    """
    theta = V3.norm(v)
    cosp = cos(0.5 * theta)
    sinp = sin(0.5 * theta)
    if theta < Constants.min_angle():
        gamma = (12.0 - theta ** 2) / 6.0
        eta = v.dot(omega) * (60.0 + theta ** 2) / 360.0
    else:
        cotp = cosp / sinp
        gamma = theta * cotp
        eta = v.dot(omega) / theta * (cotp - 2.0 / theta)
    vdot = (gamma * omega + V3.cross(omega, v) - eta * v) * 0.5
    return vdot


def dQdv_i(v, i):
    """
    Partial derivative of quaternion wrt i'th component of exponential map vector v

    :param v:
    :param i:
    :return:
    """
    assert i >= 0 and i < 3, "illegal index range"
    theta = V3.norm(v)
    i1 = (i + 1) % 3
    i2 = (i + 2) % 3

    dQdvi = Q.identity()
    if theta < Constants.min_angle():
        Tsinc = 0.5 - theta ** 2 / 48.0
        vTerm = v[i] * (theta ** 2 / 40.0 - 1.0) / 24.0
        dQdvi[0] = -0.5 * v[i] * Tsinc
        dQdvi[i + 1] = v[i] * vTerm + Tsinc
        dQdvi[i1 + 1] = v[i1] * vTerm
        dQdvi[i2 + 1] = v[i2] * vTerm
    else:
        cosp = cos(0.5 * theta)
        sinp = sin(0.5 * theta)
        ang = 1.0 / theta
        sang = sinp * ang
        cterm = v[i] * (ang ** 2) * (0.5 * cosp - sang)
        dQdvi[0] = -0.5 * v[i] * sang        
        dQdvi[i + 1] = cterm * v[i] + sang
        dQdvi[i1 + 1] = cterm * v[i1]
        dQdvi[i2 + 1] = cterm * v[i2]
    return dQdvi


def __dRdv_i(q, dqdvi):
    """
    Given a quaternion 'q' computed from the current 2 or 3 degree of freedom exponential
    map vector 'v', and the partial derivative of the quaternion with respect
    to the i'th element of 'v' in 'dqdvi', compute and store in 'dRdvi' the i'th partial
    derivative of the rotation matrix 'R' with respect to the i'th element of 'v'.

    :param q:
    :param dqdvi:
    :return:
    """
    # Thi efficient formulation is arrived at by writing out the entire chain
    # rule product dRdq * dqdv in terms of 'q' and noticing that all the
    # entries are formed from sums of just nine products of 'q' and 'dqdv'
    W = 0
    X = 1
    Y = 2
    Z = 3
    prod0 = -4 * q[X] * dqdvi[X]
    prod1 = -4 * q[Y] * dqdvi[Y]
    prod2 = -4 * q[Z] * dqdvi[Z]
    prod3 = 2 * (q[Y] * dqdvi[X] + q[X] * dqdvi[Y])
    prod4 = 2 * (q[W] * dqdvi[Z] + q[Z] * dqdvi[W])
    prod5 = 2 * (q[Z] * dqdvi[X] + q[X] * dqdvi[Z])
    prod6 = 2 * (q[W] * dqdvi[Y] + q[Y] * dqdvi[W])
    prod7 = 2 * (q[Z] * dqdvi[Y] + q[Y] * dqdvi[Z])
    prod8 = 2 * (q[W] * dqdvi[X] + q[X] * dqdvi[W])
    dRdvi = M3.zero()
    dRdvi[0][0] = prod1 + prod2
    dRdvi[0][1] = prod3 - prod4
    dRdvi[0][2] = prod5 + prod6
    dRdvi[1][0] = prod3 + prod4
    dRdvi[1][1] = prod0 + prod2
    dRdvi[1][2] = prod7 - prod8
    dRdvi[2][0] = prod5 - prod6
    dRdvi[2][1] = prod7 + prod8
    dRdvi[2][2] = prod0 + prod1
    return dRdvi


def dRdv_i(v, i):
    """
    Compute the i'th partial derivative of
    the rotation matrix with respect to EM parameter 'v', storing result
    in 'dRdvi'.  If 'v' is near a singularity, it will be dynamically reparameterized
    in place and the value 1 is returned; otherwise,  0 is returned.

    :param v:
    :param i:
    :return:
    """
    q = exp(v)
    dQdvi = dQdv_i(v, i)
    dRdvi = __dRdv_i(q, dQdvi)
    return dRdvi