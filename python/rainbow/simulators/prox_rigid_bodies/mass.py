import numpy as np
import rainbow.math.vector3 as V3
import rainbow.math.matrix3 as M3
import rainbow.math.quaternion as Q


class MassProperties:
    """
    Mass properties.
    This class holds information about mass properties, which is the total
    mass, the center of mass potion, and the inertia tensor. As the inertia tensor
    is always symmetric not all terms are saved, but only the symmetric part.
    """

    def __init__(self):
        self.mass = 0.0  # Total mass
        self.x = 0.0  # Center of mass
        self.y = 0.0
        self.z = 0.0
        self.Ixx = 0.0  # Inertia tensor
        self.Iyy = 0.0
        self.Izz = 0.0
        self.Ixy = 0.0
        self.Ixz = 0.0
        self.Iyz = 0.0

    def __str__(self):
        layout = ".2f"
        txt = "( m="
        txt = txt + format(self.mass, layout)
        txt += ", x="
        txt = txt + format(self.x, layout)
        txt += ", y="
        txt = txt + format(self.y, layout)
        txt += ", z="
        txt = txt + format(self.z, layout)
        txt += ", Ixx="
        txt = txt + format(self.Ixx, layout)
        txt += ", Iyy="
        txt = txt + format(self.Iyy, layout)
        txt += ", Izz="
        txt = txt + format(self.Izz, layout)
        txt += ", Ixy="
        txt = txt + format(self.Ixy, layout)
        txt += ", Ixz="
        txt = txt + format(self.Ixz, layout)
        txt += ", Iyz="
        txt = txt + format(self.Iyz, layout)
        txt += ")"
        return txt


def __compute_projected_face_integral(V, T, idx, X, Y):
    """
    This function computes face integrals of triangles that have been projected onto a coordinate plane.

    Details can be found in this paper: https://doi.org/10.1080/10867651.1996.10487458

    :param V:
    :param T:
    :param idx:
    :param X:
    :param Y:
    :return:
    """
    P1 = 0.0
    Pa = 0.0
    Paa = 0.0
    Paaa = 0.0
    Pb = 0.0
    Pbb = 0.0
    Pbbb = 0.0
    Pab = 0.0
    Paab = 0.0
    Pabb = 0.0
    for e in range(3):
        i0 = e
        i1 = (e + 1) % 3
        v0 = V[T[idx, i0], :]
        v1 = V[T[idx, i1], :]
        a0 = v0[X]
        b0 = v0[Y]
        a1 = v1[X]
        b1 = v1[Y]
        da = a1 - a0
        db = b1 - b0
        a0_2 = a0 * a0
        a0_3 = a0_2 * a0
        a0_4 = a0_3 * a0
        b0_2 = b0 * b0
        b0_3 = b0_2 * b0
        b0_4 = b0_3 * b0
        a1_2 = a1 * a1
        a1_3 = a1_2 * a1
        b1_2 = b1 * b1
        b1_3 = b1_2 * b1
        C1 = a1 + a0
        Ca = a1 * C1 + a0_2
        Caa = a1 * Ca + a0_3
        Caaa = a1 * Caa + a0_4
        Cb = b1 * (b1 + b0) + b0_2
        Cbb = b1 * Cb + b0_3
        Cbbb = b1 * Cbb + b0_4
        Cab = 3.0 * a1_2 + 2.0 * a1 * a0 + a0_2
        Kab = a1_2 + 2.0 * a1 * a0 + 3.0 * a0_2
        Caab = a0 * Cab + 4.0 * a1_3
        Kaab = a1 * Kab + 4.0 * a0_3
        Cabb = 4.0 * b1_3 + 3.0 * b1_2 * b0 + 2.0 * b1 * b0_2 + b0_3
        Kabb = b1_3 + 2.0 * b1_2 * b0 + 3.0 * b1 * b0_2 + 4.0 * b0_3
        P1 += db * C1
        Pa += db * Ca
        Paa += db * Caa
        Paaa += db * Caaa
        Pb += da * Cb
        Pbb += da * Cbb
        Pbbb += da * Cbbb
        Pab += db * (b1 * Cab + b0 * Kab)
        Paab += db * (b1 * Caab + b0 * Kaab)
        Pabb += da * (a1 * Cabb + a0 * Kabb)
    P1 /= 2.0
    Pa /= 6.0
    Paa /= 12.0
    Paaa /= 20.0
    Pb /= -6.0
    Pbb /= -12.0
    Pbbb /= -20.0
    Pab /= 24.0
    Paab /= 60.0
    Pabb /= -60.0
    return P1, Pa, Paa, Paaa, Pb, Pbb, Pbbb, Pab, Paab, Pabb


def compute_mass_properties(V, T, density: float):
    """
    Compute mass properties of the given triangle surface mesh. It is assumed
    that the mesh is watertight and not self-intersecting.

    Details can be found in this paper: https://doi.org/10.1080/10867651.1996.10487458

    :param V:        The vertex coordinates of the triangle mesh. Assumed to be N-by-3 where N is number of nodes.
    :param T:        The triangle elements of the triangle mesh. Assumes to be K-by-3 where K is number of triangles.
    :param density:  The mass density value to use.
    :return:         The mass properties wrt the model space (aka the space the input mesh lives in)
    """
    prop = MassProperties()

    T0 = 0.0
    T1 = V3.zero()
    T2 = V3.zero()
    TP = V3.zero()

    for idx in range(len(T)):

        # Extract vertex indices of the triangle face
        i, j, k = T[idx, :]

        # Extract coordinates of the vertices
        pi = V[i, :]
        pj = V[j, :]
        pk = V[k, :]

        # Compute face plane of triangle, n: normal, w: offset to origin
        n = V3.unit(V3.cross(pj - pi, pk - pi))
        p_avg = (pi + pj + pk) / 3.0
        w = -np.dot(n, p_avg)

        Z = V3.max_abs_component(n)
        X = (Z + 1) % 3
        Y = (X + 1) % 3

        nA = n[X]
        nB = n[Y]
        nC = n[Z]

        k1 = 1.0 / nC
        k2 = k1 * k1
        k3 = k2 * k1
        k4 = k3 * k1

        (
            P1,
            Pa,
            Paa,
            Paaa,
            Pb,
            Pbb,
            Pbbb,
            Pab,
            Paab,
            Pabb,
        ) = __compute_projected_face_integral(V, T, idx, X, Y)

        Fa = k1 * Pa
        Fb = k1 * Pb
        Fc = -k2 * (nA * Pa + nB * Pb + w * P1)
        Faa = k1 * Paa
        Fbb = k1 * Pbb
        Fcc = k3 * (
            nA * nA * Paa
            + 2.0 * nA * nB * Pab
            + nB * nB * Pbb
            + w * (2.0 * (nA * Pa + nB * Pb) + w * P1)
        )
        Faaa = k1 * Paaa
        Fbbb = k1 * Pbbb
        Fccc = -k4 * (
            nA * nA * nA * Paaa
            + 3.0 * nA * nA * nB * Paab
            + 3.0 * nA * nB * nB * Pabb
            + nB * nB * nB * Pbbb
            + 3.0 * w * (nA * nA * Paa + 2.0 * nA * nB * Pab + nB * nB * Pbb)
            + w * w * (3.0 * (nA * Pa + nB * Pb) + w * P1)
        )
        Faab = k1 * Paab
        Fbbc = -k2 * (nA * Pabb + nB * Pbbb + w * Pbb)
        Fcca = k3 * (
            nA * nA * Paaa
            + 2.0 * nA * nB * Paab
            + nB * nB * Pabb
            + w * (2.0 * (nA * Paa + nB * Pab) + w * Pa)
        )
        if X == 0:
            T0 += n[0] * Fa
        elif Y == 0:
            T0 += n[0] * Fb
        else:
            T0 += n[0] * Fc

        T1[X] += nA * Faa
        T1[Y] += nB * Fbb
        T1[Z] += nC * Fcc

        T2[X] += nA * Faaa
        T2[Y] += nB * Fbbb
        T2[Z] += nC * Fccc

        TP[X] += nA * Faab
        TP[Y] += nB * Fbbc
        TP[Z] += nC * Fcca

    T1[0] /= 2.0
    T1[1] /= 2.0
    T1[2] /= 2.0
    T2[0] /= 3.0
    T2[1] /= 3.0
    T2[2] /= 3.0
    TP[0] /= 2.0
    TP[1] /= 2.0
    TP[2] /= 2.0
    if T0 > 0.0:
        prop.mass = density * T0
        prop.x = T1[0] / T0
        prop.y = T1[1] / T0
        prop.z = T1[2] / T0
        prop.Ixx = density * (T2[1] + T2[2])
        prop.Iyy = density * (T2[2] + T2[0])
        prop.Izz = density * (T2[0] + T2[1])
        prop.Ixy = -density * TP[0]
        prop.Iyz = -density * TP[1]
        prop.Ixz = -density * TP[2]
    else:
        raise RuntimeError(
            "compute_mass_properties(): internal error, bad mesh encountered"
        )
    # We clamp computed values to zero if they are very tiny numbers. This is mainly due making inertia tensor more
    # human-readable when printing it.
    too_small = 1.0e-10
    prop.mass = prop.mass if prop.mass > too_small else 0.0
    prop.Ixx = prop.Ixx if prop.Ixx > too_small else 0.0
    prop.Iyy = prop.Iyy if prop.Iyy > too_small else 0.0
    prop.Izz = prop.Izz if prop.Izz > too_small else 0.0
    prop.Ixy = prop.Ixy if abs(prop.Ixy) > too_small else 0.0
    prop.Iyz = prop.Iyz if abs(prop.Iyz) > too_small else 0.0
    prop.Ixz = prop.Ixz if abs(prop.Ixz) > too_small else 0.0
    prop.x = prop.x if abs(prop.x) > too_small else 0.0
    prop.y = prop.y if abs(prop.y) > too_small else 0.0
    prop.z = prop.z if abs(prop.z) > too_small else 0.0
    return prop


def __rotate_to_model_space_orientation(prop):
    """
    Retrieve body space to model space rotation and inertia tensor wrt body space orientation
    :param prop:   Mass properties given in some "model space"
    :return:       Mass properties rotated into body frame.
    """
    A = M3.make(
        prop.Ixx,
        prop.Ixy,
        prop.Ixz,
        prop.Ixy,
        prop.Iyy,
        prop.Iyz,
        prop.Ixz,
        prop.Iyz,
        prop.Izz,
    )

    d, R = np.linalg.eig(A)

    permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 0, 1], [1, 2, 0], [2, 1, 0]]

    order = [0, 0, 0]
    for p in permutations:
        if d[p[0]] >= d[p[1]] >= d[p[2]]:
            order = p
            break

    # Create rotation matrix corresponding the new order
    R = R[:, order]

    # Ensure we have a special orthogonal matrix
    if np.linalg.det(R) < 0.0:
        R[:, 0] = -R[:, 0]

    q = Q.from_matrix(R)  # from body space to model space

    I_body = d[order]

    return q, I_body


def __translate_to_body_space_origin(prop_model):
    """
    Transforms inertia tensor given wrt model space origin to inertia tensor given wrt body space origin.

    :param prop_model: Mass properties wrt to model space.
    :return:           Mass properties translated into body space.
    """
    prop_body = MassProperties()

    m = prop_model.mass

    # Translation to apply to model space coordinates to get into body space coordinates
    #  (assuming coordinate frames are aligned)
    x = -prop_model.x
    y = -prop_model.y
    z = -prop_model.z

    prop_body.Ixx = prop_model.Ixx - m * (y * y + z * z)
    prop_body.Iyy = prop_model.Iyy - m * (x * x + z * z)
    prop_body.Izz = prop_model.Izz - m * (x * x + y * y)

    too_small = 1.0e-10
    prop_body.Ixy = prop_model.Ixy + m * (x * y)
    prop_body.Ixy = prop_body.Ixy if np.abs(prop_body.Ixy) > too_small else 0.0
    prop_body.Ixz = prop_model.Ixz + m * (x * z)
    prop_body.Ixz = prop_body.Ixz if np.abs(prop_body.Ixz) > too_small else 0.0
    prop_body.Iyz = prop_model.Iyz + m * (y * z)
    prop_body.Iyz = prop_body.Iyz if np.abs(prop_body.Iyz) > too_small else 0.0
    prop_body.mass = m
    prop_body.x = 0.0
    prop_body.y = 0.0
    prop_body.z = 0.0

    r = V3.make(x, y, z)

    return r, prop_body


def xform_model_2_body_space(prop):
    """
    Transform mass properties from model space into body space.

    :param prop:   Model space mass properties
    :return:       A tuple that gives the rigid body transformation from body space to model space and
                   the body space mass properties.
    """
    m = prop.mass
    (r, prop) = __translate_to_body_space_origin(prop)
    (q, I_body) = __rotate_to_model_space_orientation(prop)

    #
    # Transformations work as the following rigid body transformation
    #
    #     [p]_mf = q * [p]_bf  - r
    #
    #   T_bf2mf = |  q -r |
    #             |  0  1 |
    #
    #   | p_mf | = |  q -r | | p_bf |
    #   |  1   |   |  0  1 | | 1    |
    #
    r_bf2mf = -r
    q_bf2mf = q

    return r_bf2mf, q_bf2mf, m, I_body


def update_inertia_tensor(R, Bv):
    """
    Inertia Update Method.
    Computes, W = R B R^T. This method has been optimized to
    avoid having to transpose the orientation matrix and exploit
    symmetry of the inertia tensor.

    Note: Further optimization may be possible by exploiting common sub-terms.

    :param R: The orientation as a rotation matrix.
    :param Bv: An inertia tensor entity in body frame (ie a vector not a matrix)
    :return: The inertia tensor entity in the corresponding world frame.
    """
    """
      KE 30-08-2004: The formulas below was computed using Matlab, that is
      syms R00 R01 R02  R10 R11 R12 R20 R21 R22 real;
      syms I00 I01 I02  I10 I11 I12 I20 I21 I22 real;
      I = [ I00, I01 , I02; I01, I11, I12; I02, I12, I22]
      R = [ R00, R01 , R02; R10, R11, R12; R20, R21, R22]
      W = R*I*R'
      simplify(W)
    """
    # RT = np.transpose(R)
    # I = M3.diag_from_array(Bv)
    # W = R.dot(I.dot(RT))
    # TODO 2017-02-12 Kenny : We can hand optimize this to exploit zero-pattern of Bv
    W = M3.zero()
    W[0, 0] = (
        R[0, 0] * R[0, 0] * Bv[0]
        + R[0, 1] * R[0, 1] * Bv[1]
        + R[0, 2] * R[0, 2] * Bv[2]
    )
    W[1, 1] = (
        R[1, 0] * R[1, 0] * Bv[0]
        + R[1, 1] * R[1, 1] * Bv[1]
        + R[1, 2] * R[1, 2] * Bv[2]
    )
    W[2, 2] = (
        R[2, 0] * R[2, 0] * Bv[0]
        + R[2, 1] * R[2, 1] * Bv[1]
        + R[2, 2] * R[2, 2] * Bv[2]
    )
    W[1, 0] = W[0, 1] = (
        R[1, 0] * R[0, 0] * Bv[0]
        + R[1, 1] * R[0, 1] * Bv[1]
        + R[1, 2] * R[0, 2] * Bv[2]
    )
    W[2, 0] = W[0, 2] = (
        R[2, 0] * R[0, 0] * Bv[0]
        + R[2, 1] * R[0, 1] * Bv[1]
        + R[2, 2] * R[0, 2] * Bv[2]
    )
    W[2, 1] = W[1, 2] = (
        R[1, 0] * R[2, 0] * Bv[0]
        + R[1, 1] * R[2, 1] * Bv[1]
        + R[1, 2] * R[2, 2] * Bv[2]
    )
    return W
