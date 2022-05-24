import numpy as np
import rainbow.math.vector3 as V3
from rainbow.geometry.tetrahedron import compute_inscribed_sphere
from rainbow.geometry.tetrahedron import compute_circumscribed_sphere
from rainbow.geometry.tetrahedron import compute_signed_volume
from rainbow.geometry.tetrahedron import is_left_orientation


def vl(pi, pj, pk, pm):
    from math import fabs

    V = fabs(compute_signed_volume(pi, pj, pk, pm))
    Lij = V3.norm(pi - pj)
    Lik = V3.norm(pi - pk)
    Lim = V3.norm(pi - pm)
    Ljk = V3.norm(pj - pk)
    Ljm = V3.norm(pj - pm)
    Lkm = V3.norm(pk - pm)
    L2 = Lij ** 2 + Lik ** 2 + Lim ** 2 + Ljk ** 2 + Ljm ** 2 + Lkm ** 2
    return 12.0 * ((3.0 * V) ** (2.0 / 3.0)) / L2  # Barry Joe style


def rl(pi, pj, pk, pm):
    from math import sqrt

    Lij = V3.norm(pi - pj)
    Lik = V3.norm(pi - pk)
    Lim = V3.norm(pi - pm)
    Ljk = V3.norm(pj - pk)
    Ljm = V3.norm(pj - pm)
    Lkm = V3.norm(pk - pm)
    L_max = max([Lij, Lik, Lim, Ljk, Ljm, Lkm])
    Rin = compute_inscribed_sphere(pi, pj, pk, pm)[1]
    return 2.0 * sqrt(6.0) * Rin / L_max


def rr(pi, pj, pk, pm):
    Rin = compute_inscribed_sphere(pi, pj, pk, pm)[1]
    Rout = compute_circumscribed_sphere(pi, pj, pk, pm)[1]
    return 3.0 * Rin / Rout


def theta(pi, pj, pk, pm):
    from math import fabs
    from math import sqrt

    V = fabs(compute_signed_volume(pi, pj, pk, pm))
    Eij = pi - pj
    Eik = pi - pk
    Eim = pi - pm
    Ejk = pj - pk
    Ejm = pj - pm
    Ekm = pk - pm
    Lij = V3.norm(Eij)
    Lik = V3.norm(Eik)
    Lim = V3.norm(Eim)
    Ljk = V3.norm(Ejk)
    Ljm = V3.norm(Ejm)
    Lkm = V3.norm(Ekm)
    Ai = V3.norm(V3.cross(Ejk, Ekm)) / 2
    Aj = V3.norm(V3.cross(Eik, Ekm)) / 2
    Ak = V3.norm(V3.cross(Eij, Eim)) / 2
    Am = V3.norm(V3.cross(Ejk, Eik)) / 2
    Sij = Lij / (Ak * Am)
    Sik = Lik / (Aj * Am)
    Sim = Lim / (Aj * Ak)
    Sjk = Ljk / (Ai * Am)
    Sjm = Ljm / (Ai * Ak)
    Skm = Lkm / (Ai * Aj)
    S_min = min([Sij, Sik, Sim, Sjk, Sjm, Skm])
    return (9.0 * sqrt(2.0) / 8.0) * V * S_min


def compute_quality_vector(V, T, quality):
    """

    :param V:
    :param T:
    :param quality:
    :return:
    """
    X = V[:, 0]
    Y = V[:, 1]
    Z = V[:, 2]
    K = T.shape[0]
    quality_vector = np.zeros((K, 1), dtype=np.float64)
    for e in range(K):
        i = T[e, 0]
        j = T[e, 1]
        k = T[e, 2]
        m = T[e, 3]
        pi = V3.make(X[i], Y[i], Z[i])
        pj = V3.make(X[j], Y[j], Z[j])
        pk = V3.make(X[k], Y[k], Z[k])
        pm = V3.make(X[m], Y[m], Z[m])
        if is_left_orientation(pi, pj, pk, pm):
            pi, pj = pj, pi
        quality_vector[e] = quality(pi, pj, pk, pm)
    return quality_vector
