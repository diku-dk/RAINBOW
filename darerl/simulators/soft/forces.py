"""NumPy force, tangent, pressure, and energy kernels."""

from __future__ import annotations

import numpy as np

from .mesh import TetMesh


def compute_elastic_forces(x, mesh: TetMesh, lam: float, mu: float, model_code: int = 0):
    p = x[mesh.elements]
    d = np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
    p1 = _pk1_stress(d @ mesh.inv_Dm, lam, mu, model_code)
    local = -np.einsum("eij,eaj->eai", p1, mesh.volume_grad_N)
    indices = mesh.elements.reshape(-1)
    out = np.empty_like(x)
    for axis in range(3):
        out[:, axis] = np.bincount(indices, weights=local[:, :, axis].reshape(-1), minlength=len(x))
    return out


def compute_directional_forces(x, direction, mesh: TetMesh, lam: float, mu: float, model_code: int, pressure_faces, pressure):
    """Return the analytical directional derivative of elastic and pressure forces."""
    p, dp = x[mesh.elements], direction[mesh.elements]
    d = np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
    dd = np.stack((dp[:, 1] - dp[:, 0], dp[:, 2] - dp[:, 0], dp[:, 3] - dp[:, 0]), axis=2)
    f, df = d @ mesh.inv_Dm, dd @ mesh.inv_Dm
    if model_code == 0:
        c = np.einsum("...ji,...jk->...ik", f, f)
        dc = np.einsum("...ji,...jk->...ik", df, f) + np.einsum("...ji,...jk->...ik", f, df)
        strain, dstrain = 0.5 * (c - np.eye(3)), 0.5 * dc
        identity = np.eye(3)
        stress = lam * np.trace(strain, axis1=1, axis2=2)[:, None, None] * identity + 2.0 * mu * strain
        dstress = lam * np.trace(dstrain, axis1=1, axis2=2)[:, None, None] * identity + 2.0 * mu * dstrain
        dp1 = np.einsum("eij,ejk->eik", df, stress) + np.einsum("eij,ejk->eik", f, dstress)
    elif model_code == 1:
        mu_hat, lam_hat = (4.0 / 3.0) * mu, lam + (5.0 / 6.0) * mu
        alpha = 1.0 + mu_hat / lam_hat - mu_hat / (4.0 * lam_hat)
        i_c, d_i_c = np.sum(f * f, axis=(1, 2)), 2.0 * np.sum(f * df, axis=(1, 2))
        cof = np.stack((np.cross(f[:, 1], f[:, 2]), np.cross(f[:, 2], f[:, 0]), np.cross(f[:, 0], f[:, 1])), axis=1)
        dcof = np.stack((
            np.cross(df[:, 1], f[:, 2]) + np.cross(f[:, 1], df[:, 2]),
            np.cross(df[:, 2], f[:, 0]) + np.cross(f[:, 2], df[:, 0]),
            np.cross(df[:, 0], f[:, 1]) + np.cross(f[:, 0], df[:, 1])), axis=1)
        j, d_j = np.linalg.det(f), np.sum(cof * df, axis=(1, 2))
        a = mu_hat * (1.0 - 1.0 / (i_c + 1.0))
        d_a = mu_hat * d_i_c / (i_c + 1.0) ** 2
        dp1 = d_a[:, None, None] * f + a[:, None, None] * df + lam_hat * (d_j[:, None, None] * cof + (j - alpha)[:, None, None] * dcof)
    else:
        raise ValueError(f"unknown material model code: {model_code}")
    local = -np.einsum("eij,eaj->eai", dp1, mesh.volume_grad_N)
    out = np.zeros_like(x)
    indices = mesh.elements.reshape(-1)
    for axis in range(3):
        out[:, axis] = np.bincount(indices, weights=local[:, :, axis].reshape(-1), minlength=len(x))
    out += compute_pressure_directional_forces(x, direction, pressure_faces, pressure, len(x))
    return out


def compute_pressure_forces(x, faces, pressure, node_count):
    out = np.zeros((node_count, 3), dtype=x.dtype)
    if len(faces) == 0:
        return out
    face_x = x[faces]
    area_vectors = 0.5 * np.cross(face_x[:, 1] - face_x[:, 0], face_x[:, 2] - face_x[:, 0])
    local = np.broadcast_to(pressure, (len(faces),))[:, None] * area_vectors / 3.0
    indices, nodal = faces.reshape(-1), np.repeat(local, 3, axis=0)
    for axis in range(3):
        out[:, axis] = np.bincount(indices, weights=nodal[:, axis], minlength=node_count)
    return out


def compute_pressure_directional_forces(x, direction, faces, pressure, node_count):
    out = np.zeros((node_count, 3), dtype=x.dtype)
    if len(faces) == 0:
        return out
    face_x, face_dx = x[faces], direction[faces]
    e1, e2 = face_x[:, 1] - face_x[:, 0], face_x[:, 2] - face_x[:, 0]
    de1, de2 = face_dx[:, 1] - face_dx[:, 0], face_dx[:, 2] - face_dx[:, 0]
    d_area = 0.5 * (np.cross(de1, e2) + np.cross(e1, de2))
    local = np.broadcast_to(pressure, (len(faces),))[:, None] * d_area / 3.0
    indices, nodal = faces.reshape(-1), np.repeat(local, 3, axis=0)
    for axis in range(3):
        out[:, axis] = np.bincount(indices, weights=nodal[:, axis], minlength=node_count)
    return out


def compute_energy_density(f, lam: float, mu: float, model_code: int):
    c = np.einsum("...ji,...jk->...ik", f, f)
    i_c = np.trace(c, axis1=1, axis2=2)
    if model_code == 0:
        strain = 0.5 * (c - np.eye(3))
        return 0.5 * lam * np.trace(strain, axis1=1, axis2=2) ** 2 + mu * np.sum(strain * strain, axis=(1, 2))
    if model_code == 1:
        mu_hat, lam_hat = (4.0 / 3.0) * mu, lam + (5.0 / 6.0) * mu
        alpha = 1.0 + mu_hat / lam_hat - mu_hat / (4.0 * lam_hat)
        j = np.linalg.det(f)
        return 0.5 * mu_hat * (i_c - 3.0) + 0.5 * lam_hat * (j - alpha) ** 2 - 0.5 * mu_hat * np.log(i_c + 1.0)
    raise ValueError(f"unknown material model code: {model_code}")


def _pk1_stress(f, lam: float, mu: float, model_code: int):
    if model_code == 0:
        c = np.einsum("...ji,...jk->...ik", f, f)
        strain = 0.5 * (c - np.eye(3))
        s = lam * np.trace(strain, axis1=1, axis2=2)[:, None, None] * np.eye(3) + 2.0 * mu * strain
        return f @ s
    if model_code == 1:
        mu_hat, lam_hat = (4.0 / 3.0) * mu, lam + (5.0 / 6.0) * mu
        alpha = 1.0 + mu_hat / lam_hat - mu_hat / (4.0 * lam_hat)
        i_c = np.sum(f * f, axis=(1, 2))
        cof = np.stack((np.cross(f[:, 1], f[:, 2]), np.cross(f[:, 2], f[:, 0]), np.cross(f[:, 0], f[:, 1])), axis=1)
        j = np.linalg.det(f)
        return mu_hat * (1.0 - 1.0 / (i_c + 1.0))[:, None, None] * f + lam_hat * (j - alpha)[:, None, None] * cof
    raise ValueError(f"unknown material model code: {model_code}")
