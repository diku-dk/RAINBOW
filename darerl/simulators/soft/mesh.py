"""Small NumPy-only mesh constructors for the soft-body prototype."""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass


@dataclass(frozen=True)
class TetMesh:
    """Reference data for an oriented first-order tetrahedral mesh."""

    x0: np.ndarray
    elements: np.ndarray
    inv_Dm: np.ndarray
    volume: np.ndarray
    grad_N: np.ndarray
    volume_grad_N: np.ndarray
    lumped_mass: np.ndarray
    inverse_lumped_mass: np.ndarray

    @classmethod
    def from_vertices(cls, vertices: np.ndarray, elements: np.ndarray, density: float = 1.0) -> "TetMesh":
        x0 = np.asarray(vertices, dtype=np.float64)
        t = np.asarray(elements, dtype=np.int32)
        if x0.ndim != 2 or x0.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if t.ndim != 2 or t.shape[1] != 4:
            raise ValueError("elements must have shape (K, 4)")
        if not np.all(np.isfinite(x0)):
            raise ValueError("vertices must be finite")
        if np.any(t < 0) or np.any(t >= len(x0)):
            raise ValueError("elements contain an invalid vertex index")
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError("density must be finite and positive")
        p = x0[t]
        dm = np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
        det = np.linalg.det(dm)
        if np.any(det <= 0.0):
            bad = int(np.flatnonzero(det <= 0.0)[0])
            raise ValueError(f"tetrahedron {bad} is inverted or degenerate in reference space")
        inv_dm = np.linalg.inv(dm)
        volume = det / 6.0
        grad_N = np.empty((len(t), 4, 3), dtype=np.float64)
        grad_N[:, 1:, :] = inv_dm
        grad_N[:, 0, :] = -np.sum(grad_N[:, 1:, :], axis=1)
        mass = np.zeros(len(x0), dtype=np.float64)
        np.add.at(mass, t.reshape(-1), np.repeat(density * volume / 4.0, 4))
        if np.any(mass <= 0.0):
            raise ValueError("tetrahedral mesh contains a node with zero lumped mass")
        return cls(x0, t, inv_dm, volume, grad_N, volume[:, None, None] * grad_N, mass, 1.0 / mass)

    @property
    def node_count(self) -> int:
        return len(self.x0)

    @property
    def tet_count(self) -> int:
        return len(self.elements)


def compute_boundary_faces(elements: np.ndarray) -> np.ndarray:
    """Extract the unique triangular boundary faces from tetrahedra."""
    tetrahedra = np.asarray(elements, dtype=np.int32)
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("elements must have shape (K, 4)")
    face_map: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for tet in tetrahedra:
        for face in ((tet[1], tet[2], tet[3]), (tet[0], tet[3], tet[2]), (tet[0], tet[1], tet[3]), (tet[0], tet[2], tet[1])):
            key = tuple(sorted(int(vertex) for vertex in face))
            if key in face_map:
                del face_map[key]
            else:
                face_map[key] = tuple(int(vertex) for vertex in face)
    return np.asarray(list(face_map.values()), dtype=np.int32).reshape((-1, 3))


def create_beam_mesh(
    i: int,
    j: int,
    k: int,
    length: float,
    height: float,
    depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an oriented, structured first-order tetrahedral beam mesh.

    ``i``, ``j`` and ``k`` are node counts along x, y and z.  Each hexahedral
    cell is split into five tetrahedra using an alternating diagonal pattern.
    The returned tetrahedra have positive reference orientation and require
    only NumPy; no geometry or igl dependency is involved.
    """
    if min(i, j, k) < 2:
        raise ValueError("i, j, and k must all be at least 2")
    if min(length, height, depth) <= 0.0:
        raise ValueError("beam dimensions must be positive")

    x = np.linspace(-length / 2.0, length / 2.0, i)
    y = np.linspace(-height / 2.0, height / 2.0, j)
    z = np.linspace(-depth / 2.0, depth / 2.0, k)
    vertices = np.array([[xx, yy, zz] for zz in z for yy in y for xx in x], dtype=np.float64)

    elements = []
    node = lambda ii, jj, kk: (kk * j + jj) * i + ii
    for kk in range(k - 1):
        for jj in range(j - 1):
            for ii in range(i - 1):
                i000 = node(ii, jj, kk)
                i001 = node(ii + 1, jj, kk)
                i010 = node(ii, jj + 1, kk)
                i011 = node(ii + 1, jj + 1, kk)
                i100 = node(ii, jj, kk + 1)
                i101 = node(ii + 1, jj, kk + 1)
                i110 = node(ii, jj + 1, kk + 1)
                i111 = node(ii + 1, jj + 1, kk + 1)
                if (ii + jj + kk) % 2:
                    cells = (
                        (i000, i001, i010, i100),
                        (i010, i001, i011, i111),
                        (i100, i110, i111, i010),
                        (i100, i111, i101, i001),
                        (i010, i111, i100, i001),
                    )
                else:
                    cells = (
                        (i000, i001, i011, i101),
                        (i000, i011, i010, i110),
                        (i100, i110, i101, i000),
                        (i101, i110, i111, i011),
                        (i000, i011, i110, i101),
                    )
                elements.extend(cells)

    return vertices, np.asarray(elements, dtype=np.int32)
