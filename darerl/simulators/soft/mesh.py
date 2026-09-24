"""Small NumPy-only mesh constructors for the soft-body prototype."""

from __future__ import annotations

import numpy as np


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
