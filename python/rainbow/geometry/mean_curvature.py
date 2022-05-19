import numpy as np
import rainbow.geometry.grid3 as GRID


def create_from_sdf(phi):
    """
    :param phi:   A grid3 signed distance field.
    :return:      A grid3 of same dimension as input phi, but will contain the mean curvature value.
    """
    # TODO 2021-03-25 Kenny: We are missing a way of setting the boundary vertex to a high default value. They are currently 0.
    kappa = GRID.Grid(phi.min_coord, phi.max_coord, phi.I, phi.J, phi.K)
    kappa.values = np.zeros(phi.values.shape, dtype=np.float64)

    dx = phi.spacing[0]
    dy = phi.spacing[1]
    dz = phi.spacing[2]
    inv_2dx = 1.0 / (2.0 * dx)
    inv_2dy = 1.0 / (2.0 * dy)
    inv_2dz = 1.0 / (2.0 * dz)
    inv_dxdx = 1.0 / (dx * dx)
    inv_dydy = 1.0 / (dy * dy)
    inv_dzdz = 1.0 / (dz * dz)
    inv_4dxdy = 1.0 / (4.0 * dx * dy)
    inv_4dxdz = 1.0 / (4.0 * dx * dz)
    inv_4dydz = 1.0 / (4.0 * dy * dz)

    kappa_max = 1 / np.amin(phi.spacing)
    kappa_min = -kappa_max

    for k in range(1, phi.K - 1):
        for j in range(1, phi.J - 1):
            for i in range(1, phi.I - 1):
                d000 = 2 * phi.get_node_value(i, j, k)
                dp00 = phi.get_node_value(i + 1, j, k)
                dm00 = phi.get_node_value(i - 1, j, k)
                d0p0 = phi.get_node_value(i, j + 1, k)
                d0m0 = phi.get_node_value(i, j - 1, k)
                d00p = phi.get_node_value(i, j, k + 1)
                d00m = phi.get_node_value(i, j, k - 1)
                dpp0 = phi.get_node_value(i + 1, j + 1, k)
                dmp0 = phi.get_node_value(i - 1, j + 1, k)
                dpm0 = phi.get_node_value(i + 1, j - 1, k)
                dmm0 = phi.get_node_value(i - 1, j - 1, k)
                dp0p = phi.get_node_value(i + 1, j, k + 1)
                dm0p = phi.get_node_value(i - 1, j, k + 1)
                dp0m = phi.get_node_value(i + 1, j, k - 1)
                dm0m = phi.get_node_value(i - 1, j, k - 1)
                d0pp = phi.get_node_value(i, j + 1, k + 1)
                d0pm = phi.get_node_value(i, j + 1, k - 1)
                d0mp = phi.get_node_value(i, j - 1, k + 1)
                d0mm = phi.get_node_value(i, j - 1, k - 1)

                Dx = (dp00 - dm00) * inv_2dx
                Dy = (d0p0 - d0m0) * inv_2dy
                Dz = (d00p - d00m) * inv_2dz
                Dxx = (dp00 + dm00 - d000) * inv_dxdx
                Dyy = (d0p0 + d0m0 - d000) * inv_dydy
                Dzz = (d00p + d00m - d000) * inv_dzdz
                Dxy = (dpp0 - dmp0 - dpm0 + dmm0) * inv_4dxdy
                Dxz = (dp0p - dm0p - dp0m + dm0m) * inv_4dxdz
                Dyz = (d0pp - d0pm - d0mp + d0mm) * inv_4dydz

                g = np.array([Dx, Dy, Dz])
                l = np.linalg.norm(g)
                if l < 0.5:
                    l = 1.0

                num = (
                    Dxx * (Dy ** 2 + Dz ** 2)
                    + Dyy * (Dx ** 2 + Dz ** 2)
                    + Dzz * (Dx ** 2 + Dy ** 2)
                    - 2 * (Dxy * Dx * Dy + Dxz * Dx * Dz + Dyz * Dy * Dz)
                )
                denom = l ** 3
                row_idx = kappa.get_linear_index(i, j, k)
                kappa.values[row_idx] = np.clip(num / denom, kappa_min, kappa_max)
    return kappa
