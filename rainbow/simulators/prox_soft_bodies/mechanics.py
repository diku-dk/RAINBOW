import rainbow.math.matrix3 as M3
import numpy as np


def right_cauchy_strain_tensor(F):
    """
    This function computes the right Cauchy strain tensor.

    :param F:  The deformation gradient
    :return:   The right Cauchy strain tensor
    """
    return np.matmul(np.transpose(F), F)


def green_strain_tensor(F):
    """
    This function computes the right Green strain tensor.

    :param F:  The deformation gradient
    :return:   The Green strain tensor
    """
    C = right_cauchy_strain_tensor(F)
    E = (C - M3.identity()) / 2
    return E


def cauchy_stress_tensor(F, S):
    """
    This function computes the right Cauchy stress tensor.

    :param F:   The deformation gradient.
    :param S:   The 2nd Piola-Kirchoff stress tensor.
    :return:    The Cauchy stress tensor.
    """
    j = np.linalg.det(F)
    sigma = np.multiply(
        (1 / j)[:, None, None], np.matmul(F, (np.matmul(S, F.transpose(0, 2, 1))))
    )
    return sigma


def create_material_parameters(name=None):
    """
    Convenience function to quickly get some "sensible" material parameters.

    :param name:  The name of the material to create parameters for.
    :return:      A triplet of Young modulus, Poisson ratio, and mass density.
    """
    E = 10e5  # Young modulus
    nu = 0.3  # Poisson ratio
    rho = 1000  # Mass density
    if name is None:
        return E, nu, rho
    if name.lower() == "cartilage":
        E = 0.69e6
        nu = 0.018
        rho = 1000
    if name.lower() == "cortical bone":
        E = 16.16e9
        nu = 0.33
        rho = 1600
    if name.lower() == "cancellous bone":
        E = 452e6
        nu = 0.3
        rho = 1600
    if name.lower() == "rubber":
        E = 0.01e9
        nu = 0.48
        rho = 1050
    if name.lower() == "concrete":
        E = 30e9
        nu = 0.20
        rho = 2320
    if name.lower() == "copper":
        E = 125e9
        nu = 0.35
        rho = 8900
    if name.lower() == "steel":
        E = 210e9
        nu = 0.31
        rho = 7800
    if name.lower() == "aluminium":
        E = 72e9
        nu = 0.34
        rho = 2700
    if name.lower() == "glass":
        E = 50e9
        nu = 0.18
        rho = 2190
    return E, nu, rho


def first_lame(E, nu):
    """
    Convert elastic parameters into Lamé parameters.

    :param E:   Young modulus
    :param nu:  Poisson ratio
    :return:    The corresponding value of the first Lamé parameter (lambda)
    """
    return (nu * E) / ((1 + nu) * (1 - 2 * nu))


def second_lame(E, nu):
    """
    Convert elastic parameters into Lamé parameters.

    :param E:   Young modulus
    :param nu:  Poisson ratio
    :return:    The corresponding value of the second Lamé parameter (mu)
    """
    return E / (2 * (1 + nu))


class SNH:
    """
    Elasticity model from "Stable Neo-Hookean Flesh Simulation".

    https://graphics.pixar.com/library/StableElasticity/paper.pdf
    """

    @staticmethod
    def pk1_stress(F, lambda_in, mu_in):
        """
        Compute stress tensors.

        :param F:           The deformation gradient
        :param lambda_in:   The first Lame parameter
        :param mu_in:       The second Lame parameter
        :return:            The first Piola-Kirchhoff stress tensor
        """
        mu_ = (4.0 / 3.0) * mu_in
        lambda_ = lambda_in + (5.0 / 6.0) * mu_in
        alpha = 1 + (mu_ / lambda_) - (mu_ / (4 * lambda_))
        J = np.linalg.det(F)
        C = np.matmul(np.transpose(F), F)
        I_C = np.trace(C)
        # J = F0 dot (F1 cross F2), from this we can compute dJdF
        dJdF = np.zeros_like(F)
        dJdF[:, 0] = np.cross(F[:, 1], F[:, 2])
        dJdF[:, 1] = np.cross(F[:, 2], F[:, 0])
        dJdF[:, 2] = np.cross(F[:, 0], F[:, 1])
        return mu_ * (1 - (1 / (I_C + 1))) * F + lambda_ * (J - alpha) * dJdF

    @staticmethod
    def energy_density(F, lambda_in, mu_in):
        """
        :param F:           The deformation gradient
        :param lambda_in:   The first Lame parameter
        :param mu_in:       The second Lame parameter
        :return:            The energy density.
        """
        mu_ = (4.0 / 3.0) * mu_in
        lambda_ = lambda_in + (5.0 / 6.0) * mu_in
        alpha = 1 + (mu_ / lambda_) - (mu_ / (4 * lambda_))
        J = np.linalg.det(F)
        C = np.matmul(np.transpose(F), F)
        I_C = np.trace(C)
        return (
            0.5 * mu_ * (I_C - 3)
            + 0.5 * lambda_ * np.square(J - alpha)
            - 0.5 * mu_ * np.log2(I_C + 1)
        )


class SVK:
    """
    Saint Venant Kirchhoff Elastic Model.
    """

    @staticmethod
    def pk1_stress(F, lambda_in, mu_in):
        """
        Compute stress tensors.

        :param F:           The deformation gradient
        :param lambda_in:   The first Lame parameter
        :param mu_in:       The second Lame parameter
        :return:            The first Piola-Kirchhoff stress tensor
        """
        C = np.matmul(F.T, F)
        E = (C - np.eye(3)) / 2
        S = lambda_in * np.trace(E) * np.eye(3) + 2 * mu_in * E
        return F.dot(S)

    @staticmethod
    def energy_density(F, lambda_in, mu_in):
        """
        :param F:           The deformation gradient
        :param lambda_in:   The first Lame parameter
        :param mu_in:       The second Lame parameter
        :return:            The energy density.
        """
        C = np.matmul(F.T, F)
        E = (C - np.eye(3)) / 2
        return 0.5 * lambda_in * np.trace(E) ** 2 + mu_in * np.tensordot(E, E)


class COR:
    """
    Corotational linear elastic model.
    """

    @staticmethod
    def pk1_stress(F, lambda_in, mu_in):
        """
        Compute stress tensors.

        :param F:           The deformation gradient
        :param lambda_in:   The first Lame parameter
        :param mu_in:       The second Lame parameter
        :return:            The first Piola-Kirchhoff stress tensor
        """
        R, S = M3.polar_decomposition_array(F)
        return (lambda_in * np.trace(S - np.eye(3))) * R + (2 * mu_in) * (F - R)

    @staticmethod
    def energy_density(F, lambda_in, mu_in):
        """
        :param F:           The deformation gradient
        :param lambda_in:   The first Lame parameter
        :param mu_in:       The second Lame parameter
        :return:            The energy density.
        """
        R, S = M3.polar_decomposition_array(F)
        #  mu ||F - R||_F^2 + (lambda/2) tr^2 (R^T F - I)
        return mu_in * np.square(np.linalg.norm(F - R, ord="fro")) + (
            lambda_in / 2
        ) * np.square(np.trace(S - np.eye(3)))
