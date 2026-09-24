"""Direct deformation-gradient and strain tests."""

import unittest

import numpy as np

from darerl.simulators.soft import SVKMaterial, SoftBody, TetMesh, create_beam_mesh


def one_tet_body():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mesh = TetMesh.from_vertices(vertices, [[0, 1, 2, 3]], density=1.0)
    return SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False), vertices


def rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TestSoftKinematics(unittest.TestCase):
    def test_reference_tet_has_identity_deformation_gradient_and_zero_strain(self):
        body, vertices = one_tet_body()
        np.testing.assert_allclose(body.compute_deformation_gradient(vertices)[0], np.eye(3), atol=1.0e-14)
        np.testing.assert_allclose(body.compute_green_lagrange_strain(vertices)[0], 0.0, atol=1.0e-14)

    def test_single_tet_recovers_arbitrary_affine_deformation_exactly(self):
        body, vertices = one_tet_body()
        deformation_gradient = np.array([[1.2, 0.1, -0.05], [0.02, 0.85, 0.08], [0.03, -0.04, 1.1]])
        deformed = vertices @ deformation_gradient.T + np.array([4.0, -2.0, 0.75])
        np.testing.assert_allclose(body.compute_deformation_gradient(deformed)[0], deformation_gradient, rtol=1.0e-13, atol=1.0e-13)

    def test_deformation_gradient_is_translation_invariant(self):
        body, vertices = one_tet_body()
        deformation_gradient = np.array([[1.1, 0.2, 0.0], [0.0, 0.9, 0.1], [0.03, 0.0, 1.05]])
        deformed = vertices @ deformation_gradient.T
        translated = deformed + np.array([-3.0, 1.25, 0.6])
        np.testing.assert_allclose(body.compute_deformation_gradient(translated), body.compute_deformation_gradient(deformed), atol=1.0e-13)

    def test_rigid_coordinate_rotations_are_recovered_and_have_zero_strain(self):
        body, vertices = one_tet_body()
        for axis in range(3):
            with self.subTest(axis=axis):
                q = rotation(axis, 0.61)
                deformed = vertices @ q.T
                np.testing.assert_allclose(body.compute_deformation_gradient(deformed)[0], q, rtol=1.0e-13, atol=1.0e-13)
                np.testing.assert_allclose(body.compute_green_lagrange_strain(deformed)[0], 0.0, atol=1.0e-13)

    def test_green_lagrange_strain_matches_definition_for_stretch_and_shear(self):
        body, vertices = one_tet_body()
        deformation_gradient = np.array([[1.15, 0.2, -0.1], [0.0, 0.9, 0.05], [0.03, 0.0, 1.05]])
        deformed = vertices @ deformation_gradient.T
        expected = 0.5 * (deformation_gradient.T @ deformation_gradient - np.eye(3))
        np.testing.assert_allclose(body.compute_green_lagrange_strain(deformed)[0], expected, rtol=1.0e-13, atol=1.0e-13)
        np.testing.assert_allclose(body.compute_green_lagrange_strain(deformed)[0], body.compute_green_lagrange_strain(deformed)[0].T, atol=1.0e-13)

    def test_all_elements_recover_same_affine_map_on_multi_tet_beam(self):
        vertices, elements = create_beam_mesh(4, 3, 3, 2.0, 0.8, 0.6)
        mesh = TetMesh.from_vertices(vertices, elements, density=1.0)
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        deformation_gradient = np.array([[1.08, 0.12, -0.03], [0.01, 0.93, 0.07], [0.02, -0.04, 1.04]])
        deformed = vertices @ deformation_gradient.T + np.array([0.2, -0.3, 0.4])
        gradients = body.compute_deformation_gradient(deformed)
        strains = body.compute_green_lagrange_strain(deformed)
        np.testing.assert_allclose(gradients, np.broadcast_to(deformation_gradient, gradients.shape), rtol=1.0e-12, atol=1.0e-12)
        expected_strain = 0.5 * (deformation_gradient.T @ deformation_gradient - np.eye(3))
        np.testing.assert_allclose(strains, np.broadcast_to(expected_strain, strains.shape), rtol=1.0e-12, atol=1.0e-12)

    def test_kinematics_reject_invalid_position_shapes(self):
        body, _ = one_tet_body()
        for invalid in (np.zeros(3), np.zeros((3, 3)), np.zeros((5, 3))):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):
                    body.compute_deformation_gradient(invalid)
                with self.assertRaises(ValueError):
                    body.compute_green_lagrange_strain(invalid)


if __name__ == "__main__":
    unittest.main()
