import unittest

import numpy as np

from darerl.simulators.soft import (
    BASELINE_DEPTH,
    BASELINE_HEIGHT,
    BASELINE_LENGTH,
    StableNeoHookeanMaterial,
    SVKMaterial,
    SoftBody,
    TetMesh,
    create_beam_mesh,
    create_bending_baseline,
    create_compress_baseline,
    create_stretch_baseline,
    create_twist_baseline,
)

try:
    import jax  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def one_tet():
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    return TetMesh.from_vertices(x, [[0, 1, 2, 3]], density=2.0), x


class TestSoftBody(unittest.TestCase):
    def test_baseline_factories_share_scale_and_create_finite_bodies(self):
        factories = (
            create_bending_baseline,
            create_stretch_baseline,
            create_compress_baseline,
            create_twist_baseline,
        )
        for factory in factories:
            case = factory(6, 3, 3)
            self.assertAlmostEqual(np.ptp(case.mesh.x0[:, 0]), BASELINE_LENGTH)
            self.assertAlmostEqual(np.ptp(case.mesh.x0[:, 1]), BASELINE_HEIGHT)
            self.assertAlmostEqual(np.ptp(case.mesh.x0[:, 2]), BASELINE_DEPTH)
            body = case.create_body(use_jax=False)
            body.step(1.0e-6, gravity=case.gravity)
            self.assertTrue(np.all(np.isfinite(body.x)))
            self.assertTrue(np.all(np.isfinite(body.v)))

    def test_numpy_beam_mesh_is_oriented_without_external_geometry_dependencies(self):
        vertices, elements = create_beam_mesh(3, 2, 2, 2.0, 1.0, 1.0)
        determinants = np.linalg.det(
            np.transpose(vertices[elements][:, 1:] - vertices[elements][:, :1], (0, 2, 1))
        )
        self.assertEqual(vertices.shape, (12, 3))
        self.assertEqual(elements.shape, (10, 4))
        self.assertTrue(np.all(determinants > 0.0))

    def test_reference_configuration_has_zero_force_and_expected_mass(self):
        mesh, x = one_tet()
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 2.0), use_jax=False)
        np.testing.assert_allclose(body.elastic_forces(x), 0.0, atol=1e-12)
        np.testing.assert_allclose(mesh.volume, 1.0 / 6.0)
        np.testing.assert_allclose(mesh.lumped_mass.sum(), 2.0 / 6.0)

    def test_stable_neo_hookean_has_zero_rest_force_and_finite_inversion_force(self):
        mesh, x = one_tet()
        body = SoftBody(mesh, StableNeoHookeanMaterial(1000.0, 0.3, 2.0), use_jax=False)
        np.testing.assert_allclose(body.elastic_forces(x), 0.0, atol=1e-12)
        inverted = x.copy()
        inverted[[1, 2]] = inverted[[2, 1]]
        self.assertTrue(np.all(np.isfinite(body.elastic_forces(inverted))))

    def test_material_force_is_negative_energy_gradient(self):
        mesh, x = one_tet()
        deformed = x.copy()
        deformed[1, 0] = 1.1
        deformed[2, 1] = 0.9
        direction = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3], [-0.1, 0.2, 0.1], [0.1, 0.1, -0.2]])
        epsilon = 1.0e-6
        for material in (SVKMaterial(1000.0, 0.3, 2.0), StableNeoHookeanMaterial(1000.0, 0.3, 2.0)):
            body = SoftBody(mesh, material, use_jax=False)
            numerical = (body.elastic_energy(deformed + epsilon * direction) - body.elastic_energy(deformed - epsilon * direction)) / (2.0 * epsilon)
            analytical = -np.sum(body.elastic_forces(deformed) * direction)
            self.assertAlmostEqual(numerical, analytical, delta=1.0e-5 * max(1.0, abs(numerical)))

    @unittest.skipUnless(JAX_AVAILABLE, "JAX is not installed")
    def test_jax_and_numpy_force_paths_agree_for_both_materials(self):
        mesh, x = one_tet()
        deformed = x.copy()
        deformed[1, 0] = 1.1
        deformed[2, 1] = 0.95
        for material in (SVKMaterial(1000.0, 0.3, 2.0), StableNeoHookeanMaterial(1000.0, 0.3, 2.0)):
            numpy_body = SoftBody(mesh, material, use_jax=False)
            jax_body = SoftBody(mesh, material, use_jax=True)
            np.testing.assert_allclose(jax_body.elastic_forces(deformed), numpy_body.elastic_forces(deformed), rtol=3.0e-5, atol=3.0e-5)
            jax_body.set_pressure_boundary([[1, 2, 3]], 2.0)
            numpy_body.set_pressure_boundary([[1, 2, 3]], 2.0)
            np.testing.assert_allclose(jax_body.neumann_forces(deformed), numpy_body.neumann_forces(deformed), rtol=3.0e-5, atol=3.0e-5)

    @unittest.skipUnless(JAX_AVAILABLE, "JAX is not installed")
    def test_jax_semi_implicit_step_matches_numpy(self):
        mesh, _ = one_tet()
        fixed = np.array([True, False, False, False])
        for material in (SVKMaterial(100.0, 0.3, 1.0), StableNeoHookeanMaterial(100.0, 0.3, 1.0)):
            numpy_body = SoftBody(mesh, material, fixed=fixed, use_jax=False)
            jax_body = SoftBody(mesh, material, fixed=fixed, use_jax=True)
            numpy_body.step(1.0e-3, gravity=(0.0, 0.0, -9.81))
            jax_body.step(1.0e-3, gravity=(0.0, 0.0, -9.81))
            np.testing.assert_allclose(jax_body.x, numpy_body.x, rtol=3.0e-5, atol=3.0e-5)
            np.testing.assert_allclose(jax_body.v, numpy_body.v, rtol=3.0e-5, atol=3.0e-5)

    def test_implicit_bfgs_converges_under_gravity_for_both_materials(self):
        mesh, _ = one_tet()
        for material in (SVKMaterial(100.0, 0.3, 1.0), StableNeoHookeanMaterial(100.0, 0.3, 1.0)):
            body = SoftBody(mesh, material, fixed=np.array([True, False, False, False]), use_jax=False)
            body.step_implicit(1.0e-3, gravity=(0.0, 0.0, -9.81), settings={"max_iterations": 30, "raise_on_failure": True})
            self.assertTrue(body.last_implicit_info["converged"])

    @unittest.skipUnless(JAX_AVAILABLE, "JAX is not installed")
    def test_jax_implicit_bfgs_matches_numpy_native_solver(self):
        mesh, _ = one_tet()
        fixed = np.array([True, False, False, False])
        settings = {"max_iterations": 30, "history_size": 5, "raise_on_failure": True}
        for material in (SVKMaterial(100.0, 0.3, 1.0), StableNeoHookeanMaterial(100.0, 0.3, 1.0)):
            numpy_body = SoftBody(mesh, material, fixed=fixed, use_jax=False)
            jax_body = SoftBody(mesh, material, fixed=fixed, use_jax=True)
            numpy_body.step_implicit(1.0e-3, gravity=(0.0, 0.0, -9.81), settings=settings)
            jax_body.step_implicit(1.0e-3, gravity=(0.0, 0.0, -9.81), settings=settings)
            self.assertTrue(jax_body.last_implicit_info["converged"])
            np.testing.assert_allclose(jax_body.x, numpy_body.x, rtol=3.0e-5, atol=3.0e-5)
            np.testing.assert_allclose(jax_body.v, numpy_body.v, rtol=3.0e-5, atol=3.0e-5)


    def test_force_is_restoring_for_small_stretch(self):
        mesh, x = one_tet()
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        stretched = x.copy()
        stretched[1, 0] = 1.1
        force = body.elastic_forces(stretched)
        self.assertLess(force[1, 0], 0.0)
        np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1e-10)

    def test_pressure_face_set_uses_linear_surface_fem_forces(self):
        mesh, x = one_tet()
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        # Face (1, 2, 3) is oriented outward from the reference tetrahedron.
        body.set_pressure_boundary([[1, 2, 3]], pressure=6.0)
        force = body.neumann_forces(x)
        expected_total = 6.0 * 0.5 * np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(force.sum(axis=0), expected_total)
        np.testing.assert_allclose(force[[1, 2, 3]], np.tile(expected_total / 3.0, (3, 1)))
        np.testing.assert_allclose(force[0], 0.0)

    def test_fixed_vertices_are_preserved_by_time_step(self):
        mesh, x = one_tet()
        body = SoftBody(mesh, SVKMaterial(100.0, 0.3, 1.0), use_jax=False)
        body.set_fixed_vertices([0])
        body.step(0.01, gravity=(0.0, 0.0, -10.0))
        np.testing.assert_allclose(body.x[0], x[0])
        np.testing.assert_allclose(body.v[0], 0.0)

    def test_implicit_bfgs_accepts_settings_and_preserves_rest_state(self):
        mesh, x = one_tet()
        body = SoftBody(mesh, SVKMaterial(100.0, 0.3, 1.0), use_jax=False)
        body.set_fixed_vertices([0, 1, 2, 3])
        body.step_implicit(
            0.01,
            gravity=(0.0, 0.0, 0.0),
            settings={"max_iterations": 8, "history_size": 3, "raise_on_failure": True},
        )
        np.testing.assert_allclose(body.x, x)
        self.assertTrue(body.last_implicit_info["converged"])


    def test_semi_implicit_step_moves_under_gravity(self):
        mesh, _ = one_tet()
        body = SoftBody(mesh, SVKMaterial(100.0, 0.3, 1.0), use_jax=False)
        body.step(0.01, gravity=(0, 0, -10))
        self.assertTrue(np.all(body.v[:, 2] < 0.0))


if __name__ == "__main__":
    unittest.main()
