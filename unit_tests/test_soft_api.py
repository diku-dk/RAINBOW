"""Smoke coverage for every public soft-body construction and state API."""

import unittest

import numpy as np

from darerl.simulators.soft import (
    StableNeoHookeanMaterial,
    SVKMaterial,
    SoftBody,
    TetMesh,
    compute_boundary_faces,
    create_beam_mesh,
    create_bending_baseline,
    create_compress_baseline,
    create_stretch_baseline,
    create_twist_baseline,
)

try:
    import jax  # noqa: F401
except ImportError:
    JAX_AVAILABLE = False
else:
    JAX_AVAILABLE = True


class TestSoftAPI(unittest.TestCase):
    def test_public_constructors_and_body_methods_are_operational(self):
        vertices, elements = create_beam_mesh(2, 2, 2, 0.1, 0.02, 0.02)
        self.assertEqual(elements.shape, (5, 4))
        self.assertEqual(compute_boundary_faces(elements).shape[1], 3)
        mesh = TetMesh.from_vertices(vertices, elements, density=1100.0)
        self.assertEqual(mesh.node_count, len(vertices))
        self.assertEqual(mesh.tet_count, len(elements))

        for material in (
            SVKMaterial(1.0e5, 0.49, 1100.0),
            StableNeoHookeanMaterial(1.0e5, 0.49, 1100.0),
        ):
            self.assertEqual(material.model_code, 0 if isinstance(material, SVKMaterial) and not isinstance(material, StableNeoHookeanMaterial) else 1)
            self.assertEqual(len(material.compute_lame_parameters()), 2)
            body = SoftBody(mesh, material, use_jax=False)
            np.testing.assert_allclose(
                body.compute_deformation_gradient(),
                np.repeat(np.eye(3)[None], mesh.tet_count, axis=0),
                atol=1.0e-14,
            )
            self.assertEqual(body.compute_green_lagrange_strain().shape, (mesh.tet_count, 3, 3))
            self.assertEqual(body.compute_elastic_forces().shape, vertices.shape)
            self.assertEqual(body.compute_neumann_forces().shape, vertices.shape)
            self.assertTrue(np.isfinite(body.compute_elastic_energy()))
            self.assertEqual(body.compute_body_forces().shape, vertices.shape)
            body.set_fixed_vertices([0])
            body.set_pressure_boundary([[1, 2, 3]], 1.0)
            body.validate_pressure_boundary()
            body.set_external_forces(np.zeros_like(vertices))
            body.clear_external_forces()
            body.step(1.0e-5, gravity=np.zeros(3))
            body.step_implicit(
                1.0e-5,
                gravity=np.zeros(3),
                settings={"max_iterations": 5, "history_size": 1},
            )
            body.synchronize()
            self.assertTrue(np.all(np.isfinite(body.x)))
            self.assertTrue(np.all(np.isfinite(body.v)))

    @unittest.skipUnless(JAX_AVAILABLE, "JAX is not installed")
    def test_public_body_api_is_operational_on_jax(self):
        vertices, elements = create_beam_mesh(2, 2, 2, 0.1, 0.02, 0.02)
        mesh = TetMesh.from_vertices(vertices, elements, density=1100.0)
        body = SoftBody(mesh, StableNeoHookeanMaterial(1.0e5, 0.49, 1100.0), use_jax=True)
        body.set_fixed_vertices([0])
        body.set_pressure_boundary([[1, 2, 3]], 1.0)
        body.set_external_forces(np.zeros_like(vertices))
        body.step(1.0e-5, gravity=np.zeros(3), sync=False)
        body.synchronize()
        body.compute_elastic_forces()
        body.compute_neumann_forces()
        body.compute_elastic_energy()
        body.step_implicit(
            1.0e-5,
            gravity=np.zeros(3),
            settings={"max_iterations": 5, "history_size": 1},
        )
        self.assertTrue(np.all(np.isfinite(body.x)))

    def test_all_canonical_baseline_factories_create_bodies(self):
        factories = (
            create_bending_baseline,
            create_stretch_baseline,
            create_compress_baseline,
            create_twist_baseline,
        )
        for factory in factories:
            baseline = factory(3, 2, 2)
            self.assertGreater(baseline.mesh.tet_count, 0)
            self.assertEqual(baseline.fixed.shape, (baseline.mesh.node_count,))
            body = baseline.create_body(use_jax=False)
            self.assertEqual(body.x.shape, baseline.mesh.x0.shape)


if __name__ == "__main__":
    unittest.main()
