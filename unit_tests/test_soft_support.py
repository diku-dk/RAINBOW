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
    compute_boundary_faces,
    create_beam_mesh,
    create_bending_baseline,
    create_compress_baseline,
    create_stretch_baseline,
    create_twist_baseline,
)
from darerl.simulators.soft.solver import _numpy_energy_density, _numpy_pk1_stress

try:
    import jax  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def one_tet():
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    return TetMesh.from_vertices(x, [[0, 1, 2, 3]], density=2.0), x


def regular_tet_body(material, use_jax=False):
    """Return a positively oriented regular tetrahedron centered at the origin."""
    vertices = np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]],
        dtype=float,
    )
    mesh = TetMesh.from_vertices(vertices, [[0, 1, 2, 3]], density=1.0)
    return SoftBody(mesh, material, use_jax=use_jax), vertices


def rotation_matrix(axis, angle):
    """Construct a right-handed coordinate-axis rotation."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def octahedron_body(material, use_jax=False):
    """Return an oriented eight-tet octahedron with its apex at the origin."""
    axes = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
         [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        dtype=float,
    )
    vertices = np.vstack((np.zeros((1, 3)), axes))
    tetrahedra = []
    for x_index in (1, 2):
        for y_index in (3, 4):
            for z_index in (5, 6):
                face = [x_index, y_index, z_index]
                points = vertices[face]
                if np.linalg.det(points.T) < 0.0:
                    face[1], face[2] = face[2], face[1]
                tetrahedra.append([0, *face])
    mesh = TetMesh.from_vertices(vertices, tetrahedra, density=1.0)
    return SoftBody(mesh, material, use_jax=use_jax), vertices


class _SoftBodyTests:
    def test_beam_lumped_masses_match_barycentric_dual_control_volumes(self):
        density = 1375.0
        beam_dimensions = (2.4, 0.8, 0.6)
        for resolution in ((2, 2, 2), (3, 3, 2), (5, 4, 3)):
            with self.subTest(resolution=resolution):
                vertices, elements = create_beam_mesh(*resolution, *beam_dimensions)
                mesh = TetMesh.from_vertices(vertices, elements, density=density)
                dual_volumes = np.zeros(mesh.node_count)
                np.add.at(dual_volumes, elements.reshape(-1), np.repeat(mesh.volume / 4.0, 4))

                # For linear tetrahedra, barycentric mass lumping assigns one
                # quarter of each tetrahedron's volume to each vertex.
                np.testing.assert_allclose(mesh.lumped_mass / density, dual_volumes, rtol=1.0e-13, atol=1.0e-14)
                np.testing.assert_allclose(mesh.lumped_mass / dual_volumes, density, rtol=1.0e-13, atol=1.0e-12)

    def test_beam_total_mass_is_density_times_analytic_volume_at_all_resolutions(self):
        density = 950.0
        length, height, depth = (3.1, 0.7, 0.45)
        expected_mass = density * length * height * depth
        measured_masses = []
        for resolution in ((2, 2, 2), (3, 2, 3), (4, 4, 2), (7, 3, 5)):
            vertices, elements = create_beam_mesh(*resolution, length, height, depth)
            mesh = TetMesh.from_vertices(vertices, elements, density=density)
            measured_masses.append(mesh.lumped_mass.sum())
            np.testing.assert_allclose(mesh.volume.sum(), length * height * depth, rtol=1.0e-13, atol=1.0e-14)
            np.testing.assert_allclose(mesh.lumped_mass.sum(), expected_mass, rtol=1.0e-13, atol=1.0e-12)
        np.testing.assert_allclose(measured_masses, expected_mass, rtol=1.0e-13, atol=1.0e-12)

    def test_beam_mass_scales_linearly_with_density(self):
        vertices, elements = create_beam_mesh(5, 4, 3, 1.7, 0.4, 0.9)
        low_density = TetMesh.from_vertices(vertices, elements, density=2.0)
        high_density = TetMesh.from_vertices(vertices, elements, density=11.0)
        np.testing.assert_allclose(high_density.lumped_mass, 5.5 * low_density.lumped_mass, rtol=1.0e-13, atol=1.0e-13)

    def test_beam_body_forces_have_correct_density_and_dual_volume_integral(self):
        density = 1200.0
        dimensions = (2.0, 0.6, 0.4)
        acceleration = np.array([1.5, -9.81, 3.25])
        expected_density = density * acceleration
        expected_resultant = density * np.prod(dimensions) * acceleration
        for resolution in ((2, 2, 2), (3, 3, 2), (6, 4, 3)):
            with self.subTest(resolution=resolution):
                vertices, elements = create_beam_mesh(*resolution, *dimensions)
                mesh = TetMesh.from_vertices(vertices, elements, density=density)
                body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, density), use_jax=False)
                nodal_forces = body.compute_body_forces(acceleration)
                dual_volumes = mesh.lumped_mass / density

                np.testing.assert_allclose(
                    nodal_forces / dual_volumes[:, None],
                    np.broadcast_to(expected_density, nodal_forces.shape),
                    rtol=1.0e-13,
                    atol=1.0e-12,
                )
                np.testing.assert_allclose(nodal_forces.sum(axis=0), expected_resultant, rtol=1.0e-13, atol=1.0e-12)

    def test_body_force_api_rejects_invalid_acceleration_shape(self):
        mesh, _ = one_tet()
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 2.0), use_jax=False)
        for invalid in ((0.0,), (0.0, 0.0), np.zeros((3, 1))):
            with self.subTest(invalid_shape=np.shape(invalid)):
                with self.assertRaises(ValueError):
                    body.compute_body_forces(invalid)

    def test_external_nodal_forces_are_persistent_and_can_be_cleared(self):
        mesh, rest = one_tet()
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 2.0), use_jax=False)
        external = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [0.5, 0.25, -0.75], [2.0, 1.0, 0.0]])
        body.set_external_forces(external)
        np.testing.assert_allclose(body.external_forces, external)
        np.testing.assert_allclose(body._total_forces(rest, np.zeros(3)), external)
        body.clear_external_forces()
        np.testing.assert_allclose(body.external_forces, 0.0)
        np.testing.assert_allclose(body._total_forces(rest, np.zeros(3)), 0.0, atol=1.0e-14)

    def test_total_force_is_additive_for_elastic_pressure_body_and_external_terms(self):
        mesh, rest = one_tet()
        deformed = rest @ np.array([[1.1, 0.1, 0.0], [0.0, 0.9, 0.05], [0.0, 0.0, 1.05]]).T
        gravity = np.array([1.0, -2.0, 0.5])
        external = np.array([[0.2, 0.3, -0.4], [-0.1, 0.5, 0.6], [0.7, -0.8, 0.9], [-0.3, 0.2, 0.1]])
        faces = np.array([[1, 2, 3]], dtype=np.int32)
        pressure = 2.5
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 2.0), use_jax=False)
        body.set_pressure_boundary(faces, pressure)
        body.set_external_forces(external)
        expected = (
            body.compute_elastic_forces(deformed)
            + body.compute_neumann_forces(deformed)
            + body.compute_body_forces(gravity)
            + external
        )
        np.testing.assert_allclose(body._total_forces(deformed, gravity), expected, rtol=1.0e-12, atol=1.0e-12)

    def test_directional_force_matches_independent_finite_difference(self):
        mesh, rest = one_tet()
        deformation = np.array([[1.1, 0.1, 0.0], [0.0, 0.9, 0.05], [0.0, 0.0, 1.05]])
        position = rest @ deformation.T
        direction = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3], [-0.1, 0.2, 0.05], [0.1, 0.1, -0.2]])
        gravity = np.array([0.3, -1.2, 0.4])
        body = SoftBody(mesh, StableNeoHookeanMaterial(1000.0, 0.3, 1.0), use_jax=False)
        body.set_pressure_boundary([[1, 2, 3]], 2.0)
        body.set_external_forces(np.array([[0.1, -0.2, 0.3], [0.2, 0.0, -0.1], [0.0, 0.4, 0.2], [-0.3, 0.1, 0.0]]))
        epsilon = 1.0e-6
        expected = (body._total_forces(position + epsilon * direction, gravity) - body._total_forces(position - epsilon * direction, gravity)) / (2.0 * epsilon)
        actual = body._directional_force(position, direction, gravity, epsilon)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-7)

    @unittest.skipUnless(JAX_AVAILABLE, "JAX is not installed")
    def test_jax_directional_force_matches_finite_difference(self):
        mesh, rest = one_tet()
        position = rest @ np.array([[1.1, 0.1, 0.0], [0.0, 0.9, 0.05], [0.0, 0.0, 1.05]]).T
        direction = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3], [-0.1, 0.2, 0.05], [0.1, 0.1, -0.2]])
        gravity = np.array([0.3, -1.2, 0.4])
        body = SoftBody(mesh, StableNeoHookeanMaterial(1000.0, 0.3, 1.0), use_jax=True)
        body.set_pressure_boundary([[1, 2, 3]], 2.0)
        body.set_external_forces(np.array([[0.1, -0.2, 0.3], [0.2, 0.0, -0.1], [0.0, 0.4, 0.2], [-0.3, 0.1, 0.0]]))
        epsilon = 1.0e-6
        expected = (body._total_forces(position + epsilon * direction, gravity) - body._total_forces(position - epsilon * direction, gravity)) / (2.0 * epsilon)
        actual = body._directional_force(position, direction, gravity, epsilon)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-7)

    def test_external_force_input_shape_is_validated(self):
        mesh, rest = one_tet()
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        for invalid in (np.zeros(3), np.zeros((3, 3)), np.zeros((5, 3))):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):
                    body.set_external_forces(invalid)
        with self.assertRaises(ValueError):
            SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), external_forces=np.zeros((3, 3)), use_jax=False)

    def test_fixed_vertex_setter_validates_indices_and_updates_constraints(self):
        mesh, rest = one_tet()
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        body.set_fixed_vertices([1, 1, 3])
        np.testing.assert_array_equal(body.fixed, [False, True, False, True])
        for invalid in ([-1], [4], [[1]]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    body.set_fixed_vertices(invalid)
        body.set_fixed_vertices([])
        np.testing.assert_array_equal(body.fixed, [False, False, False, False])

    def test_step_validates_timestep_method_and_gravity(self):
        mesh, _ = one_tet()
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        with self.assertRaises(ValueError):
            body.step(0.0)
        with self.assertRaises(ValueError):
            body.step(-1.0e-3)
        with self.assertRaises(ValueError):
            body.step(1.0e-3, gravity=(0.0, 0.0))
        with self.assertRaises(ValueError):
            body.step(1.0e-3, method="unknown")

    def test_implicit_settings_are_validated(self):
        mesh, _ = one_tet()
        body = SoftBody(mesh, SVKMaterial(100.0, 0.3, 1.0), fixed=np.ones(4, dtype=bool), use_jax=False)
        with self.assertRaises(ValueError):
            body.step_implicit(1.0e-3, settings={"unknown": 1})
        with self.assertRaises(ValueError):
            body.step_implicit(1.0e-3, settings={"max_iterations": 0})
        with self.assertRaises(ValueError):
            body.step_implicit(1.0e-3, settings={"history_size": -1})
        with self.assertRaises(ValueError):
            body.step_implicit(1.0e-3, settings={"max_line_search_iterations": 0})

    def test_pressure_boundary_input_is_validated(self):
        mesh, _ = one_tet()
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        with self.assertRaises(ValueError):
            body.set_pressure_boundary(np.zeros((3, 2), dtype=np.int32), 1.0)
        with self.assertRaises(ValueError):
            body.set_pressure_boundary(np.array([[0, 1, 4]], dtype=np.int32), 1.0)
        with self.assertRaises(ValueError):
            body.set_pressure_boundary(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32), np.array([1.0]))
        with self.assertRaises(ValueError):
            body.set_pressure_boundary(np.array([[0, 0, 2]], dtype=np.int32), 1.0)
        with self.assertRaises(ValueError):
            body.set_pressure_boundary(np.array([[0, 1, 2]], dtype=np.int32), np.nan)

    def test_state_and_load_inputs_reject_nonfinite_values(self):
        mesh, rest = one_tet()
        material = SVKMaterial(1000.0, 0.3, 1.0)
        with self.assertRaises(ValueError):
            SoftBody(mesh, material, x=rest.copy() * np.nan, use_jax=False)
        with self.assertRaises(ValueError):
            SoftBody(mesh, material, v=np.full_like(rest, np.inf), use_jax=False)
        with self.assertRaises(ValueError):
            SoftBody(mesh, material, external_forces=np.full_like(rest, np.nan), use_jax=False)
        body = SoftBody(mesh, material, use_jax=False)
        with self.assertRaises(ValueError):
            body.set_external_forces(np.full_like(rest, np.inf))
        with self.assertRaises(ValueError):
            body.step(1.0e-3, gravity=(0.0, np.nan, 0.0))
        with self.assertRaises(ValueError):
            body.step_implicit(1.0e-3, gravity=(0.0, np.inf, 0.0))

    @unittest.skipUnless(JAX_AVAILABLE, "JAX is not installed")
    def test_jax_and_numpy_timestep_agree_with_external_and_body_forces(self):
        mesh, _ = one_tet()
        fixed = np.array([True, False, False, False])
        external = np.array([[0.0, 0.0, 0.0], [0.4, -0.2, 0.1], [-0.1, 0.3, 0.2], [0.2, 0.1, -0.4]])
        numpy_body = SoftBody(mesh, StableNeoHookeanMaterial(100.0, 0.3, 1.0), fixed=fixed, use_jax=False)
        jax_body = SoftBody(mesh, StableNeoHookeanMaterial(100.0, 0.3, 1.0), fixed=fixed, use_jax=True)
        numpy_body.set_external_forces(external)
        jax_body.set_external_forces(external)
        gravity = (0.0, 0.0, -9.81)
        numpy_body.step(1.0e-3, gravity=gravity)
        jax_body.step(1.0e-3, gravity=gravity)
        np.testing.assert_allclose(jax_body.x, numpy_body.x, rtol=3.0e-11, atol=3.0e-11)
        np.testing.assert_allclose(jax_body.v, numpy_body.v, rtol=3.0e-11, atol=3.0e-11)

    @unittest.skipUnless(JAX_AVAILABLE, "JAX is not installed")
    def test_jax_and_numpy_restore_initially_displaced_fixed_vertices(self):
        mesh, rest = one_tet()
        material = SVKMaterial(100.0, 0.3, 1.0)
        fixed = np.array([True, False, False, False])
        initial = rest + np.array([0.2, -0.1, 0.3])
        numpy_body = SoftBody(mesh, material, x=initial, fixed=fixed, use_jax=False)
        jax_body = SoftBody(mesh, material, x=initial, fixed=fixed, use_jax=True)
        gravity = np.zeros(3)
        numpy_body.step(1.0e-3, gravity=gravity)
        jax_body.step(1.0e-3, gravity=gravity)
        np.testing.assert_allclose(numpy_body.x[fixed], rest[fixed], atol=1.0e-14)
        np.testing.assert_allclose(jax_body.x[fixed], rest[fixed], atol=1.0e-14)
        np.testing.assert_allclose(jax_body.x, numpy_body.x, rtol=3.0e-11, atol=3.0e-11)
        np.testing.assert_allclose(jax_body.v, numpy_body.v, rtol=3.0e-11, atol=3.0e-11)

    def test_regular_tet_is_centered_and_equiangular(self):
        _, vertices = regular_tet_body(SVKMaterial(1000.0, 0.3, 1.0))
        np.testing.assert_allclose(vertices.mean(axis=0), 0.0, atol=1.0e-14)
        edge_lengths = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=2)
        np.testing.assert_allclose(edge_lengths[np.triu_indices(4, 1)], np.sqrt(8.0), atol=1.0e-14)

    def test_uniform_tet_scale_produces_central_restoring_forces(self):
        materials = (SVKMaterial(1000.0, 0.3, 1.0), StableNeoHookeanMaterial(1000.0, 0.3, 1.0))
        backends = (False, True) if JAX_AVAILABLE else (False,)
        for material in materials:
            for use_jax in backends:
                with self.subTest(material=type(material).__name__, use_jax=use_jax):
                    body, rest = regular_tet_body(material, use_jax)
                    for scale in (0.9, 1.1):
                        deformed = scale * rest
                        force = body.compute_elastic_forces(deformed)
                        # Isotropic expansion is opposed by inward force; isotropic
                        # compression is opposed by outward force.
                        radial_work = np.sum(force * (deformed - rest))
                        self.assertLess(radial_work, 0.0)
                        for position, nodal_force in zip(deformed, force):
                            self.assertLess(np.linalg.norm(nodal_force), np.inf)
                            self.assertAlmostEqual(np.linalg.norm(np.cross(position, nodal_force)), 0.0, delta=1.0e-10)
                            radial_force = np.dot(nodal_force, position)
                            if scale > 1.0:
                                self.assertLess(radial_force, 0.0)
                            else:
                                self.assertGreater(radial_force, 0.0)
                        np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1.0e-10)

    def test_octahedron_scale_has_axis_aligned_restoring_forces_and_zero_center_force(self):
        materials = (SVKMaterial(1000.0, 0.3, 1.0), StableNeoHookeanMaterial(1000.0, 0.3, 1.0))
        backends = (False, True) if JAX_AVAILABLE else (False,)
        for material in materials:
            for use_jax in backends:
                with self.subTest(material=type(material).__name__, use_jax=use_jax):
                    body, rest = octahedron_body(material, use_jax)
                    self.assertEqual(body.mesh.tet_count, 8)
                    np.testing.assert_allclose(rest[0], 0.0, atol=1.0e-14)
                    for scale in (0.9, 1.1):
                        deformed = scale * rest
                        force = body.compute_elastic_forces(deformed)
                        # The central node is shared by every tetrahedron;
                        # symmetry requires its accumulated force to vanish.
                        np.testing.assert_allclose(force[0], 0.0, atol=1.0e-10)
                        np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1.0e-10)
                        for position, nodal_force in zip(deformed[1:], force[1:]):
                            axis_force = np.dot(nodal_force, position) / np.dot(position, position) * position
                            np.testing.assert_allclose(nodal_force, axis_force, atol=1.0e-10)
                            radial_force = np.dot(nodal_force, position)
                            if scale > 1.0:
                                self.assertLess(radial_force, 0.0)
                            else:
                                self.assertGreater(radial_force, 0.0)

    def test_tet_translation_has_zero_elastic_force(self):
        translation_directions = np.eye(3)
        materials = (SVKMaterial(1000.0, 0.3, 1.0), StableNeoHookeanMaterial(1000.0, 0.3, 1.0))
        for material in materials:
            for direction in translation_directions:
                body, rest = regular_tet_body(material, use_jax=False)
                force = body.compute_elastic_forces(rest + 0.35 * direction)
                np.testing.assert_allclose(force, 0.0, atol=1.0e-10)

    def test_tet_rigid_rotation_about_each_coordinate_axis_has_zero_force(self):
        materials = (SVKMaterial(1000.0, 0.3, 1.0), StableNeoHookeanMaterial(1000.0, 0.3, 1.0))
        for material in materials:
            for axis in range(3):
                body, rest = regular_tet_body(material, use_jax=False)
                rotated = rest @ rotation_matrix(axis, 0.37).T
                np.testing.assert_allclose(body.compute_elastic_forces(rotated), 0.0, atol=1.0e-10)

    def test_unit_cube_shear_force_restores_material_coordinates(self):
        vertices, elements = create_beam_mesh(2, 2, 2, 1.0, 1.0, 1.0)
        mesh = TetMesh.from_vertices(vertices, elements, density=1.0)
        shear_displacement = np.zeros_like(vertices)
        shear_displacement[:, 0] = vertices[:, 1]
        materials = (SVKMaterial(1000.0, 0.3, 1.0), StableNeoHookeanMaterial(1000.0, 0.3, 1.0))
        backends = (False, True) if JAX_AVAILABLE else (False,)
        for material in materials:
            for use_jax in backends:
                with self.subTest(material=type(material).__name__, use_jax=use_jax):
                    body = SoftBody(mesh, material, use_jax=use_jax)
                    for shear in (-0.1, 0.1):
                        force = body.compute_elastic_forces(vertices + shear * shear_displacement)
                        restoring_work = np.sum(force * (shear * shear_displacement))
                        self.assertLess(restoring_work, 0.0)
                        np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1.0e-10)

    def test_material_parameters_reject_nonphysical_edge_cases(self):
        for material_type in (SVKMaterial, StableNeoHookeanMaterial):
            for youngs_modulus, poisson_ratio, density in (
                (0.0, 0.3, 1.0),
                (-1.0, 0.3, 1.0),
                (1.0, 0.5, 1.0),
                (1.0, -1.0, 1.0),
                (1.0, 0.3, 0.0),
                (1.0, 0.3, -1.0),
            ):
                with self.subTest(material=material_type.__name__, values=(youngs_modulus, poisson_ratio, density)):
                    with self.assertRaises(ValueError):
                        material_type(youngs_modulus, poisson_ratio, density).compute_lame_parameters()

    def test_constitutive_stress_is_zero_at_rest_for_both_models(self):
        identity = np.eye(3)[None, :, :]
        for model_code in (0, 1):
            lam, mu = SVKMaterial(1200.0, 0.3, 1.0).compute_lame_parameters()
            stress = _numpy_pk1_stress(identity, lam, mu, model_code)
            np.testing.assert_allclose(stress, 0.0, atol=1.0e-12)

    def test_both_models_have_expected_linear_elastic_limit(self):
        material = SVKMaterial(1200.0, 0.3, 1.0)
        lam, mu = material.compute_lame_parameters()
        h = np.array([[0.2, -0.1, 0.05], [0.03, -0.15, 0.08], [-0.02, 0.04, 0.1]])
        symmetric_strain = 0.5 * (h + h.T)
        expected_tangent = lam * np.trace(symmetric_strain) * np.eye(3) + 2.0 * mu * symmetric_strain
        epsilon = 1.0e-7
        for model_code in (0, 1):
            stress = _numpy_pk1_stress((np.eye(3) + epsilon * h)[None], lam, mu, model_code)[0]
            np.testing.assert_allclose(stress / epsilon, expected_tangent, rtol=2.0e-5, atol=2.0e-5)

    def test_constitutive_energy_gradient_matches_pk1_stress(self):
        rng = np.random.default_rng(7)
        for model_code in (0, 1):
            material = SVKMaterial(1200.0, 0.3, 1.0)
            lam, mu = material.compute_lame_parameters()
            # Include moderate tension, shear, compression, and an inverted
            # state; the stable model is specifically expected to stay finite
            # for the latter.
            deformation_gradients = (
                np.array([[1.2, 0.1, 0.0], [0.0, 0.8, 0.05], [0.0, 0.0, 1.1]]),
                np.array([[0.8, 0.2, 0.0], [0.0, 1.1, 0.1], [0.0, 0.0, 0.9]]),
                np.array([[1.0, 0.0, 0.0], [0.0, -0.7, 0.0], [0.0, 0.0, 1.1]]),
            )
            for f in deformation_gradients:
                direction = rng.normal(size=(3, 3))
                epsilon = 1.0e-6
                plus = _numpy_energy_density((f + epsilon * direction)[None], lam, mu, model_code)[0]
                minus = _numpy_energy_density((f - epsilon * direction)[None], lam, mu, model_code)[0]
                numerical = (plus - minus) / (2.0 * epsilon)
                analytical = np.sum(_numpy_pk1_stress(f[None], lam, mu, model_code)[0] * direction)
                self.assertTrue(np.isfinite(numerical))
                self.assertAlmostEqual(numerical, analytical, delta=2.0e-5 * max(1.0, abs(numerical)))

    def test_constitutive_objectivity_and_isotropy_invariants(self):
        rng = np.random.default_rng(11)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0.0:
            q[:, 0] *= -1.0
        right, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(right) < 0.0:
            right[:, 0] *= -1.0
        f = np.array([[1.2, 0.15, -0.05], [0.0, 0.85, 0.1], [0.0, 0.0, 1.1]])
        material = SVKMaterial(1200.0, 0.3, 1.0)
        lam, mu = material.compute_lame_parameters()
        for model_code in (0, 1):
            energy = _numpy_energy_density(f[None], lam, mu, model_code)[0]
            stress = _numpy_pk1_stress(f[None], lam, mu, model_code)[0]
            np.testing.assert_allclose(_numpy_energy_density((q @ f)[None], lam, mu, model_code)[0], energy, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(_numpy_energy_density((f @ right)[None], lam, mu, model_code)[0], energy, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(_numpy_pk1_stress((q @ f)[None], lam, mu, model_code)[0], q @ stress, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(_numpy_pk1_stress((f @ right)[None], lam, mu, model_code)[0], stress @ right, rtol=1e-11, atol=1e-11)
            # Objectivity and isotropy imply a symmetric Cauchy stress P F^T.
            cauchy_numerator = stress @ f.T
            np.testing.assert_allclose(cauchy_numerator, cauchy_numerator.T, rtol=1e-11, atol=1e-11)

    def test_constitutive_edge_states_are_finite_and_stable_model_handles_inversion(self):
        material = SVKMaterial(1200.0, 0.3, 1.0)
        lam, mu = material.compute_lame_parameters()
        states = np.array(
            [
                np.diag([1.0e-8, 1.0, 1.0]),
                np.diag([-1.0, 1.0, 1.0]),
                np.array([[1.0e3, 0.2, 0.0], [0.0, 1.0e-3, 0.0], [0.0, 0.0, 1.0]]),
            ]
        )
        for model_code in (0, 1):
            stress = _numpy_pk1_stress(states, lam, mu, model_code)
            energy = _numpy_energy_density(states, lam, mu, model_code)
            self.assertTrue(np.all(np.isfinite(stress)))
            self.assertTrue(np.all(np.isfinite(energy)))

    def test_single_tet_force_assembly_matches_shape_function_formula(self):
        mesh, rest = one_tet()
        deformation_gradient = np.array([[1.15, 0.12, -0.04], [0.03, 0.9, 0.08], [0.02, -0.05, 1.1]])
        deformed = rest @ deformation_gradient.T
        for material in (SVKMaterial(1000.0, 0.3, 1.0), StableNeoHookeanMaterial(1000.0, 0.3, 1.0)):
            body = SoftBody(mesh, material, use_jax=False)
            force = body.compute_elastic_forces(deformed)
            lam, mu = material.compute_lame_parameters()
            model_code = material.model_code
            stress = _numpy_pk1_stress(deformation_gradient[None], lam, mu, model_code)[0]
            expected = -np.einsum("ij,aj->ai", stress, mesh.volume_grad_N[0])
            np.testing.assert_allclose(force, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_affine_multi_tet_patch_test_has_zero_interior_force(self):
        vertices, elements = create_beam_mesh(3, 3, 3, 1.0, 1.0, 1.0)
        mesh = TetMesh.from_vertices(vertices, elements, density=1.0)
        deformation_gradient = np.array([[1.1, 0.08, 0.02], [0.01, 0.92, -0.04], [0.03, 0.02, 1.06]])
        deformed = vertices @ deformation_gradient.T + np.array([0.3, -0.2, 0.4])
        center_index = (1 * 3 + 1) * 3 + 1
        for material in (SVKMaterial(1000.0, 0.3, 1.0), StableNeoHookeanMaterial(1000.0, 0.3, 1.0)):
            force = SoftBody(mesh, material, use_jax=False).compute_elastic_forces(deformed)
            np.testing.assert_allclose(force[center_index], 0.0, atol=1.0e-10)

    def test_internal_forces_conserve_linear_and_angular_momentum(self):
        vertices, elements = create_beam_mesh(2, 2, 2, 1.0, 1.0, 1.0)
        mesh = TetMesh.from_vertices(vertices, elements, density=1.0)
        deformation = np.array([[1.1, 0.2, -0.1], [0.0, 0.9, 0.12], [0.05, 0.0, 1.05]])
        deformed = vertices @ deformation.T + np.array([0.7, -0.4, 0.25])
        for material in (SVKMaterial(1000.0, 0.3, 1.0), StableNeoHookeanMaterial(1000.0, 0.3, 1.0)):
            force = SoftBody(mesh, material, use_jax=False).compute_elastic_forces(deformed)
            np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1.0e-10)
            torque = np.cross(deformed, force).sum(axis=0)
            np.testing.assert_allclose(torque, 0.0, atol=1.0e-10)

    def test_tetrahedron_order_does_not_change_assembled_force(self):
        vertices, elements = create_beam_mesh(2, 2, 2, 1.0, 1.0, 1.0)
        mesh = TetMesh.from_vertices(vertices, elements, density=1.0)
        reversed_mesh = TetMesh.from_vertices(vertices, elements[::-1], density=1.0)
        deformation = np.array([[1.08, 0.1, 0.0], [0.0, 0.94, 0.05], [0.02, 0.0, 1.04]])
        deformed = vertices @ deformation.T
        for material in (SVKMaterial(1000.0, 0.3, 1.0), StableNeoHookeanMaterial(1000.0, 0.3, 1.0)):
            force = SoftBody(mesh, material, use_jax=False).compute_elastic_forces(deformed)
            reversed_force = SoftBody(reversed_mesh, material, use_jax=False).compute_elastic_forces(deformed)
            np.testing.assert_allclose(force, reversed_force, rtol=1.0e-12, atol=1.0e-12)

    def test_mesh_rejects_degenerate_and_inverted_reference_tetrahedra(self):
        degenerate = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        inverted = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        with self.assertRaises(ValueError):
            TetMesh.from_vertices(degenerate, [[0, 1, 2, 3]])
        with self.assertRaises(ValueError):
            TetMesh.from_vertices(inverted, [[0, 1, 2, 3]])

    def test_mesh_validates_shapes_indices_finiteness_and_density(self):
        vertices = np.zeros((4, 3))
        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
        with self.assertRaises(ValueError):
            TetMesh.from_vertices(np.zeros(3), elements)
        with self.assertRaises(ValueError):
            TetMesh.from_vertices(vertices, np.zeros(4, dtype=np.int32))
        with self.assertRaises(ValueError):
            TetMesh.from_vertices(vertices, [[0, 1, 2, 4]])
        nonfinite = vertices.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaises(ValueError):
            TetMesh.from_vertices(nonfinite, elements)
        valid_vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for density in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(density=density):
                with self.assertRaises(ValueError):
                    TetMesh.from_vertices(valid_vertices, elements, density=density)

    def test_boundary_faces_cancel_interior_faces_and_have_no_duplicates(self):
        vertices, elements = create_beam_mesh(3, 3, 2, 1.0, 1.0, 1.0)
        faces = compute_boundary_faces(elements)
        canonical = [tuple(sorted(face)) for face in faces]
        self.assertEqual(len(canonical), len(set(canonical)))
        all_faces = []
        for tet in elements:
            all_faces.extend(((tet[1], tet[2], tet[3]), (tet[0], tet[3], tet[2]), (tet[0], tet[1], tet[3]), (tet[0], tet[2], tet[1])))
        counts = {}
        for face in all_faces:
            key = tuple(sorted(int(index) for index in face))
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(set(canonical), {key for key, count in counts.items() if count == 1})
        self.assertTrue(all(count in (1, 2) for count in counts.values()))

    def test_boundary_faces_reject_malformed_elements(self):
        with self.assertRaises(ValueError):
            compute_boundary_faces(np.zeros((2, 3), dtype=np.int32))
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

    def test_baseline_loads_have_expected_bending_stretch_compression_and_twist_resultants(self):
        bending = create_bending_baseline(4, 3, 3)
        self.assertLess(np.sum(bending.mesh.lumped_mass[:, None] * np.asarray(bending.gravity), axis=0)[1], 0.0)
        stretch = create_stretch_baseline(4, 3, 3)
        compression = create_compress_baseline(4, 3, 3)
        stretch_resultant = stretch.create_body(False).compute_neumann_forces(stretch.mesh.x0).sum(axis=0)
        compression_resultant = compression.create_body(False).compute_neumann_forces(compression.mesh.x0).sum(axis=0)
        self.assertGreater(stretch_resultant[0], 0.0)
        self.assertLess(compression_resultant[0], 0.0)
        twist = create_twist_baseline(4, 3, 3)
        twist_force = twist.external_forces
        self.assertAlmostEqual(np.sum(twist_force[:, 1]), 0.0, delta=1.0e-14)
        self.assertAlmostEqual(np.sum(twist_force[:, 2]), 0.0, delta=1.0e-14)
        torque = np.cross(twist.mesh.x0, twist_force).sum(axis=0)
        self.assertGreater(torque[0], 0.0)

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
        np.testing.assert_allclose(body.compute_elastic_forces(x), 0.0, atol=1e-12)
        np.testing.assert_allclose(mesh.volume, 1.0 / 6.0)
        np.testing.assert_allclose(mesh.lumped_mass.sum(), 2.0 / 6.0)

    def test_stable_neo_hookean_has_zero_rest_force_and_finite_inversion_force(self):
        mesh, x = one_tet()
        body = SoftBody(mesh, StableNeoHookeanMaterial(1000.0, 0.3, 2.0), use_jax=False)
        np.testing.assert_allclose(body.compute_elastic_forces(x), 0.0, atol=1e-12)
        inverted = x.copy()
        inverted[[1, 2]] = inverted[[2, 1]]
        self.assertTrue(np.all(np.isfinite(body.compute_elastic_forces(inverted))))

    def test_material_force_is_negative_energy_gradient(self):
        mesh, x = one_tet()
        deformed = x.copy()
        deformed[1, 0] = 1.1
        deformed[2, 1] = 0.9
        direction = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3], [-0.1, 0.2, 0.1], [0.1, 0.1, -0.2]])
        epsilon = 1.0e-6
        for material in (SVKMaterial(1000.0, 0.3, 2.0), StableNeoHookeanMaterial(1000.0, 0.3, 2.0)):
            body = SoftBody(mesh, material, use_jax=False)
            numerical = (body.compute_elastic_energy(deformed + epsilon * direction) - body.compute_elastic_energy(deformed - epsilon * direction)) / (2.0 * epsilon)
            analytical = -np.sum(body.compute_elastic_forces(deformed) * direction)
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
            np.testing.assert_allclose(jax_body.compute_elastic_forces(deformed), numpy_body.compute_elastic_forces(deformed), rtol=3.0e-5, atol=3.0e-5)
            jax_body.set_pressure_boundary([[1, 2, 3]], 2.0)
            numpy_body.set_pressure_boundary([[1, 2, 3]], 2.0)
            np.testing.assert_allclose(jax_body.compute_neumann_forces(deformed), numpy_body.compute_neumann_forces(deformed), rtol=3.0e-5, atol=3.0e-5)

    @unittest.skipUnless(JAX_AVAILABLE, "JAX is not installed")
    def test_jax_and_numpy_assembled_forces_agree_on_multi_tet_mesh(self):
        vertices, elements = create_beam_mesh(3, 3, 3, 1.0, 1.0, 1.0)
        mesh = TetMesh.from_vertices(vertices, elements, density=1.0)
        deformation = np.array([[1.05, 0.12, -0.03], [0.02, 0.95, 0.07], [0.01, -0.04, 1.08]])
        deformed = vertices @ deformation.T + np.array([0.1, 0.2, -0.15])
        for material in (SVKMaterial(1000.0, 0.3, 1.0), StableNeoHookeanMaterial(1000.0, 0.3, 1.0)):
            numpy_force = SoftBody(mesh, material, use_jax=False).compute_elastic_forces(deformed)
            jax_force = SoftBody(mesh, material, use_jax=True).compute_elastic_forces(deformed)
            np.testing.assert_allclose(jax_force, numpy_force, rtol=3.0e-11, atol=3.0e-11)

    @unittest.skipUnless(JAX_AVAILABLE, "JAX is not installed")
    def test_jax_autodiff_energy_gradient_agrees_with_analytical_pk1(self):
        import jax
        import jax.numpy as jnp

        f = jnp.asarray([[1.2, 0.1, -0.05], [0.0, 0.85, 0.1], [0.0, 0.0, 1.1]])
        lam, mu = SVKMaterial(1200.0, 0.3, 1.0).compute_lame_parameters()
        for model_code in (0, 1):
            def energy(flat_f):
                deformation_gradient = flat_f.reshape((3, 3))
                c = deformation_gradient.T @ deformation_gradient
                i_c = jnp.trace(c)
                if model_code == 0:
                    strain = 0.5 * (c - jnp.eye(3, dtype=flat_f.dtype))
                    return 0.5 * lam * jnp.trace(strain) ** 2 + mu * jnp.sum(strain * strain)
                mu_hat = (4.0 / 3.0) * mu
                lam_hat = lam + (5.0 / 6.0) * mu
                alpha = 1.0 + mu_hat / lam_hat - mu_hat / (4.0 * lam_hat)
                return 0.5 * mu_hat * (i_c - 3.0) + 0.5 * lam_hat * (jnp.linalg.det(deformation_gradient) - alpha) ** 2 - 0.5 * mu_hat * jnp.log(i_c + 1.0)

            autodiff_pk1 = np.asarray(jax.grad(energy)(f.reshape(-1))).reshape(3, 3)
            analytical_pk1 = _numpy_pk1_stress(np.asarray(f)[None], lam, mu, model_code)[0]
            np.testing.assert_allclose(autodiff_pk1, analytical_pk1, rtol=1.0e-11, atol=1.0e-11)

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

    def test_implicit_bfgs_converges_with_pressure_and_external_loading(self):
        mesh, _ = one_tet()
        fixed = np.array([True, False, False, False])
        external = np.array([[0.0, 0.0, 0.0], [0.01, -0.02, 0.01], [-0.01, 0.02, 0.0], [0.0, 0.01, -0.01]])
        for use_jax in (False, True) if JAX_AVAILABLE else (False,):
            with self.subTest(use_jax=use_jax):
                body = SoftBody(mesh, StableNeoHookeanMaterial(100.0, 0.3, 1.0), fixed=fixed, use_jax=use_jax)
                body.set_pressure_boundary([[1, 2, 3]], 0.05)
                body.set_external_forces(external)
                body.step_implicit(
                    1.0e-3,
                    gravity=(0.0, 0.0, -0.2),
                    settings={"max_iterations": 30, "history_size": 5, "raise_on_failure": True},
                )
                self.assertTrue(body.last_implicit_info["converged"])
                self.assertTrue(np.all(np.isfinite(body.x)))

    def test_all_fixed_implicit_system_is_supported_by_numpy_and_jax(self):
        mesh, rest = one_tet()
        for use_jax in (False, True) if JAX_AVAILABLE else (False,):
            with self.subTest(use_jax=use_jax):
                body = SoftBody(mesh, SVKMaterial(100.0, 0.3, 1.0), fixed=np.ones(4, dtype=bool), use_jax=use_jax)
                body.step_implicit(1.0e-3, gravity=(0.0, 0.0, -9.81), settings={"raise_on_failure": True})
                np.testing.assert_allclose(body.x, rest, atol=1.0e-12)
                np.testing.assert_allclose(body.v, 0.0, atol=1.0e-12)

    def test_jax_fixed_vertex_setter_updates_device_constraints(self):
        if not JAX_AVAILABLE:
            self.skipTest("JAX is not installed")
        mesh, rest = one_tet()
        body = SoftBody(mesh, SVKMaterial(100.0, 0.3, 1.0), use_jax=True)
        body.set_fixed_vertices([0, 1])
        body.step(1.0e-3, gravity=(0.0, 0.0, -9.81))
        np.testing.assert_allclose(body.x[[0, 1]], rest[[0, 1]], atol=1.0e-12)
        np.testing.assert_allclose(body.v[[0, 1]], 0.0, atol=1.0e-12)

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
        force = body.compute_elastic_forces(stretched)
        self.assertLess(force[1, 0], 0.0)
        np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1e-10)

    def test_pressure_face_set_uses_linear_surface_fem_forces(self):
        mesh, x = one_tet()
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        # Face (1, 2, 3) is oriented outward from the reference tetrahedron.
        body.set_pressure_boundary([[1, 2, 3]], pressure=6.0)
        force = body.compute_neumann_forces(x)
        expected_total = 6.0 * 0.5 * np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(force.sum(axis=0), expected_total)
        np.testing.assert_allclose(force[[1, 2, 3]], np.tile(expected_total / 3.0, (3, 1)))
        np.testing.assert_allclose(force[0], 0.0)

    def test_pressure_forces_match_current_area_vectors_and_per_face_pressures(self):
        mesh, rest = one_tet()
        deformation = np.array([[1.2, 0.1, 0.0], [0.0, 0.8, 0.05], [0.0, 0.0, 1.1]])
        deformed = rest @ deformation.T + np.array([0.4, -0.3, 0.2])
        faces = np.array([[1, 2, 3], [0, 3, 2]], dtype=np.int32)
        pressures = np.array([4.0, -2.5])
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        body.set_pressure_boundary(faces, pressures)
        force = body.compute_neumann_forces(deformed)
        expected = np.zeros_like(deformed)
        expected_resultant = np.zeros(3)
        for face, pressure in zip(faces, pressures):
            area_vector = 0.5 * np.cross(deformed[face[1]] - deformed[face[0]], deformed[face[2]] - deformed[face[0]])
            face_resultant = pressure * area_vector
            expected_resultant += face_resultant
            expected[face] += face_resultant / 3.0
        np.testing.assert_allclose(force, expected, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(force.sum(axis=0), expected_resultant, rtol=1.0e-12, atol=1.0e-12)

    def test_pressure_forces_are_translation_invariant(self):
        mesh, rest = one_tet()
        faces = np.array([[1, 2, 3], [0, 3, 2]], dtype=np.int32)
        pressures = np.array([3.0, 1.5])
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        body.set_pressure_boundary(faces, pressures)
        translation = np.array([2.0, -1.5, 0.75])
        np.testing.assert_allclose(body.compute_neumann_forces(rest + translation), body.compute_neumann_forces(rest), atol=1.0e-12)

    def test_pressure_forces_rotate_covariantly_and_scalar_matches_per_face_values(self):
        mesh, rest = one_tet()
        faces = np.array([[1, 2, 3], [0, 3, 2]], dtype=np.int32)
        scalar_body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        array_body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        scalar_body.set_pressure_boundary(faces, 2.0)
        array_body.set_pressure_boundary(faces, np.full(len(faces), 2.0))
        q = rotation_matrix(2, 0.41)
        np.testing.assert_allclose(scalar_body.compute_neumann_forces(rest), array_body.compute_neumann_forces(rest), atol=1.0e-12)
        rotated_force = scalar_body.compute_neumann_forces(rest @ q.T)
        np.testing.assert_allclose(rotated_force, scalar_body.compute_neumann_forces(rest) @ q.T, rtol=1.0e-12, atol=1.0e-12)

    def test_uniform_pressure_on_closed_tetrahedron_has_zero_resultant(self):
        mesh, rest = one_tet()
        faces = compute_boundary_faces(mesh.elements)
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        body.set_pressure_boundary(faces, 3.0)
        np.testing.assert_allclose(body.compute_neumann_forces(rest).sum(axis=0), 0.0, atol=1.0e-12)

    def test_empty_pressure_face_set_has_zero_force(self):
        mesh, rest = one_tet()
        body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        body.set_pressure_boundary(np.empty((0, 3), dtype=np.int32), pressure=np.empty(0))
        np.testing.assert_allclose(body.compute_neumann_forces(rest), 0.0, atol=1.0e-14)

    @unittest.skipUnless(JAX_AVAILABLE, "JAX is not installed")
    def test_jax_and_numpy_pressure_forces_agree_on_multiple_faces(self):
        mesh, rest = one_tet()
        faces = np.array([[1, 2, 3], [0, 3, 2]], dtype=np.int32)
        pressures = np.array([4.0, -2.5])
        numpy_body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=False)
        jax_body = SoftBody(mesh, SVKMaterial(1000.0, 0.3, 1.0), use_jax=True)
        numpy_body.set_pressure_boundary(faces, pressures)
        jax_body.set_pressure_boundary(faces, pressures)
        deformed = rest @ np.array([[1.2, 0.1, 0.0], [0.0, 0.8, 0.05], [0.0, 0.0, 1.1]]).T
        np.testing.assert_allclose(jax_body.compute_neumann_forces(deformed), numpy_body.compute_neumann_forces(deformed), rtol=3.0e-11, atol=3.0e-11)

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
