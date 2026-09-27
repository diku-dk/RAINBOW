"""Time integration, constraints, and backend agreement tests."""

import unittest

import numpy as np

from darerl.simulators.soft import create_bending_baseline

from unit_tests.test_soft_support import JAX_AVAILABLE, _SoftBodyTests


class TestSoftTimeStepping(unittest.TestCase):
    test_fixed_vertices_are_preserved_by_time_step = _SoftBodyTests.test_fixed_vertices_are_preserved_by_time_step
    test_semi_implicit_step_moves_under_gravity = _SoftBodyTests.test_semi_implicit_step_moves_under_gravity
    test_implicit_bfgs_accepts_settings_and_preserves_rest_state = _SoftBodyTests.test_implicit_bfgs_accepts_settings_and_preserves_rest_state
    test_implicit_bfgs_converges_under_gravity_for_both_materials = _SoftBodyTests.test_implicit_bfgs_converges_under_gravity_for_both_materials
    test_jax_semi_implicit_step_matches_numpy = _SoftBodyTests.test_jax_semi_implicit_step_matches_numpy
    test_jax_implicit_bfgs_matches_numpy_native_solver = _SoftBodyTests.test_jax_implicit_bfgs_matches_numpy_native_solver
    test_jax_and_numpy_timestep_agree_with_external_and_body_forces = _SoftBodyTests.test_jax_and_numpy_timestep_agree_with_external_and_body_forces
    test_jax_and_numpy_restore_initially_displaced_fixed_vertices = _SoftBodyTests.test_jax_and_numpy_restore_initially_displaced_fixed_vertices
    test_step_validates_timestep_method_and_gravity = _SoftBodyTests.test_step_validates_timestep_method_and_gravity
    test_implicit_settings_are_validated = _SoftBodyTests.test_implicit_settings_are_validated
    test_all_directional_residual_strategies_converge = _SoftBodyTests.test_all_directional_residual_strategies_converge
    test_fixed_vertex_setter_validates_indices_and_updates_constraints = _SoftBodyTests.test_fixed_vertex_setter_validates_indices_and_updates_constraints
    test_implicit_bfgs_converges_with_pressure_and_external_loading = _SoftBodyTests.test_implicit_bfgs_converges_with_pressure_and_external_loading
    test_implicit_failure_keeps_finite_state_and_honors_raise_on_failure = _SoftBodyTests.test_implicit_failure_keeps_finite_state_and_honors_raise_on_failure
    test_all_fixed_implicit_system_is_supported_by_numpy_and_jax = _SoftBodyTests.test_all_fixed_implicit_system_is_supported_by_numpy_and_jax
    test_jax_fixed_vertex_setter_updates_device_constraints = _SoftBodyTests.test_jax_fixed_vertex_setter_updates_device_constraints
    test_jax_state_setter_invalidates_device_cache = _SoftBodyTests.test_jax_state_setter_invalidates_device_cache
    test_state_getters_return_copies_and_setters_synchronize_state = _SoftBodyTests.test_state_getters_return_copies_and_setters_synchronize_state
    test_jax_unsynchronized_step_keeps_solver_on_device_until_getter_or_sync = _SoftBodyTests.test_jax_unsynchronized_step_keeps_solver_on_device_until_getter_or_sync
    test_jax_implicit_reduced_line_search_step_remains_finite = _SoftBodyTests.test_jax_implicit_reduced_line_search_step_remains_finite
    test_dirichlet_positions_and_velocities_are_enforced_by_both_steppers = _SoftBodyTests.test_dirichlet_positions_and_velocities_are_enforced_by_both_steppers

    def test_new_implicit_methods_preserve_rest_state(self):
        baseline = create_bending_baseline(3, 2, 2)
        for method in ("implicit_midpoint", "trapezoidal", "newmark"):
            body = baseline.create_body(use_jax=False)
            rest = body.get_x()
            body.step(
                1.0e-4,
                gravity=np.zeros(3),
                method=method,
                settings={"max_iterations": 10, "absolute_tolerance": 1.0e-10, "relative_tolerance": 1.0e-10},
            )
            np.testing.assert_allclose(body.get_x(), rest, rtol=0.0, atol=1.0e-12)
            np.testing.assert_allclose(body.get_v(), 0.0, rtol=0.0, atol=1.0e-12)

    @unittest.skipUnless(JAX_AVAILABLE, "JAX is not installed")
    def test_new_implicit_methods_agree_between_numpy_and_jax(self):
        baseline = create_bending_baseline(3, 2, 2)
        settings = {
            "max_iterations": 20,
            "history_size": 4,
            "absolute_tolerance": 1.0e-8,
            "relative_tolerance": 1.0e-6,
            "line_search": True,
            "max_line_search_iterations": 12,
        }
        for method in ("implicit_midpoint", "trapezoidal", "newmark"):
            numpy_body = baseline.create_body(use_jax=False)
            jax_body = baseline.create_body(use_jax=True)
            numpy_body.step(1.0e-4, gravity=baseline.gravity, method=method, settings=settings)
            jax_body.step(1.0e-4, gravity=baseline.gravity, method=method, settings=settings)
            np.testing.assert_allclose(jax_body.get_x(), numpy_body.get_x(), rtol=2.0e-6, atol=2.0e-9)
            np.testing.assert_allclose(jax_body.get_v(), numpy_body.get_v(), rtol=2.0e-6, atol=2.0e-8)
