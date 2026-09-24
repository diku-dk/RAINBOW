"""Unit tests for reusable L-BFGS and line-search solver routines."""

import unittest

import numpy as np

from darerl.simulators.soft.nonlinear import (
    compute_backtracking_line_search,
    compute_lbfgs_direction,
    solve_lbfgs,
    update_lbfgs_history,
)


def solver_settings(**overrides):
    settings = {
        "max_iterations": 30,
        "tolerance": 1.0e-10,
        "history_size": 5,
        "line_search": True,
        "max_line_search_iterations": 12,
        "line_search_reduction": 0.5,
        "line_search_c1": 1.0e-4,
        "curvature_tolerance": 1.0e-10,
    }
    settings.update(overrides)
    return settings


class TestSoftNonlinear(unittest.TestCase):
    def test_lbfgs_direction_without_history_is_preconditioned_negative_gradient(self):
        gradient = np.array([2.0, -3.0, 4.0])
        diagonal = np.array([0.5, 2.0, 0.25])
        direction = compute_lbfgs_direction(gradient, [], [], [], diagonal)
        np.testing.assert_allclose(direction, -diagonal * gradient)

    def test_lbfgs_history_update_accepts_positive_curvature_and_limits_size(self):
        history_s, history_y, history_rho = [], [], []
        accepted = update_lbfgs_history(
            history_s, history_y, history_rho,
            np.array([1.0, 0.0]), np.array([2.0, 0.0]),
            history_size=1, curvature_tolerance=1.0e-10,
        )
        self.assertTrue(accepted)
        accepted = update_lbfgs_history(
            history_s, history_y, history_rho,
            np.array([0.0, 1.0]), np.array([0.0, 3.0]),
            history_size=1, curvature_tolerance=1.0e-10,
        )
        self.assertTrue(accepted)
        self.assertEqual(len(history_s), 1)
        np.testing.assert_allclose(history_s[0], [0.0, 1.0])
        np.testing.assert_allclose(history_y[0], [0.0, 3.0])
        self.assertAlmostEqual(history_rho[0], 1.0 / 3.0)

    def test_lbfgs_history_update_rejects_nonpositive_or_tiny_curvature(self):
        for y in (np.array([-1.0, 0.0]), np.array([0.0, 0.0])):
            history_s, history_y, history_rho = [], [], []
            accepted = update_lbfgs_history(
                history_s, history_y, history_rho,
                np.array([1.0, 0.0]), y,
                history_size=5, curvature_tolerance=1.0e-10,
            )
            self.assertFalse(accepted)
            self.assertEqual(history_s, [])

    def test_lbfgs_direction_with_history_remains_a_descent_direction(self):
        gradient = np.array([1.0, -2.0])
        history_s, history_y, history_rho = [], [], []
        update_lbfgs_history(
            history_s, history_y, history_rho,
            np.array([1.0, 0.0]), np.array([2.0, 0.0]),
            history_size=5, curvature_tolerance=1.0e-10,
        )
        direction = compute_lbfgs_direction(gradient, history_s, history_y, history_rho, np.ones(2))
        self.assertLess(np.dot(direction, gradient), 0.0)
        self.assertTrue(np.all(np.isfinite(direction)))

    def test_armijo_line_search_accepts_full_step_when_it_decreases_objective(self):
        residual = lambda x: x - 1.0
        result = compute_backtracking_line_search(
            np.array([0.0]), np.array([-1.0]), np.array([1.0]), residual,
            enabled=True, max_iterations=5, reduction=0.5, c1=1.0e-4,
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["iterations"], 1)
        np.testing.assert_allclose(result["position"], [1.0])

    def test_armijo_line_search_backtracks_and_reports_failure(self):
        residual = lambda x: x - 1.0
        result = compute_backtracking_line_search(
            np.array([0.0]), np.array([-1.0]), np.array([10.0]), residual,
            enabled=True, max_iterations=8, reduction=0.5, c1=1.0e-4,
        )
        self.assertTrue(result["accepted"])
        self.assertLess(result["step_length"], 1.0)
        constant_residual = lambda x: np.ones_like(x)
        failed = compute_backtracking_line_search(
            np.array([0.0]), np.array([1.0]), np.array([-1.0]), constant_residual,
            enabled=True, max_iterations=3, reduction=0.5, c1=1.0e-4,
        )
        self.assertFalse(failed["accepted"])
        self.assertEqual(failed["iterations"], 3)

    def test_lbfgs_solver_converges_on_quadratic_residual(self):
        matrix = np.diag([2.0, 5.0, 9.0])
        target = np.array([1.0, -2.0, 0.5])
        residual = lambda x: matrix @ x - target
        directional = lambda x, direction: matrix @ direction
        solution, residual_at_solution, info = solve_lbfgs(
            np.zeros(3), residual, directional, np.ones(3), solver_settings()
        )
        np.testing.assert_allclose(solution, np.linalg.solve(matrix, target), atol=1.0e-8)
        self.assertTrue(info["converged"])
        self.assertLess(np.linalg.norm(residual_at_solution), 1.0e-8)
        self.assertGreaterEqual(info["iterations"], 1)
        self.assertGreater(info["history_length"], 0)

    def test_lbfgs_solver_reports_nonconvergence_at_iteration_limit(self):
        residual = lambda x: np.array([x[0] - 1.0])
        directional = lambda x, direction: np.array([direction[0]])
        _, _, info = solve_lbfgs(
            np.array([0.0]), residual, directional, np.ones(1),
            solver_settings(max_iterations=1, tolerance=1.0e-14),
        )
        self.assertFalse(info["converged"])
        self.assertEqual(info["iterations"], 1)


if __name__ == "__main__":
    unittest.main()
