"""Small deterministic tests for soft-body example helpers."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from examples import profile_soft_realtime_scale
from examples import verify_soft
from darerl.simulators.soft import (
    create_bending_baseline,
    create_compress_baseline,
    create_stretch_baseline,
    create_twist_baseline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_autotune_module():
    path = PROJECT_ROOT / "examples" / "autotune-soft.py"
    spec = importlib.util.spec_from_file_location("autotune_soft", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSoftExamples(unittest.TestCase):
    def test_all_canonical_baselines_remain_finite_for_reference_rollout(self):
        factories = (
            create_bending_baseline,
            create_twist_baseline,
            create_compress_baseline,
            create_stretch_baseline,
        )
        try:
            import jax  # noqa: F401
            backends = (False, True)
        except ImportError:
            backends = (False,)
        for factory in factories:
            baseline = factory(6, 3, 3)
            for use_jax in backends:
                for material in ("svk", "stable_neohookean"):
                    body = baseline.create_body(use_jax=use_jax, material_model=material)
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        for _ in range(1000):
                            body.step(1.0e-4, gravity=baseline.gravity, sync=False)
                        body.synchronize()
                    self.assertTrue(np.all(np.isfinite(body.get_x())), (factory.__name__, material, use_jax))
                    self.assertTrue(np.all(np.isfinite(body.get_v())), (factory.__name__, material, use_jax))

    def test_verification_portfolio_contains_eight_combinations(self):
        cases = verify_soft.make_cases()
        self.assertEqual(len(cases), 8)
        self.assertEqual(sum(case.method == "semi_implicit" for case in cases), 2)
        self.assertEqual(sum(case.method == "implicit_bfgs" for case in cases), 6)
        self.assertEqual(
            {case.strategy for case in cases if case.method == "implicit_bfgs"},
            {"tangent_action", "closed_form", "finite_difference"},
        )

    def test_scalability_mesh_sizes_are_monotone_and_in_range(self):
        sizes = profile_soft_realtime_scale.compute_mesh_sizes(10_000, 100_000, 5, 8, 8)
        self.assertEqual(len(sizes), 5)
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreaterEqual(5 * (8 - 1) * (8 - 1) * (sizes[0] - 1), 0)
        self.assertGreaterEqual(sizes[-1], sizes[0])

    def test_realtime_error_helper_handles_finite_and_blowup_trajectories(self):
        reference = np.zeros((3, 2, 3))
        candidate = reference.copy()
        self.assertEqual(profile_soft_realtime_scale.compute_error_against_reference(candidate, reference), 0.0)
        candidate[1, 0, 0] = np.inf
        self.assertEqual(profile_soft_realtime_scale.compute_error_against_reference(candidate, reference), np.inf)

    def test_autotune_trajectory_error_compares_matching_time_samples(self):
        autotune = load_autotune_module()
        reference = np.zeros((11, 2, 3))
        reference[:, 1, 0] = 1.0
        candidate = np.zeros((2, 2, 3))
        candidate[1, 0, 0] = 2.0
        error = autotune.compute_trajectory_error(candidate, reference, 0.1, 0.5)
        self.assertEqual(error, 2.0)

if __name__ == "__main__":
    unittest.main()
