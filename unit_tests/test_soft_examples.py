"""Small deterministic tests for soft-body example helpers."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from examples import profile_soft_body_realtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_autotune_module():
    path = PROJECT_ROOT / "examples" / "autotune-soft-on-bending-beam.py"
    spec = importlib.util.spec_from_file_location("autotune_soft_on_bending_beam", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSoftExamples(unittest.TestCase):
    def test_scalability_mesh_sizes_are_monotone_and_in_range(self):
        sizes = profile_soft_body_realtime.compute_mesh_sizes(10_000, 100_000, 5, 8, 8)
        self.assertEqual(len(sizes), 5)
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreaterEqual(5 * (8 - 1) * (8 - 1) * (sizes[0] - 1), 0)
        self.assertGreaterEqual(sizes[-1], sizes[0])

    def test_realtime_error_helper_handles_finite_and_blowup_trajectories(self):
        reference = np.zeros((3, 2, 3))
        candidate = reference.copy()
        self.assertEqual(profile_soft_body_realtime.compute_error_against_reference(candidate, reference), 0.0)
        candidate[1, 0, 0] = np.inf
        self.assertEqual(profile_soft_body_realtime.compute_error_against_reference(candidate, reference), np.inf)

    def test_autotune_trajectory_error_compares_matching_time_samples(self):
        autotune = load_autotune_module()
        reference = np.zeros((11, 1, 3))
        candidate = np.zeros((2, 1, 3))
        candidate[1, 0, 0] = 2.0
        error = autotune.compute_trajectory_error(candidate, reference, 0.1, 0.5)
        self.assertEqual(error, 2.0)

if __name__ == "__main__":
    unittest.main()
