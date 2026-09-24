"""Verify implicit BFGS stepping with Stable Neo-Hookean elasticity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.verify_soft_body_common import verify


if __name__ == "__main__":
    verify("implicit_bfgs", "stable_neo_hookean")
