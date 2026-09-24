"""Verify implicit BFGS stepping with Stable Neo-Hookean elasticity."""

from examples.verify_soft_body_common import verify


if __name__ == "__main__":
    verify("implicit_bfgs", "stable_neo_hookean")
