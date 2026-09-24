"""Verify semi-implicit Euler with Stable Neo-Hookean elasticity."""

from examples.verify_soft_body_common import verify


if __name__ == "__main__":
    verify("semi_implicit", "stable_neo_hookean")
