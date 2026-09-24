"""Verify implicit BFGS stepping with Saint Venant--Kirchhoff elasticity."""

from examples.verify_soft_body_common import verify


if __name__ == "__main__":
    verify("implicit_bfgs", "svk")
