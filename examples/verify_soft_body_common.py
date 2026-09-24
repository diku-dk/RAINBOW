"""Shared setup for the four soft-body verification examples."""

from __future__ import annotations

import numpy as np

from darerl.simulators.soft import (
    BENDING_GRAVITY,
    SoftBody,
    create_bending_baseline,
)


def make_body(material_name: str, use_jax: bool = True) -> tuple[SoftBody, np.ndarray]:
    baseline = create_bending_baseline(6, 3, 3)
    return baseline.create_body(use_jax=use_jax, material_model=material_name), baseline.fixed


def verify(method: str, material_name: str, use_jax: bool = True) -> None:
    body, fixed = make_body(material_name, use_jax)
    initial = body.x.copy()
    dt = 1.0e-4
    steps = 200
    implicit_settings = {
        "max_iterations": 50,
        "tolerance": 1.0e-6,
        "history_size": 10,
        "line_search": True,
        "raise_on_failure": True,
    }

    for _ in range(steps):
        if method == "implicit_bfgs":
            body.step_implicit(dt, gravity=BENDING_GRAVITY, settings=implicit_settings)
        else:
            body.step(dt, gravity=BENDING_GRAVITY, sync=False)
            body.synchronize()

    if not np.all(np.isfinite(body.x)) or not np.all(np.isfinite(body.v)):
        raise AssertionError("verification produced non-finite state")
    np.testing.assert_allclose(body.x[fixed], initial[fixed], atol=1.0e-12)
    np.testing.assert_allclose(body.v[fixed], 0.0, atol=1.0e-12)

    displacement = np.linalg.norm(body.x - initial, axis=1).max()
    print(f"material={material_name} method={method} nodes={body.mesh.node_count} tets={body.mesh.tet_count}")
    print(f"steps={steps} dt={dt:g} max_displacement={displacement:.6e}")
    print(f"elastic_energy={body.elastic_energy():.6e}")
    if method == "implicit_bfgs":
        print(f"implicit_info={body.last_implicit_info}")


if __name__ == "__main__":
    raise SystemExit("Run one of the material/method-specific verification examples.")
