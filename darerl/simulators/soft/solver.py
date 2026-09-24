"""A small JAX-first linear-tetrahedron soft-body solver.

The solver intentionally has no collision or contact code.  Its hot path is
the element-wise Saint Venant--Kirchhoff force kernel followed by a nodal
scatter and a semi-implicit Euler update.

JAX is optional while this prototype is being integrated into RAINBOW.  When
installed, force and time-step kernels are JIT compiled.  NumPy is used as a
portable reference implementation otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .nonlinear import solve_lbfgs

try:  # Keep importing the package possible without installing JAX.
    import jax
    import jax.numpy as jnp

    # The NumPy reference path and the implicit residual tolerances use
    # float64.  Without this, JAX silently truncates mesh data and residuals
    # to float32, which can make a converged Newton/BFGS solve appear to fail
    # on stiff meshes.
    jax.config.update("jax_enable_x64", True)
    _HAS_JAX = True
except ImportError:  # pragma: no cover - exercised in environments without JAX
    jax = None
    jnp = None
    _HAS_JAX = False


Array = np.ndarray


class Material(Protocol):
    """Constitutive model interface used by future material implementations."""

    def compute_lame_parameters(self) -> tuple[float, float]: ...

    @property
    def model_code(self) -> int: ...


@dataclass(frozen=True)
class SVKMaterial:
    """Saint Venant--Kirchhoff material parameters."""

    youngs_modulus: float
    poisson_ratio: float
    density: float

    @property
    def model_code(self) -> int:
        return 0

    def compute_lame_parameters(self) -> tuple[float, float]:
        e, nu = self.youngs_modulus, self.poisson_ratio
        if not np.isfinite(e) or e <= 0.0:
            raise ValueError("youngs_modulus must be finite and positive")
        if not (-1.0 < nu < 0.5):
            raise ValueError("poisson_ratio must be in (-1, 0.5)")
        if not np.isfinite(self.density) or self.density <= 0.0:
            raise ValueError("density must be finite and positive")
        return (nu * e / ((1.0 + nu) * (1.0 - 2.0 * nu)), e / (2.0 * (1.0 + nu)))


@dataclass(frozen=True)
class StableNeoHookeanMaterial(SVKMaterial):
    """Stable Neo-Hookean material from Smith, de Goes, and Kim (2018)."""

    @property
    def model_code(self) -> int:
        return 1


@dataclass(frozen=True)
class TetMesh:
    """Reference data for an oriented first-order tetrahedral mesh."""

    x0: Array
    elements: Array
    inv_Dm: Array
    volume: Array
    grad_N: Array
    volume_grad_N: Array
    lumped_mass: Array
    inverse_lumped_mass: Array

    @classmethod
    def from_vertices(cls, vertices: Array, elements: Array, density: float = 1.0) -> "TetMesh":
        x0 = np.asarray(vertices, dtype=np.float64)
        t = np.asarray(elements, dtype=np.int32)
        if x0.ndim != 2 or x0.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if t.ndim != 2 or t.shape[1] != 4:
            raise ValueError("elements must have shape (K, 4)")
        if not np.all(np.isfinite(x0)):
            raise ValueError("vertices must be finite")
        if np.any(t < 0) or np.any(t >= len(x0)):
            raise ValueError("elements contain an invalid vertex index")
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError("density must be finite and positive")
        p = x0[t]
        dm = np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
        det = np.linalg.det(dm)
        if np.any(det <= 0.0):
            bad = int(np.flatnonzero(det <= 0.0)[0])
            raise ValueError(f"tetrahedron {bad} is inverted or degenerate in reference space")
        inv_dm = np.linalg.inv(dm)
        volume = det / 6.0
        # grad_N[e, a] is grad of shape function a in reference coordinates.
        grad_N = np.empty((len(t), 4, 3), dtype=np.float64)
        # For F = D Dm^{-1}, grad(N_a) is row a of Dm^{-1}.
        grad_N[:, 1:, :] = inv_dm
        grad_N[:, 0, :] = -np.sum(grad_N[:, 1:, :], axis=1)
        mass = np.zeros(len(x0), dtype=np.float64)
        np.add.at(mass, t.reshape(-1), np.repeat(density * volume / 4.0, 4))
        if np.any(mass <= 0.0):
            raise ValueError("tetrahedral mesh contains a node with zero lumped mass")
        return cls(x0, t, inv_dm, volume, grad_N, volume[:, None, None] * grad_N, mass, 1.0 / mass)

    @property
    def node_count(self) -> int:
        return len(self.x0)

    @property
    def tet_count(self) -> int:
        return len(self.elements)


@dataclass
class SoftBody:
    """Mutable state and operations for one tetrahedral soft body."""

    mesh: TetMesh
    material: Material
    x: Array | None = None
    v: Array | None = None
    fixed: Array | None = None
    use_jax: bool = True
    pressure_faces: Array | None = None
    pressure: float | Array = 0.0
    external_forces: Array | None = None

    def __post_init__(self) -> None:
        self.x = np.array(self.mesh.x0 if self.x is None else self.x, dtype=np.float64, copy=True)
        self.v = np.zeros_like(self.x) if self.v is None else np.array(self.v, dtype=np.float64, copy=True)
        self.fixed = np.zeros(self.mesh.node_count, dtype=bool) if self.fixed is None else np.asarray(self.fixed, dtype=bool)
        if self.x.shape != (self.mesh.node_count, 3) or self.v.shape != self.x.shape:
            raise ValueError("x and v must have shape (N, 3)")
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.v)):
            raise ValueError("x and v must be finite")
        if self.fixed.shape != (self.mesh.node_count,):
            raise ValueError("fixed must have shape (N,)")
        self.external_forces = np.zeros_like(self.x) if self.external_forces is None else np.asarray(self.external_forces, dtype=np.float64)
        if self.external_forces.shape != self.x.shape:
            raise ValueError("external_forces must have shape (N, 3)")
        if not np.all(np.isfinite(self.external_forces)):
            raise ValueError("external_forces must be finite")
        self.pressure_faces = np.empty((0, 3), dtype=np.int32) if self.pressure_faces is None else np.asarray(self.pressure_faces, dtype=np.int32)
        self.pressure = np.asarray(self.pressure, dtype=np.float64)
        self.validate_pressure_boundary()
        self._jax_enabled = bool(self.use_jax and _HAS_JAX)
        self._jax_x = None
        self._jax_v = None
        self._static = (
            tuple(map(jnp.asarray, (self.mesh.elements, self.mesh.inv_Dm, self.mesh.volume_grad_N, self.mesh.inverse_lumped_mass, self.fixed)))
            if self._jax_enabled
            else None
        )
        self._pressure_static = (jnp.asarray(self.pressure_faces), jnp.asarray(self.pressure)) if self._jax_enabled else None
        self._external_forces_device = jnp.asarray(self.external_forces) if self._jax_enabled else None
        self._jax_x0 = jnp.asarray(self.mesh.x0) if self._jax_enabled else None

    def validate_pressure_boundary(self) -> None:
        """Validate the configured pressure-face set and pressure values."""
        if self.pressure_faces.ndim != 2 or self.pressure_faces.shape[1] != 3:
            raise ValueError("pressure_faces must have shape (M, 3)")
        if np.any(self.pressure_faces < 0) or np.any(self.pressure_faces >= self.mesh.node_count):
            raise ValueError("pressure_faces contains an invalid vertex index")
        if np.any(
            (self.pressure_faces[:, 0] == self.pressure_faces[:, 1])
            | (self.pressure_faces[:, 0] == self.pressure_faces[:, 2])
            | (self.pressure_faces[:, 1] == self.pressure_faces[:, 2])
        ):
            raise ValueError("pressure_faces must contain three distinct vertices per face")
        if self.pressure.ndim == 0:
            if not np.isfinite(self.pressure):
                raise ValueError("pressure must be finite")
            return
        if self.pressure.shape != (len(self.pressure_faces),):
            raise ValueError("pressure must be a scalar or have one value per pressure face")
        if not np.all(np.isfinite(self.pressure)):
            raise ValueError("pressure must be finite")

    def set_fixed_vertices(self, vertices: Array) -> None:
        """Set Dirichlet conditions from a sequence of fixed vertex indices."""
        mask = np.zeros(self.mesh.node_count, dtype=bool)
        indices = np.asarray(vertices, dtype=np.int32)
        if indices.ndim != 1:
            raise ValueError("fixed vertex list must have shape (K,)")
        if np.any(indices < 0) or np.any(indices >= self.mesh.node_count):
            raise ValueError("fixed vertex list contains an invalid index")
        mask[indices] = True
        self.fixed = mask
        if self._jax_enabled:
            self._static = self._static[:4] + (jnp.asarray(self.fixed),)

    def set_pressure_boundary(self, faces: Array, pressure: float | Array) -> None:
        """Set a von Neumann pressure condition on oriented triangular faces.

        Faces must be ordered so their cross product points in the desired
        normal direction. Positive pressure acts in that direction. For a
        conventional inward pressure, pass a negative value. A scalar applies
        uniformly; an array supplies one pressure per face.
        """
        faces_array = np.asarray(faces, dtype=np.int32)
        pressure_array = np.asarray(pressure, dtype=np.float64)
        old_faces, old_pressure = self.pressure_faces, self.pressure
        self.pressure_faces, self.pressure = faces_array, pressure_array
        try:
            self.validate_pressure_boundary()
        except ValueError:
            self.pressure_faces, self.pressure = old_faces, old_pressure
            raise
        if self._jax_enabled:
            self._pressure_static = (jnp.asarray(self.pressure_faces), jnp.asarray(self.pressure))

    def set_external_forces(self, forces: Array) -> None:
        """Set persistent nodal von Neumann loads in world coordinates."""
        forces = np.asarray(forces, dtype=np.float64)
        if forces.shape != self.x.shape:
            raise ValueError("external_forces must have shape (N, 3)")
        if not np.all(np.isfinite(forces)):
            raise ValueError("external_forces must be finite")
        self.external_forces = forces.copy()
        if self._jax_enabled:
            self._external_forces_device = jnp.asarray(self.external_forces)

    def clear_external_forces(self) -> None:
        """Remove all persistent nodal loads."""
        self.set_external_forces(np.zeros_like(self.x))

    def step(
        self,
        dt: float,
        gravity: Array = (0.0, -9.81, 0.0),
        sync: bool = True,
        method: str = "semi_implicit",
        settings: dict | None = None,
    ) -> tuple[Array, Array]:
        """Advance one step with the selected time-integration method.

        ``method`` is ``"semi_implicit"`` by default or ``"implicit_bfgs"``
        for fully implicit backward Euler solved with matrix-free L-BFGS.
        ``settings`` is passed to the implicit solver; see
        :meth:`step_implicit` for supported keys.
        """
        if method == "implicit_bfgs":
            return self.step_implicit(dt, gravity, settings)
        if method != "semi_implicit":
            raise ValueError("method must be 'semi_implicit' or 'implicit_bfgs'")
        return self._step_semi_implicit(dt, gravity, sync)

    def _step_semi_implicit(self, dt: float, gravity: Array, sync: bool) -> tuple[Array, Array]:
        """Advance with semi-implicit Euler."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        gravity = np.asarray(gravity, dtype=np.float64)
        if gravity.shape != (3,):
            raise ValueError("gravity must have shape (3,)")
        if not np.all(np.isfinite(gravity)):
            raise ValueError("gravity must be finite")
        if self._jax_enabled:
            if self._jax_x is None:
                self._jax_x = jnp.asarray(self.x)
                self._jax_v = jnp.asarray(self.v)
            self._jax_x, self._jax_v = _jax_step(
                self._jax_x,
                self._jax_v,
                *self._static,
                self._jax_x0,
                *self._pressure_static,
                self._external_forces_device,
                jnp.asarray(gravity),
                *self.material.compute_lame_parameters(),
                self.material.model_code,
                dt,
            )
            if sync:
                self.synchronize()
            else:
                self.x, self.v = self._jax_x, self._jax_v
        else:
            f = self._noninertial_forces(np.asarray(self.x))
            active = ~self.fixed
            self.v[active] += dt * (f[active] * self.mesh.inverse_lumped_mass[active, None] + gravity)
            self.x[active] += dt * self.v[active]
            self.v[self.fixed] = 0.0
            self.x[self.fixed] = self.mesh.x0[self.fixed]
        return self.x, self.v

    def step_implicit(
        self,
        dt: float,
        gravity: Array = (0.0, -9.81, 0.0),
        settings: dict | None = None,
    ) -> tuple[Array, Array]:
        """Advance with fully implicit backward Euler and matrix-free L-BFGS.

        The nonlinear solve is performed on free vertex positions. The
        implicit residual is the gradient of the backward-Euler objective,

        ``g(x) = M/dt² * (x - x_n - dt*v_n) - f(x)``.

        Directional derivatives ``J_g(x) @ s`` reuse the force computation.
        JAX uses ``jax.jvp``; the NumPy fallback uses a configurable forward
        directional difference. Supported settings are:

        ``max_iterations`` (25), ``tolerance`` (1e-6), ``history_size`` (10),
        ``line_search`` (True), ``max_line_search_iterations`` (12),
        ``line_search_reduction`` (0.5), ``line_search_c1`` (1e-4),
        ``curvature_tolerance`` (1e-10), ``directional_epsilon`` (1e-6), and
        ``raise_on_failure`` (False).
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        gravity = np.asarray(gravity, dtype=np.float64)
        if gravity.shape != (3,):
            raise ValueError("gravity must have shape (3,)")
        if not np.all(np.isfinite(gravity)):
            raise ValueError("gravity must be finite")
        cfg = {
            "max_iterations": 25,
            "tolerance": 1.0e-6,
            "history_size": 10,
            "line_search": True,
            "max_line_search_iterations": 12,
            "line_search_reduction": 0.5,
            "line_search_c1": 1.0e-4,
            "curvature_tolerance": 1.0e-10,
            "directional_epsilon": 1.0e-6,
            "raise_on_failure": False,
        }
        if settings is not None:
            unknown = set(settings) - set(cfg)
            if unknown:
                raise ValueError(f"unknown implicit solver settings: {sorted(unknown)}")
            cfg.update(settings)
        if cfg["max_iterations"] < 1 or cfg["history_size"] < 0:
            raise ValueError("max_iterations must be positive and history_size cannot be negative")
        if cfg["max_line_search_iterations"] < 1:
            raise ValueError("max_line_search_iterations must be positive")
        if not (0.0 < cfg["line_search_reduction"] < 1.0):
            raise ValueError("line_search_reduction must be in (0, 1)")
        if not (0.0 <= cfg["line_search_c1"] < 1.0):
            raise ValueError("line_search_c1 must be in [0, 1)")
        if cfg["tolerance"] <= 0.0 or cfg["directional_epsilon"] <= 0.0 or cfg["curvature_tolerance"] < 0.0:
            raise ValueError("solver tolerances and directional_epsilon must be valid positive values")

        x_n = np.asarray(self.x, dtype=np.float64).copy()
        v_n = np.asarray(self.v, dtype=np.float64).copy()
        free = np.flatnonzero(~self.fixed)
        dofs = np.arange(self.mesh.node_count * 3).reshape((-1, 3))[free].reshape(-1)

        if self._jax_enabled:
            x_device, v_device, info = _jax_implicit_step(
                jnp.asarray(x_n),
                jnp.asarray(v_n),
                jnp.asarray(self.mesh.x0),
                *self._static,
                jnp.asarray(self.mesh.lumped_mass),
                *self._pressure_static,
                self._external_forces_device,
                jnp.asarray(gravity),
                *self.material.compute_lame_parameters(),
                self.material.model_code,
                dt,
                jnp.asarray(dofs, dtype=jnp.int32),
                jnp.asarray(np.repeat(self.mesh.lumped_mass[free], 3) / (dt * dt)),
                jnp.asarray(np.repeat(self.mesh.inverse_lumped_mass[free], 3)),
                cfg["tolerance"],
                int(cfg["max_iterations"]),
                int(cfg["history_size"]),
                bool(cfg["line_search"]),
                int(cfg["max_line_search_iterations"]),
                cfg["line_search_reduction"],
                cfg["line_search_c1"],
                cfg["curvature_tolerance"],
            )
            x_device.block_until_ready()
            self._jax_x, self._jax_v = x_device, v_device
            self.x, self.v = np.asarray(x_device), np.asarray(v_device)
            self.last_implicit_info = {
                "converged": bool(info[0]),
                "iterations": int(info[1]),
                "final_residual_norm": float(info[2]),
                "initial_residual_norm": float(info[3]),
                "line_search_steps": int(info[4]),
                "history_length": int(info[5]),
            }
            if not self.last_implicit_info["converged"] and cfg["raise_on_failure"]:
                raise RuntimeError(f"implicit BFGS solve did not converge: {self.last_implicit_info}")
            return self.x, self.v

        x = x_n + dt * v_n
        x[self.fixed] = self.mesh.x0[self.fixed]
        mass_dof = np.repeat(self.mesh.lumped_mass[free], 3)
        inv_mass_dof = np.repeat(self.mesh.inverse_lumped_mass[free], 3)
        scale = mass_dof / (dt * dt)

        def residual(position: Array) -> Array:
            position_array = np.asarray(position).reshape(self.x.shape)
            force = self._total_forces(position_array, gravity)
            result = np.zeros(position.size, dtype=np.float64)
            result[dofs] = scale * (
                position_array.reshape(-1)[dofs] - x_n.reshape(-1)[dofs] - dt * v_n.reshape(-1)[dofs]
            ) - force[free].reshape(-1)
            return result

        def directional_residual(position: Array, direction: Array) -> Array:
            position_array = np.asarray(position).reshape(self.x.shape)
            direction_array = np.asarray(direction).reshape(self.x.shape)
            df = self._directional_force(position_array, direction_array, gravity, cfg["directional_epsilon"])
            result = np.zeros(position.size, dtype=np.float64)
            result[dofs] = scale * direction_array.reshape(-1)[dofs] - df[free].reshape(-1)
            return result

        x = x.reshape(-1)
        diagonal = np.zeros(x.size, dtype=np.float64)
        diagonal[dofs] = dt * dt * inv_mass_dof
        x, _, self.last_implicit_info = solve_lbfgs(
            x,
            residual,
            directional_residual,
            diagonal,
            cfg,
        )
        if not self.last_implicit_info["converged"] and cfg["raise_on_failure"]:
            raise RuntimeError(f"implicit BFGS solve did not converge: {self.last_implicit_info}")
        self.x = x.reshape(x_n.shape)
        self.v = (self.x - x_n) / dt
        self.x[self.fixed] = self.mesh.x0[self.fixed]
        self.v[self.fixed] = 0.0
        if self._jax_enabled:
            self._jax_x = jnp.asarray(self.x)
            self._jax_v = jnp.asarray(self.v)
        return self.x, self.v

    def _total_forces(self, x: Array, gravity: Array) -> Array:
        forces = self._noninertial_forces(x)
        forces += self.compute_body_forces(gravity)
        return forces

    def compute_body_forces(self, gravity: Array = (0.0, -9.81, 0.0)) -> Array:
        """Return lumped nodal forces from a uniform body acceleration.

        ``gravity`` is an acceleration in m/s².  The corresponding volumetric
        body-force density is ``material.density * gravity`` in N/m³, and the
        nodal force is that density integrated over each barycentric dual
        control volume.
        """
        acceleration = np.asarray(gravity, dtype=np.float64)
        if acceleration.shape != (3,):
            raise ValueError("gravity must have shape (3,)")
        if not np.all(np.isfinite(acceleration)):
            raise ValueError("gravity must be finite")
        return self.mesh.lumped_mass[:, None] * acceleration

    def _noninertial_forces(self, x: Array) -> Array:
        forces = _numpy_forces(x, self.mesh, *self.material.compute_lame_parameters(), self.material.model_code)
        forces += _numpy_pressure_forces(x, self.pressure_faces, self.pressure, self.mesh.node_count)
        forces += self.external_forces
        return forces

    def _directional_force(self, x: Array, direction: Array, gravity: Array, epsilon: float) -> Array:
        if self._jax_enabled:
            def force_kernel(position):
                elastic = _jax_forces(position, *self._static, *self.material.compute_lame_parameters(), self.material.model_code)
                pressure = _jax_pressure_forces(position, *self._pressure_static)
                return elastic + pressure + self._external_forces_device + jnp.asarray(self.mesh.lumped_mass)[:, None] * jnp.asarray(gravity)

            _, derivative = jax.jvp(force_kernel, (jnp.asarray(x),), (jnp.asarray(direction),))
            return np.asarray(derivative)
        length = max(float(np.linalg.norm(direction)), 1.0)
        h = epsilon * max(1.0, float(np.linalg.norm(x))) / length
        return (self._total_forces(x + h * direction, gravity) - self._total_forces(x, gravity)) / h

    def compute_deformation_gradient(self, x: Array | None = None) -> Array:
        x = self.x if x is None else np.asarray(x, dtype=np.float64)
        if x.shape != (self.mesh.node_count, 3):
            raise ValueError("x must have shape (node_count, 3)")
        p = x[self.mesh.elements]
        d = np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
        return d @ self.mesh.inv_Dm

    def compute_green_lagrange_strain(self, x: Array | None = None) -> Array:
        """Return the element Green--Lagrange strain tensors.

        The returned array has shape ``(tet_count, 3, 3)`` and is computed as
        ``E = 0.5 * (F.T @ F - I)`` for each element.
        """
        deformation_gradient = self.compute_deformation_gradient(x)
        identity = np.eye(3, dtype=deformation_gradient.dtype)
        right_cauchy_green = np.einsum("...ji,...jk->...ik", deformation_gradient, deformation_gradient)
        return 0.5 * (right_cauchy_green - identity)

    def compute_elastic_forces(self, x: Array | None = None) -> Array:
        """Return internal elastic forces, with shape ``(N, 3)``."""
        x = self.x if x is None else x
        lam, mu = self.material.compute_lame_parameters()
        if self._jax_enabled:
            x_device = self._jax_x if x is self.x and self._jax_x is not None else jnp.asarray(x)
            return np.asarray(_jax_forces(x_device, *self._static, lam, mu, self.material.model_code))
        return _numpy_forces(np.asarray(x), self.mesh, lam, mu, self.material.model_code)

    def compute_neumann_forces(self, x: Array | None = None) -> Array:
        """Return pressure nodal forces from the configured face set."""
        x = self.x if x is None else x
        if self._jax_enabled:
            x_device = self._jax_x if x is self.x and self._jax_x is not None else jnp.asarray(x)
            return np.asarray(_jax_pressure_forces(x_device, *self._pressure_static))
        return _numpy_pressure_forces(np.asarray(x), self.pressure_faces, self.pressure, self.mesh.node_count)

    def compute_elastic_energy(self, x: Array | None = None) -> float:
        x = self.x if x is None else x
        f = self.compute_deformation_gradient(x)
        lam, mu = self.material.compute_lame_parameters()
        density = _numpy_energy_density(f, lam, mu, self.material.model_code)
        return float(np.sum(self.mesh.volume * density))

    def synchronize(self) -> tuple[Array, Array]:
        """Synchronize device state and expose it as NumPy arrays."""
        if self._jax_enabled and self._jax_x is not None:
            self._jax_x.block_until_ready()
            self.x, self.v = np.asarray(self._jax_x), np.asarray(self._jax_v)
        return self.x, self.v


def _numpy_forces(x: Array, mesh: TetMesh, lam: float, mu: float, model_code: int = 0) -> Array:
    p = x[mesh.elements]
    d = np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
    f = d @ mesh.inv_Dm
    p1 = _numpy_pk1_stress(f, lam, mu, model_code)
    local = -np.einsum("eij,eaj->eai", p1, mesh.volume_grad_N)
    indices = mesh.elements.reshape(-1)
    out = np.empty_like(x)
    for axis in range(3):
        out[:, axis] = np.bincount(indices, weights=local[:, :, axis].reshape(-1), minlength=len(x))
    return out


def _numpy_pk1_stress(f: Array, lam: float, mu: float, model_code: int) -> Array:
    if model_code == 0:
        c = np.einsum("...ji,...jk->...ik", f, f)
        strain = 0.5 * (c - np.eye(3))
        s = lam * np.trace(strain, axis1=1, axis2=2)[:, None, None] * np.eye(3) + 2.0 * mu * strain
        return f @ s
    if model_code == 1:
        mu_hat = (4.0 / 3.0) * mu
        lam_hat = lam + (5.0 / 6.0) * mu
        alpha = 1.0 + mu_hat / lam_hat - mu_hat / (4.0 * lam_hat)
        i_c = np.sum(f * f, axis=(1, 2))
        d_j = np.stack(
            (
                np.cross(f[:, 1], f[:, 2]),
                np.cross(f[:, 2], f[:, 0]),
                np.cross(f[:, 0], f[:, 1]),
            ),
            axis=1,
        )
        j = np.linalg.det(f)
        return mu_hat * (1.0 - 1.0 / (i_c + 1.0))[:, None, None] * f + lam_hat * (j - alpha)[:, None, None] * d_j
    raise ValueError(f"unknown material model code: {model_code}")


def _numpy_energy_density(f: Array, lam: float, mu: float, model_code: int) -> Array:
    c = np.einsum("...ji,...jk->...ik", f, f)
    i_c = np.trace(c, axis1=1, axis2=2)
    if model_code == 0:
        strain = 0.5 * (c - np.eye(3))
        return 0.5 * lam * np.trace(strain, axis1=1, axis2=2) ** 2 + mu * np.sum(strain * strain, axis=(1, 2))
    if model_code == 1:
        mu_hat = (4.0 / 3.0) * mu
        lam_hat = lam + (5.0 / 6.0) * mu
        alpha = 1.0 + mu_hat / lam_hat - mu_hat / (4.0 * lam_hat)
        j = np.linalg.det(f)
        return 0.5 * mu_hat * (i_c - 3.0) + 0.5 * lam_hat * (j - alpha) ** 2 - 0.5 * mu_hat * np.log(i_c + 1.0)
    raise ValueError(f"unknown material model code: {model_code}")


def _numpy_pressure_forces(x: Array, faces: Array, pressure: Array, node_count: int) -> Array:
    out = np.zeros((node_count, 3), dtype=x.dtype)
    if len(faces) == 0:
        return out
    face_x = x[faces]
    area_vectors = 0.5 * np.cross(face_x[:, 1] - face_x[:, 0], face_x[:, 2] - face_x[:, 0])
    values = np.broadcast_to(pressure, (len(faces),))
    local = values[:, None] * area_vectors / 3.0
    indices = faces.reshape(-1)
    nodal = np.repeat(local, 3, axis=0)
    for axis in range(3):
        out[:, axis] = np.bincount(indices, weights=nodal[:, axis], minlength=node_count)
    return out


if _HAS_JAX:
    @jax.jit(static_argnums=(6,))
    def _jax_element_forces(x, elements, inv_dm, volume_grad_n, lam, mu, model_code):
        p = x[elements]
        d = jnp.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
        f = d @ inv_dm
        if model_code == 0:
            c = jnp.einsum("...ji,...jk->...ik", f, f)
            strain = 0.5 * (c - jnp.eye(3))
            s = lam * jnp.trace(strain, axis1=1, axis2=2)[:, None, None] * jnp.eye(3) + 2.0 * mu * strain
            p1 = f @ s
        elif model_code == 1:
            mu_hat = (4.0 / 3.0) * mu
            lam_hat = lam + (5.0 / 6.0) * mu
            alpha = 1.0 + mu_hat / lam_hat - mu_hat / (4.0 * lam_hat)
            i_c = jnp.sum(f * f, axis=(1, 2))
            d_j = jnp.stack(
                (
                    jnp.cross(f[:, 1], f[:, 2]),
                    jnp.cross(f[:, 2], f[:, 0]),
                    jnp.cross(f[:, 0], f[:, 1]),
                ),
                axis=1,
            )
            j = jnp.linalg.det(f)
            p1 = mu_hat * (1.0 - 1.0 / (i_c + 1.0))[:, None, None] * f + lam_hat * (j - alpha)[:, None, None] * d_j
        else:
            raise ValueError("unknown material model code")
        return -jnp.einsum("eij,eaj->eai", p1, volume_grad_n), elements

    @jax.jit(static_argnums=(8,))
    def _jax_forces(x, elements, inv_dm, volume_grad_n, inverse_mass, fixed, lam, mu, model_code):
        local, elements = _jax_element_forces(x, elements, inv_dm, volume_grad_n, lam, mu, model_code)
        return jax.ops.segment_sum(local.reshape((-1, 3)), elements.reshape(-1), x.shape[0])

    @jax.jit
    def _jax_pressure_forces(x, faces, pressure):
        face_x = x[faces]
        area_vectors = 0.5 * jnp.cross(face_x[:, 1] - face_x[:, 0], face_x[:, 2] - face_x[:, 0])
        values = jnp.broadcast_to(pressure, (faces.shape[0],))
        local = values[:, None] * area_vectors / 3.0
        return jax.ops.segment_sum(jnp.repeat(local, 3, axis=0), faces.reshape(-1), x.shape[0])

    @jax.jit(static_argnums=(14,))
    def _jax_step(x, v, elements, inv_dm, volume_grad_n, inverse_mass, fixed, x0, pressure_faces, pressure, external_forces, gravity, lam, mu, model_code, dt):
        local, elements = _jax_element_forces(x, elements, inv_dm, volume_grad_n, lam, mu, model_code)
        force = jax.ops.segment_sum(local.reshape((-1, 3)), elements.reshape(-1), x.shape[0])
        force += _jax_pressure_forces(x, pressure_faces, pressure)
        force += external_forces
        active = ~fixed
        v_new = jnp.where(active[:, None], v + dt * (force * inverse_mass[:, None] + gravity), 0.0)
        x_new = jnp.where(active[:, None], x + dt * v_new, x0)
        return x_new, v_new

    @jax.jit(static_argnums=(15, 21, 22, 23, 24))
    def _jax_implicit_step(
        x_n,
        v_n,
        x0,
        elements,
        inv_dm,
        volume_grad_n,
        inverse_mass,
        fixed,
        lumped_mass,
        pressure_faces,
        pressure,
        external_forces,
        gravity,
        lam,
        mu,
        model_code,
        dt,
        dofs,
        scale,
        inv_mass_dof,
        tolerance,
        max_iterations,
        history_size,
        line_search,
        max_line_search_iterations,
        line_search_reduction,
        line_search_c1,
        curvature_tolerance,
    ):
        """Fully device-resident JAX L-BFGS backward-Euler step.

        The iteration and line-search bounds are static so the whole solve is
        compiled once per solver configuration.  The history is stored in
        fixed-size device arrays, which avoids Python lists and host/device
        transfers in the nonlinear solve.
        """
        free_count = dofs.shape[0]

        def force(position):
            elastic = _jax_forces(position, elements, inv_dm, volume_grad_n, inverse_mass, fixed, lam, mu, model_code)
            pressure_force = _jax_pressure_forces(position, pressure_faces, pressure)
            return elastic + pressure_force + external_forces + lumped_mass[:, None] * gravity

        def residual(position):
            total = force(position)
            displacement = position.reshape(-1)[dofs] - x_n.reshape(-1)[dofs] - dt * v_n.reshape(-1)[dofs]
            return scale * displacement - total.reshape(-1)[dofs]

        x_initial = x_n + dt * v_n
        x_initial = jnp.where(fixed[:, None], x0, x_initial)
        g_initial = residual(x_initial)
        initial_norm = jnp.maximum(jnp.linalg.norm(g_initial), 1.0)
        history_capacity = max(history_size, 1)
        history_s = jnp.zeros((history_capacity, free_count), dtype=x_n.dtype)
        history_y = jnp.zeros((history_capacity, free_count), dtype=x_n.dtype)
        history_rho = jnp.zeros((history_capacity,), dtype=x_n.dtype)

        def compute_lbfgs_direction(g, hist_s, hist_y, hist_rho, count):
            direction = -g
            alpha_values = jnp.zeros((history_capacity,), dtype=g.dtype)

            def first_loop(index, carry):
                current, alphas = carry
                valid = index < count
                history_index = jnp.maximum(count - 1 - index, 0)
                s = hist_s[history_index]
                y = hist_y[history_index]
                rho = hist_rho[history_index]
                alpha = jnp.where(valid, rho * jnp.dot(s, current), 0.0)
                current = jnp.where(valid, current - alpha * y, current)
                alphas = alphas.at[index].set(alpha)
                return current, alphas

            direction, alpha_values = jax.lax.fori_loop(
                0, history_capacity, first_loop, (direction, alpha_values)
            )
            direction = direction * (dt * dt) * inv_mass_dof

            def second_loop(index, current):
                valid = index < count
                history_index = index
                s = hist_s[history_index]
                y = hist_y[history_index]
                rho = hist_rho[history_index]
                alpha_index = jnp.maximum(count - 1 - index, 0)
                beta = rho * jnp.dot(y, current)
                updated = current + s * (alpha_values[alpha_index] - beta)
                return jnp.where(valid, updated, current)

            return jax.lax.fori_loop(0, history_capacity, second_loop, direction)

        def append_history(hist_s, hist_y, hist_rho, count, s, y, curvature):
            valid = curvature > curvature_tolerance * jnp.linalg.norm(s) * jnp.maximum(jnp.linalg.norm(y), 1.0e-30)
            rho = jnp.where(valid, 1.0 / curvature, 0.0)

            def append_at_end(values):
                old_s, old_y, old_rho = values
                return (
                    old_s.at[count].set(s),
                    old_y.at[count].set(y),
                    old_rho.at[count].set(rho),
                )

            def append_after_shift(values):
                old_s, old_y, old_rho = values
                return (
                    jnp.concatenate((old_s[1:], s[None, :]), axis=0),
                    jnp.concatenate((old_y[1:], y[None, :]), axis=0),
                    jnp.concatenate((old_rho[1:], jnp.asarray([rho])), axis=0),
                )

            def append_valid(values):
                return jax.lax.cond(count < history_size, append_at_end, append_after_shift, values)

            new_values = jax.lax.cond(valid & (history_size > 0), append_valid, lambda values: values, (hist_s, hist_y, hist_rho))
            new_count = jnp.where(valid & (history_size > 0), jnp.minimum(count + 1, history_size), count)
            return new_values[0], new_values[1], new_values[2], new_count

        def iteration_body(iteration, state):
            x, g, hist_s, hist_y, hist_rho, count, done, converged, iterations, line_steps = state

            def no_op(current_state):
                return current_state

            def solve(current_state):
                x, g, hist_s, hist_y, hist_rho, count, done, converged, iterations, line_steps = current_state
                norm_g = jnp.linalg.norm(g)
                already_converged = norm_g <= tolerance * initial_norm

                def converged_state():
                    return x, g, hist_s, hist_y, hist_rho, count, True, True, iteration + 1, line_steps

                def search_state():
                    direction = compute_lbfgs_direction(g, hist_s, hist_y, hist_rho, count)
                    direction = jnp.where(
                        (jnp.dot(direction, g) < 0.0) & jnp.all(jnp.isfinite(direction)),
                        direction,
                        -(dt * dt) * inv_mass_dof * g,
                    )
                    phi = 0.5 * jnp.dot(g, g)
                    slope = jnp.dot(g, direction)

                    def line_search_loop(search_iteration, search_state):
                        trial_x, trial_g, step_length, accepted, attempts = search_state
                        trial_flat = x.reshape(-1).at[dofs].add(step_length * direction)
                        candidate_x = trial_flat.reshape(x.shape)
                        candidate_x = jnp.where(fixed[:, None], x0, candidate_x)
                        candidate_g = residual(candidate_x)
                        sufficient_decrease = 0.5 * jnp.dot(candidate_g, candidate_g) <= phi + line_search_c1 * step_length * slope
                        accept_now = (~accepted) & ((not line_search) | sufficient_decrease)
                        trial_x = jnp.where(accept_now, candidate_x, trial_x)
                        trial_g = jnp.where(accept_now, candidate_g, trial_g)
                        step_length = jnp.where((~accepted) & (~sufficient_decrease) & line_search, step_length * line_search_reduction, step_length)
                        attempts = attempts + (~accepted).astype(attempts.dtype)
                        return trial_x, trial_g, step_length, accepted | accept_now, attempts

                    trial_x = x
                    trial_g = g
                    trial_x, trial_g, step_length, accepted, search_steps = jax.lax.fori_loop(
                        0,
                        max_line_search_iterations,
                        line_search_loop,
                        (trial_x, trial_g, jnp.asarray(1.0, dtype=x.dtype), jnp.asarray(False), jnp.asarray(0)),
                    )
                    s = step_length * direction
                    full_direction = jnp.zeros_like(x).reshape(-1).at[dofs].set(s).reshape(x.shape)
                    _, df = jax.jvp(force, (trial_x,), (full_direction,))
                    y = scale * s - df.reshape(-1)[dofs]
                    curvature = jnp.where(accepted, jnp.dot(s, y), 0.0)
                    hist_s_new, hist_y_new, hist_rho_new, count_new = append_history(
                        hist_s, hist_y, hist_rho, count, s, y, curvature
                    )
                    return (
                        jnp.where(accepted, trial_x, x),
                        jnp.where(accepted, trial_g, g),
                        hist_s_new,
                        hist_y_new,
                        hist_rho_new,
                        count_new,
                        ~accepted,
                        jnp.asarray(False),
                        iteration + 1,
                        line_steps + search_steps,
                    )

                return jax.lax.cond(already_converged, converged_state, search_state)

            return jax.lax.cond(done, no_op, solve, state)

        state = (x_initial, g_initial, history_s, history_y, history_rho, jnp.asarray(0), jnp.asarray(False), jnp.asarray(False), jnp.asarray(0), jnp.asarray(0))
        x, g, _, _, _, history_length, done, converged, iterations, line_steps = jax.lax.fori_loop(
            0, max_iterations, iteration_body, state
        )
        velocity = (x - x_n) / dt
        velocity = jnp.where(fixed[:, None], 0.0, velocity)
        return x, velocity, (converged, iterations, jnp.linalg.norm(g), initial_norm, line_steps, history_length)
