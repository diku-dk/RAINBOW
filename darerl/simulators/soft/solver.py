"""A small JAX-first linear-tetrahedron soft-body solver.

The solver intentionally has no collision or contact code.  Its hot path is
the element-wise Saint Venant--Kirchhoff force kernel followed by a nodal
scatter and a semi-implicit Euler update.

JAX is optional while this prototype is being integrated into RAINBOW.  When
installed, force and time-step kernels are JIT compiled.  NumPy is used as a
portable reference implementation otherwise.
"""

from __future__ import annotations

import numpy as np

from .material import Material, StableNeoHookeanMaterial, SVKMaterial
from .mesh import TetMesh
from .time_stepper import step_implicit, step_semi_implicit
from .types import Array
from .forces import (
    compute_directional_forces,
    compute_elastic_forces as compute_numpy_elastic_forces,
    compute_energy_density,
    compute_pressure_forces,
    compute_pk1_stress,
)

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


from dataclasses import dataclass


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
            return step_implicit(self, dt, np.asarray(gravity, dtype=np.float64), settings=settings)
        if method != "semi_implicit":
            raise ValueError("method must be 'semi_implicit' or 'implicit_bfgs'")
        return step_semi_implicit(self, dt, np.asarray(gravity, dtype=np.float64), sync=sync)

    def step_implicit(
        self,
        dt: float,
        gravity: Array = (0.0, -9.81, 0.0),
        settings: dict | None = None,
    ) -> tuple[Array, Array]:
        """Advance with the implicit time-stepper and matrix-free L-BFGS."""
        return step_implicit(self, dt, gravity, settings)

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
        forces = compute_numpy_elastic_forces(x, self.mesh, *self.material.compute_lame_parameters(), self.material.model_code)
        forces += compute_pressure_forces(x, self.pressure_faces, self.pressure, self.mesh.node_count)
        forces += self.external_forces
        return forces

    def _directional_force(
        self,
        x: Array,
        direction: Array,
        gravity: Array,
        epsilon: float,
        strategy: str = "tangent_action",
    ) -> Array:
        if strategy == "finite_difference":
            length = max(float(np.linalg.norm(direction)), 1.0)
            h = epsilon * max(1.0, float(np.linalg.norm(x))) / length
            return (self._total_forces(x + h * direction, gravity) - self._total_forces(x, gravity)) / h
        if strategy == "tangent_action" and self._jax_enabled:
            def force_kernel(position):
                elastic = _jax_forces(position, *self._static, *self.material.compute_lame_parameters(), self.material.model_code)
                pressure = _jax_pressure_forces(position, *self._pressure_static)
                return elastic + pressure + self._external_forces_device + jnp.asarray(self.mesh.lumped_mass)[:, None] * jnp.asarray(gravity)

            _, derivative = jax.jvp(force_kernel, (jnp.asarray(x),), (jnp.asarray(direction),))
            return np.asarray(derivative)
        # NumPy has no built-in forward-mode AD. Its tangent-action strategy
        # therefore uses the same analytical local tangent as ``closed_form``.
        return compute_directional_forces(
            x,
            direction,
            self.mesh,
            *self.material.compute_lame_parameters(),
            self.material.model_code,
            self.pressure_faces,
            self.pressure,
        )

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
        return compute_numpy_elastic_forces(np.asarray(x), self.mesh, lam, mu, self.material.model_code)

    def compute_neumann_forces(self, x: Array | None = None) -> Array:
        """Return pressure nodal forces from the configured face set."""
        x = self.x if x is None else x
        if self._jax_enabled:
            x_device = self._jax_x if x is self.x and self._jax_x is not None else jnp.asarray(x)
            return np.asarray(_jax_pressure_forces(x_device, *self._pressure_static))
        return compute_pressure_forces(np.asarray(x), self.pressure_faces, self.pressure, self.mesh.node_count)

    def compute_elastic_energy(self, x: Array | None = None) -> float:
        x = self.x if x is None else x
        f = self.compute_deformation_gradient(x)
        lam, mu = self.material.compute_lame_parameters()
        density = compute_energy_density(f, lam, mu, self.material.model_code)
        return float(np.sum(self.mesh.volume * density))

    def synchronize(self) -> tuple[Array, Array]:
        """Synchronize device state and expose it as NumPy arrays."""
        if self._jax_enabled and self._jax_x is not None:
            self._jax_x.block_until_ready()
            self.x, self.v = np.asarray(self._jax_x), np.asarray(self._jax_v)
        return self.x, self.v


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
    def _jax_element_jacobians(x, elements, inv_dm):
        p = x[elements]
        d = jnp.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
        return jnp.linalg.det(d @ inv_dm)

    @jax.jit
    def _jax_pressure_forces(x, faces, pressure):
        face_x = x[faces]
        area_vectors = 0.5 * jnp.cross(face_x[:, 1] - face_x[:, 0], face_x[:, 2] - face_x[:, 0])
        values = jnp.broadcast_to(pressure, (faces.shape[0],))
        local = values[:, None] * area_vectors / 3.0
        return jax.ops.segment_sum(jnp.repeat(local, 3, axis=0), faces.reshape(-1), x.shape[0])

    @jax.jit(static_argnums=(9,))
    def _jax_directional_forces(x, direction, elements, inv_dm, volume_grad_n, pressure_faces, pressure, lam, mu, model_code):
        p = x[elements]
        dp = direction[elements]
        d = jnp.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
        dd = jnp.stack((dp[:, 1] - dp[:, 0], dp[:, 2] - dp[:, 0], dp[:, 3] - dp[:, 0]), axis=2)
        f = d @ inv_dm
        df = dd @ inv_dm

        if model_code == 0:
            c = jnp.einsum("...ji,...jk->...ik", f, f)
            dc = jnp.einsum("...ji,...jk->...ik", df, f) + jnp.einsum("...ji,...jk->...ik", f, df)
            dstrain = 0.5 * dc
            strain = 0.5 * (c - jnp.eye(3))
            identity = jnp.eye(3)
            stress = lam * jnp.trace(strain, axis1=1, axis2=2)[:, None, None] * identity + 2.0 * mu * strain
            dstress = lam * jnp.trace(dstrain, axis1=1, axis2=2)[:, None, None] * identity + 2.0 * mu * dstrain
            dp1 = jnp.einsum("eij,ejk->eik", df, stress) + jnp.einsum("eij,ejk->eik", f, dstress)
        elif model_code == 1:
            mu_hat = (4.0 / 3.0) * mu
            lam_hat = lam + (5.0 / 6.0) * mu
            alpha = 1.0 + mu_hat / lam_hat - mu_hat / (4.0 * lam_hat)
            i_c = jnp.sum(f * f, axis=(1, 2))
            d_i_c = 2.0 * jnp.sum(f * df, axis=(1, 2))
            cof = jnp.stack(
                (jnp.cross(f[:, 1], f[:, 2]), jnp.cross(f[:, 2], f[:, 0]), jnp.cross(f[:, 0], f[:, 1])),
                axis=1,
            )
            dcof = jnp.stack(
                (
                    jnp.cross(df[:, 1], f[:, 2]) + jnp.cross(f[:, 1], df[:, 2]),
                    jnp.cross(df[:, 2], f[:, 0]) + jnp.cross(f[:, 2], df[:, 0]),
                    jnp.cross(df[:, 0], f[:, 1]) + jnp.cross(f[:, 0], df[:, 1]),
                ),
                axis=1,
            )
            j = jnp.linalg.det(f)
            d_j = jnp.sum(cof * df, axis=(1, 2))
            d_a = mu_hat * d_i_c / (i_c + 1.0) ** 2
            a = mu_hat * (1.0 - 1.0 / (i_c + 1.0))
            dp1 = d_a[:, None, None] * f + a[:, None, None] * df + lam_hat * (
                d_j[:, None, None] * cof + (j - alpha)[:, None, None] * dcof
            )
        else:
            raise ValueError("unknown material model code")

        local = -jnp.einsum("eij,eaj->eai", dp1, volume_grad_n)
        elastic = jax.ops.segment_sum(local.reshape((-1, 3)), elements.reshape(-1), x.shape[0])
        face_x = x[pressure_faces]
        face_dx = direction[pressure_faces]
        edge_1 = face_x[:, 1] - face_x[:, 0]
        edge_2 = face_x[:, 2] - face_x[:, 0]
        d_edge_1 = face_dx[:, 1] - face_dx[:, 0]
        d_edge_2 = face_dx[:, 2] - face_dx[:, 0]
        d_area_vectors = 0.5 * (jnp.cross(d_edge_1, edge_2) + jnp.cross(edge_1, d_edge_2))
        values = jnp.broadcast_to(pressure, (pressure_faces.shape[0],))
        local_pressure = values[:, None] * d_area_vectors / 3.0
        pressure_derivative = jax.ops.segment_sum(
            jnp.repeat(local_pressure, 3, axis=0), pressure_faces.reshape(-1), x.shape[0]
        )
        return elastic + pressure_derivative

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

    @jax.jit(static_argnums=(15, 21, 22, 23, 24, 29, 31))
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
        directional_epsilon,
        directional_strategy,
        minimum_jacobian,
        prevent_inversion,
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

        def directional_force_action(position, direction):
            if directional_strategy == 0:
                _, derivative = jax.jvp(force, (position,), (direction,))
                return derivative
            if directional_strategy == 1:
                return _jax_directional_forces(
                    position,
                    direction,
                    elements,
                    inv_dm,
                    volume_grad_n,
                    pressure_faces,
                    pressure,
                    lam,
                    mu,
                    model_code,
                )
            length = jnp.maximum(jnp.linalg.norm(direction), 1.0)
            h = directional_epsilon * jnp.maximum(1.0, jnp.linalg.norm(position)) / length
            return (force(position + h * direction) - force(position)) / h

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
                        trial_x, trial_g, step_length, accepted, attempts, best_x, best_g, best_phi, best_step = search_state
                        trial_flat = x.reshape(-1).at[dofs].add(step_length * direction)
                        candidate_x = trial_flat.reshape(x.shape)
                        candidate_x = jnp.where(fixed[:, None], x0, candidate_x)

                        if prevent_inversion:
                            candidate_jacobians = _jax_element_jacobians(candidate_x, elements, inv_dm)
                            feasible = jnp.all(candidate_jacobians > minimum_jacobian)
                        else:
                            feasible = jnp.asarray(True)

                        # Avoid evaluating elastic forces for an infeasible trial.
                        candidate_g = jax.lax.cond(
                            feasible,
                            lambda _: residual(candidate_x),
                            lambda _: g,
                            operand=None,
                        )
                        armijo = 0.5 * jnp.dot(candidate_g, candidate_g) <= phi + line_search_c1 * step_length * slope
                        sufficient_decrease = feasible & armijo
                        candidate_phi = 0.5 * jnp.dot(candidate_g, candidate_g)
                        finite_candidate = jnp.isfinite(candidate_phi)
                        accept_now = (~accepted) & feasible & finite_candidate & ((not line_search) | armijo)
                        best_now = feasible & jnp.isfinite(candidate_phi) & (candidate_phi < best_phi)
                        trial_x = jnp.where(accept_now, candidate_x, trial_x)
                        trial_g = jnp.where(accept_now, candidate_g, trial_g)
                        best_x = jnp.where(best_now, candidate_x, best_x)
                        best_g = jnp.where(best_now, candidate_g, best_g)
                        best_phi = jnp.where(best_now, candidate_phi, best_phi)
                        best_step = jnp.where(best_now, step_length, best_step)
                        step_length = jnp.where((~accepted) & (~sufficient_decrease) & line_search, step_length * line_search_reduction, step_length)
                        attempts = attempts + (~accepted).astype(attempts.dtype)
                        return trial_x, trial_g, step_length, accepted | accept_now, attempts, best_x, best_g, best_phi, best_step

                    trial_x = x
                    trial_g = g
                    best_x = x
                    best_g = g
                    best_phi = phi
                    best_step = jnp.asarray(1.0, dtype=x.dtype)
                    trial_x, trial_g, step_length, accepted, search_steps, best_x, best_g, best_phi, best_step = jax.lax.fori_loop(
                        0,
                        max_line_search_iterations,
                        line_search_loop,
                        (trial_x, trial_g, jnp.asarray(1.0, dtype=x.dtype), jnp.asarray(False), jnp.asarray(0), best_x, best_g, best_phi, best_step),
                    )
                    rescue = (~accepted) & (best_phi < phi)
                    trial_x = jnp.where(rescue, best_x, trial_x)
                    trial_g = jnp.where(rescue, best_g, trial_g)
                    step_length = jnp.where(rescue, best_step, step_length)
                    accepted = accepted | rescue
                    s = step_length * direction
                    full_direction = jnp.zeros_like(x).reshape(-1).at[dofs].set(s).reshape(x.shape)
                    df = directional_force_action(trial_x, full_direction)
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
        final_norm = jnp.linalg.norm(g)
        reduction_factor = final_norm / initial_norm
        return x, velocity, (converged, iterations, final_norm, initial_norm, line_steps, history_length, reduction_factor)
