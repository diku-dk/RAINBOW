# Canonical soft-body baselines

`darerl.simulators.soft.baseline` owns the reference scenes used by the
examples and profiling studies. This prevents a change to one example's
geometry, material, or loading from silently invalidating comparisons with
the other examples.

The public factories are:

- `create_bending_baseline()` — fixed-root cantilever under amplified gravity;
- `create_stretch_baseline()` — fixed-root cantilever with positive normal
  pressure on the free end;
- `create_compress_baseline()` — fixed-root cantilever with negative normal
  pressure on the free end;
- `create_twist_baseline()` — fixed-root cantilever with a distributed tip
  torque.

Each function returns a `SoftBaseline`. Call `baseline.create_body()` to
obtain a fresh NumPy or JAX `SoftBody`; the same object also exposes the mesh,
fixed vertices, pressure faces, external forces, and gravity used by plotting
and energy-reporting code.

The default geometry is 0.10 m long with a 0.02 m square section. The
material is a soft, nearly incompressible skin approximation (`E=100 kPa`,
`nu=0.49`, density `1100 kg/m^3`). Skin properties vary substantially, so
these values are intended for repeatable examples rather than biological
calibration. Loads are intentionally amplified to make nonlinear motion
visible in short runs.
