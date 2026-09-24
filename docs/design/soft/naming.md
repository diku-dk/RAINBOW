# Naming convention

Soft-simulator methods and functions use an action-oriented `snake_case` name.
The first word should describe the operation:

- `make_xxx` for lightweight example or configuration assembly;
- `create_xxx` for constructing a persistent object or data structure;
- `compute_xxx` for numerical calculations and derived quantities;
- `get_xxx` for retrieving a value through an operation;
- `set_xxx`, `add_xxx`, `clear_xxx`, and `remove_xxx` for state mutation;
- `validate_xxx` for explicit validation routines;
- `test_xxx` and `verify_xxx` for tests and executable verification examples;
- `write_xxx` and `read_xxx` for serialization or report I/O.

Names use `baseline`, never `base_line` or `base-line`. Example filenames use
the same `snake_case` convention, except for the explicitly named
`autotune-soft.py` command-line example.

Standard Python protocol names, properties, constructors such as
`from_vertices`, lifecycle hooks such as `__post_init__`, and private numerical
kernels prefixed with `_` are deliberate exceptions. New public APIs should
follow the operation prefixes above.

Current public numerical queries include `compute_deformation_gradient`,
`compute_green_lagrange_strain`, `compute_elastic_forces`,
`compute_neumann_forces`, `compute_body_forces`, and `compute_elastic_energy`.
