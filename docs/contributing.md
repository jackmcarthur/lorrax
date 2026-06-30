# Contributing

!!! note "TODO"
    A full contributor guide (humanized from `AGENTS.md`) is planned. The current coding
    standards live in `AGENTS.md` in the repository root.

## Coding standards (summary)

- Use NumPy-style docstrings. Document shapes, units, and shardings for array parameters.
- Match existing formatting; do not reformat unrelated lines.
- Every function implementing a physics equation should reference what it computes.

### JAX sharding rules

- Never hard-code mesh shapes; refer to mesh axes by name (`'x'`, `'y'`).
- Use `NamedSharding` / `PartitionSpec` for all layouts; let XLA handle communication
  (no `np.concatenate` or host-side gathers).
- LORRAX is memory-constrained: avoid operations that rematerialize large arrays on a
  subset of processors.

### Before committing

- Run `uv run python -m pytest -q` after longer branches.
- Do not commit `__pycache__/`, `.venv/`, or cache directories.

See `AGENTS.md` for the full module map and the per-subsystem read order.
