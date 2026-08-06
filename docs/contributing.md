# Contributing

The full module map, the per-subsystem read order and the coding standards
live in **`AGENTS.md`** in the repository root — that file is the authority
and this page does not restate it. What follows is the short version plus the
two rules that are easiest to break without noticing.

## Before you write

Read the [register on the front page](index.md#register) and find the page
that owns the thing you are changing. If your change makes a documented fact
false, the fix goes on the owner page — not in a new paragraph beside it.

## Coding standards (summary)

- NumPy-style docstrings. For array parameters, document shapes, units **and
  shardings**.
- Match existing formatting; do not reformat unrelated lines.
- A function implementing a physics equation says which equation.

## The two rules that bite

**Never hard-code a mesh shape.** Refer to mesh axes by name (`'x'`, `'y'`);
use `NamedSharding` / `PartitionSpec` for every layout and let XLA move the
data. No `np.concatenate`, no host-side gathers.

**Never require a whole array on one rank.** LORRAX's design envelope is
arrays that need hundreds of GPUs to hold. An operation that rematerialises a
global array on a subset of processes is not a slow path, it is a path that
cannot run the workload — see
[Design decisions](architecture/decisions.md), 2026-08-05.

Both rules have machine enforcement: `tests/test_layering.py` for the import
direction and the driver plumbing budgets ([Layers](architecture/layers.md)),
`tests/test_env_registry.py` for the environment surface.

## Before committing

- `uv run python -m pytest -q` after anything longer than a one-liner.
- `tools/release_check.sh` is the one command that runs the pre-push set: the
  login-node AST suites (layering, cross-file, env registry, env grammar, FFT
  shard-map), the input-reference drift check, and the origin-delta blob and
  secrets scan. Add `--with-allocation` for the end-to-end leg.
- On Perlmutter, a test result counts only if it was produced on a **compute
  node**. A login-node pytest has no GPU, no container and a different device
  count, so it is green for reasons unrelated to your change.
- Do not commit `__pycache__/`, `.venv/`, or cache directories.

Certification scope — which platforms and process counts are currently
green, and against which jobs — is not kept on this page, because a page that
records it goes stale silently. It lives in the sandbox measurement ledger
(`CLAIMS.md`), which is append-only and dated.
