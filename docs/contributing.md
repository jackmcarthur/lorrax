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

## Test tiers

> **THE FOUR-GPU RULE — every GPU verification leg runs at P=4.** It applies to
> both tiers below. A P=1-only verification is never sufficient for landing;
> unit and CPU cells are exempt. The owner's rationale, verbatim: *"use four
> gpus for 100% of all testing so that never ever do we run something on one
> GPU and then learn it doesn't generalize later"*. `lx test` already takes all
> four GPUs on the node; a driver leg wants `-G 4` rather than the one-GPU
> default.

The ordinary developer verdict is a two-minute core over tiny cached systems.
The old suite is the nightly full tier.

```bash
uv run python -m pytest -q                 # default core
uv run python -m pytest -q --full          # nightly full tier
lx test                                    # developer pre-push
lx test --full                             # nightly
```

The default core authenticates and runs only the tiny A/B fixture family plus
the exact service/runtime roster in `tests/core/manifest.py`. It covers the
major modules and one hostile size per contract without running a production
deck. `--core-extended` adds redundant standalone tiny drivers and optional
provider checks while staying below ten minutes.

The **full tier** owns the historical real-deck regressions, per-defect twins,
and `tests/KNOWN_FAILURES.md` accounting. `--census` and `-m census` remain
compatible aliases. A named path, `-m`, `-k`, or service-selection option is
an explicit selection and stands the default narrowing down.

Fixture generation is not part of either timing. `lx test --build-fixtures`
builds or hits the hash-addressed QE cache, verifies the committed SHA-256
reference stamps, and exits without running tests.

## Before committing

- `lx test` is the developer pre-push verdict. Run `lx test --full` in the
  nightly/release lane.
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
