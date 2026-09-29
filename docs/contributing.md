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

## The test suite

> **THE FOUR-GPU RULE — every GPU verification leg runs at P=4.** A P=1-only
> verification is never sufficient for landing. The owner's rationale,
> verbatim: *"use four gpus for 100% of all testing so that never ever do we
> run something on one GPU and then learn it doesn't generalize later"*.

The suite is `tests/hsuite`: the production drivers run end to end on one tiny
magnetic system (two H atoms, one electron, noncollinear with spin-orbit, time
reversal broken), at P4 on one node. The chain is kmeans → kin_ion → dipole →
gwjax GN-PPM one-shot → gwjax shared-pole QSGW (2 maps) → BSE → htransform →
exciton bands, then restarted COHSEX, GN-PPM SC (one map) and a shared-pole
one-shot with its W and pole exports, all in one process per rank. Each stage
is checked on its outputs (eqp columns, the numeric members of the h5 files it
writes, eigenvalue tables) against the stored
references in `tests/hsuite/reference/`, and every rank log is scanned for
failure signatures. [`tests/hsuite/README.md`](../tests/hsuite/README.md) owns
the fixture, the coverage table and the tolerances.

```bash
source config/perlmutter/gpu_env.sh   # Perlmutter machine settings
# the verdict: four pytest ranks, each running the drivers in its own process
srun --jobid=$JOBID -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh \
  .venv/bin/python -m pytest tests/hsuite -q -p no:cacheprovider
# regenerate the stored outputs after an intended change; review the diff
srun --jobid=$JOBID -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh \
  .venv/bin/python -m tests.hsuite.chain --out DIR --regenerate
```

With `lx` (project m4598) the same two lines are
`lx run -N 1 -G 4 -n 4 -- python -m pytest tests/hsuite` and
`lx run -N 1 -G 4 -n 4 -- python -m tests.hsuite.chain --out DIR --regenerate`.
pytest captures the per-stage walls; add `-s` to see them. The suite's compile
cache is `.hsuite_jax_cache` in the clone (`HSUITE_CACHE_DIR` moves it).

The P4 verdict is the four-rank `srun` line above (`lx run -n 4` with `lx`).
A one-rank launch (`lx test`, or `srun -n 1 --gpus-per-node=1`) runs the same
cell at P1, which is a smoke run, not the verdict.

Beside the suite are the five static AST suites (`test_layering.py`,
`test_crossfile_requests.py`, `test_env_registry.py`, `test_env_grammar.py`,
`test_fft_shardmap_context.py`). They run as scripts on a login node and also
collect under pytest.

`tests/bench` holds performance tools: standalone benchmark and profiling
drivers used to measure kernels and stages. They are not tests, and pytest does
not collect them.

## Before committing

- The P4 suite run above is the pre-push verdict for a driver or physics
  change.
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
