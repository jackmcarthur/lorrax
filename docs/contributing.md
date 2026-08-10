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

## Two test tiers: the default gate and the census

There are exactly two ways to run the suite, and they answer different
questions.

```bash
uv run python -m pytest -q            # DEFAULT GATE — minutes
uv run python -m pytest -q --census   # THE CENSUS — everything
lx test                               # Perlmutter: the default gate
lx test --census                      # Perlmutter: the census
```

The **default gate** is the owner's sentence, implemented: run the Si
end-to-end test calculation for the drivers your branch actually touched,
plus every service's own suite, and have that basically be it. It is what you
run after a change. It is minutes, not tens of minutes, and it is minutes
because it is a handful of real end-to-end driver runs rather than a thousand
unit cells.

The **census** is everything, and it is what a bare `pytest` used to be —
the same collected set, unchanged. It is what `tests/KNOWN_FAILURES.md`
accounts for, and it is what you run before a release, after a merge wave, or
whenever you want the whole accounting rather than a verdict on your change.

Why the split exists, in the owner's words: the suite accreted a per-fix unit
cell for every bug the tree ever had, and running all of them on every branch
stopped being a signal and started being a tax. **Nothing was deleted.** Every
red-twin and gate cell still exists, still has to be green, and still runs —
under `--census` rather than on every invocation.

Which drivers count as "touched" is decided by a deliberately coarse
file→driver map in `tests/fast_gate.py` — `src/bse/*`
reaches the BSE drivers, `src/gw/*` the GW drivers, `src/psp/*` and
`src/common/*` reach everything, a service reaches its own suite and its
dependents, and **anything unmapped reaches everything**, which is the
fail-safe direction. The same file names the drivers that have no runnable
in-tree deck; touch one of those and the run tells you, out loud, that the
default gate says nothing about it.

The gate stands down and you get the census whenever you have already said
what you want: `--census`, `-m`, `-k`, a named path, `--no-services`,
`--only-service`, or `LX_CENSUS=1` in the environment. `LX_GATE_DRIVERS`
(`all` / `none` / a comma-separated list) overrides the diff, and
`LX_GATE_REF` changes the base the diff is taken against (default:
merge-base with `origin/main`).

## Before committing

- `uv run python -m pytest -q` after anything longer than a one-liner — the
  default gate. Run `--census` before you ask anyone to land it.
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
