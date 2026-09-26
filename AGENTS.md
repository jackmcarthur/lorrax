# LORRAX Agent Guide

**LORRAX** (**Lo**w-scaling **R**eal-space **R**eal-**A**xis e**X**cited state package) —
JAX-based GW with ISDF compression. The GW driver is called **GWJAX**.

You are most likely arriving here from the `lorrax_sandbox` project, where test runs
and BGW comparisons are organized. This repo contains the source code you may need to
read or modify. Read this file upon first inspection of the LORRAX source before editing any code.

> **If you are a dispatched lane, [`AGENT_PREAMBLE.md`](AGENT_PREAMBLE.md) is the
> entry document — read it first, once.** It carries the **efficiency doctrine**
> (fan out independent legs — measured duty cycle 0.41, 32.4 h idle vs 17.5 h
> compute; one combined P=4 leg per lane; harvest the index before measuring; warm
> worker for repeated legs, 2.4 s vs 16 s; lane weights; this contract read once;
> ledger discipline), the **measurement-discipline** rules, **THE FOUR-GPU RULE**
> (*"never ever do we run something on one GPU and then learn it doesn't generalize
> later"*), and the machine — certificate, pool, EXIT codes, traps, allocator,
> etiquette. This file covers the code; that one covers the method and the machine.

## Where things are

The module map is [`docs/codebase.md`](docs/codebase.md). The drivers, in chain
order, are [`docs/drivers.md`](docs/drivers.md). The standalone services are
listed in [`docs/architecture/services.md`](docs/architecture/services.md). The
register at the top of [`docs/index.md`](docs/index.md#register) names the page
that owns each fact.

## Key documentation

| Doc | What it covers |
|-----|---------------|
| `docs/theory/physics.md` | ISDF theory, GW equations, COHSEX, CTSP formalism |
| `docs/codebase.md` | Module map, data flow, key classes, sharding patterns |
| `docs/architecture/memory-model.md` | Per-stage memory formulas, chunk sizing, bottleneck arrays |
| `docs/theory/minimax-quadrature.md` | GL/HGL quadrature, error scaling, crossing windows |

## How to run

### Local dev (single machine, uv)

```bash
# Preprocessing (centroids, dipole, kin+ion)
uv run python -m centroid.kmeans_cli 600 --seed 42
uv run python -m psp.get_dipole_mtxels -i cohsex.in
uv run python -m gw.kin_ion_io -i cohsex.in

# GW calculation
uv run python -m gw.gw_jax -i cohsex.in

# Tests -- the DEFAULT CORE: tiny cached A/B systems plus one basic contract
# for every major module.  Target: two minutes.
uv run python -m pytest -q

# The nightly FULL tier: historical real decks and defect twins.
uv run python -m pytest -q --full
```

### Perlmutter (the `lx` harness)

`lx run` puts one step on a compute node and selects the `lorrax_A` base module.
The [machine page](docs/environment/machines/perlmutter.md) owns the launch
contract, the one-rank-per-GPU geometry and the evidence rules. Never `sbatch`
an iteration. On Frontera this differs; see
`docs/environment/machines/frontera.md` and the examples below.

See [`config/README.md`](config/README.md) for the full cluster reference. Docs: [`docs/environment/overview.md`](docs/environment/overview.md).

### Frontera (TACC, CPU: apptainer + srun --mpi=pmi2)

Working invocations from the certified scripts (`config/frontera/templates/gw_dev.sbatch`; mos2_4x4_test sbatch family):

```bash
# preprocessing, single node / single process (deck_b300.sbatch steps 3-4):
python3 -u -m centroid.kmeans_cli 3000 --orbit --out-suffix _b300_c3000
python3 -u -m gw.kin_ion_io -i deck_b300.in -o kin_ion_b300.h5 -n 300

# multi-node GW via the certified launch block (gw_ht_b300.sbatch):
export LORRAX_ROOT=... LORRAX_RUN_DIR=... LORRAX_INPUT=gw.in
bash $LORRAX_ROOT/config/frontera/templates/gw_dev.sbatch
```

## Coding standards

- Use NumPy-style docstrings. Document shapes, units, and shardings for array parameters.
- Match existing formatting. Do not reformat unrelated lines.
- Every function implementing a physics equation should reference what it is computing for human readability standards.

## CONVENTIONS (load before editing GW code)

These are the norms that make the codebase legible to humans and one-shottable by models.
They are enforced by review and the regression gate, not by ceremony. When a convention
forces a bigger change than the task, flag it — don't silently violate it. The sandbox
claims ledger (`lorrax_sandbox/CLAIMS.md`) records what has been verified about the
pipeline; the old refactor-map reports directory was purged.

### Structure & style
- **Procedural on plain arrays, not new API layers.** LORRAX is scientific code read by
  physicists. Do not introduce classes/dataclasses/wrappers for what a function on numpy/jax
  arrays does. No `BzIbzTable`, no `SymAction` object — augment the existing bundle/table with
  an accessor instead. New abstractions cost human bandwidth; justify them or skip them.
- **`main()` reads as a physics outline.** The driver is a sequence of named stage calls
  (ζ-fit → V_q → χ₀/W → Σ → eqp), not inlined machinery. Machinery lives in the stage helper.
- **Minimal signatures — pass bundles, not 15 arrays.** Thread `(wfns, meta, config/opts)`
  bundles through stages. If a function takes >~6 positional arrays, it wants a bundle.
- **Single source of truth. No parallel old/new paths.** Never add `fetch_X_dyn` beside
  `fetch_X`, never leave a deprecated facade on the import path "for now". If you change a
  routine, delete the old one in the same change. Duplicated logic (the cohsex.in parser ×3,
  the eqp/Z math ×4) is a defect to collapse, not a pattern to extend.

### JAX / arrays
- **FFTs go through their owners.** Never call `jnp.fft.*` directly in a stage kernel.
  The k-axis transforms and convolutions go through the `ffi.fft` router, and sphere↔box
  and plane transforms through `LocalFourierPlan`
  ([ffi_layout.md](docs/architecture/ffi_layout.md#k-convolution-router-and-the-mathdx-family)).
  A raw `jnp.fft` in a kernel is a bug.
- **k/q dimensions are FLAT axes, never folded into the FFT grid.** Store and shard k-points
  (and q-points) as an explicit leading flat axis; do the spatial FFT over the grid axes only.
  This keeps the k-axis independent of the spatial transform so FFT→NUFFT and flat-k batching
  (see `project_flat_k_chi0_pipeline`) stay drop-in. Do not reshape k into the FFT box.
- **Big read-only host caches go through `io_callback`, never jit args.** ψ(G) and other large
  read-only arrays live on host (`common/psi_G_store.py`) and are pulled per-slice inside the
  jit via `io_callback`. Passing them as jit arguments replicates them on every device — an OOM.
- **Sharding: mesh axes by name (`'x'`, `'y'`), always `NamedSharding`/`PartitionSpec`.** Never
  hard-code mesh shapes. Let XLA move data — no `np.concatenate`, no host-side gathers.
- **No replicated large intermediates.** We are memory-constrained; most large arrays only fit
  tiled over the XY grid. Any op that rematerializes a large array on a subset of processors is
  a defect to fix, not a budget to work around (`feedback_zero_replicated_intermediates_principle`).
  Python-unrolled inner loops inside jit pile up N× unsharded slots — use `scan` *inside*
  `shard_map`, not a naive `fori_loop` (`feedback_path_d_scaffolding_pattern`).

### Symmetry
- **One IBZ table + one sym-action helper.** ψ, ζ, V_q, W transform as the same kind of object
  under space-group + TRS. Route every unfold through the canonical `SymMaps` table and a single
  sym-action helper. Do not add per-object "rotate X at q" variants (there are historically ≥6;
  they are being retired — `feedback_unified_sym_action`). TRS index handling must be explicit;
  never silently clip or nearest-fallback an unmapped k (that was the TRS-blind bug).

### Physics reporting
- **Don't blame residuals on "ISDF rank" without evidence.** Plateau-shaped LORRAX-vs-BGW
  disagreement rules out basis error — chase an algorithm/convention difference instead
  (`feedback_no_isdf_rank_excuse`).

### JAX sharding rules (restated)
- Never hard-code mesh shapes. Refer to mesh axes by name (`'x'`, `'y'`).
- Use `NamedSharding` / `PartitionSpec` for all layouts. Let XLA handle communication, no `np.concatenate` or
  host-side gathers.
- We are very memory constrained and most large arrays can only be stored when tiled over the XY processor grid. Avoid at all costs operations that rematerialize large arrays on subsets of all processors.

## Before committing

Run `uv run python -m pytest -q` after long running branches (5+ small commits) -- that is the
two-minute DEFAULT CORE. Run `uv run python -m pytest -q --full` in the nightly/release lane;
it is the suite KNOWN_FAILURES.md accounts for. See `tests/README.md` and `docs/contributing.md`.
GPU verification follows [the four-GPU rule](AGENT_PREAMBLE.md#the-four-gpu-rule).
Name your evidence directory, as a path, in the report.
Do not commit `__pycache__/`, `.venv/`, or `uv_cache/`, etc. directories.

## Environment

Use `uv` as the package manager. One `.venv/` (gitignored) per machine. No alternative
envs. Let uv use its global cache — do not create project-local uv cache directories.

On Perlmutter: `lx` and the `lorrax_A` base module
([Perlmutter](docs/environment/machines/perlmutter.md)). On Frontera this differs; see
`docs/environment/machines/frontera.md`. See `config/README.md`.
