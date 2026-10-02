# LORRAX Agent Guide

**LORRAX** (**Lo**w-scaling **R**eal-space **R**eal-**A**xis e**X**cited state package) is
a JAX GW and BSE code with ISDF compression. The GW driver is **GWJAX** (`gw.gw_jax`).

This file is the entry point for an agent or developer about to read or change
the source. It says where the code and its owner pages are, how to run the
drivers and the test suite, and the coding conventions a change must follow.
Read it before editing code.

## Where things are

The module map is [`docs/codebase.md`](docs/codebase.md). The drivers, in chain
order, are [`docs/drivers.md`](docs/drivers.md). The standalone services are
listed in [`docs/architecture/services.md`](docs/architecture/services.md). The
register at the top of [`docs/index.md`](docs/index.md#register) names the page
that owns each fact.

## Key documentation

| Doc | What it covers |
|-----|---------------|
| `docs/theory/physics.md` | ISDF factorization, the Coulomb matrix, χ₀, screening, Σ and the QP Hamiltonian shared by every mode |
| `docs/codebase.md` | Module map: one line per source module |
| `docs/architecture/memory-model.md` | Per-rank budget, per-stage memory closed forms, the compiled check, the communication model |
| `docs/theory/minimax-quadrature.md` | Laplace kernels and minimax rules for χ₀, and the Σ denominator-box rules |

## How to run

Each driver's invocation, flags and outputs are in
[`docs/drivers.md`](docs/drivers.md); a complete first run from the bundled
fixture is the [Quickstart](docs/quickstart.md). Every driver is a module run
with `python -m` (`centroid.kmeans_cli`, `psp.get_dipole_mtxels`,
`gw.kin_ion_io`, `gw.gw_jax`, `bse.bse_jax`, …) under `srun` on a compute node,
one rank per GPU. The launch contract and the per-machine settings are on the
machine pages: [Perlmutter](docs/environment/machines/perlmutter.md),
[Frontera](docs/environment/machines/frontera.md). Installation:
[`docs/installation/index.md`](docs/installation/index.md).

## How to test

[`docs/contributing.md`](docs/contributing.md#the-test-suite) owns the test
suite and the pre-push checklist. In short, there are two kinds of tests:

- `tests/hsuite`: the production drivers run end to end on two small fixtures
  (a magnetic H2⁻ cell and bcc Na 3³) at P4 on one node, each stage compared
  with stored references and every rank's log scanned for failure signatures.
  It is the verdict for a driver or physics change, and only a compute-node run
  counts.
- The five static AST suites in `tests/` (`test_layering.py`,
  `test_crossfile_requests.py`, `test_env_registry.py`, `test_env_grammar.py`,
  `test_fft_shardmap_context.py`), which run on a login node.

`tools/release_check.sh` runs the login-node pre-push set. Service benchmarks
live in `services/<svc>/bench/` and are not collected by pytest.

## Coding standards

- Use NumPy-style docstrings. Document shapes, units, and shardings for array parameters.
- Match existing formatting. Do not reformat unrelated lines.
- A function implementing a physics equation names the equation it computes, so a
  reader can check it against the theory page.

## Conventions (read before editing GW code)

These norms keep the code legible to physicists and changeable by models. Review
and the static gates enforce them. When a convention forces a bigger change than
the task, flag it; do not silently violate it.

### Structure and style
- **Procedural on plain arrays, not new API layers.** Do not introduce
  classes, dataclasses or wrappers for what a function on NumPy/JAX arrays
  does; augment the existing bundle or table with an accessor instead. Each
  new abstraction costs every later reader time, so justify it or skip it.
- **`main()` reads as a physics outline.** A driver is a sequence of named stage calls
  (ζ fit → V_q → χ₀/W → Σ → eqp), not inlined machinery. Machinery lives in the stage helper.
- **Minimal signatures: pass bundles, not 15 arrays.** Thread `(wfns, meta, config/opts)`
  bundles through stages. A function taking more than about six positional arrays wants a bundle.
- **Single source of truth; no parallel old/new paths.** Never add `fetch_X_dyn` beside
  `fetch_X`, and never leave a deprecated facade on the import path. A changed
  routine replaces the old one in the same change, because two copies of one
  computation drift apart. Duplicated logic is a defect to collapse.

### Arrays, FFTs and sharding
- **FFTs go through their owners.** Never call `jnp.fft.*` directly in a stage kernel.
  The k-axis transforms and convolutions go through the `ffi.fft` router, and sphere↔box
  and plane transforms through `LocalFourierPlan`
  ([ffi_layout.md](docs/architecture/ffi_layout.md#k-convolution-router-and-the-mathdx-family)).
- **k and q are flat axes, never folded into the FFT grid.** Store and shard k
  (and q) as an explicit leading axis and transform over the grid axes only, so
  the k axis stays independent of the spatial transform.
- **Large read-only host caches go through `io_callback`, never jit arguments.**
  ψ(G) lives on the host (`common/psi_G_store.py`) and is pulled per slice
  inside the jit; a jit argument is replicated on every device.
- **Loops inside jit are `lax.scan` inside `shard_map`.** A Python-unrolled loop
  keeps one unsharded slot per trip alive.
- **Sharding.** Mesh axes are named (`'x'`, `'y'`), layouts are
  `NamedSharding`/`PartitionSpec`, and no large array is ever held on fewer
  than all ranks. The rules and their reasons are in
  [contributing](docs/contributing.md#the-two-rules-that-bite); the import
  direction is in [layers](docs/architecture/layers.md).

### Symmetry
- **One IBZ table and one symmetry-action helper.** ψ, ζ, V_q and W transform as
  the same kind of object under the space group and time reversal. Route every
  unfold through the `symmetry_maps` service (`SymMaps`); do not add a
  per-object "rotate X at q" variant. Handle time-reversal indices explicitly
  and never clip or nearest-map an unmapped k: a silently wrong partner passes
  norm checks while giving wrong physics.

### Physics reporting
- **Do not blame residuals on ISDF rank without evidence.** A LORRAX-vs-BerkeleyGW
  difference that plateaus as the centroid count grows is not basis error;
  look for an algorithm or convention difference instead.

## Before committing

- Run the checks [contributing](docs/contributing.md#before-committing) lists:
  the P4 `tests/hsuite` run for a driver or physics change, and
  `tools/release_check.sh` for the static set.
- Name your evidence directory, as a path, in the commit or report.
- Do not commit `__pycache__/`, `.venv/` or cache directories.
