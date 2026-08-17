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

| Path | What | When to read |
|------|------|-------------|
| `src/gw/gw_jax.py` | Main GW driver | Any GW debugging |
| `src/gw/gw_init.py` | Input parsing, chunking strategy, pipeline orchestration | Input file questions, chunk sizing |
| `src/gw/gw_config.py` | `LorraxConfig` runtime options dataclass | Flag plumbing, memory budget |
| `src/gw/sigma_dispatch.py` | Mode-agnostic Σ dispatch (one call per compute mode) | Driver wiring |
| `src/gw/w_isdf.py` | χ₀ → W screening pipeline (CTSP, Dyson solve) | Screening / epsilon issues |
| `src/gw/ppm_sigma.py` | GN-PPM dynamic self-energy Σ^c(ω) | Frequency-dependent sigma issues |
| `src/gw/band_extrapolation.py` | Σ_c band-convergence extrapolation: the disjoint band-bracket plan (interior cuts prefer a clean multiplet boundary via `common/band_degeneracy.py`, falling back rather than refusing), the two-parameter 1/N fit and its trust diagnostics. Sampling fractions are of the TOTAL band count and are MEASURED against BerkeleyGW `ch_converge.dat` — read the comment at `BRACKET_FRACTIONS` before changing them. Deck key `sigma_band_extrapolation`, GN/HL-PPM only, which is a correctness guard | "how many bands does Σ_c need"; the Σ cube's leading bracket axis |
| `src/gw/minimax_screening.py` | PPM extraction, minimax window helpers | PPM parameter issues |
| `src/gw/minimax_config.py` | Shared minimax / sigma quadrature config | Quadrature setup |
| `src/gw/head_correction.py` | q=0 head / wing correction | Head corrections |
| `src/gw/vcoul.py`, `compute_vcoul.py`, `compute_vcoul_0d.py` | Coulomb potential (3D / 2D slab / 0D box) | Truncation, V_q build |
| `src/gw/greens_function_kernel.py` | `build_G` occupied/all Green's function | G-matrix construction |
| `src/gw/wavefunction_bundle.py` | `Wavefunctions` bundle + `project` / `project_ri` (Σ_μν → Σ_ij band projection) | Band-basis projection |
| `src/gw/qsgw_utils.py` | QSGW fixed-point solver, Σ^xc I/O | Self-consistent GW |
| `src/gw/kin_ion_io.py` | Kinetic + ionic Hamiltonian I/O | `kin_ion.h5` issues |
| `src/isdf/core.py` + `src/gw/isdf_fitting.py` | CCT/ZCT, pair-density kernels, zeta solve / stage orchestration | Zeta fitting, pair density |
| `src/common/wfn_transforms.py` | Wavefunction loading + band-chunked FFT | WFN load path |
| `services/distrib_la/` | **The distributed dense-linalg service**: one door for `eigh` / `cholesky` / `solve_lu` over scalapack, slate, cusolvermp and native (incl. the 2D-blocked `native2d` Cholesky that was `src/common/cholesky_2d.py`). Read `docs/services/distrib_la.md` first | Cholesky / eigh / LU on a mesh; backend refusals; `.so` pins |
| `src/common/fft_helpers.py` | Flat-k FFT helpers | FFT plumbing |
| `src/common/gvec_fft_box.py` | Sphere ↔ FFT-box gather | V_q G-space build |
| `services/symmetry_maps/` | **The crystal-symmetry service**: one door for `SymMaps` (IBZ→full BZ tables, spinor rotations), the k-star index map, the sharded q-axis unfolds, the real-space orbit machinery and the time-reversal MEASUREMENT (was `src/common/symmetry_maps.py`, `src/centroid/orbit_syms.py`, `src/common/density_symmetry_check.py`). Read `docs/services/symmetry_maps.md` first | Symmetry / k-point unfolding; star conjugation; TRS verdicts |
| `services/minimax/` | **The certified-quadrature service**: one door for `serve` / `lookup` (shipped tables with provenance and refusals), the target and family vocabulary as data, and the offline solvers behind an announced escape hatch (was `src/common/minimax.py` + `src/common/minimax_assets/`). Reach it as `import minimax`; never a submodule path. | Quadrature node/weight issues; "no certified table" refusals; uncertified-solve announcements |
| `src/common/meta.py` | `Meta` system-parameters dataclass | k/q-grid, band ranges |
| `services/wfn_loader/` | **The ψ(G) loading service**: one door for `WfnLoader`, the canonical WFN.h5 reader, with `backend='auto'` picking the eager or the phdf5 (parallel HDF5) collective read and the two held byte-identical (was `src/file_io/wfn_loader.py`). Reach it as `import wfn_loader`, or as `from file_io import WfnLoader` / `WFNReader`; never a submodule path. Read `docs/services/wfn_loader.md` first | Wavefunction loading, H5 read backends |
| `src/common/gpu_utils.py` | Host-side GPU memory detection | Chunk auto-sizing |
| `src/file_io/slab_io.py` | `SlabIO`: phdf5 writer wrapper for zeta_q / V_qmunu | Big HDF5 writes |
| `src/file_io/sigma_output.py` | Σ output (eqp.dat, sigma.h5) | Output formats |
| `src/ffi/` | XLA FFI bridge, lorrax's half: `phdf5`, `fft`/`mklfft`, `gemm`, `cusolvermp` context + `cublasmp`, and the C++ tree `cpp/` (which still builds the slate / scalapack / cusolvermp handlers). The distributed-linalg python side moved to `services/distrib_la` | Native-library entry points |
| `src/solvers/` | Davidson, Lanczos, Chebyshev, pseudobands | Iterative eigensolvers |
| `src/centroid/kmeans_cli.py` / `src/centroid/kmeans_isdf.py` | ISDF centroid generation — `kmeans_cli` is the CLI (`python -m centroid.kmeans_cli`); `kmeans_isdf` is the algorithm library, no `__main__` | Centroid count / quality |
| `src/psp/` | Pseudopotentials, dipole / kin+ion generators | `dipole.h5` or `kin_ion.h5` issues |
| `src/bse/` | Bethe–Salpeter equation | Optical spectra; exciton dispersion `E_S(Q)` at arbitrary Q. Read `src/bse/STATUS.md`, then `BGW_COMPARE.md` (absorption) or `EXCITON_BANDS.md` (arbitrary Q) before running either |
| `src/bandstructure/` | H-matrix interpolation (experimental) | Band-structure plots |

## Key documentation

| Doc | What it covers |
|-----|---------------|
| `docs/theory/physics.md` | ISDF theory, GW equations, COHSEX, CTSP formalism |
| `docs/architecture/codebase.md` | Module map, data flow, key classes, sharding patterns |
| `docs/architecture/memory-model.md` | Per-stage memory formulas, chunk sizing, bottleneck arrays |
| `docs/theory/minimax-quadrature.md` | GL/HGL quadrature, error scaling, crossing windows |
| `docs/dev/notes/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` | PPM sigma pipeline: phases, windows, evaluation loop |
| `docs/dev/progress/SIGMA_FREQ_AUDIT_STATUS.md` | Current BGW vs GWJAX comparison status, known offsets |

## How to run

### Local dev (single machine, uv)

```bash
# Preprocessing (centroids, dipole, kin+ion)
uv run python -m centroid.kmeans_cli 600 --seed 42
uv run python -m psp.get_dipole_mtxels -i cohsex.in
uv run python -m gw.kin_ion_io -i cohsex.in

# GW calculation
uv run python -m gw.gw_jax -i cohsex.in

# Tests -- the DEFAULT GATE: the Si end-to-end calculation for the drivers
# this branch touched, plus the services' own suites.  Minutes.
uv run python -m pytest -q

# The CENSUS: everything.  What a bare `pytest` used to be, and what
# tests/KNOWN_FAILURES.md accounts for.
uv run python -m pytest -q --census
```

### Perlmutter (Shifter, via the `lx` harness)

```bash
export LX_BASE_MODULE=lorrax_J070             # without it you get the wrong jax
lx run -N 1 -G 4 -n 4 python3 -u -m gw.gw_jax -i cohsex.in   # one P=4 step, allocates or attaches
lx test                                       # the default gate, on a compute node, in cwd
lx status                                     # who is running where
```

`lx` allocates or attaches by itself, so never `sbatch` an iteration and never
`lx release --all`. The older `module load lorrax_X` + `lxalloc`/`lxrun`/`lxpre`
workflow is superseded; `docs/environment/machines/perlmutter.md` is the current
reference and keeps the old one as history. Fan out independent legs rather than
running them serially, and combine one branch's verification into one P=4 leg —
`AGENT_PREAMBLE.md` has the measurements. On Frontera this differs entirely; see
`docs/environment/machines/frontera.md` and the examples below.

See [`config/README.md`](config/README.md) for the full cluster reference. Docs: [`docs/environment/overview.md`](docs/environment/overview.md).

### Frontera (TACC, CPU: apptainer + srun --mpi=pmi2)

Working invocations from the certified scripts (`config/frontera/templates/gw_dev.sbatch`; mos2_4x4_test sbatch family):

```bash
# preprocessing, single node / single process (deck_b300.sbatch steps 3-4):
python3 -u -m centroid.kmeans_cli 3000 --orbit --qe-save ../b300_out/MoS2.save --out-suffix _b300_c3000
python3 -u -m gw.kin_ion_io -i deck_b300.in -o kin_ion_b300.h5 -n 300 --hartree

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
- **One FFT path: the sharded FFT helpers in `common/fft_helpers.py`.** Never call
  `jnp.fft.*` directly in a stage kernel. All G↔r transforms go through the helper factories
  so sharding, box placement, and Bloch phases stay consistent — and so the whole package can
  be upgraded FFT→NUFFT in one place. A raw `jnp.fft` in a kernel is a bug.
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
  (`feedback_no_isdf_rank_excuse`). Common convention gotchas live in `FLAGS.md` (`sys_dim`,
  `bare_coulomb_cutoff`, velocity-operator sign).

### JAX sharding rules (restated)
- Never hard-code mesh shapes. Refer to mesh axes by name (`'x'`, `'y'`).
- Use `NamedSharding` / `PartitionSpec` for all layouts. Let XLA handle communication, no `np.concatenate` or
  host-side gathers.
- We are very memory constrained and most large arrays can only be stored when tiled over the XY processor grid. Avoid at all costs operations that rematerialize large arrays on subsets of all processors.

## Before committing

Run `uv run python -m pytest -q` after long running branches (5+ small commits) -- that is the
DEFAULT GATE (Si end-to-end smoke for the drivers you touched + the services' suites; minutes).
Run `uv run python -m pytest -q --census` -- the full suite, and the run KNOWN_FAILURES.md
accounts for -- before asking for a landing.  See `tests/README.md` and `docs/contributing.md`.
**Every GPU verification leg runs at P=4, in ONE combined leg** -- gates, driver and red
twin together, not one leg per gate; a P=1-only verification is never sufficient for
landing (unit and CPU cells are exempt). `AGENT_PREAMBLE.md` owns that rule and its
rationale. Name your evidence directory, as a path, in the report.
Do not commit `__pycache__/`, `.venv/`, or `uv_cache/`, etc. directories.

## Environment

Use `uv` as the package manager. One `.venv/` (gitignored) per machine. No alternative
envs. Let uv use its global cache — do not create project-local uv cache directories.

On Perlmutter: use Shifter with the NVIDIA JAX container (Perlmutter workflow —
on Frontera this differs; see `docs/environment/machines/frontera.md`). See
`config/README.md`.
