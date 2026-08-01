# LORRAX Agent Guide

**LORRAX** (**Lo**w-scaling **R**eal-space **R**eal-**A**xis e**X**cited state package) —
JAX-based GW with ISDF compression. The GW driver is called **GWJAX**.

You are most likely arriving here from the `lorrax_sandbox` project, where test runs
and BGW comparisons are organized. This repo contains the source code you may need to
read or modify. Read this file upon first inspection of the LORRAX source before editing any code.

## Where things are

| Path | What | When to read |
|------|------|-------------|
| `src/gw/gw_jax.py` | Main GW driver | Any GW debugging |
| `src/gw/gw_init.py` | Input parsing, chunking strategy, pipeline orchestration | Input file questions, chunk sizing |
| `src/gw/gw_config.py` | `LorraxConfig` runtime options dataclass | Flag plumbing, memory budget |
| `src/gw/gw_driver_helpers.py` | Screening/PPM setup helpers | Driver wiring |
| `src/gw/w_isdf.py` | χ₀ → W screening pipeline (CTSP, Dyson solve) | Screening / epsilon issues |
| `src/gw/ppm_sigma.py` | GN-PPM dynamic self-energy Σ^c(ω) | Frequency-dependent sigma issues |
| `src/gw/minimax_screening.py` | PPM extraction, minimax window helpers | PPM parameter issues |
| `src/gw/minimax_config.py` | Shared minimax / sigma quadrature config | Quadrature setup |
| `src/gw/head_correction.py` | q=0 head / wing correction | Head corrections |
| `src/gw/vcoul.py`, `compute_vcoul.py`, `compute_vcoul_0d.py` | Coulomb potential (3D / 2D slab / 0D box) | Truncation, V_q build |
| `src/gw/greens_function_kernel.py` | `build_G` occupied/all Green's function | G-matrix construction |
| `src/gw/wavefunction_bundle.py` | `Wavefunctions` bundle + `project` / `project_ri` (Σ_μν → Σ_ij band projection) | Band-basis projection |
| `src/gw/qsgw_utils.py` | QSGW fixed-point solver, Σ^xc I/O | Self-consistent GW |
| `src/gw/kin_ion_io.py` | Kinetic + ionic Hamiltonian I/O | `kin_ion.h5` issues |
| `src/common/isdf_fitting.py` | CCT/ZCT, pair-density kernels, zeta solve | Zeta fitting, pair density |
| `src/common/wfn_transforms.py` | Wavefunction loading + band-chunked FFT | WFN load path |
| `src/common/cholesky_2d.py` | 2D-blocked Cholesky for sharded CCT | Cholesky issues |
| `src/common/fft_helpers.py` | Flat-k FFT helpers | FFT plumbing |
| `src/common/gvec_fft_box.py` | Sphere ↔ FFT-box gather | V_q G-space build |
| `src/common/symmetry_maps.py` | `SymMaps`: IBZ→full BZ unfolding, spinor rotations | Symmetry / k-point unfolding |
| `src/common/minimax.py` | Minimax quadrature solvers | Quadrature node/weight issues |
| `src/common/meta.py` | `Meta` system-parameters dataclass | k/q-grid, band ranges |
| `src/file_io/wfn_loader.py` | `WfnLoader`: canonical WFN.h5 reader; `backend='auto'` picks the phdf5 (parallel HDF5) async path | Wavefunction loading, async H5 reads |
| `src/common/gpu_utils.py` | Host-side GPU memory detection | Chunk auto-sizing |
| `src/file_io/slab_io.py` | `SlabIO`: phdf5 writer wrapper for zeta_q / V_qmunu | Big HDF5 writes |
| `src/file_io/sigma_output.py` | Σ output (eqp.dat, sigma.h5) | Output formats |
| `src/ffi/` | XLA FFI bridge: `cusolvermp`, `cublasmp`, `phdf5`, `slate` | Native-library entry points |
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

# Tests (~15s, JAX compilation overhead)
uv run python -m pytest -q
```

### Perlmutter (Shifter via Lmod module)

Perlmutter workflow (lxrun/module) — on Frontera this differs; see `docs/environment/machines/frontera.md` and the working examples below.

```bash
module load lorrax_X           # X = A | B | C
lxalloc                        # 1 node / 4 GPUs / 2 h
lxpre cohsex.in 640            # all 3 preprocessing steps (single-GPU)
lxrun python3 -u -m gw.gw_jax -i cohsex.in        # 4-GPU GW
LORRAX_NGPU=1 lxrun ...                           # single-GPU override
```

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

Run `uv run python -m pytest -q` after long running branches (5+ small commits); it takes 15 seconds.
Do not commit `__pycache__/`, `.venv/`, or `uv_cache/`, etc. directories.

## Environment

Use `uv` as the package manager. One `.venv/` (gitignored) per machine. No alternative
envs. Let uv use its global cache — do not create project-local uv cache directories.

On Perlmutter: use Shifter with the NVIDIA JAX container (Perlmutter workflow —
on Frontera this differs; see `docs/environment/machines/frontera.md`). See
`config/README.md`.
