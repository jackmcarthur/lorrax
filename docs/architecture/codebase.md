# Codebase Structure & Architecture

Module organization, key classes, data flow, and file formats. Read this when working on implementation tasks. For physics/theory see [Core ISDF and GW theory](../theory/physics.md); for environment setup see [`environment/overview.md`](../environment/overview.md).

> **Read [`layers.md`](layers.md) first if you are about to add or move a
> module.** This page says *where things are*; that one says **what a module is
> allowed to know about**, and it is enforced.
>
> Three levels — **L1 physics** (bands, q-points, ζ, Σ, decks), **L2 numerical
> routines** (matrices, quadrature, convergence — nothing physical), **L3
> substrate** (devices, meshes, processes, native libraries, files — nothing
> mathematical). **Imports run downhill only.** `tests/test_layering.py` fails
> when they do not: it is pure AST, needs neither jax nor an importable `src`,
> and runs on a login node in under ten seconds.
>
> The one-line version: a driver should read as physics. If `gw_jax.py` has to
> say `Mesh`, `shard_map` or `os.environ`, the thing it is saying belongs in
> the substrate.

---

## Table of Contents

1. [Module Structure](#1-module-structure)
2. [Key Classes](#2-key-classes)
3. [Device mesh and sharding conventions](#3-device-mesh-and-sharding-conventions)
4. [Flat-k / flat-q convention](#4-flat-k--flat-q-convention)
5. [Data Flow Pipeline](#5-data-flow-pipeline)
6. [File Formats](#6-file-formats)
7. [Entry Points](#7-entry-points)
8. [Function Call Hierarchy](#8-function-call-hierarchy)
9. [Code Locations Quick Reference](#9-code-locations-quick-reference)

---

## 1. Module Structure

```
src/
├── gw/                   # GW driver + sigma kernels
│   ├── gw_jax.py              main(): mesh, config, ISDF, χ₀/W, Σ, head, QSGW
│   ├── gw_config.py           LorraxConfig dataclass (from cohsex.in)
│   ├── gw_init.py             chunk sizing, prepare_isdf_and_wavefunctions
│   ├── gw_driver_helpers.py   build_*_runtime_options (config → kernel kwargs)
│   ├── gw_output.py           banner / section / GWResults / write_results
│   ├── minimax_config.py      MinimaxConfig / SigmaQuadratureConfig
│   ├── minimax_screening.py   PPM extraction, window helpers, shipped-table lookup
│   ├── w_isdf.py              χ₀ minimax kernel + W Dyson solve (flat-q)
│   ├── ppm_sigma.py           GN-PPM build + Σ^c(ω) branch/window/τ pipeline
│   ├── greens_function_kernel.py  build_G (single entry point)
│   ├── head_correction.py     q→0 head sample + exact static head terms
│   ├── head_channel.py        WHERE the mini-BZ q≠0 average is applied (mc_average_placement)
│   ├── head_densify.py        Γ head split off before the coarse→fine densifier, re-attached analytically
│   ├── downfold.py            transfer solve, orbit-floored selection, child unfold tables
│   ├── downfold_run.py        the driver's stages + its gates (see docs/downfold.md)
│   ├── downfold_cli.py / downfold_config.py   entry point and deck
│   ├── experimental/head_wing_schur.py
│   │                          sharded head/wing/body Schur reconstruction of W in the
│   │                          centroid basis. NOT in the BSE and NOT wired into
│   │                          production W: it is an algebraic kernel behind the
│   │                          pytest `extra` marker (tests/test_head_wing_schur.py).
│   │                          head_channel.py reuses its extract_V_body_sharded
│   ├── wavefunction_bundle.py BandSlices + Wavefunctions (4 sharded ψ copies); project, project_ri (band-space contractions)
│   ├── coulomb/               dimension-aware Coulomb kernels behind get_kernel():
│   │                            base.py (SysDim, dispatcher, mini-BZ sampler),
│   │                            bulk_3d.py, slab_2d.py, box_0d.py
│   ├── vcoul.py               MC q=0 averages, Voronoi wrap helpers
│   ├── compute_vcoul.py       V_qμν from ζ (μ-chunked FFT + H5 prefetch); compute_v_q_per_G
│   ├── compute_vcoul_0d.py    box-truncated Coulomb driver (molecules)
│   ├── qsgw_utils.py          diagonal fixed-point + QSGW from sigma_mnk.h5
│   ├── scissor.py             valence/conduction scissor fit (out-of-grid bands)
│   └── kin_ion_io.py          kinetic + ionic H matrix elements
│
├── common/               # Shared kernels + utilities
│   ├── meta.py                Meta dataclass (system params, band edges)
│   ├── (symmetry_maps.py, density_symmetry_check.py → services/symmetry_maps/, 2026-08-07)
│   ├── wfn_transforms.py      WFN.h5 reads, per-k FFT, kchunk helpers (was load_wfns.py)
│   ├── isdf_fitting.py        CCT/ZCT kernels, Cholesky, zeta solve, full pipeline
│   ├── cholesky_2d.py         2D blocked Cholesky (sharded)
│   ├── fft_helpers.py         flat-k ↔ 3D k FFT helpers via custom_partitioning
│   ├── gvec_fft_box.py        sphere ↔ FFT-box gather (G-space layouts)
│   ├── chi_from_dipole.py     S(ω) tensor from dipole matrix elements
│   ├── gamma_matrices.py      Pauli / bispinor helpers
│   ├── bispinor_init.py       spinor → bispinor lift
│   ├── jax_compile_cache.py   XLA persistent cache activator
│   ├── jax_profile.py         annotation / trace_section context mgrs
│   ├── gpu_utils.py           device-memory budget probes
│   ├── progress.py            LoopProgress (rank-0 progress bar)
│   ├── timing.py              named sections, aggregate report
│   │   (minimax.py and minimax_assets/ left for services/minimax/ 2026-08-08)
│   ├── (bench/smoke drivers → tests/bench/, 2026-07-31)
│   │                          FFI smoke tests / benchmarks (run via lxrun)
│   └── phdf5_*                phdf5 benchmarks + plumbing tests
│
├── file_io/              # Canonical file-format I/O (used by gw_jax)
│   ├── (wfn_loader.py, zeta_loader.py → services/, 2026-08-07; __init__.py re-exports
│   │                          WfnLoader / WFNReader / ZetaLoader off the doors)
│   ├── wfn_writer.py          BGW-compatible WFN.h5 writer
│   ├── epsreader.py           EPSReader (eps0mat / epsmat)
│   ├── qe_save_reader.py      CrystalData from QE .save
│   ├── tagged_arrays.py       ISDF restart state serialization
│   ├── sigma_output.py        eqp.dat, eqp1.dat, sigma_mnk.h5 writers
│   ├── kin_ion.py             load_kin_ion_submatrix
│   ├── qp_wfn.py              write_qp_rotations_h5
│   ├── centroids.py           centroid file loader
│   ├── paths.py               path resolution helpers
│   ├── read_bgw_vcoul.py      BGW vcoul table reader (diagnostic override)
│   └── slab_io.py             SlabIO: MPI-IO-like phdf5 writer (+ allgather fallback)
│       _slab_io_ffi.py          phdf5 FFI backend
│       _slab_io_allgather.py    plain-h5py rank-0 backend
│
├── ffi/                  # XLA FFI bridge to native libraries
│   ├── common/                ffi_loader (ctypes) + cpp (CMake) → liblorrax_ffi.so
│   ├── cusolvermp/            distributed eigh (multi-proc multi-GPU, CAL+NCCL)
│   ├── phdf5/                 parallel HDF5 read/write (MPI-IO)
│   └── slate/                 SLATE Cholesky / trsm / heev (p×q grid)
│
├── solvers/              # Iterative eigensolvers + quadrature / DOS
│   ├── davidson.py, lanczos.py, chebyshev.py
│   ├── pseudobands.py, pseudobands_v2.py     (DFT-band compression)
│   ├── quadrature.py, dos.py
│   └── docs/                  solver-specific notes
│
├── centroid/             # ISDF centroid selection
│   ├── kmeans_cli.py          CLI entrypoint: `python -m centroid.kmeans_cli`
│   ├── kmeans_isdf.py         k-means algorithm (k-means++ init, charge-weighted)
│   ├── charge_density.py      ρ(r) build for centroid weights (get_charge_density)
│   └── pivoted_cholesky.py    Cholesky-pruned candidate list
│
├── psp/                  # Pseudopotentials + DFT operators
│   ├── pseudos.py, upf/, radial/
│   ├── get_dipole_mtxels.py   dipole.h5 generator
│   ├── get_DFT_mtxels.py      DFT matrix elements
│   ├── h_dft.py, dft_operators.py, dft_precond.py
│   ├── run_nscf.py, nscf_input.py
│   ├── kpm_dos.py
│   └── tests/                 psp unit tests
│
├── mixing/               # Self-consistent acceleration
│   └── acceleration.py        Anderson / rcrop fixed-point solver
│
├── bse/                  # Bethe–Salpeter (experimental; see src/bse/)
│   ├── bse_isdf.py, bse_jax.py, bse_w_exact.py
│   ├── bse_feast.py, bse_lanczos.py, bse_kpm.py
│   ├── bse_preconditioner.py, bse_pseudopoles.py
│   ├── bse_ring_comm.py, bse_serial.py
│   └── context/  (test_bse.py → tests/bench/)
│
├── bandstructure/        # H-matrix interpolation (experimental)
│   └── htransform.py
│
└── postprocess/
    └── rotate_wfn_to_qp.py
```

---

## 2. Key Classes

### 2.1 `LorraxConfig` — `src/gw/gw_config.py`

Loaded from `cohsex.in` via `LorraxConfig.from_input_file()`. Holds every flag the driver consults: file paths, band ranges, ISDF parameters, screening / PPM knobs, head-correction policy, `use_ffi_io`, `use_ppm_sigma`, `use_bgw_vcoul`, `memory_per_device_gb`, `sys_dim`, `bispinor`, `self_consistent`, etc. Two nested configs:

- `MinimaxConfig` (`minimax_config.py`) — static/imag-ω quadrature parameters.
- `SigmaQuadratureConfig` — Σ^c(ω) window quadrature parameters.

### 2.2 `Meta` — `src/common/meta.py`

Immutable-ish dataclass with all system parameters derived from `WFNReader` + `SymMaps`:

```python
@dataclass
class Meta:
    rank, n_proc                     # MPI / jax.process info
    b_id_0 .. b_id_4                 # band edges
    fft_grid, cell_volume
    n_rtot, n_rmu                    # total r-grid, centroids
    npol, nfreq, nspin, nspinor, nspinor_wfnfile
    nkx, nky, nkz, nk_tot
    nbnd_jax, n_rtot_jax, n_rmu_jax  # round-up-to-n_proc versions

    # derived in __post_init__:
    nelec = b_id_2
    nb_sigma = b_id_3 - b_id_0
    kgrid = (nkx, nky, nkz)
    band_edges = (b0, b1, b2, b3, b4)
```

`Meta` carries the band EDGES only.  Band *windows* come from
`gw.wavefunction_bundle.BandSlices` — the single source of truth.  (A
duplicate `Meta.band_ranges` namespace existed until AD; it was dead code
with a conflicting `sigma` convention and is gone.)

Constructed via `Meta.from_system(wfn, sym, nval, ncond, nband, n_rmu, bispinor)` in the driver.

### 2.3 `BandSlices` + `Wavefunctions` — `src/gw/wavefunction_bundle.py`

The canonical wavefunction container. **Four** sharded copies of ψ_nk(r_μ), one per contraction direction:

| Field | Shape | Spec | Used as |
|---|---|---|---|
| `psi_xn` | `(nk, s, μ_X, n)` | `P(None, None, 'x', None)` | G LHS, χ₀ LHS (direct, conj inside `build_G`) |
| `psi_xr` | `(nk, n, s, μ_X)` | `P(None, None, None, 'x')` | Σ-projection LHS (conjugated) |
| `psi_yr` | `(nk, n, s, μ_Y)` | `P(None, None, None, 'y')` | G RHS, χ₀ RHS |
| `psi_yn` | `(nk, s, μ_Y, n)` | `P(None, None, 'y', None)` | Σ-projection RHS |

Plus `enk (nk, nb_full)` and `occ (nk, nb_full)`, both replicated. `slices: BandSlices` carries local `val / cond / sigma / full / occ` slices and `b0..b4`.

Built via `build_wavefunctions(psi_l_yr, psi_r_yr, ...)` or `build_wavefunctions_from_full(...)`; `Wavefunctions` is registered as a JAX pytree so it threads through `@jax.jit` cleanly.

Accessors: `wfns.xn(s.val)`, `wfns.xr(s.sigma)`, etc.

### 2.4 `WfnLoader` — `services/wfn_loader/src/wfn_loader/loader.py`

**Moved 2026-08-07** (wave-1 service extraction), and the transitional shim that briefly stood at `src/file_io/wfn_loader.py` was deleted by the phase-wide cleanup: that path no longer exists, `from file_io.wfn_loader import WfnLoader` raises, and re-creating the file to green a branch is a red cell at `tests/test_service_path_bootstrap.py::test_the_retired_shim_files_are_gone`. Reach the class as `import wfn_loader` (in a process where the service-path bootstrap has run) or as `from file_io import WfnLoader` / `WFNReader`. Full page: [docs/services/wfn_loader.md](../services/wfn_loader.md).

Reads BerkeleyGW `WFN.h5`. `backend='auto'` (default) picks `eager` (h5py + numpy) or `phdf5` (one collective read through `SlabIO.read_slabs` + on-device unfold), and the two are held **byte-identical**. Symmetry unfolding is the loader's: `SymMaps.get_gvecs_kfull` / `get_cnk_fullzone[_batch]` moved into it, and `SymMaps` keeps the sym tables and the IBZ k/q maps. The legacy `WFNReader` / `PhdfWfnReader` readers (formerly in both `common/` and `file_io/`) were consolidated into this single class; `PhdfWfnReader` is gone and `WFNReader` remains as a back-compat alias (`from file_io import WfnLoader as WFNReader`) used by the GW driver, BSE, and benchmarks.

### 2.5 `SymMaps` — `services/symmetry_maps/` (the door)

> **Read [`symmetry_register.md`](symmetry_register.md) first** if your question
> is "where does this kind of symmetry live". It maps every symmetry operation
> in the tree to its backend and call sites, grouped by OPERATION rather than by
> file, and names the two conventions (`mtrx` vs `mtrx.T`; raw `tnp` vs τ) that
> cause the most damage when confused.

IBZ → full BZ unfolding. Builds the full mesh, k→q maps, spinor SU(2) rotations (Markley quaternion), and fractional-translation phases for non-symmorphic operators. Key methods:

- `get_gvecs_kfull(wfn, nk)` — rotated G-vectors at full-BZ k-point
- `get_cnk_fullzone(wfn, nb, nk)` — rotated coefficients with spinor rotation + TR conjugation
- `get_cnk_fullzone_batch(wfn, band_indices, nk)` — vectorized

Since 2026-08-07 this class, the k-star index map, the sharded q-axis
unfolds, the real-space orbit machinery and the TRS measurement live in
`services/symmetry_maps/` and are reached by `import symmetry_maps` —
never through a submodule path. `src/common/symmetry_maps.py`,
`src/centroid/orbit_syms.py` and `src/common/density_symmetry_check.py`
were forwarding shims only briefly, and the phase-wide cleanup deleted
them the same day; those three paths are gone, their absence is pinned by
`tests/test_service_path_bootstrap.py::test_the_retired_shim_files_are_gone`,
and the old spellings now raise rather than forward. Read
[`docs/services/symmetry_maps.md`](../services/symmetry_maps.md) for the
contract — in particular which conjugation predicate goes with which
operand flavour, which is a 183.61 eV question.

### 2.6 `EPSReader` — `src/file_io/epsreader.py`

Reads `eps0mat.h5` / `epsmat.h5`. Used only by the `epshead` head source (`head_correction.resolve_head_sample`) and BSE.

### 2.7 `SlabIO` — `src/file_io/slab_io.py`

Sharded HDF5 writer with three backends (selected by `SlabIOBackend` enum; the deprecated `use_ffi_io: bool` kwarg/input-key is still coerced for back-compat):

- `PHDF5_FFI` → `_slab_io_ffi.py` (parallel HDF5 via the phdf5 FFI). Each rank writes its own hyperslab; collective MPI-IO writes are the default (`LORRAX_PHDF5_COLLECTIVE_WRITES=0` reverts to independent — same knob, same default as the Python host writer since 2026-07-27), with rank-local replica dedup (`LORRAX_PHDF5_DEDUP_REPLICAS=0` disables). Available on BOTH backends: the C++ core compiles into the CUDA lib and, under `LORRAX_FFI_NO_CUDA`, into the CUDA-free host lib, where the D2H staging collapses to an in-place read of the XLA host buffer (workstream AE).
- `PHDF5_HOST` → `_slab_io_mpi_host.py` (parallel HDF5 via mpi4py + h5py(parallel)). Same per-rank parallel-write semantics, driven from Python. Second CPU tier — for a host lib built without the write handler; needs the mpi4py overlay that the FFI path does not.
- `H5PY_ALLGATHER` → `_slab_io_allgather.py` (all-gather to rank 0 then serial h5py). Last-resort fallback for systems without parallel HDF5; slow at scale, and the gather is the dominant collective in a large run.

The `LorraxConfig.from_input_file` builder routes `slab_io = auto` (the default) UNCONDITIONALLY through a capability-probed router — no other input key gates it. On CPU, `_route_cpu_slab_io` probes `ffi_loader.probe_target('lorrax_phdf5_write', 'cpu')` and picks PHDF5_FFI → PHDF5_HOST (the tier-2 probe really runs `MPI_Init_thread`, so a PMI-mismatched harness demotes instead of dying) → H5PY_ALLGATHER. On GPU, `_route_gpu_slab_io` applies the same two conditions — the CUDA lib exports the write handler, and `_probe_mpi_bootstrap_ffi('CUDA')` says MPI can bootstrap — else PHDF5_HOST/H5PY_ALLGATHER. **Node count is not a condition.** It was until 2026-08-05, when the router declined PHDF5_FFI unconditionally at `SLURM_JOB_NUM_NODES > 1` without probing; that arm generalised an Intel-MPI-on-Frontera launcher misconfiguration to the Cray-MPICH/Shifter GPU path, and was deleted after 16 ranks on 4 Perlmutter nodes wrote and read bit-exactly through PHDF5_FFI with `MPI_Comm_size` asserted (see `docs/architecture/slab_io.md`). Every decision is logged with the tier and, on a demotion, the probe's reason. The deprecated `use_ffi_io` input key: `false` forces H5PY_ALLGATHER (warned), `true` is a no-op, and it is ignored when `slab_io` is explicit.

Used for `zeta_q.h5` and `V_qmunu.h5` (big files), and for `sigma_mnk.h5` via `write_sigma_omega_h5`.

---

## 3. Device mesh and sharding conventions

### 3.1 The one mesh

`gw_jax._build_mesh()` returns a single 2-D mesh:

```python
total = jax.process_count() * jax.local_device_count()
gx = int(sqrt(total)); while total % gx: gx -= 1
Mesh(devices.reshape(gx, total//gx), ('x', 'y'))
```

Most-square factorization. There is **no** 1-D `'bands'` mesh — every pjit/shard_map in the driver uses axes `'x'` and `'y'`. On 4 GPUs this is 2×2; on 16 GPUs, 4×4.

### 3.2 Canonical specs

Centroid-space (μ_X, ν_Y) is the dominant sharding. Flat-k / flat-q arrays are `(nk_tot, …)` with `P(None, …)` on the leading axis.

| Array | Spec | Comment |
|---|---|---|
| ψ_xn | `P(None, None, 'x', None)` | G/χ₀ build |
| ψ_xr | `P(None, None, None, 'x')` | Σ projection LHS |
| ψ_yr | `P(None, None, None, 'y')` | G/χ₀ build |
| ψ_yn | `P(None, None, 'y', None)` | Σ projection RHS |
| Pair density `P_k(μ, ν)` | `P(None, 'x', 'y')` | scalar (spin-traced) |
| Pair density `P_k(s, s', μ, ν)` | `P(None, None, None, 'x', 'y')` | spin-matrix mode |
| `C_q(μ, ν)` | `P(None, 'x', 'y')` | CCT |
| `L_q(μ, ν)` | `P(None, 'x', 'y')` | Cholesky factor |
| `L_q` as tiles | `P(None, 'x', 'y', None, None)` | 2-D blocked algo |
| `Z_q(μ, r)` | `P(None, 'x', 'y')` | ZCT z-chunk |
| `ζ_q(μ, r_XY)` | `P(None, None, ('x','y'))` | triangular-solve output |
| `V_q` flat-q | `P(None, 'x', 'y')` | (nq, μ_X, ν_Y) |
| `χ₀_R` in tau step | `P(None, 'y', 'x')` | **Note**: μ on y, ν on x — matches einsum output |
| G(k) 5D flat-k | `P(None, None, 'x', None, 'y')` | (nk, s, μ_X, s', ν_Y) |
| G/W on 7D (3-D k) | `P(None, None, None, None, 'x', None, 'y')` | reshape inside FFT helper |
| V on 5D (3-D k) | `P(None, None, None, 'x', 'y')` | reshape inside FFT helper |
| W solve reshard | `P(('x','y'), None, None)` | per-q LU via shard_map |
| Σ_k(s, μ, s', ν) | `P(None, None, 'x', None, 'y')` | σ^τ output |
| Σ_k(m, n) | `P(None, 'x', 'y')` | band basis, reduce-scattered |
| `enk`, `occ`, `efermi` | replicated | small |
| `B_q`, `Omega_q` (PPM) | `P(None, 'x', 'y')` | |

### 3.3 Key collective patterns

- **Static COHSEX Σ path** (`gw_jax.sigma_sx/sigma_coh/hartree`): `_convolve(G, W)` uses per-device IFFT/FFT on the replicated 3-D k-axes (via `fft_helpers`) while keeping μ on `'x'` and ν on `'y'`. The final `project(psi_xr, psi_yn, Σ_k)` does two einsum contractions that end up replicated on (k, m, n).

- **χ₀ minimax kernel** (`w_isdf._get_chi_minimax_kernel`): `Gv` and `Gc` use *swapped* μ/ν ⇄ 'x'/'y' assignments (`_Gv_spec` μ=x, ν=y; `_Gc_spec` μ=y, ν=x) so that the final `einsum('Rambn,Rbnam->Rmn')` contracts over the two local axes and leaves output sharded `(μ_Y, ν_X)` — explicit `with_sharding_constraint` on `_chi_R_spec = P(None, 'y', 'x')` prevents XLA from replicating a 23 GB intermediate at Si 4×4×4 60 Ry μ=2400.

- **Dyson solve** (`w_isdf._get_w_solve_fn`): V and χ₀ both arrive as `P(None, 'x', 'y')`. They are padded to a multiple of `mesh.size`, constrained to `P(('x','y'), None, None)` via two `with_sharding_constraint` stages (replicate → q-shard), then a `shard_map` runs `jsp_linalg.lu_factor/lu_solve` per-q inside a `fori_loop`. Output is resharded back to fully-replicated `rep_3d`.

- **Σ^c(ω) project** (`ppm_sigma._make_project_ri_reduce_scatter`): replaces the naive `project_ri(ψ* σ ψ)` with a `shard_map` that uses two `psum_scatter` operations — one over `'x'` scattering the `m` index, one over `'y'` scattering `n`. Same NCCL byte volume as two psums, but output arrives `(m_X, n_Y)`-sharded, so every downstream `coeff·σ` multiply stays local.

- **Cholesky** (`cholesky_2d.cholesky_2d_batched`): operates on tile layout `P(None, 'x', 'y', None, None)`. Panel broadcasts + triangular updates use `lax.psum` over axis `'x'`. Falls back to dense `jnp.linalg.cholesky` when `mesh.size == 1`.

- **Triangular solve** (`isdf_fitting.solve_zeta_from_L_q`): loops over q-batches of size `q_chunk_size` (default 1); gathers `L_q[q0:q1]` to replicated, runs vmapped `solve_triangular` inside a `shard_map` over `('x','y')` flattened as a single scatter axis for the column dimension. Python loop with `donate_argnums` forces sequential GPU execution (fori_loop SPMD-replicates the carry; scan unrolled OOMs).

---

## 4. Flat-k / flat-q convention

Every physics array uses flattened k/q indexing: `(nk_tot, *trail)` with `q_flat = qx*nky*nkz + qy*nkz + qz`. The 3-D `(nkx, nky, nkz)` shape only appears **inside** `common.fft_helpers`:

```python
from common.fft_helpers import make_flat_k_fftn, make_flat_k_ifftn

spec_3d = P(None, None, None, None, 'x', None, 'y')   # (nkx, nky, nkz, ..., μ, ..., ν)
ifft = make_flat_k_ifftn(mesh_xy, kgrid, spec_3d, norm='ortho')

x_R = ifft(x_k)   # callers only see (nk, *trail) ↦ (nk, *trail)
```

The helper reshapes `(nk, *trail) → (nkx, nky, nkz, *trail)`, constrains to `spec_3d`, runs a `custom_partitioning`-wrapped `jnp.fft.fft` along axes `(0, 1, 2)` on each device locally (replicated k-axes guarantee the full FFT is visible), then reshapes back. `norm='ortho'` is used for physics FFTs; `'forward'` for the convolution identity in CCT/ZCT. The driver, `w_isdf`, `ppm_sigma`, and `isdf_fitting` all build their FFTs through this helper — **never** `jnp.fft.fftn` directly on a sharded array.

---

## 5. Data Flow Pipeline

The driver runs in one `main()` in `gw/gw_jax.py`; read it top-to-bottom.

```
cohsex.in
    │  LorraxConfig.from_input_file()
    ▼
WFN.h5 + WFNq.h5 + centroids_frac.h5 + (eps0mat.h5, dipole.h5 optional)
    │  WFNReader, SymMaps, load_centroids
    │  Meta.from_system, BandSlices.from_band_edges
    │  mesh_xy = _build_mesh()
    │  if slab_io is PHDF5_FFI: phdf5_init_mpi() (eager MPI_THREAD_MULTIPLE init;
    │                            PHDF5_HOST warms mpi4py the same way)
    │  ensure_jax_compile_cache()
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ISDF  —  gw_init.prepare_isdf_and_wavefunctions                       │
│ ─────────────────────────────────────────────────────────────────────│
│ Returns an isdf bundle with V_qmunu + Wavefunctions.                  │
│ Two paths:                                                            │
│   restart=True  → load_restart_state_from_h5 (tagged_arrays.h5)       │
│   restart=False →                                                     │
│     isdf_fitting.fit_zeta_chunked_to_h5                               │
│       ├─ load_centroids_band_chunked       (band-chunked FFT at r_μ)  │
│       ├─ compute_pair_density_spin_{traced,matrix}   (both L and R)   │
│       ├─ compute_CCT_from_left_right{_spin_matrix}   (k → R → q conv) │
│       ├─ compute_L_q_from_CCT               (2-D blocked chol)        │
│       └─ z-chunk loop:                                                │
│           ├─ compute_ZCT_from_left_right_zchunk{_spin_matrix}          │
│           ├─ solve_zeta_from_L_q             (triangular solve)       │
│           └─ zeta_io.write_slab('zeta_q', ...)  via SlabIO            │
│     compute_vcoul.compute_all_V_q  (dispatcher → G-flat path)         │
│       └─ v_q_g_flat.compute_all_V_q_g_flat                            │
│           ├─ compute_v_q_per_G  (v(q+G) on per-q WFN.h5 sphere)       │
│           └─ per-q, G-chunked contract → V_qmunu                      │
│     build_wavefunctions(...) → Wavefunctions bundle                   │
└──────────────────────────────────────────────────────────────────────┘
    │
    │  V_q = flatten_V_qmunu(V_qmunu)   (nq, μ, μ) on P(None, 'x', 'y')
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Screening  —  if config.do_screened                                   │
│ ─────────────────────────────────────────────────────────────────────│
│   quad, e_ref = w_isdf.build_static_quadrature(wfns, minimax_config)  │
│   χ₀_q       = w_isdf.compute_chi0(wfns, quad, meta, mesh_xy, e_ref)   │
│                (minimax kernel accumulates χ_R across τ via fori loop)│
│   W_q        = w_isdf.solve_w(V_q, χ₀_q, meta, mesh_xy)                │
│                (per-q LU via shard_map, returns replicated)           │
│ else: W_q = V_q                                                       │
└──────────────────────────────────────────────────────────────────────┘
    │
    │  Gij = _build_Gij(meta, mesh_xy)   (nk, nb_sigma, nb_sigma) occupation proj
    │  static_head_terms = head_correction.compute_static_head_terms_from_sample(...)
    │    iff do_G0 and not use_ppm_sigma
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Static COHSEX                                                         │
│ ─────────────────────────────────────────────────────────────────────│
│   sigma_sx(wfns, Gij, W_q) :   -project[ IFFT(G_occ)·IFFT(W) ·√Nk⁻¹ ]  │
│   sigma_coh(wfns, W_q, V_q):  +½ project[ IFFT(G_all)·IFFT(W-V) ...]   │
│   hartree(wfns, Gij, V_q) :      project[ V(q=0) · ρ ]                 │
│   sig_x = sigma_sx(wfns, Gij, V_q)   # bare-exchange diagnostic      │
│   + static_head_terms_to_kij (exact q→0 band-diagonal shifts)         │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ GN-PPM Σ^c(ω)  —  if config.use_ppm_sigma                             │
│ ─────────────────────────────────────────────────────────────────────│
│   quad_imag = w_isdf.build_imag_quadrature(quad, ωp, minimax_config)  │
│   χ₀(iωp)   = w_isdf.compute_chi0(..., quad_imag)                     │
│   W(iωp)    = w_isdf.solve_w(V_q, χ₀(iωp), ...)                        │
│   ppm       = ppm_sigma.fit_gn_ppm(W(0), W(iωp), V_q, ωp, mesh_xy)     │
│                → B_q, Ω_q, valid_mask_q  (all P(None,'x','y'))        │
│   Σ^c(ω)    = ppm_sigma.compute_sigma_c_ppm_omega_grid(...)           │
│     └─ 4 branches ×   (+ω cond / +ω val / -ω cond / -ω val)           │
│        each branch:                                                   │
│          _build_windows_for_branch   (host-side minimax window build) │
│          _integrate_tau_windows_for_branch                            │
│            └─ for each window (Laplace / crossing / slab):            │
│                 _get_sigma_tau_scan_kernel                            │
│                   lax.scan over τ nodes:                              │
│                     tau_kernel: σ^τ = project[IFFT(G·W_τ)/√Nk]        │
│                     _project_tau_onto_omega: apply e^{iω·τ} kernel    │
│                       onto acc(n_ω, nk, m_X, n_Y)                     │
│          accumulator:                                                 │
│            _ReduceScatterGpuAccumulator   (kij in GPU)                │
│            _StreamedH5Accumulator          (sigma_kij.h5, 1-proc only)│
│   + diagonal SC fixed-point (qsgw_utils.solve_diagonal_sigma_fixed_point)│
│   + scissor fit for out-of-grid bands (scissor.fit_scissor)           │
│   + QSGW Σ^xc construction (qsgw_utils.build_qsgw_sigma_xc_from_h5)   │
│   → sigma_mnk.h5                                                      │
└──────────────────────────────────────────────────────────────────────┘
    │
    │  sigma_total = sig_sx + sig_coh + sig_h
    │  kin_ion = load_kin_ion_submatrix(kin_ion_file, b0, b3)
    │  H_qp = (kin_ion + sigma_total) hermitianized
    │  E_qp, U_qp = jax.vmap(jnp.linalg.eigh)(H_qp)
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Optional self-consistent COHSEX  —  if config.self_consistent         │
│ ─────────────────────────────────────────────────────────────────────│
│   rcrop_nojit Anderson fixed-point on flattened upper-Hermitian Σ:    │
│     _sc_step: diag H_qp → U → G_ij = U f U†  → rebuild Σ_sx+Σ_coh+V_H │
│   converges on |ΔΣ| < 1e-5 with m=3 Anderson history                   │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
GWResults → write_results → eqp.dat / eqp1.dat / sigma_mnk.h5 / qp_rotations.h5
                            + timing.report
```

---

## 6. File Formats

### 6.1 Input — `cohsex.in`

Parsed by `LorraxConfig.from_input_file()`. Canonical reference: [`docs/input_reference.md`](../input_reference.md), generated from the parser by `tools/gen_input_reference.py` and drift-checked in `tools/release_check.sh`. (A prose companion, `docs/docs_gwjax/COHSEX_INPUT.md`, lives in the sandbox repo — not here, which is why the relative link that stood here could never resolve.) Between them they cover every flag (band ranges, ISDF parameters, memory budget, screening knobs, head-correction policy, frequency grid, etc.).

### 6.2 Input — `centroids_frac.h5`

Generated by `centroid.kmeans_cli` (algorithm in `centroid.kmeans_isdf`). Stores fractional centroid coordinates (legacy layouts also supported).

### 6.3 Intermediate — `zeta_q.h5`

Written by `isdf_fitting.fit_zeta_chunked_to_h5` via `SlabIO`.

```
/zeta_q : (n_q_flat, n_rtot, n_rmu)  complex128
          chunks = (1, n_rchunk, n_rmu)
          layout = flat-q:  q_flat = qx*nky*nkz + qy*nkz + qz
```

**Layout note**: dataset is `(nq, n_rtot, n_rmu)` with n_rmu innermost — so each per-r-chunk write spans the full μ axis contiguously. The old `(nq, n_rmu, n_rtot)` layout scattered 480K × 1920-B strips per rank per write and ran 8× slower on Perlmutter pscratch. V_q reads remain contiguous (per-q slab at `(q, 0, 0)` is one block); the downstream kernel transposes on GPU (~50 µs/q).

### 6.4 Intermediate — `V_qmunu.h5` (or returned in-memory)

```
/V_qmunu : (nqx, nqy, nqz, n_rmu, n_rmu)  complex128
           sharded P(None, None, None, 'x', 'y') when in GPU mem
```

### 6.5 Intermediate — `isdf_tensors_{n_rmu}.h5` (tagged-array restart file)

`file_io.tagged_arrays.read_restart_state_from_h5` / `write_restart_state_to_h5`. Lets you skip the zeta-fit stage on a rerun. Contains V_q, ψ bundle arrays, kin_ion submatrix pointers, and related metadata.

### 6.6 Output — `eqp0.dat` / `eqp1.dat`

BGW-compatible text, on the **IBZ wedge** (one block per `wfn.kpoints` entry, k-coordinates in the BGW `(3f13.9,i8)` block header). Written by `gw.eqp_bgw.write_bgw_eqp`, via `assemble_eqp` — one assembly shared by the live driver (`gw_output.write_results`) and the post-hoc CLI (`gw.eqp_bgw.make_eqp_bgw`), so the V_H seam, the mean-field gate, the Z-factor and the formatter each exist once. `eqp1` differs from `eqp0` only in the `E_QP` column (Z-linearized vs zeroth-order; Z=1 in static modes ⇒ identical).

Not to be confused with the **full-BZ** `sigma_diag.dat` / `eqp_g0w0.dat` (`file_io.sigma_output.write_sigma_to_file` / `write_eqp_g0w0`), which use `k-point N:` blocks plus a `# kcrys` line. The two bases are deliberate — see `docs/drivers.md` and `tests/test_eqp_kpoint_basis.py`.

### 6.7 Output — `sigma_mnk.h5` (new format)

Written by `write_sigma_omega_h5` (phdf5-capable). Datasets in eV:

```
/omega_ev          (n_omega,)
/sigma_c_kij_ev    (n_omega, nk, nb_sigma, nb_sigma)  complex128
/sigma_sx_kij_ev   (nk, nb_sigma, nb_sigma)           complex128
/hartree_kij_ev    (nk, nb_sigma, nb_sigma)           complex128
```

### 6.8 Output — `qp_rotations.h5`

`file_io.qp_wfn.write_qp_rotations_h5`. U matrices + eigenvalues for QP rotation.

### 6.9 Sandbox (not this repo) — `manifest.yaml`, `reports/*/report.md`

Run-tracking metadata in the `lorrax_sandbox` superproject.

---

## 7. Entry Points

### 7.1 Console scripts (pyproject.toml)

```toml
lorrax-gw        = "gw.gw_jax:main"
gw_jax           = "gw.gw_jax:main"          # alias
lorrax-centroids = "centroid.kmeans_cli:main"
lorrax-bse       = "bse.bse_isdf:main"
```

### 7.2 Common module invocations

```bash
# Perlmutter canonical (see config/README.md)
lxpre cohsex.in 640                              # centroids + dipole + kin_ion
lxrun python3 -u -m gw.gw_jax -i cohsex.in       # 4-GPU GW

# Local dev
uv run python -m centroid.kmeans_cli 640 --seed 42
uv run python -m psp.get_dipole_mtxels -i cohsex.in
uv run python -m gw.kin_ion_io -i cohsex.in
uv run python -m gw.gw_jax -i cohsex.in

# FFI smoke tests (module-load + lxalloc first)
lxrun python3 tests/bench/slate_batched_test.py
lxrun python3 tests/bench/cusolvermp_eigh_test.py
lxrun python3 -m common.phdf5_write_test
```

---

## 8. Function Call Hierarchy

### 8.1 ISDF fitting

```
prepare_isdf_and_wavefunctions          [gw/gw_init.py]
 ├─ fit_zeta_chunked_to_h5               [common/isdf_fitting.py]
 │   ├─ load_centroids_band_chunked       [common/wfn_transforms.py]
 │   │   └─ read_Gvecs_to_devices           [common/wfn_transforms.py]
 │   ├─ compute_pair_density_spin_{traced,matrix}  [common/isdf_fitting.py]
 │   ├─ compute_CCT_from_left_right[_spin_matrix]  [common/isdf_fitting.py]
 │   │   └─ make_flat_k_{fftn,ifftn}        [common/fft_helpers.py]
 │   ├─ compute_L_q_from_CCT               [common/isdf_fitting.py]
 │   │   └─ cholesky_2d_batched            [common/cholesky_2d.py]
 │   └─ z-chunk loop:
 │       ├─ compute_ZCT_from_left_right_zchunk    [common/isdf_fitting.py]
 │       ├─ solve_zeta_from_L_q             [common/isdf_fitting.py]
 │       └─ SlabIO.write_slab               [file_io/slab_io.py]
 └─ compute_all_V_q  (dispatcher)        [gw/compute_vcoul.py]
     └─ compute_all_V_q_g_flat            [gw/v_q_g_flat.py]
         ├─ compute_v_q_per_G             [gw/compute_vcoul.py]
         └─ ZetaReader.read_slab          [file_io/zeta_reader.py]
```

### 8.2 Screening + static Σ

```
main                                       [gw/gw_jax.py]
 ├─ build_static_quadrature                 [gw/w_isdf.py]
 │   └─ build_static_minimax_window_pair     [gw/minimax_screening.py]
 ├─ compute_chi0 → compute_chi0_minimax     [gw/w_isdf.py]
 │   └─ _get_chi_minimax_kernel (cached)
 │       ├─ build_G                          [gw/greens_function_kernel.py]
 │       └─ flat-k FFTs via fft_helpers
 ├─ solve_w → _get_w_solve_fn (cached)      [gw/w_isdf.py]
 │   └─ shard_map( fori_loop( lu_factor + lu_solve ) )
 ├─ sigma_sx / sigma_coh / hartree          [gw/gw_jax.py]
 │   ├─ build_G                              [gw/greens_function_kernel.py]
 │   ├─ _convolve (G·W/√Nk via IFFT/FFT)     [gw/gw_jax.py]
 │   └─ project                              [gw/wavefunction_bundle.py]
 └─ _compute_static_head                     [gw/gw_jax.py]
     └─ head_correction.resolve_head_sample  [gw/head_correction.py]
         ├─ EPSReader.epshead                [file_io/epsreader.py]
         └─ chi_from_dipole.compute_S_omega  [common/chi_from_dipole.py]
```

### 8.3 GN-PPM Σ^c(ω)

```
main                                       [gw/gw_jax.py]
 ├─ build_imag_quadrature                   [gw/w_isdf.py]
 ├─ compute_chi0 (imag ω)                   [gw/w_isdf.py]
 ├─ solve_w (imag ω)                        [gw/w_isdf.py]
 ├─ fit_gn_ppm                              [gw/ppm_sigma.py]
 │   └─ fit_gn_ppm_from_wc_pair              [gw/minimax_screening.py]
 └─ compute_sigma_c_ppm_omega_grid          [gw/ppm_sigma.py]
     ├─ _prepare_sigma_state                 (fused Fermi / masks)
     ├─ _iter_branches                       (4-branch enumerator)
     └─ _run_sigma_branch (×4)
         ├─ _build_windows_for_branch       (host: window + minimax)
         │   ├─ _build_single_sigma_window  (Laplace)
         │   └─ _build_three_sigma_windows  (Laplace + crossing + slab)
         └─ _integrate_tau_windows_for_branch
             └─ _get_sigma_tau_scan_kernel  (device: scan over τ)
                 ├─ _get_sigma_tau_kernel
                 │   ├─ _build_tau_operands  (Gij, W_τ_q)
                 │   └─ _get_sigma_kij_kernel
                 │       ├─ build_G         [greens_function_kernel.py]
                 │       ├─ _convolve via fft_helpers
                 │       └─ _make_project_ri_reduce_scatter  (reduce-scatter on m,n)
                 └─ _project_tau_onto_omega  (e^{iωτ} kernel ⊗ σ^τ)
     → _ReduceScatterGpuAccumulator  or  _StreamedH5Accumulator
```

### 8.4 Post-processing (all rank-0 host)

```
main                                       [gw/gw_jax.py]
 ├─ solve_diagonal_sigma_fixed_point         [gw/qsgw_utils.py]
 ├─ fit_scissor                              [gw/scissor.py]
 ├─ build_qsgw_sigma_xc_from_h5              [gw/qsgw_utils.py]
 ├─ rcrop_nojit                              [mixing/acceleration.py]    (if self_consistent)
 ├─ write_sigma_omega_h5                     [file_io/sigma_output.py]
 ├─ write_results → write_eqp_table          [gw/gw_output.py, file_io/sigma_output.py]
 └─ write_qp_rotations_h5                    [file_io/qp_wfn.py]
```

---

## 9. Code Locations Quick Reference

| Task | File : function |
|------|-----------------|
| **Main driver** | `gw/gw_jax.py : main` |
| **Parse cohsex.in** | `gw/gw_config.py : LorraxConfig.from_input_file` |
| **Chunk auto-sizing** | `gw/gw_init.py : compute_optimal_chunks`, `prepare_isdf_and_wavefunctions` |
| **Minimax / sigma quad config** | `gw/minimax_config.py : MinimaxConfig`, `SigmaQuadratureConfig` |
| **Build `Meta`** | `common/meta.py : Meta.from_system` |
| **Build wavefunction bundle** | `gw/wavefunction_bundle.py : build_wavefunctions_from_full` |
| **Load WFN.h5** | `services/wfn_loader/ : WfnLoader` (`import wfn_loader`, after `ffi._services.ensure_on_path()`); band-chunked FFT `common/wfn_transforms.py` |
| **Symmetry unfolding** | `symmetry_maps : SymMaps.get_cnk_fullzone[_batch]` (service door) |
| **Flat-k FFT helpers** | `common/fft_helpers.py : make_flat_k_fftn`, `make_flat_k_ifftn` |
| **Pair density (spin-traced)** | `common/isdf_fitting.py : compute_pair_density_spin_traced` |
| **CCT matrix** | `common/isdf_fitting.py : compute_CCT_from_left_right[_spin_matrix]` |
| **Cholesky factorization** | `common/isdf_fitting.py : compute_L_q_from_CCT` → `common/cholesky_2d.py : cholesky_2d_batched` |
| **ZCT matrix (z-chunk)** | `common/isdf_fitting.py : compute_ZCT_from_left_right_zchunk` |
| **ζ solve** | `common/isdf_fitting.py : solve_zeta_from_L_q` |
| **Full ζ pipeline** | `common/isdf_fitting.py : fit_zeta_chunked_to_h5` |
| **Compute V_q** | `gw/compute_vcoul.py : compute_all_V_q` → `gw/v_q_g_flat.py : compute_all_V_q_g_flat` (charge) / `gw/v_q_bispinor.py : compute_V_q_bispinor_g_flat_to_h5` (bispinor) |
| **Voronoi MC for q=0** | `gw/vcoul.py : compute_q0_averages`, or `gw/coulomb/base.py : sample_minibz_qpoints` + each kernel's `q0_average` |
| **Coulomb kernel + truncation** | `gw/coulomb/` — `get_kernel(meta.sys_dim)` → `Bulk3D` (3) \| `Slab2D` (2) \| `Box0D` (0). Production `v(q+G)` enters at `gw/compute_vcoul.py : compute_v_q_per_G`. *(This row named `gw/vcoul.py : compute_V_qfullG_for_q` until 2026-08-06; that function does not exist anywhere in `src/`.)* **On `feat/vcoul-consolidation-2026-08-06` (unmerged)** `compute_v_q_per_G` becomes a thin dispatcher over one `coulomb/base.py : v_qG_table` driver: each kernel contributes only `_v_bare_per_q` (the dimension's bare formula at one q), and the per-q loop, the `vcoul_cutoff_ry` mask, the G=0 head-slot injection and the `(n_q, ngkmax)` float64 contract live once — so the cutoff, head injection and batching that used to exist only in `compute_vcoul` become available in every dimension, and `Box0D` refuses q≠0 rather than returning a wrong number. |
| **χ₀ minimax kernel** | `gw/w_isdf.py : _get_chi_minimax_kernel`, `compute_chi0_minimax` |
| **W Dyson solve** | `gw/w_isdf.py : solve_w`, `_get_w_solve_fn` |
| **Static minimax lookup** | `gw/minimax_screening.py : build_static_minimax_window_pair` |
| **Build G** | `gw/greens_function_kernel.py : build_G` |
| **Σ band projection** | `gw/wavefunction_bundle.py : project`, `project_ri` |
| **Reduce-scatter projection** | `gw/ppm_sigma.py : _make_project_ri_reduce_scatter` |
| **Static Σ_SX / Σ_COH / V_H** | `gw/gw_jax.py : sigma_sx`, `sigma_coh`, `hartree` |
| **q→0 head resolution** | `gw/head_correction.py : resolve_head_sample` |
| **Exact static head terms** | `gw/head_correction.py : compute_static_head_terms`, `static_head_terms_to_kij` |
| **Dipole S(ω)** | `common/chi_from_dipole.py : compute_S_omega` |
| **GN-PPM fit** | `gw/ppm_sigma.py : fit_gn_ppm` → `gw/minimax_screening.py : fit_gn_ppm_from_wc_pair` |
| **Σ^c(ω) driver** | `gw/ppm_sigma.py : compute_sigma_c_ppm_omega_grid` |
| **σ^τ kernel** | `gw/ppm_sigma.py : _get_sigma_tau_kernel`, `_get_sigma_tau_scan_kernel` |
| **Diagonal fixed-point** | `gw/qsgw_utils.py : solve_diagonal_sigma_fixed_point` |
| **Scissor extrapolation** | `gw/scissor.py : fit_scissor` |
| **QSGW Σ^xc** | `gw/qsgw_utils.py : build_qsgw_sigma_xc_from_h5` |
| **Anderson mixing (SC)** | `mixing/acceleration.py : rcrop_nojit`, `hermitian_to_upper_flat` |
| **Write sigma_mnk.h5** | `file_io/sigma_output.py : write_sigma_omega_h5` |
| **Write eqp0.dat / eqp1.dat** (IBZ wedge) | `gw/eqp_bgw.py : assemble_eqp`, `write_bgw_eqp` |
| **Write sigma_diag.dat / eqp_g0w0.dat** (full BZ) | `file_io/sigma_output.py : write_sigma_to_file`, `write_eqp_g0w0` |
| **SlabIO (phdf5 writer)** | `file_io/slab_io.py : SlabIO` (backends in `_slab_io_ffi.py` / `_slab_io_allgather.py`) |
| **Centroid selection** | `centroid/kmeans_cli.py : main` (algorithm: `centroid/kmeans_isdf.py`) |
| **Dipole generation** | `psp/get_dipole_mtxels.py : main` |
| **kin_ion generation** | `gw/kin_ion_io.py : main` |
| **FFI loader** | `ffi/common/ffi_loader.py : phdf5_init_mpi`, `register_handlers` |
| **cuSOLVERMp eigh** | `ffi/cusolvermp/eigh.py`, smoke test `tests/bench/cusolvermp_eigh_test.py` |
| **SLATE eigh / Cholesky / trsm** | `ffi/slate/` , tests `tests/bench/slate_*_test.py` |
| **phdf5 R/W** | `ffi/phdf5/` , benchmarks `common/phdf5_*.py` |

---

## Next Steps

- Physics / theory: [Core ISDF and GW theory](../theory/physics.md)
- Memory model: [`MEMORY_MODEL.md`](memory-model.md)
- Environment / Perlmutter: [`environment/overview.md`](../environment/overview.md), [`environment/machines/perlmutter.md`](../environment/machines/perlmutter.md), [`../config/README.md`](../../config/README.md)
- FFI internals: [`../src/ffi/AGENTS.md`](../../src/ffi/AGENTS.md)
- GN-PPM Σ details: see developer notes under `docs/dev/notes/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md`
- Current BGW-vs-LORRAX status: see developer notes under `docs/dev/progress/SIGMA_FREQ_AUDIT_STATUS.md`
- Agent todos: `docs/dev/notes/AGENT_TODO.md` is **superseded** and should not be
  worked from — it describes a `src/isdf/` package layout the tree no longer has,
  and one of its suggestions is now forbidden by a test. It carries a banner
  explaining why. Current work items live in the campaign ledgers, not here.
