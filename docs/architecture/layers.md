# The three levels

*Companion to the [codebase map](../codebase.md), which says where modules
are. This page says **what a module is allowed to know about**, and why.*

A module's level is decided by its vocabulary, not by its directory:

| | level | it may know about | it may not |
|---|---|---|---|
| **L1** | **physics** | bands, q-points, ζ, Σ, symmetry, decks, file formats | — |
| **L2** | **numerical routines** | matrices, quadrature nodes, residuals, convergence | anything physical |
| **L3** | **substrate** | devices, meshes, processes, native libraries, files | anything mathematical |

**Imports run downhill only: L1 → L2 → L3.** `tests/test_layering.py` holds
the map and fails when a rule breaks ([§6](#6-the-gate)). Everything under
`src/` that the map does not name is L1: physics is the bulk, so a new module
has to argue its way down.

L3 is called the *substrate* because the other candidate words already mean
something else in this tree: a *backend* is which native library answered, a
*kernel* is an L1 physics kernel (`ppm_tau_kernel`), a *platform* is CPU or
CUDA, and a *primitive* is `jax.core.Primitive`.

**Services.** The gate scans `src/` and every `services/*/src`; a module's
name is relative to its source root. `lxkit` and `distrib_la` are L3 packages
(`lxkit.deck_doctor`, which parses a GW deck, is L1). `wfn_loader`,
`zeta_loader`, `symmetry_maps`, `vcoul` and `minimax` take the L1 default.
Levels order modules within one distribution unit. An import of a service's
top-level package is not ranked; rule 6 governs it instead: lorrax reaches a
service through its door and nowhere else.

---

## 1. L1 — physics

**The drivers**, the modules a physicist opens: `gw/gw_jax.py` ·
`bse/bse_jax.py` · `bse/exciton_bands.py` · `bandstructure/htransform.py` ·
`gw/kin_ion_io.py` · `gw/downfold_cli.py` · `centroid/kmeans_cli.py` ·
`psp/{run_nscf,run_sternheimer,get_DFT_mtxels,get_dipole_mtxels,kpm_dos,orbital_magnetization,finite_q_head_interp}.py` ·
`bse/{bse_feast,bse_pseudopoles,bse_w_exact,bse_kpm}.py` · `gw/eqp_bgw.py` ·
`gw/compute_vcoul_0d.py` · `postprocess/rotate_wfn_to_qp.py`.

**The physics kernels**: everything in `gw/`, `bse/`, `psp/`, `isdf/`,
`bandstructure/`, most of `centroid/`, the format readers and writers in
`file_io/`, and the physics-aware modules in `common/` (`zeta_projection`,
`wfn_transforms`, `gvec_fft_box`, `psi_G_store`, `kq_mapping`, `meta`,
`units`, `gamma_matrices`, `bispinor_init`, `chi_from_dipole`, …).

**A driver should contain only physics on inspection** (owner). Rule 1 makes
that concrete: a driver imports none of `jax.sharding`,
`jax.experimental.shard_map`, `multihost_utils`, `mesh_utils` or `jax._src`,
nor `shard_map` / `Mesh` / `NamedSharding` / `PartitionSpec` / `make_mesh`
off `jax`, at module scope or lazily. Fifteen drivers carry none. Six carry
one import each, and `tests/test_layering.py::_DRIVER_PLUMBING_BUDGET` pins
that count:

- `bse/exciton_bands.py` is a driver and a library: it owns the
  interpolated-band assembly its own CLI consumes.
- `bandstructure/htransform.py` is the fH interpolation library with a CLI.
- `bse/{bse_feast,bse_pseudopoles,bse_w_exact,bse_kpm}.py` each import
  `jax.sharding` for a `mesh_xy: Mesh` annotation.

The budget is a ratchet: exceeding it fails, and so does coming in under it
without lowering the table, so a budget cannot become a licence.

An L1 **library** that must read the environment does so only through a
module-level `resolve*` function, and only the variables pinned to it
(rule 2b):

| module | variables |
|---|---|
| `gw.sigma_plan` | `LORRAX_SIGMA_PLAN` |
| `gw.sigma_box_plan` | `LORRAX_UNIFORM_RULE_TRACE` |
| `bandstructure.htransform` | `LORRAX_EXTRA_RANK_PAD` |
| `bandstructure.bse_setup` | `LORRAX_FACE_TO_BATCH_ROUTE`, `LORRAX_FI_FSHOULDER_TOL` |

---

## 2. L2 — numerical routines

`solvers/` (Davidson, Lanczos, Chebyshev, MINRES, contour quadrature, KPM DOS,
pseudobands, subspace projectors, the Sternheimer preconditioner) · `mixing/`
(fixed-point acceleration) · `common/rank_criterion.py` (the pseudo-inverse
truncation criterion) · `common/spectral_closure.py` (where that truncation
may land) · `common/pivoted_cholesky.py` (greedy row selection and its
numerical certificates) · `centroid/kmeans_isdf.py` (the density-weighted
Lloyd loop).

**The test for L2: could this module be lifted into another physics code
unchanged?** A Lanczos that knows what a band index is has failed it.

Two rules follow, and both are enforced:

* **An L2 module reads no environment** (rule 2). A solve that behaves
  differently because of an exported variable is not a function of its
  arguments. Dials are parameters. Where a dial has to come from outside, the
  pattern is one level down: `common/contract_bands.py` makes no `os.environ`
  call and consumes `ffi.mklblas.gemm.GATE`, a typed capability object.
* **An L2 module imports no L1 module** (rule 3). One exception survives
  ([§5](#5-the-sanctioned-exceptions), R2).

---

## 3. L3 — substrate

**Process bootstrap**: `runtime` and `runtime.{aot_memory, padding,
production_stream, xla_memory, jax_support, pjrt_log_filter, source_closure,
network_env, env_flags}`, plus `common/grouped_layout.py`.
`runtime.initialize_communicator_stack()` is the single startup entry point.
In order it installs the fail-fast excepthook, seals the source closure, sets
the environment before jax reads it, selects the CPU-collectives transport,
runs `jax.distributed`, picks GPU or CPU, enforces the JAX generation, builds
the run's mesh with every communicator created, enforces the required FFI
backends, enables the compile cache, and reports every choice on rank 0. It
imports jax only inside function bodies.

**Distribution**: `common/collectives.py` is the cross-process layer and
`centroid/distribution.py` its one policy client. A caller of `collectives`
need not know what a `Mesh`, `NamedSharding`, `shard_map` or
`multihost_utils` is. The two calls a driver wants are `prepare_mesh()` (the
run's mesh with every communicator it will need already created) and
`gather_k_blocks()` (run this per-k block on every k and return all of them).

**Movement primitives**: `common/contract_bands.py` (the staged band
projection and reshard), `common/staged_reshard.py` (its movement-only
sibling), `common/sharding_fit.py` (a `PartitionSpec` that is legal for the
extents in hand).

**FFT front doors**: `common/fft_helpers.py`, the flat-k transform and the
k-convolution router's doors re-exported from `ffi.fft`
([FFI layer](ffi_layout.md#k-convolution-router-and-the-mathdx-family)), plus
the `shard_map`-interior `local_*fftn3` aliases.

**jax glue and instrumentation**: `common/{jax_compile_cache, jax_profile,
timing, progress, gpu_utils, async_io, sanity}.py`, and the two version
shims `common/shard_map.py` (which symbol, which kwarg spelling) and
`common/vma.py` (which spelling marks a loop carry device-varying).

**Native libraries**: the whole `ffi` package, meaning location, probing,
gating and dispatch. `ffi/gate.py` owns grammar, platform, probe and
announce-or-refuse for every gated capability. The `lxkit` and `distrib_la`
services are L3 packages for the same reason.

**Sharded-file transport**: `file_io/{slab_io, _slab_io_ffi, _slab_io_serial,
commit_state, paths, hdf5_owner, h5_journal, io_timing}.py`. The format
readers above it (`epsreader`, `mf_header`, `sigma_output`, `tagged_arrays`,
…) are L1, because they know what a band and a ζ are.

---

## 4. Contested assignments

1. **`common/contract_bands.py` and `common/staged_reshard.py`: L3, despite
   "bands" in the name.** Every policy they encode is about the machine: axis
   order, de-promotion, divisibility refusal, which GEMM backend answered.
2. **`centroid/kmeans_isdf.py`: L2.** Lloyd's algorithm under a metric tensor
   lifts into any code unchanged. Its orbit canonicalisation comes through the
   `symmetry_maps` door, which rule 6 governs. The registered fix is to inject
   the orbit map as a parameter, which is a signature change.
3. **Pivoted Cholesky: split.** `centroid/pivoted_cholesky.py` is L1: it owns
   ψ-based Gram construction, ISDF candidate policy and reporting. The greedy
   recurrence and its rank/PSD certificates are `common/pivoted_cholesky.py`
   at L2, which knows only matrices, group labels, active rows and a
   caller-supplied mesh. Centroid construction and GW downfolding both call
   the L2 owner directly (`test_pivoted_cholesky_selection_has_one_l2_owner`).
4. **`solvers/sternheimer_solve.py`: filed L2, and it fails its own test.** It
   applies `psp.dft_operators.apply_H_k_from_G` in its matvec. It stays at L2
   with the violation named (R2), because reclassifying it would hide the
   clean L2 CG core inside it.
5. **`common/sanity.py`: L3, though it reads as physics.** Each check is a
   cheap array reduction plus one `device_get` and knows nothing about what
   the numbers mean.
6. **`file_io/`: split, not assigned.** A sharded-slab transport (L3) and a
   shelf of BerkeleyGW/QE format readers (L1).
7. **`gw/gw_config.py`: L1.** It is the GW deck parser (`read_lorrax_input`,
   `LorraxConfig`). The XLA-memory policy and the boolean-env grammar live at
   L3 in `runtime.xla_memory` and `runtime.env_flags`; `gw_config`
   re-exports them, so the substrate never imports the deck parser.
8. **`bandstructure/htransform.py`: a driver and a library in one file.** The
   reusable centroid Galerkin fit lives in `isdf/galerkin.py`; htransform
   keeps the fH interpolation and its deck/environment adapter.
9. **`runtime/padding.py`: L3, though it is arithmetic.** It exists only
   because a mesh axis has to divide an extent. It owns logical-to-carrier
   receipts, exact-zero producer padding, consumer masks and spec-derived
   divisors ([Mesh-padded axes](padding.md)).
10. **`common/vma.py`: L3, not L2.** It contains no arithmetic. Its subject is
    mesh axes and whether a value may differ per device, which is L3's
    vocabulary. Between two levels that both fit, L3 is the tighter
    assignment, since L3 may import nothing above itself.
11. **Minimax: a service plus two L1 modules.** The physics-free quadrature
    solver is the `minimax` service (reached through its door).
    `gw/minimax_config.py`, which picks a target error for a screening
    integral, and `gw/minimax_screening.py`, which extracts PPM poles, are L1.

---

## 5. The sanctioned exceptions

Each exception is listed in `tests/test_layering.py` with its reason and is
asserted to be still needed: an exception that outlives its violation fails
the suite. An import of a service's top-level package is not an exception:
`upward_edges()` does not rank it, and rule 6 governs it.

| # | exception | rule | why it is still here |
|---|---|---|---|
| **R2** | `solvers.sternheimer_solve` → `psp.dft_operators` | 3 | Is this an L2 CG solve or an L1 Sternheimer kernel? Split `SternheimerOp`'s operator out, or move the file to `psp/`. Physics decision. |
| **R4** | `mixing.acceleration` sets `JAX_ENABLE_X64` at module scope | 2, 5 | Its consumer (`gw.sc_iteration`) imports it lazily after bootstrap, but a bare `import mixing.acceleration` in a fresh process would otherwise run the accelerator in f32 silently. Physics decision. |
| — | `gw/__init__.py` sets `JAX_ENABLE_X64` | 5 | Inert for both GW drivers, whose startup call owns x64 (`runtime.set_x64_on_imported_jax`; a resolved `False` refuses, override `LORRAX_ALLOW_X64_OFF=1`). Kept for import paths with no bootstrap. |
| — | `centroid/kmeans_plot.py` sets `MPLBACKEND=Agg` | 5 | Not a compute knob; a plotting helper choosing a headless renderer is the right owner. |
| — | `ffi.cusolvermp.{batched, eigh, context}` reach past the `distrib_la` door (5 + 1 + 1 edges) | 6 | `ffi/cublasmp/batched.py` takes `get_or_init_context` from `ffi.cusolvermp`, and cuBLASMp is a future `gemm` service, not `distrib_la`. The edges go when that service is extracted. |

**Mesh construction (rule 4).** Only these modules may call `Mesh(`:
`common.collectives` (`resolve_mesh`, `single_device_mesh`), `runtime`
(`RuntimeStack.reshape`, which then calls `prepare_mesh`),
`bse.bse_ring_comm` (`create_mesh_xy`, which also warms the `impl=mpi`
cliques the BSE Lanczos needs), `ffi.cpp.gate_one_odr` (a build gate that
builds a CPU mesh inside a CUDA process on purpose) and `lxkit.deck_doctor`
(a one-device provider probe). Inside `src/bse/` only `bse_ring_comm` builds
a mesh, and no BSE driver gives `--px`/`--py` a default, so an omitted flag
means the run's mesh and never 1×1.

---

## What NOT to unify

These refusals are load-bearing; each was reached by measurement.

1. **Do not merge `runtime.nccl_warmup` with `collectives.warm_mesh_cliques`.**
   `warm_mesh_cliques` creates each `impl=mpi` clique from a jit small enough
   that XLA runs it inline on the calling thread, which satisfies
   `MPI_Is_thread_main` at communicator creation. `nccl_warmup` pays
   `ncclCommInitRank` topology discovery, which has no thread constraint.
   Unify the call site (`prepare_mesh()`), never the bodies.
2. **Do not build a generic `shard_map` wrapper.** `in_specs`/`out_specs` are
   the distributed algorithm; a wrapper taking them as arguments abstracts
   nothing and hides the one thing the reader came to read. `contract_bands`
   shows the right abstraction is specific: it owns one named pattern, which
   is why it can carry a contract document.
3. **Do not collapse the boolean-env grammar into the gate grammar.**
   `runtime.env_flags` answers "did someone turn this off", with a stated
   default. `ffi.gate.MODE_SPELLINGS` and `Gate.mode` answer "which of my
   declared modes is this" (`off`, `on`, and `auto` for an optional
   accelerator with a certified fallback) and announce a token they do not
   recognise. Any unified rank resolver must keep the launcher variables
   ahead of `jax.process_index()`, which initialises the XLA backend.
4. **Do not give `ffi/` a shared C++ handler base.** The NVRTC-compiled
   mathdx kernel family with its cubin cache, the FFTW3-ABI advanced-layout
   plans with OpenMP chunking, and a scratch-free BLAS call share an FFI
   signature and nothing else.
5. **Do not force the bench drivers in `tests/bench/` through
   `resolve_mesh`.** They build meshes differently on purpose:
   `tests/bench/profile_batched.py` parses `--mesh 2x2` and exits on a
   mismatch, which is right for a benchmark that sweeps geometries.

**Bench drivers live in `tests/bench/`**: argv-driven scripts that pytest does
not collect, run as `python3 tests/bench/<name>.py` with `src/` on
`PYTHONPATH`. A module under `src/` with a bench-shaped name (`test_*`,
`*_test`, `*_bench`, `benchmark*`, `profile_*`, `.tests.`, `.archive.`) fails
the gate.

---

## 6. The gate

`tests/test_layering.py` is pure AST: it imports neither jax nor anything
under `src/`, so it runs on a login node, in a container and in CI
(`python3 tests/test_layering.py` also runs it without pytest).

| rule | what fails it |
|---|---|
| 1 | a budgeted L1 driver imports jax plumbing, at module scope or lazily, beyond its budget, or comes in under budget without the budget being lowered |
| 2 | an L2 module touches `os.environ` |
| 2b | a budgeted L1 library reads a variable not pinned to it, or reads one outside a module-level `resolve*` function |
| 3 | a module imports one at a lower-numbered level (an edge into a service's top-level package is not ranked) |
| 4 | `Mesh(` is called outside the mesh owners; in `src/bse/`, a second mesh builder or a `--px`/`--py` default |
| 5 | a module that is not `runtime` and not an entry point writes an environment variable at import time |
| 6 | a `src/` module imports a service past its top-level package |

Map hygiene: every named module and package exists, no module is in two
levels, every module gets a level, each level has at least four modules, and
every exception is still needed.

**Every scanner has a red twin.** Each `test_*_can_fail` feeds the same scanner
function a source that violates its rule. Re-implementing the scan inside the
twin would test the twin, not the instrument.
