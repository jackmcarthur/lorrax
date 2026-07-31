# The three levels

*Companion to [`codebase.md`](codebase.md), which says where things are. This
says **what a module is allowed to know about**, and why.*

LORRAX has three levels. A module's level is decided by its vocabulary, not by
its directory:

| | level | it may know about | it may not |
|---|---|---|---|
| **L1** | **physics** | bands, q-points, ζ, Σ, symmetry, decks, file formats | — |
| **L2** | **numerical routines** | matrices, quadrature nodes, residuals, convergence | anything physical |
| **L3** | **substrate** | devices, meshes, processes, native libraries, files | anything mathematical |

**Imports run downhill only: L1 → L2 → L3.** Nothing else is allowed, and
`tests/test_layering.py` fails when it happens.

Today: **141 L1, 18 L2, 56 L3**, plus 26 bench/smoke drivers that live under
`src/` and are exempt (see [below](#the-26-exempt-bench-drivers)). Four upward
edges survive, all named in [§5](#5-the-sanctioned-exceptions).

---

## Why "substrate" and not "backend", "platform", "primitives" or "kernels"

The owner asked for a better name than "primitive wrappers" and invited
candidates. **Substrate** wins on one criterion — it is the only one that is
not already taken by something else in this tree:

Measured with `grep -rioE '\b<word>s?\b' --include='*.py' src/` on
2026-07-31:

| candidate | occurrences under `src/` | why it loses |
|---|---:|---|
| backend | 708 | `SlabIOBackend`, `resolve_backend`, `eigh_backend_choices`, `LORRAX_WFN_BACKEND`, `jax.default_backend()`. "The backend" already means "which native library answered", which is *one thing inside* this level. |
| kernel | 415 | `ppm_tau_kernel`, `greens_function_kernel` — "physics kernel" is the project's own word for **L1**. Naming L3 "kernels" inverts the vocabulary. |
| platform | 405 | `Gate(platforms=("cpu","CUDA"))`, `mesh_platform`, `mesh_ffi_platform`, `JAX_PLATFORMS`. Already means "CPU or CUDA". |
| primitive | 79 | collides with `jax.core.Primitive`, and `docs/dev/staged_reshard_primitive.md` uses it for `contract_bands` — a *pattern*, one level up from what it would name. |
| **substrate** | **2** | one is prose in `gw/gflat_memory_model.py:4`, not an identifier; the other is `runtime/xla_memory.py:3`, written under this name on the day it was chosen. **Zero collisions.** |

It also reads right in the sentence the name has to survive:

> *"That belongs in the substrate, not in the physics driver."*

and in its negation, which is the one that gets said in review:

> *"`resolve_mesh` is substrate. A ζ-projection factory calling it to create
> communicators is the substrate being driven from L1, and that is the bug."*

---

## 1. L1 — physics

**The drivers** — what a physicist opens:

`gw/gw_jax.py` · `bse/bse_jax.py` · `bse/exciton_bands.py` ·
`bandstructure/htransform.py` · `gw/kin_ion_io.py` · `centroid/kmeans_cli.py` ·
`psp/{run_nscf,run_sternheimer,get_DFT_mtxels,get_dipole_mtxels,kpm_dos,orbital_magnetization,finite_q_head_interp}.py` ·
`bse/{bse_feast,bse_pseudopoles,bse_w_exact,bse_kpm}.py` · `gw/eqp_bgw.py` ·
`postprocess/rotate_wfn_to_qp.py`

**The physics kernels** — everything in `gw/`, `bse/`, `psp/`, `isdf/`,
`bandstructure/`, most of `centroid/`, the format readers and writers in
`file_io/`, and the physics-aware modules in `common/`
(`zeta_projection`, `wfn_transforms`, `symmetry_maps`, `gvec_fft_box`,
`psi_G_store`, `kq_mapping`, `coulomb_sphere`, `meta`, `units`,
`gamma_matrices`, `bispinor_init`, `chi_from_dipole`,
`density_symmetry_check`).

**The acceptance test for a driver** is the one the owner stated: *it should
appear to contain only physics on inspection*. Concretely — no `jax.sharding`,
no `shard_map`, no `multihost_utils`, no `Mesh(`, no `os.environ`.

**8 of the 20 gated drivers pass all three today** (`gw/gw_jax.py`,
`bse/bse_jax.py`, `centroid/kmeans_cli.py`, `psp/run_nscf.py`,
`psp/finite_q_head_interp.py`, `gw/eqp_bgw.py`, `gw/compute_vcoul_0d.py`,
`postprocess/rotate_wfn_to_qp.py`), and **14 of 20 carry no jax plumbing at
all**. `gw/gw_jax.py` and `bse/bse_jax.py` are the existence proof;
`gw/kin_ion_io.py` is the proof that a driver can be *brought* there — 6 raw
jax imports and 4 module-scope env writes on 2026-07-30 16:47, and 0 and 0 now.
Its three surviving `os.environ` reads are one launcher-mismatch check
(`JAX_PROCESS_COUNT` / `JAX_NUM_PROCESSES` / `SLURM_NTASKS`, refusing when the
launcher advertises P tasks and jax joined a world of 1), which is a check
worth keeping — only its resolver is duplicated from `runtime`.

`tests/test_layering.py::_DRIVER_PLUMBING_BUDGET` holds each driver's current
count. It is a **ratchet**: exceeding it fails, and so does coming in *under*
it without editing the table, so a budget cannot quietly become a licence.

---

## 2. L2 — numerical routines

`solvers/` (Davidson, Lanczos, Chebyshev, MINRES, contour quadrature, KPM DOS,
pseudobands, subspace projectors, Sternheimer preconditioner) · `mixing/`
(Anderson, CROP, rCROP) · `common/minimax.py` (minimax quadrature) ·
`common/cholesky_2d.py` (blocked Cholesky) · `common/rank_criterion.py` (the
pseudo-inverse truncation criterion) · `centroid/kmeans_isdf.py` (the
density-weighted Lloyd loop).

**The test for L2 is: could this module be lifted into another physics code
unchanged?** A Lanczos that knows what a band index is has failed it.

Two rules follow, and both are enforced:

* **An L2 module reads no environment.** A Davidson solve that behaves
  differently because of an exported variable is not a function of its
  arguments and the caller cannot see why. Dials are **parameters**. Where a
  dial genuinely has to come from outside, the model already exists one level
  down: `common/contract_bands.py` makes no `os.environ` call at all — it
  consumes `ffi.mklblas.gemm.GATE`, a typed capability object. *A kernel taking
  a `Gate` rather than a string is the target state.*
* **An L2 module imports no L1 module.** Two exceptions survive, both listed
  in §5, and both are really "this file is L1 wearing an L2 name".

---

## 3. L3 — substrate

**Process bootstrap** — `runtime/` (`__init__`, `aot_memory`, `padding`,
`xla_memory`). `initialize_communicator_stack()` is the single startup entry
point: env before `import jax`, failfast excepthook, CPU-collectives transport,
`jax.distributed`, GPU-or-CPU, the run's mesh with every communicator already
created, the compile cache, and a rank-0 report of every choice where more than
one was possible. It imports jax **only inside function bodies**, deliberately.

**Distribution** — `common/collectives.py` is *the* cross-process layer, and
`centroid/distribution.py` is the one policy client above it. Nothing in
`collectives` requires the caller to know what a `Mesh`, a `NamedSharding`, a
`shard_map` or `multihost_utils` is; that is the point. The two calls a driver
actually wants are `prepare_mesh()` ("give me the run's mesh with every
communicator it will need already created") and `gather_k_blocks()` ("run this
per-k block on every k and hand me all of them").

**Movement primitives** — `common/contract_bands.py` (the staged band
projection + reshard), `common/staged_reshard.py` (its movement-only sibling),
`common/sharding_fit.py` (a `PartitionSpec` that is *legal* for the extents in
hand).

**FFT kernels** — `common/fft_helpers.py`, which now gets its gate and its two
bodies from `ffi.mklfft`.

**jax glue and instrumentation** — `common/{jax_compile_cache, jax_profile,
timing, progress, gpu_utils, async_io, sanity}.py`.

**Native libraries** — all of `ffi/`: location, probing, gating, dispatch.
`ffi/common/gate.py` owns four things that used to be four drifting copies —
grammar, platform, probe, announce-or-refuse.

**Sharded-file transport** — `file_io/{slab_io, _slab_io_ffi,
_slab_io_allgather, _slab_io_mpi_host, paths}.py`. Note the split inside
`file_io/`: the *transport* is L3, the *format readers* above it
(`wfn_loader`, `zeta_loader`, `epsreader`, `mf_header`, `sigma_output`,
`tagged_arrays`, …) are L1, because they know what a band and a ζ are.

---

## 4. The ten calls that were genuinely ambiguous

A level assignment nobody argued is a level assignment nobody will keep.

1. **`common/contract_bands.py` and `common/staged_reshard.py` — L3, despite
   "bands" in the name.** They read as mathematics (a projection, a reshard)
   but every policy they encode is about the machine: axis order, de-promotion,
   divisibility refusal, which GEMM backend answered. `contract_bands` reads no
   environment and consumes a typed `Gate`. That is L3 behaviour with an L1
   noun in the name; the noun loses.
2. **`centroid/kmeans_isdf.py` — L2, with a named exception.** Lloyd's
   algorithm under a metric tensor is textbook clustering and would lift into
   any code unchanged. Its one lazy import of `centroid.orbit_syms` folds a
   real-space point group into the assignment step, and *that* is physics. The
   fix is to inject the orbit map as a parameter; that is a signature change,
   so it is a request, not a landing.
3. **`centroid/pivoted_cholesky.py` — L1, not L2.** Tempting as "pivoted
   Cholesky, a numerical routine", but it imports `file_io`,
   `common.symmetry_maps`, `common.meta`, `common.wfn_transforms` and `isdf`.
   It prunes ISDF *candidate points* using ψ. The algorithm inside is L2; the
   module is not, and pretending otherwise would have put five upward edges
   under an exception list.
4. **`solvers/sternheimer_solve.py` — filed L2, and it fails its own test.**
   `from psp.dft_operators import apply_H_k_from_G` at module scope, used in
   the matvec. A solver that applies a *specific* Hamiltonian is not
   physics-agnostic. Left at L2 with the violation named rather than silently
   reclassified to L1, because the reclassification would hide the fact that
   there is a clean L2 CG core inside it.
5. **`common/cholesky_2d.py` — L2, even though it is distributed.** Being
   sharded does not make something substrate. It factorises an arbitrary SPD
   matrix; the mesh is an argument. Contrast `common/contract_bands.py`, whose
   *subject* is the movement.
6. **`common/sanity.py` — L3, though it reads as physics.** Its checks are
   named for physics stages, but every one is a cheap array reduction plus one
   `device_get`, and it knows nothing about what the numbers mean. It is
   instrumentation, and filing it L1 would have made `common.collectives` and
   `solvers.lanczos` look like violators for using it.
7. **`file_io/` — split, not assigned.** The audit treated it as one package.
   It is two: a sharded-slab transport (L3) and a shelf of BerkeleyGW/QE format
   readers (L1). Any single assignment produces false violations in one
   direction or the other.
8. **`gw/gw_config.py` — L1, and it should be less.** It is the project's
   config module living inside the GW driver package, and five packages import
   it. Its *deck* half (`read_lorrax_input`, `LorraxConfig`, 900 lines of
   dataclasses) is genuinely GW's and should stay. Its *environment* half is
   not: the XLA-memory quarter moved out to `runtime/xla_memory.py` on
   2026-07-31 (§6); `SlabIOBackend` still has not (§5).
9. **`bandstructure/htransform.py` — an L1 driver and an L1 library in one
   file.** Same shape `gw/kin_ion_io.py` had before it was split. Its
   `shard_map` Galerkin solve is a kernel; its `main()` is a driver. Its
   hand-rolled `_build_mesh_xy` was the separable part and is gone.
10. **`runtime/padding.py` — L3, though it is arithmetic.** `round_up` looks
    like L2 mathematics. It exists **only** because a mesh axis has to divide
    an extent; delete the mesh and the module has no reason to exist. Level
    follows the reason, not the body.
11. **`common/minimax.py` vs `gw/minimax_config.py` vs
    `gw/minimax_screening.py` — L2, L1, L1.** Same word, three levels. The
    solver for the minimax quadrature problem is physics-free (L2); the
    dataclass that picks a target error for a *screening* integral and the
    module that extracts PPM poles are not.

---

## 5. The sanctioned exceptions

Four upward edges and four environment touches survive. Each is in
`tests/test_layering.py` with its reason, and each is **asserted to be still
needed** — an exception that outlives its violation fails the suite.

| # | violation | why it is still here |
|---|---|---|
| **R1** | `file_io.slab_io` → `gw.gw_config` (2 lazy sites) — `SlabIOBackend`, the enum that types the file-IO service's own `backend` parameter, is defined in the GW driver's deck parser. `slab_io`'s own public docstring tells users to `from gw.gw_config import SlabIOBackend`. | The obvious new home is `file_io`, but `file_io/__init__.py` and `slab_io.py` import jax and h5py at module scope, and `gw_config` is nearly jax-free by design. **The enum needs a home neither package owns.** That is a structural decision, not an enum move. |
| **R2** | `solvers.sternheimer_solve` → `psp.dft_operators`, plus `STERN_DEBUG` read at module scope as `bool(int(...))` — so a word spelling raises `ValueError` on **import**. | Both are the same question: is this an L2 CG solve or an L1 Sternheimer kernel? Split `SternheimerOp`'s operator out, or move the file to `psp/`. Physics decision. |
| **R3** | `centroid.kmeans_isdf` → `centroid.orbit_syms` (lazy) | See §4.2. Inject the orbit map; signature change. |
| **R4** | `mixing.acceleration` sets `JAX_ENABLE_X64` at module scope — a library mutating global jax configuration for whoever imports it. Same class as `centroid/kmeans_isdf.py`'s `config.update`, removed 2026-07-30. | Not free: the only consumer (`gw/sc_iteration.py:577`) is lazy and always post-`bootstrap()`, but a bare `import mixing.acceleration` in a fresh process would then run Anderson/CROP in **f32, silently**. Physics decision. |
| — | `gw/__init__.py` sets `JAX_ENABLE_X64` | Argued in place, and shown inert for both GW drivers (they call `bootstrap()` then `jax.config.update` explicitly). Kept for import paths with no bootstrap. |
| — | `centroid/kmeans_plot.py` sets `MPLBACKEND=Agg` | Not a compute knob. A plotting helper choosing a headless renderer is the correct owner of that choice. |

---

## What NOT to unify

A named reason not to abstract is worth as much as an abstraction. These five
refusals are load-bearing; each was reached by measurement, and re-deriving
them costs a campaign.

1. **Do not merge `runtime.nccl_warmup` with `collectives.warm_mesh_cliques`.**
   `warm_mesh_cliques` is correct *only because* its jit is small enough that
   XLA takes `ThunkExecutor::ExecuteSequential` and runs the thunk **inline on
   the calling thread** — that inlining is the mechanism that satisfies
   `MPI_Is_thread_main` at communicator creation. `nccl_warmup` deliberately
   avoids `lax.psum`; its job is to pay `ncclCommInitRank` topology discovery,
   which has no thread constraint. **Unify the call site — `prepare_mesh()` —
   never the bodies.**
2. **Do not build a generic `shard_map` wrapper.** `in_specs`/`out_specs`
   **are** the distributed algorithm; a wrapper taking them as arguments has
   abstracted nothing and put a layer between the reader and the only thing
   they came to read. `contract_bands` is the proof the right abstraction is
   *specific*: it replaces a family of `shard_map` sites by owning **one named
   pattern**, which is why it can carry a contract document. The
   counter-example is decisive — `fft_helpers`'s two functions are the only
   ones in the tree written *to be* reusable `shard_map` wrappers, and they are
   two of the sites that omit `check_rep`.
3. **Do not collapse `runtime._FALSY_TOKENS` into `gate.MODE_SPELLINGS`.**
   Two-valued vs three-valued, on purpose: `auto` must be distinguishable from
   an explicit `on`, because an AUTO default may demote silently while an
   explicit `on` must announce-or-raise. Share the token table; keep two
   resolvers. And any unified rank resolver must keep launcher env vars **ahead
   of** `jax.process_index()` — the latter goes through `get_backend()` and
   **initialises the XLA backend**, destroying the before-`initialize` promise
   and the kernel-cache keys.
4. **Do not give `ffi/` a shared C++ handler base.** The cuFFT plan workspace,
   MKL's OpenMP chunking and a scratch-free BLAS call are three measured
   designs that happen to have the same silhouette.
5. **Do not force the 26 bench drivers under `src/` through `resolve_mesh`.**
   They construct meshes differently on purpose —
   `ffi/cusolvermp/profile_batched.py` parses `--mesh 2x2` and exits on a
   mismatch, which is exactly right for a benchmark whose job is sweeping
   geometries. Forcing them through the service deletes the sweep. **Move them
   to `tests/`; leave their mesh construction alone.**

### The 26 exempt bench drivers

They are exempt from every rule here and their number is pinned, so the
exemption cannot widen. Their cost is measured, not hypothetical: their
copy-pasted jax preamble is **104 of `common/`'s 121** `os.environ`/`os.getenv`
matches (2026-07-31) — the single largest duplication in the environment sweep.
The fix is `git mv`, not a new abstraction.

---

## 6. The gate

`tests/test_layering.py` — pure AST, imports neither jax nor anything under
`src/`, so it runs on a login node, in a container and in CI. Five rules:

| rule | what fails it |
|---|---|
| 1 | a named L1 driver imports `jax.sharding` / `shard_map` / `multihost_utils` / `jax._src`, at module scope **or lazily**, beyond its budget — or comes under budget without the budget being lowered |
| 2 | an L2 module touches `os.environ` |
| 3 | any module imports one at a lower-numbered level |
| 4 | a `Mesh(` is constructed outside `common.collectives`, `runtime`, or `bse.bse_ring_comm` |
| 5 | a module that is not `runtime` and not a declared entry point writes an environment variable at **import** time |

plus map hygiene: every named module exists, no module is in two levels, every
module gets a level, every exception is still needed.

**Every scanner has a red twin** — `test_*_can_fail` feeds the *same* function
a source that does violate the rule. Re-implementing the scan inside the twin
would test the twin, not the instrument.

The suite was also seeded with ten deliberate violations injected into a
throwaway copy of `src/` (2026-07-31) and each went red on the expected test:
a plumbing import added to `gw_jax`; a plumbing import *removed* from
`exciton_bands` (slack budget); an env read added to `solvers/davidson.py`; the
excused `STERN_DEBUG` read deleted while its exception stayed; a lazy
`gw.gw_config` import added to `common/collectives.py`; a third
`slab_io → gw_config` site; a `Mesh(` added to `gw_jax`; a module-scope
`setdefault` added to `zeta_projection`; a mapped module renamed away; and
`bse_ring_comm` stripped of its `Mesh(` while keeping its owner licence.

---

## 7. Landed 2026-07-31

* **`runtime/xla_memory.py`** — `resolve_xla_gpu_memory_env`,
  `classify_xla_pool` and their two dataclasses moved out of `gw/gw_config.py`
  (~280 lines). `runtime.collect_startup_facts` used to reach *up* into the GW
  driver package for them, through an import that was lazy and
  exception-guarded precisely because the direction was wrong; the code said so
  itself, as numbered request R9. `gw.gw_config` re-exports both names, so
  `gw_init`, `gw_output` and `tests/test_env_grammar.py` are unchanged.
* **`common.collectives.single_device_mesh` now owns the process-local mesh.**
  It lived in `common/wfn_transforms.py` — a ψ-transform kernel — and the
  cross-process service imported it from there to learn where its own device
  was. `wfn_transforms.process_local_mesh` is now an **alias**, not a second
  implementation, and that is load-bearing: the shape-keyed jit caches in
  `wfn_transforms` key on a `NamedSharding` that embeds the mesh *object*, so
  two equal-but-distinct 1×1 meshes double every one of them. `Mesh(` no longer
  appears in `wfn_transforms.py` at all.
* **`bandstructure/htransform.py::_build_mesh_xy`** now calls
  `collectives.resolve_mesh` instead of re-typing its body. That retires the
  `process_count() * local_device_count()` mesh dialect — one of five — and
  adds the addressability refusal, so a mesh this process cannot compute on is
  named here instead of surfacing as a bare `StopIteration` inside a jit.
  Deliberately **not** `prepare_mesh`: that also warms the `impl=mpi` cliques,
  which is a new collective on this path and belongs to the open warm-up
  decision, not to a rename.

Production `Mesh(` construction sites: **21 (2026-07-30 16:47) → 8 (17:08) →
4**, in three modules, all sanctioned.
