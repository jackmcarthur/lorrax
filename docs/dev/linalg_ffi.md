# Distributed dense linear algebra: LORRAX's side of `distrib_la`

[`docs/services/distrib_la.md`](../services/distrib_la.md) owns the service:
API, `Plan`/`FactorToken`/`matmul`, backend vocabularies, the guard ladder,
promise semantics, layouts, donation, refusals and performance. The ζ-fit
channel policy (charge rank truncation, transverse ridge, the whole-tile
back-solve) is owned by
[Face-ψ ζ fitting](../architecture/zeta_fit_face_psi_cct.md). This page owns
the one deck dial, the procedure for adding a backend, and the failure modes
inside the ScaLAPACK handlers that no Python guard can see.

## The deck dial

`linalg = local | distributed` (default `local`) is the only dense-linear-algebra
deck key. `gw_config.resolve_linalg` interprets it once into a
`LinalgResolution`; stage code reads that record and never re-interprets the
dial.

| resolved field | `local` | `distributed` |
|---|---|---|
| W Dyson solve | per-q local LU | `plan('solve_lu', backend='distributed').batched` |
| transverse ζ LU (`distributed_lu`) | `auto` | `distributed` (ScaLAPACK on cpu, cuSOLVERMp on CUDA) |
| batched route | `DISTRIB_LA_BATCHED_ROUTE_DEFAULT` | `auto` |
| eigensolves (`eigh_backend`, `sc_eigh`) | `auto` (native, q-batched) | `distributed` (cpu → `scalapack`, CUDA → `cusolvermp`) |
| charge ζ factor | `rank_truncate` | `rank_truncate` |
| transverse ζ factor | `ridge` | `ridge` |

The charge factor is `rank_truncate` in both layouts: it is the only route
that carries the rank-conditioning of `V_q`, and no deck spelling bypasses it.
The former stage/backend keys (`distributed_zeta_solve`, `distributed_cholesky`,
`distributed_lu`, `w_dyson_solver`, `distrib_la_batched_route`,
`charge_zeta_solve`, `transverse_zeta_solve`, `eigh_backend`, `sc_eigh`,
`use_low_mem_eigh`) refuse by name with "use `linalg = local | distributed`".
CLI overrides (`--eigh-backend`, …) are debugging controls.

**Replication cap.** Block-cyclic distributed factors are grid-dependent
(partial-sum order changes with the process grid), and GN-PPM amplifies that
drift. Under both layouts the rank-truncating charge factor therefore runs
replicated and mesh-invariant, one q-batch at a time; there is no distributed
charge factor. When one batch `q_batch·n_μ²·16 B` exceeds
`LORRAX_ZETA_REPLICATE_CAP_GIB` (default 4 GiB),
`isdf.core._rank_truncate_capacity_error` refuses and names the cap value that
would clear it. Raising the cap makes the route resolve, not finish: the
replicated factor is a dense whole-tile eigh per q, `ceil(nq/P)·n_μ³` per rank.

## Adding a backend

Every step is inside `services/distrib_la/`; nothing in `src/` changes except
the lorrax-side loader row in step 2.

1. Write `services/distrib_la/src/distrib_la/_<name>.py`, copying
   `_cusolvermp.py` (its docstring names the three per-routine decisions:
   donation, handle-versus-array return, output normalisation).
2. Register the C++ handler symbol in `distrib_la/loader.py`
   (`_CUDA_TARGET_SYMBOLS` / `_HOST_TARGET_SYMBOLS`). This makes
   `probe_target()` / `has_target()` and the capability guard work, with the
   absent-versus-broken split. If `src/ffi/common/ffi_loader.py` also reaches
   the handler, add the row there: both loaders open the same `.so`.
3. In `distrib_la/resolve.py`: one entry per op in `BACKEND_CHOICES`, one
   `(op, backend) → (target, platforms)` row in `_SPEC`, any geometry rule in
   `_check_geometry`, a `_DISTRIBUTED_DEFAULT` row per platform if it is that
   platform's `distributed` answer, and a branch in `backend_module`. A
   declared-untested tier still gets its rows.
4. Add one row to `distrib_la/plan.py`'s `_IMPL`: the single-tile entry, the
   stacked entry, and an output normaliser if the library's convention
   differs. Either entry may be `None`: a missing stacked entry is filled by
   `lax.scan` over the single-tile one, and a missing single-tile entry makes
   the stacked entry the only route. That row is the capability test
   (`Plan.batched_route` reads it; nothing probes the module with `getattr`).
   A single-tile entry that returns a library handle sets `one_handle=True`.
   Normalise conventions here, never at call sites. Donation is per op in
   `DONATES` (`eigh` none, `cholesky` operand 0, `solve_lu` operands 0 and 1).
5. An opaque factor gets a branch in `distrib_la/factor.py` so it arrives as a
   `FactorToken` and leaves through `solve()`. A raw handle never crosses a
   `jit` boundary.
6. `eigh` vocabulary reaches LORRAX through `gw_config.eigh_backend_choices()`,
   which reads `distrib_la.BACKEND_CHOICES`; nothing else changes.
7. Tests with the backend, not after it: an L-a algebra cell, an L-b
   emulated-mesh cell, an L-c body in `_CLI_CELLS`, a contract cell per refusal
   it can emit, each with its false case, and a machine-profile row if the
   backend is expected present somewhere, so an unexpected skip fails.
8. Update the Backends table in `docs/services/distrib_la.md`.

## Inside the ScaLAPACK handlers

These live in `src/ffi/cpp/scalapack/` and are invisible to the resolver:
the handler is compiled, the probe passes, and the failure is numerical or
environmental.

* **Eigenvector workspace.** `pXheevd` can return `INFO = 0` with correct
  eigenvalues and a garbage `Z`: the back-transform (`pXunmtr`/`pXormtr`)
  needs more `WORK` than `pXheevd`'s published `LWORK` formula or MKL's query
  gives. `eigh_ffi.cc` floors `LWORK` at
  `max(NB(NB−1)/2, (NP0+MQ0)·NB) + NB² + 8N`. An eigenvalue-only test does not
  test an eigensolver: always assert `A Z = Z diag(W)`.
* **The workspace query is mandatory.** MKL's `pzheevd` rejects the netlib
  minimum with `INFO = −16` and asks for far more on multi-rank grids, so a
  failed query is fatal and the handler uses `max(query, formula)`. The
  workspace is `malloc`'d inside the handler, outside the JAX memory planner;
  `LORRAX_DEBUG_PRINT=1` prints it per call. Both netlib and MKL implement only
  `JOBZ='V'` for `pzheevd`.
* **SLATE's ScaLAPACK overlay is refused.** `libslate_scalapack_api.so`
  redefines `pzheevd_`, `pdsyevd_`, `pzgetrf_`, `pdgetrf_`, `pzgetrs_`,
  `pdgetrs_` (every operation this backend performs), so an `LD_PRELOAD`
  silently replaces them while `resolve` still returns `scalapack`. The overlay
  assumes rank `mx + my·p` for shard `(mx, my)`, LORRAX's C-order mesh puts it
  on `mx·q + my`, and its shims hard-wire `info = 0`. `blacs_grid.h` resolves
  the provider of each routine (`dlsym` + `dladdr`) and refuses naming it;
  `LORRAX_SCALAPACK_ALLOW_SLATE_API=1` downgrades the refusal to one stderr
  line for deliberate measurement.
* **MKL thread team.** At production grids `pzheevd`/`pzgetrf` issue thousands
  of small BLAS calls between latency-bound BLACS collectives, and a wide MKL
  team starves MPI progress (24× at a 12×12 grid, n = 2448). The handlers pin
  the calling thread's team through `mkl_set_num_threads_local`
  (`common/mkl_thread_pin.h`) to `min(current, 4)`; `LORRAX_SCALAPACK_MKL_THREADS`
  overrides, and the pin is a no-op on non-MKL ScaLAPACK.

## Verification

* `tests/test_charge_zeta_route.py`, `tests/test_zeta_mesh_invariance.py`:
  route pins and the replication-cap refusal (4+ host devices).
* `services/distrib_la/tests/test_distrib_la_contract.py` (marker
  `distrib_la`): wrapper shape and layout contracts against the real builds.
