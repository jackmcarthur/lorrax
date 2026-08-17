# LORRAX environment variables — the registry

*Every environment variable LORRAX reads, what it defaults to, how it
parses, and whether it should still be an env var at all.  Rows also name
where the variable is read, but that is an ADVISORY convenience, not
something this page owns or anything gates — see [What this page is, and
what it is not](#what-this-page-is-and-what-it-is-not).*

First generated 2026-07-25 from an audit of every `os.environ` /
`os.getenv` / `getenv()` read under `src/` (Python **and** C++);
reorganized 2026-07-27 (workstream AV) around the rule the campaign keeps
re-learning (QUALITY_PATTERNS #8):

> **Environment may grant capability; it must not silently select
> policy.**  Physics- and routing-relevant choices change only via
> declared inputs (the input file), where they are parsed, validated,
> echoed into the run log, and captured by provenance.  Env vars are for
> *machine facts* (where a library lives, which fabric, how many ranks)
> and for *debug switches*.  Where an env var still overrides an input
> key, that override is DEPRECATED and prints loudly.

## What this page is, and what it is not

**This page owns four columns and nothing else: the SPELLING, the DEFAULT,
the CLASS, and the PARSE GRAMMAR of every variable LORRAX reads.** Those four
are what `tests/test_env_registry.py` enforces and what nothing else in the
tree records.

**The read site is not a fifth column.** Most rows name the file or the
function where the variable is read, and that pointer is worth having, but
it is advisory and nothing checks it. `test_env_registry.py` is pure
name-token coverage — it walks `source_roots()` for spellings and never
resolves a path — so a row keeps passing after the module it cites has
moved or been deleted, which is how the `wfn_loader` and `zeta_loader`
rows came to point at shims that had already been deleted. Some
paths are *deliberately* historical: the four `_slab_io_mpi_host.py` rows
record where a since-deleted transport read its knobs, and other rows
point into `cpp/`, `context.cc` or `run_shifter.sh`, outside `src/`
entirely. Treat a read site as a lead to follow, not as a fact this page
warrants; if it is wrong, fixing it is welcome and gates nothing.

**It does not own the explanation.** Why collective writes are on, what the
nvhpc stage selects, how the allocator changes what `memory_stats()` reports
— each of those has an owner page, named in the
[register](../index.md#register), and each row below links to it instead of
repeating it. Rows that used to carry a paragraph of measurement now carry
one sentence and a link.

That is a deliberate reduction and it is the point. This page had grown to
carry a second copy of `slab_io.md`'s tuning campaign, a second copy of
`environment/overview.md`'s allocator table and a second copy of
`mpi_collectives.md`'s transport argument. Three copies of a fact is three
places for it to go stale, and it did: the `LORRAX_FFT_FFI` row still said
the CPU engine was "MKL FFT (DFTI API)" five days after `DftiCreateDescriptor`
was deleted from the translation unit.

Parse grammar stays here in full, including the ugly parts, because a knob's
grammar *is* its interface and there is nowhere else it is written down. Where
two sites read one variable with two different parses, that is recorded in the
[consistency audit](#consistency-audit) — a split parse is a defect this page
is responsible for surfacing.

## How to read this page

Three classes, in three sections:

| class | meaning | section |
|---|---|---|
| **input-file keys (env twins deprecated)** | policy: physics / numerics / routing. The key is the record; any env twin still honoured prints a deprecation notice. | §1 |
| **machine-capability env** | runtime machine/library facts and resource caps. Legitimately env vars. May affect perf, never physics. | §2 |
| **debug / diagnostic env** | printing, dumping, timing, test hooks. Never changes numerics. | §3 |

Build-time-only variables (§4) and external variables LORRAX sets or
depends on (§5) close the page.  The **Measured scope** column records
the conditions under which a default was chosen (QUALITY_PATTERNS #9:
every performance claim carries its measured domain) — an empty cell
means the default is a design constant, not a measurement.

---

## 0. Do not read this page to find out what a RUN resolved

This page is the registry of what *can* be set.  What a particular run
*did* resolve is printed by the run itself, in one rank-0 block, by
`runtime.initialize_communicator_stack()` — the single startup call every
core driver makes.  Read the block; it is authoritative in a way this page
cannot be, because several of these knobs interact and two of them
(`XLA_PYTHON_CLIENT_ALLOCATOR`, `XLA_PYTHON_CLIENT_PREALLOCATE`) are read
from the environment only *before* backend init, after which `os.environ`
is a **false witness** — measured, job 7882443: two runs with byte-identical
`os.environ` and `bytes_limit` 11.805 GB vs 0.000 GB.

The block states, in complete sentences, every choice where more than one
outcome was possible:

* the process count, the resolved platform, the device count and kind, and
  the device mesh with a note that its communicator cliques were warmed;
* **every demotion**, tagged `DEMOTION:` — the CPU pin, the CUDA-plugin
  skip, a failed `jax.distributed` auto-detect, a malloc tuning that did
  not arm.  (Rank-0's demotions only; a demotion that happened on one other
  rank is announced from that rank, and the block says so.);
* the CPU collectives transport and **why** it resolved that way, with the
  gloo silent-corruption warning when it is gloo;
* the XLA pool **read from `jax.local_devices()[0].memory_stats()`**, and a
  warning when the live client disagrees with the environment;
* which FFI `.so` loaded and from which variable; `LORRAX_BANDS_GEMM_FFI`,
  `LORRAX_FFT_FFI` and `LORRAX_FFT_FFI_FUSED` each with their resolved
  mode and route — *including when they are off*, because silence about an
  off dial is indistinguishable from silence about an on one.  (The FFI
  layer is REQUIRED since 2026-08-01: startup enforcement —
  `Gate.enforce`, step 6b — refuses a missing library before this block
  prints, so an off dial in the block is always an explicit `=0` opt-out,
  stated as uncertified or as a refusal per the dial's `off_policy`);
* the distributed linalg backends available for `eigh` / `cholesky` /
  `solve_lu` on this mesh (capability; the CHOICE is an input-file key);
* CPU affinity and the thread-count variables, with an oversubscription
  warning;
* the persistent compile-cache state and directory, always with the caveat
  that the key includes every array shape;
* whether the fail-fast excepthook and the glibc malloc tuning are armed;
* how long each startup phase took, so the 43.8 s `jax.distributed` init at
  P=16 and the 75.0 s cold-node import storm are visible without a profiler.

Adding a dial and *not* adding it to that block is a bug, and
`tests/test_runtime_startup_report.py` fails on it: it scans `src/ffi/**`
for `Gate(env=…)` and requires `runtime._ffi_dial_facts` to collect every
one it finds.

---

## 1. Input-file keys — and the deprecated env twins

These change the numbers a run produces.  Their home is `cohsex.in`.

### 1a. Env twins that still work (deprecated, loud)

The SC family was promoted first (`gw_config.py` `_sc_env` pattern); the
ζ conditioning pair was promoted by workstream AV (2026-07-27): the key
is the source of truth, a non-empty env var still overrides, and the
override prints a rank-0 deprecation notice (once per process,
`isdf/core.py::_deprecated_env_float`).  ζ-fit provenance records the
EFFECTIVE (post-override) value, so a rerun that drops the env cannot
silently reuse a ζ fit at a different conditioning cutoff.

| env twin (deprecated) | input key | default | effect |
|---|---|---|---|
| `LORRAX_ZETA_RCOND` | `zeta_rcond` | `1e-8` | Relative eigenvalue cutoff of the rank-revealing ζ pseudo-inverse (both the replicated and the distributed `rank_truncate` factor, `isdf/core.py`). THE conditioning knob of the μ ladder: at μ=1998 it alone decides `n_keep ≈ 1015/1998` with no spectral gap (scorecard K). |
| `LORRAX_ZETA_RIDGE` | `zeta_ridge` | `0` | Additive Tikhonov ridge on `C_q` before the ζ factorization. Superseded by `rank_truncate` as the conditioning cure; kept for the cholesky route. |
| `LORRAX_SC_MAX_ITER` | `sc_max_iter` | — | self-consistency loop cap |
| `LORRAX_SC_TOL_EV` | `sc_tol_ev` | — | SC convergence tol |
| `LORRAX_SC_ACCEL` | `sc_accelerator` | — | SC accelerator choice |
| `LORRAX_SC_DEPTH` | `sc_history_depth` | — | SC history depth |
| `LORRAX_SC_MIXING` | `sc_mixing` | — | SC mixing |
| `LORRAX_SC_DUMP_DIR` | `sc_dump_dir` | — | SC dump dir |

No live harness depends on the ζ env twins (audited 2026-07-27: the one
historical user is the completed one-off
`run_B_c1998_rcond10/run72.sbatch`).  This deprecation pattern is the one
to copy for any future promotion.

**Stale doc reference:** `docs/theory/isdf-zeta-vq.md:479` mentions
`LORRAX_GFLAT_CHUNK_SIZE`.  Nothing reads it — the knob is the input key
`gflat_chunk_size`.

### 1b. Promotion candidates (physics/routing knobs still living in env)

Ranked.  These escape input-file validation, the run log, and provenance.

1. **`LORRAX_ZETA_REPLICATE_CAP_GIB` → a memory-section key.**  Byte cap
   (GiB, default `4`; production raises to 16) on the `(nq, μ, μ)` c128
   stack below which the charge factor is fully **replicated** and
   therefore mesh-invariant (`isdf/core.py`).  Above the cap
   `rank_truncate` REFUSES rather than downgrading.  It gates *which
   route carried the conditioning cure*, so "which route did this run
   take" currently depends on an env var.  Raising it is already a
   deliberate, logged act — which is why it ranks below the (now-done)
   rcond/ridge promotion.
2. `ISDF_CHUNK_TARGET_UTILIZATION` / `ISDF_ZCT_STAGE_CAP_GB` /
   `ISDF_ZCT_STAGE_CAP_FRAC` → memory-section keys, for symmetry with
   `band_chunk_size` / `r_chunk_size` / `gflat_chunk_size`, which are
   already keys (planner dials in `gw_config.py`: utilization target
   clamped to [0.85, 1.0]; absolute/fractional caps on the ζ-contraction
   stage transient).
3. `LORRAX_WFN_BACKEND` (`""` → config/auto; forces `eager` \| `phdf5`,
   `services/wfn_loader/src/wfn_loader/loader.py`) → the `slab_io`/backend config
   section, so the read path is recorded alongside the write path.

**Guard escape hatch (physics, not a capability):**
`LORRAX_OCCUPATION_SUPPORT` defaults to unset (strict refusal); the sole
case-insensitive nonempty value `allow-truncated` logs the support bounds and
continues, and every other value refuses by name.

**Do NOT promote:** anything in §3 (debug), §4 (build), or the compile
cache (§2 — a machine fact; its mandatory-`""` status during regression
timing belongs in the job script, not the physics input).

---

## 2. Machine-capability env — runtime

Machine and library facts, transport/placement, and resource caps.
Legitimately env vars.  None of these may change physics; several change
wall time, and those carry their measured scope.

### 2a. Transport / placement / distributed

| var | default | effect | measured scope |
|---|---|---|---|
| `LORRAX_COLLECTIVE_CHUNK_MB` | `128` | Upper bound on ONE emitted collective's payload in the ζ `distributed` tier (C⁺ formation + back-solve GEMM, `isdf/core.py`), the distributed W Dyson A-build (`gw/w_isdf.py`), and the rank-0 owner gather of k-partitioned sweeps (`common/collectives.py::gather_indexed_blocks_to_owner` — dipole/kin-ion h5 tables), enforced as a host-level chunk loop XLA cannot fuse back (AF).  `0`/negative = unbounded — reproduction escape hatch only.  A per-instruction TRANSPORT cap, deliberately orthogonal to the LIVE-bytes memory cap `LORRAX_ZETA_GATHER_CAP_GIB`. | Bracketed at P=144 on **em1-era Gloo**: 1.15 GB single-shot AllGather fatal, 0.104 GB good (AF).  Re-priced 2026-07-27 (scorecard AV matrix), caps {64,128,256,512,∞}: at 785c/P=16 impl=mpi/mlx the cap is indistinguishable from unbounded (596-714 ms back-solve, no trend); at P=64 ib0-Gloo the chunk loop costs +35-70 ms/r-chunk (~2-3 % of the ζ-fit) vs a healthy 192 MB single-shot, and at P=144 the chunked tier measured 1.9× FASTER than its per_q control (AF.4).  Default stays 128 = protective at the one measured fatality point, near-free everywhere else; on ib0/mlx it is not known to be *necessary* below ~370 MB payloads. |
| `LORRAX_ZETA_GATHER_CAP_GIB` | `4` | Byte budget for the ζ back-solve's replicated-factor all-gather transient (LIVE bytes; also bounds the distributed tier's eager q-batch, `isdf/core.py`). | |
| `LORRAX_ZETA_QPARALLEL` | unset (→ `auto`) | Execution SCHEDULE of the replicated charge ζ-fit factor (`isdf/core.py::_qparallel_factor_ok`): `auto` scatters the q axis over all devices above the fold threshold `_QPARALLEL_MIN_NQ_MU3` (each device factors whole per-q tiles, factors resharded back — bit-identical to the all-ranks execution by construction, gate `tests/test_zeta_mesh_invariance.py`); `0` forces the pre-fold all-ranks execution (A/B control); `1` forces the fold (test hook).  Never a numerical route — same plan, same bits. | Motivating measurement: job 7884656 (b300, P=16, nq_ibz=10, μ=2979) spent 105.1 s in `zeta_fit.cholesky`, one dense eigh per q on EVERY rank. |
| `LORRAX_ZETA_REPLICATE_CAP_GIB` | `4` | See §1b(1) — promotion candidate. | production 12×12 runs raise to 16 (overnight A/B) |
| `LORRAX_MAX_RCHUNKS` | unset | Ceiling on the r-chunk count the planner may pick (`gw/isdf_fitting.py`).  Memory/perf, not numerics — but chunking is load-bearing at large μ (scale ladder). | |
| `LORRAX_GALERKIN_CHUNK_GIB` | `6` | htransform's Galerkin accumulation chunk budget (`bandstructure/htransform.py`).  Perf only. | |
| `LORRAX_PPM_FIT_ARENA_GIB` | `8` | Temp-arena budget for the GN-PPM fit's q-chunk loop (`gw/minimax_screening.py::_gn_ppm_fit_q_block`).  Sizes `q_block` so `q_block · tile_bytes · _GN_PPM_FIT_LIVE_TILES` fits; `q_block ≥ nq` takes the untouched single-shot path.  Bit-exact by construction: the kernel is elementwise in q and its two reductions are exact integer counts, so chunking changes evaluation order/placement only.  Perf/memory only. | Arena 74.27 → 4.64 GiB at μ=24,933/P=64; gated bit-exact on the pinned Σ reference on the single-shot AND a forced `q_block=1` path (ladder notes R32/R33). |
| `LORRAX_CENTROID_RANK_TOL` | `0.01` | Tolerance fraction for the centroid rank-shortfall HARD REFUSAL (`centroid/kmeans_cli.py`): after pivoted-Cholesky pruning, `rank < ceil((1−tol)·n_orbit_keep)` is FATAL — the candidate pool cannot supply the requested independent ISDF directions (the pre-2026-07-29 clamped prune window produced 630/897 silently; see `_resolve_sigma_window`).  Fail-closed; raising it is a deliberate, logged act.  Promotion candidate (it gates run acceptance). | b1024 rung: 630/897 (70.2%) refused; widened window: 897/897 passes (ladder notes R12). |
| `LORRAX_CENTROID_PC_TOL` | unset (→ `sqrt(eps)` ≈ 1.49e-08) | Stopping tolerance of the pivoted-Cholesky select (`centroid/pivoted_cholesky.py`), RELATIVE to the largest initial Gram diagonal: the recurrence stops when the max residual Schur diagonal falls to `tol·max(diag G)`, and `rank` is then the number of pivots actually taken (`piv[rank:] == -1`).  Scale-relative so the answer does not depend on how G is normalised; a `pmax` in the sharded kernel so it is identical at every shard count.  The default is the number this kernel has always computed for its `rank` report, kept so no existing deck's reported rank moves — LAPACK `?pstrf`'s own policy is `n·eps` (5.7e-13 at the production M=2580, four orders looser), available by passing it.  Numerics-affecting: loosening it certifies fewer directions and REFUSES earlier, never silently. | Pre-2026-08-07 the floor was computed and used only as a label: past the numerical rank the divisor clamped to `sqrt(eps)` and L blew up to Inf then NaN (first non-finite column twelve iterations past a true rank of 10), after which argmax over NaN returned candidate indices in array order. |
| `LORRAX_CENTROID_POINT_RANK_CAP` | `4096` | Size cap on the POINT-granularity rank check the orbit-aware prune reports (`centroid/pivoted_cholesky.py::point_granularity_rank`).  In orbit mode the select certifies ORBITS — it deflates the Schur complement by one direction per orbit while removing all `n_sym` members from contention — so its `rank` is not comparable to the point count of the file it blesses.  The check is a dense `eigvalsh` on the kept sub-Gram, O(n^3), so above this many delivered points it reports NOT MEASURED with the reason rather than paying minutes and gigabytes.  Diagnostic only: it never changes the selection or refuses. | D3 passed its gate at "42 of 42 directions certified" and wrote 1908 points; the zeta back-solve then truncated to 1440-1455 modes per q (23.98% mean) across eight q-points, logged every leg and read by nobody.  Sibling arms: D1 (1008) drops 0.00%, D2 (1464) 2.66%. |
| `LORRAX_FAILFAST` | `1` (on) | **One variable, two mechanisms — this page carried them as two rows in two sections until 2026-08-06, which read as a default disagreement and was not one.** (a) CLI failure propagation in `runtime/__init__.py::bootstrap`: an uncaught exception aborts the whole step rather than leaving P−1 ranks in a collective (QUALITY_PATTERNS #7). (b) `install_failfast_excepthook`, **P>1 only**: a per-rank uncaught exception prints a rank-tagged banner and calls `os._exit(1)`, so the *job* fails instead of the peers hanging in a collective the dead rank never joins; no-op single-process. `0`/`off` disables both. `SystemExit(0)` (e.g. `LORRAX_EXIT_AFTER_ZETA`) stays a clean exit. | |
| `LORRAX_MALLOC_TUNE` / `LORRAX_MALLOC_MMAP_MB` (`1`) / `LORRAX_MALLOC_TRIM_MB` (`128`) | on | glibc malloc tuning at bootstrap (`runtime/__init__.py:tune_glibc_malloc`): pins `M_MMAP_THRESHOLD`/`M_TRIM_THRESHOLD` so freed XLA:CPU transients return to the OS — the arena-retention cure (scorecard T), ≤4 % wall cost. `LORRAX_MALLOC_TUNE=0` disables. | RSS-∝-FLOPs ramp root-caused + cured at 12×12/P=80 (T.2); it had OOM'd the 1998c fits |
| `LORRAX_MALLOC_TRIM` | `1` (on) | Per-r-chunk `malloc_trim` in the ζ-fit loop (`gw/isdf_fitting.py`); glibc-only, no-op elsewhere. | same (T.2) |
| `LORRAX_BSE_MATVEC_OPT` | unset (→ none) | Comma-set of BSE stack-matvec strategy opts.  As of 2026-08-08 the ONLY token is `gspmd` (`bse/bse_stack_matvec.py::matvec_opts`); an unknown token REFUSES rather than silently running the baseline under an optimised label.  `gspmd` is an AUDIT ROUTE, default off: it rebuilds the W term with no manual `shard_map` and lets XLA's SPMD partitioner pick the collectives, so it is an A/B instrument, not a proposed default.  Two tokens were retired the same day, in OPPOSITE directions, and the difference is the point: `yhoist` was made PERMANENT (1.007×, 16 KB/rank, bit-identical, no scale cliff — a free and harmless win is a default, not a dial), while `krep` was DELETED (0.05% of the eigensolve, invisible in a wall A/B, and never honoured on the shipped `block_size = 1` route — FEAST_KPM_PASS.md §1).  Setting either now refuses. |
| `LORRAX_DAV_MVSCAN` | unset (→ off) | Comma list of block widths (e.g. `1,2,4,8`).  DIAGNOSTIC, off by default: times `apply_H` at each width inside the Davidson route so the block-width choice can be measured rather than assumed (`bse/bse_lanczos.py`).  Landed with the 2026-08-08 Davidson-competitiveness lane; it only times, it changes no result. |
| `LORRAX_DAV_TRACE` | unset (→ off) | Path to an `.npz`.  DIAGNOSTIC, off by default: dumps the Davidson per-iteration history (iter, matvec count, subspace size, eigenvalues, residuals, wall) plus the distinct-program count, written on process 0 only (`bse/bse_lanczos.py`).  Landed with the 2026-08-08 Davidson-competitiveness lane; read-only instrumentation. |
| `LORRAX_VQ_LR_GZ_TRIM` | unset (→ off, `=1` opts in) | Trim the structurally dead G_z columns out of the long-range v(q) design basis before the fit (`bse/vq_interp.py::lr_gset`).  On the MoS2 slab 176 of 337 columns are dead by construction (`max|A[:, dead]| = 0.000e+00`) and nG drops 337 → 161.  **DEFAULT OFF, and it must stay off until the null is understood**: with the trim on, `run_nulls` reads `F_own_rebuild_vs_cleaned_LR_tile_max = 5.793e-03` against a `1e-6` tolerance (FAIL), where the default reads `6.004e-11` (OK).  At the default the code reproduces pre-trim `013aad92` exactly.  Landed by the 2026-08-08 vq-interp lane; see FIX_vq_interp.md.  **Since 2026-08-10** the trim follows `lr_fit_degrees` — the energy cutoff with a two-shell floor (`docs/architecture/decisions.md`) — rather than `DEG_B26P`'s keys, so on a thick slab it now keeps every channel the model actually fits instead of a fixed four.  That widens what the trim retains; it does not change this knob's default, and the coverage-null question above is untouched by it. |
| `LORRAX_LANCZOS_REORTH` | unset (→ `cgs2`) | Lanczos reorthogonalisation route, read by `bse/bse_lanczos.py::reorth_route` and passed down as a token to `solvers/lanczos.py::reorth_kind` (`solvers` is L2 and may not read the environment -- `tests/test_layering.py::test_no_l2_module_reads_the_environment`; the first draft of this dial read it there and that gate caught it). Steers `lanczos_eig_jit` and both block variants. `cgs2` (**default**) is batched classical Gram-Schmidt applied twice: every overlap of a sweep in ONE matrix product, so **2·max_iter** all-reduces of an `(m,)` vector. `mgs` is the legacy per-vector sweep, **max_iter(max_iter+1)/2** all-reduces of a `c128[]` SCALAR — 20 100 of them on the Si 4×4×4 record deck at 200 iters, 364 ms of GPU time at 0 % occupancy to do 165 MFLOP. `mgs` is kept for bisects and for reproducing pre-2026-08-08 runs; it is a FALLBACK, not a tuning knob. An unknown token REFUSES — in both directions, since a typo must not silently hand back the slow route either. Collective STRUCTURE only: the `--n-reorth` window is identical on both routes (measured: max\|Δλ\| 9.77e-15 eV on the record deck's 20 excitons, identical `max\|V^H V − I\|`). |
| `LORRAX_FACE_TO_BATCH_ROUTE` | unset (→ `staged_reshard.DEFAULT_ROUTE`) | Which `common.staged_reshard` schedule performs the fH_q face→batch move (`bandstructure/bse_setup.py::resolve_reshard_route` — the ONE resolver; the `reshard_route` kwarg wins when the caller passes it).  Exists so the two routes can be A/B'd through the production `bse.exciton_bands` driver.  Movement only, value-identical.  An unrecognised token is ANNOUNCED (`*** LORRAX SANITY`) and the default runs — never silently. |
| `LORRAX_KIN_ION_LOOKAHEAD` | `2` | Host-ahead-of-device depth for `sweep_local_k` (`common/collectives.py::sweep_lookahead`).  `1` restores the serialised no-overlap behaviour so the pipelining win can be MEASURED (the D10 harness control cell sets it).  Malformed values REFUSE naming the variable (the swallowed-`ValueError` parse it replaced returned the default silently). | |

### 2b. Libraries, backends, caches, I/O placement

| var | default | effect |
|---|---|---|
| `LORRAX_FFI_SO` | in-tree `src/ffi/cpp/build/liblorrax_ffi.so` | Path to the **CUDA** FFI library (`ffi/common/ffi_loader.py:96`). |
| `LORRAX_FFI_HOST_SO` | in-tree `src/ffi/cpp/build_host/liblorrax_ffi_host.so` | Path to the **host** FFI library (`ffi_loader.py`, `_PLATFORMS["cpu"]`). The in-tree default is what `src/ffi/cpp/build_host.sh` produces; `config/frontera/build_ffi_host.sh` (the Frontera SLATE+ScaLAPACK build) writes `$LORRAX_FFI_STAGE_WTA/build_host/liblorrax_ffi_host.so` (default `$WORK/lorrax_ffi_wtA/build_host/…`) — point this variable there for that build. FFI dependency note: the linalg handlers call the ScaLAPACK API, which MKL supplies on Frontera and Cray LibSci elsewhere (`-DLORRAX_SCALAPACK_LIBRARIES` takes any vendor's link line); the GEMM handler below builds against any standard CBLAS — the batched `cblas_?gemm_batch` entry when the BLAS provides it, plain-GEMM loop otherwise, decided at RUN time and announced on first use (there is deliberately no configure-time probe). Works in principle with Intel MKL or Cray LibSci; **tested with Intel only so far**. |
| `LORRAX_FFI_ABI_STRICT` | unset (→ announce, do not refuse) | What both loaders do at `dlopen` with a library that exports no `lorrax_ffi_{host,cuda}_abi_version` — i.e. one built before the handler-signature stamp existed (2026-08-08). Unset, they say so once per library and proceed, because every library the pinning worktrees hold today is unstamped and refusing them all would make the check mean "old" far more often than "wrong". Set to `1`, an unstamped library is REFUSED, which is what a certification run wants. A library that IS stamped and disagrees with `src/ffi/cpp/common/lorrax_ffi_abi.h` is refused either way — that is not a dial. The artifact-level twin of this knob is `LORRAX_FFI_VERIFY_STRICT` in `scripts/verify_ffi_build.sh` (GATE 11); see `docs/building_ffi.md`. |
| `LORRAX_BANDS_GEMM_FFI` | unset (→ `on` — REQUIRED) | The `contract_bands_block_reshard` FFI GEMM dial (`common/contract_bands.py`, read at kernel-FACTORY time — consumers key their kernel caches on it). **REQUIRED since the 2026-08-01 ruling** (`docs/architecture/decisions.md`): unset/`1` = the vendor-BLAS host handler (`lorrax_mklblas_gemm_batch`); a missing/unloadable handler on a CPU mesh REFUSES at startup (`Gate.enforce`, wired into `runtime.initialize_communicator_stack`) and at the factory, naming `liblorrax_ffi_host.so` / `LORRAX_FFI_HOST_SO` / `docs/environment/overview.md` — never a silent demotion. The pre-ruling `auto` mode is DELETED; a stale `=auto` resolves to the default with an announced grammar note. `0` = explicit debug opt-OUT onto the retained native XLA einsum arm, announced once as UNCERTIFIED (the arm is retained because `extra="minor"` structurally cannot ride a batched GEMM — that order quietly keeps the XLA plan under every mode). On CUDA the dial does not exist (host symbol table only; XLA:GPU's dot lowering already dispatches cuBLAS — the required path there IS the native lowering; silent by declared design). PERFORMANCE PURPOSE: XLA:CPU lowers the LARGE right contraction through Eigen dots **1.6–1.9× below vendor BLAS at full threads**; the handler measured project_rs 29.4→19.6 s and sigma.exec 58.3→49.2 at nb=128/P=64 (jobs 7879008/7879010). **All four BLAS precisions served** — f64/f32/c128/c64 onto `cblas_{d,s,z,c}gemm[_batch]` (BSE fp32-GMRES c64 rides it); an unserveable dtype (f16/bf16/mismatched pair) REFUSES with the fix named. Perf only, value-level identical (1e-12 gate class, not bit-exact). |
| `LORRAX_MKLBLAS_THREADS` | unset (→ `auto`) | C++ (`cpp/mklblas/gemm_batch_ffi.cc`): BLAS team size pinned (thread-locally, via `MKL_Set_Num_Threads_Local` when the linked BLAS provides it — a no-op on non-MKL BLAS) for the GEMM handler call. `auto` = ambient `omp_get_max_threads()` (the production 28/rank under taskset); `off` = 1; integer pins exactly. Strict full-string grammar; unrecognized values announce on stderr and fall back to `auto`. |
| `LORRAX_FFT_FFI_THREADS` | unset (→ `auto`) | C++ (`cpp/mklfft/fft_flat_k_ffi.cc::team_threads`): OpenMP team size for the flat-k FFT chunk loop.  `auto` = ambient `omp_get_max_threads()`; `off` = 1; an integer 1–4096 pins exactly.  Strict full-string grammar; an unrecognized value announces on stderr and falls back to `auto` (the blacs_grid AW-audit lesson: a typo must not silently pick a known-bad policy).  Alias policy: `LORRAX_MKLFFT_THREADS` is a deprecated alias, honored with a one-time announcement (`mklpin::knob_value`); the shared spelling wins when both are set. |
| `LORRAX_FFT_FFI_CHUNK` | unset (→ auto) | C++ (`fft_flat_k_ffi.cc::chunk_elems`): trail elements per FFT chunk.  Auto sizes the per-thread compact buffer (`nk·chunk·16 B`) to ~512 KiB so it stays in a CLX core's 1 MiB L2 (the strided→strided form measured 2.8× slower single-thread).  Experiment knob; the unit gate sweeps it and exercises ragged chunks.  Unrecognized values announce and use the auto policy.  Alias policy: `LORRAX_MKLFFT_CHUNK` deprecated, warned once. |
| `LORRAX_CPU_SKIP_GPU_PLUGINS` | `1` (on) | Skip jax's CUDA PJRT plugin discovery when the run is CPU-only (`JAX_PLATFORMS=cpu`, or a GPU was requested but no NVIDIA device exists on the node) — `runtime/__init__.py::skip_gpu_plugin_discovery`.  Loading the plugin on a cold Frontera node measured **76.9 s** of extra start (job 7882076).  `0` re-enables discovery; that is recorded as a DEMOTION and announced.  Two-valued `runtime._env_falsy` grammar.  Never touches a run where a GPU is stated AND present. |
| `LORRAX_GRAM_COL_BLOCK` | `""` (→ auto) | Column-block width for the pivoted-Cholesky Gram update (`centroid/pivoted_cholesky.py:819`).  Falsy tokens (`""`/`0`/`false`/`no`/`off`) select the auto policy (sized from `memory_per_device_gb`, floor 256); a positive integer pins (floor 256); anything else REFUSES naming the accepted grammar.  Perf/memory only. |
| `LORRAX_FH_ORTHO_TOL` | `1e-6` | Cap for the `build_fH_R` ctilde-orthonormality HARD GATE (`bandstructure/htransform.py::resolve_fh_ortho_tol`): `max|C Cᴴ − I|` above the cap refuses, because fH's eigenvalues are then not f(ε) and on-grid energies are wrong by ≈9.0e3× the residual, silently.  `0` disables — only to reproduce a known-bad run.  Blank/unset → the default (a blank used to silently DISABLE the gate — fixed by the P1 audit); garbage REFUSES (`gw_config.env_float` refuse mode); a non-default value is announced. |
| `LORRAX_SCALAPACK_ALLOW_SLATE_API` | off | C++ (`cpp/scalapack/blacs_grid.h:273`): waives the refusal against SLATE's `libslate_scalapack_api` overlay answering the pzheevd/pzgetrf-family symbols.  Standard boolean spellings, case-insensitive; a malformed value is announced and takes the SAFE direction (still refused).  See `SLATE_SCALAPACK_TARGET` in §5 for the overlay's own target demotion. |
| `LORRAX_FFT_FFI` | unset (→ `on` — REQUIRED) | The flat-k FFT backend (`common/fft_helpers.make_flat_k_*` → `ffi/fft.py`; factory-time read, kernel caches key on it). **REQUIRED since the 2026-08-01 ruling**: `0` REFUSES — the XLA flat-k twin was deleted, so there is nothing to opt out to. c128-only; any other dtype refuses at trace time. **SCOPE:** the dial reaches ONLY `make_flat_k_*` call sites — `gw/` + `bandstructure/htransform`. **BSE has no `make_flat_k_*` call site and is NOT affected on either platform**; its FFTs ride `fft_helpers.local_ifftn3`/`local_fftn3`, aliases of `jnp.fft.ifftn`/`fftn` kept deliberately by the ruling. *Which engine answers is not this variable's business and is not recorded here — see `docs/architecture/ffi_layout.md` §3. This row previously said "MKL FFT (DFTI API) on cpu meshes", which has been false since 2026-08-05: `mklfft/fft_flat_k_ffi.cc` contains zero `DftiCreateDescriptor` calls and four `fftw_plan_many_dft`, and the engine is resolved by `dlsym` against whatever the `.so` links. Re-verified 2026-08-06.* |
| `LORRAX_FFTW3_SO` | unset (→ the candidate ladder) | **Names the FILE the host FFT engine is `dlopen`'d from** (`ffi/cpp/mklfft/fft_flat_k_ffi.cc::fftw3_candidates`, run time, first FFT). Deployment plumbing for a site whose SONAME nobody guessed; it is tried FIRST, ahead of the build's own `LORRAX_FFTW3_SO_HINT` (a **compile-time** `-D`, not an env var) and ahead of `libfftw3.so.3` / `libfftw3.so.mpi31.3` / `libmkl_rt.so` / `libfftw3.so`. Bad path → skipped silently, the ladder continues; no engine at all → startup refusal naming every candidate tried. **This selects WHICH ENGINE ANSWERS, and nothing verifies the one it names is a CPU FFTW3.** The Shifter image ships `libcufftw.so.11`, which exports all three entry points the ladder binds (`fftw_plan_many_dft`/`fftw_execute_dft`/`fftw_destroy_plan`), so pointing this at it turns every FFT cell green while the *host* handler transforms on the GPU. The gate for that (GATE 8, `ffi/cpp/gate_one_fftw.sh`) is written but **not merged into this branch** — see `docs/architecture/ffi_layout.md` §3. Unregistered until 2026-08-06: it was the one name hidden by the universal-glob defect in this page's own enforcement line. |
| `LORRAX_FFT_FFI_FUSED` | unset (→ `on`) | The fused IFFT·(G·W)·FFT τ-kernel entry (`gw/ppm_tau_kernel.py` → `make_flat_k_gw_conv`); independent of `LORRAX_FFT_FFI`. Default ON since 2026-08-01 (the certified production form). `0` is a real, announced opt-out onto the decomposed three-transform chain — which is itself FFI-served through the same required handlers, so this is a structural choice between two certified FFI forms, not a native-JAX fallback. |
| `LORRAX_WFN_BACKEND` | `""` (→ config/auto) | Forces the WFN read backend: `eager` \| `phdf5` (`services/wfn_loader/src/wfn_loader/loader.py::_auto_pick_backend`).  **`phdf5_host` was a third value until 2026-08-06 and is now REFUSED, not remapped** — it named an h5py union read that used the eager backend's own POSIX transport with a different unfold kernel, and was auto-selected only by a missing FFI `.so`; silently resolving a deleted spelling to another backend is how an A/B measures the arm nobody asked for.  With no loadable FFI at P>1 the auto-pick now refuses quoting `probe_target`, and this variable set to `eager` is the way through. |
| `ISDF_JAX_CACHE_DIR` | `$SCRATCH/lorrax_jax_cache`, else `$XDG_CACHE_HOME/isdf_jax_compilation` (→ `~/.cache/...`) | JAX persistent compile-cache dir (`common/jax_compile_cache.py:1063`, where the variable is read; the default chain resolves in the lines below it). Default chain is **$SCRATCH-first** (announced once, rank 0): a populated world size is several hundred small files (886 measured for a big deck), and the home-filesystem fallback eats the inode quota that locks a user out of the machine — with the cache on by default at every P, the safe location is the CODE's default, not a docstring plea (release audit 2026-07-28).  The no-`$SCRATCH` fallback announces the inode warning.  `""` (or whitespace-only) opts out entirely — **announced on rank 0 since AT** (the silent opt-out let harnesses keep exporting `""` for the pre-AH deadlock reason long after AH removed the deadlock). **Entries live in ONE shared `{base}/np{P}/` per world size and the cache is ON at every P** (scorecard AH). It used to be refused at `process_count() > 1` because the old per-rank `rank{i}/` layout combined with JAX's process-0-only write (`jax/_src/compiler.py::_cache_write`) guaranteed a divergent hit/miss pattern, and XLA:GPU compilation is a collective (`xla::gpu::AutotunerPass` → `MultiProcessKeyValueStore` → `CoordinationServiceAgent::GetKeyValue`) → permanent silent hang (scorecard AG). AH replaces the refusal with a real fix: a process-invariant cache key, a coordination-service agreement on the usable entry set taken before any compile, atomic writes, and JAX's rank-asymmetric XLA sub-caches disabled. |
| ⚠ **the whole `LORRAX_JAX_CACHE_*` family, on the GPU leg** | — | **Currently inert at P=1 and the failure is silent.** Measured 2026-08-06: with `ISDF_JAX_CACHE_DIR` unset, JAX's native cache works and writes entries; with LORRAX's patch armed, `enabled` reports `True` and **zero entries are written**. Not the container — the native path works in the same image. Every row in this family below describes a design that is currently reaching the **CPU leg only**; read them that way until it is fixed. `docs/environment/overview.md` §2 carries the finding. |
| `LORRAX_JAX_CACHE_MULTIPROCESS` | `1` (**on** — was `0`) | Historically the escape hatch that "re-armed the deadlock"; now the DEFAULT path. Set it to `0` to restore the scorecard-AG refusal (no persistent cache at all when P > 1) if you ever need to bisect against it (`common/jax_compile_cache.py`). |
| `LORRAX_JAX_CACHE_INVARIANT_KEY` | `1` (on, P>1 only) | Makes jax's persistent-cache key process-invariant (forces the GPU-only device-assignment strip on every platform and canonicalises the accelerator-config hash). MEASURED without it (4 CPU ranks): rank 0 hits 7/7 while ranks 1-3 hit 0/7 and compile — which is the AG divergence, so `0` does not merely lose the win, it switches the cache OFF at P > 1 rather than let process 0 hit alone. The per-rank component is `accelerator_config` (the serialized topology), not the device assignment. |
| `LORRAX_JAX_CACHE_AGREE_TIMEOUT_S` | `300` | Timeout for the P>1 hit/miss agreement. On expiry the run degrades to cache-off with a printed reason; it never hangs. |
| `LORRAX_JAX_CACHE_STRICT` | `1` | An entry all ranks agreed on but this rank cannot load is a divergence hazard → abort loudly. `0` downgrades to a warning (**unsafe on GPU: can hang**). |
| `LORRAX_JAX_CACHE_FORCE_DIVERGE` | `0` | TEST HOOK / positive control: every rank != 0 pretends its N alphabetically-last cache entries are missing. The agreement must drop them and say so; the run must NOT hang. |
| `LORRAX_JAX_CACHE_NO_AGREE` | `0` | TEST HOOK: shared dir with the agreement layer OFF — the naive design, i.e. the deadlock reproducer. Never use in production. |
| `LORRAX_JAX_CACHE_EXPLAIN` | `0` | Turns on `jax_explain_cache_misses` (JAX logs why each entry was not written/read). |
| `LORRAX_JAX_CACHE_KEYDUMP` | unset | A directory into which **every rank** writes the SET of persistent-cache keys it asked about, `<dir>/rank{i:03d}_of{N:03d}.json`, at exit (tmp+rename, so a reader never sees a short file). This is what makes the cache contract's SYMMETRY arm falsifiable: `xla_compiles` and `vetoed` are per-rank COUNTS, and four ranks that each compiled a private program report exactly the same counts as four ranks that shared one — only the key set separates them. Not `LORRAX_JAX_CACHE_EXPLAIN`, which names only the keys that MISSED and therefore prints nothing at all on a run where every rank hits its own private entries. Consumed by `tests/test_jax_cache_contract.py` via `tests/mesh_launch.py::read_keydumps`; the same list is on `compile_cache_stats()["keys"]` for in-process probes. Diagnostic only — it changes no cache decision. |
| `LX_MESH4_MODE` | unset | Test-harness knob (`tests/mesh_launch.py`): pins how a four-PROCESS leg is launched — `srun` (inside a Slurm allocation), `local-cpu` (four local processes wired by an explicit jax coordinator, CPU backend), or `none`. Unset auto-detects, and `srun` always wins when it is usable so the CPU emulation can never quietly stand in for the real P=4 leg. |
| `LX_MESH4_DECKS` | `0` | Test-harness knob: run the cache contract's full DRIVER-DECK arms even when the only available launch is `local-cpu`. Off by default because a CPU mesh is fine for device-count logic and never substitutes for the P=4 leg on a GPU path (`AGENT_PREAMBLE.md`, the four-GPU rule), so off-cluster the deck arms skip with that reason while the red twins still run. |
| `LORRAX_JAX_CACHE_SHARD_SLICE` | `1` (on, P>1 only) | Canonicalizes jax's own `ArrayImpl._multi_slice` so every rank compiles ONE program (`common/jax_compile_cache.py::_install_shard_slice_patch`). That jit is `static_argnums=(1,2,3)` and `jax/_src/array.py::shard_device_array` calls it with **this rank's** shard bounds (`sharding.addressable_devices_indices_map`), so out of the box each rank compiles `slice(x, [r·n/P, …], …)` — a different module, a different cache key, and, because JAX writes entries from process 0 only, rank 0 hitting while its peers miss and compile. The patch keeps the shard SIZES static and passes the OFFSETS as dynamic `lax.dynamic_slice` operands; in-bounds unit-stride `dynamic_slice` is exactly `slice`, so values are bit-identical. MEASURED A/B at P=4 on `si444`, warm, one env var apart: **`0`** gives `rank 0: xla_compiles=0 hits=36` against `ranks 1,2,3: xla_compiles=1 hits=35 vetoed=1` with three distinct `jit__multi_slice-*` keys; **`1`** gives `xla_compiles=0 hits=38 vetoed=0` on all four ranks and no `jit__multi_slice` at all. `0` is the TEST HOOK / red twin — it restores the divergence and is never a production setting. Reached whenever a single-device `jax.Array` meets a **partitioned** sharding (a fully-replicated target short-circuits before the jit, and a numpy source never enters it); `runtime/__init__.py::nccl_warmup` does exactly that at mesh bootstrap, so this is on the path of **every** P>1 GPU run, not just BSE. The residual it does NOT fix is a ragged sharding, where the shard SHAPES differ across ranks and no single program can serve them. |
| `LORRAX_JAX_CACHE_PREFETCH` | `1` (on, P>1) | After the agreement, pulls the agreed entries into the page cache from a thread pool. Cache entries are tiny (876 kB for 140 of them) so reading them is pure per-file Lustre latency: measured **29 s serial** at 606c/P=16, against ~4.5 s of XLA compile saved — without this the cache is a net loss on a cold-read CPU run. `LORRAX_JAX_CACHE_PREFETCH_THREADS` (16) sets the pool size. |
| `LORRAX_MINIMAX_CACHE_DIR` | `~/.cache/lorrax/minimax_quadratures` | Where the minimax service caches rules it had to solve at runtime (`services/minimax/src/minimax/cache.py`). Cached rules are NEVER certified, and since 2026-08-08 the key records the solver version and the numerics backend; entries written under the old unversioned key are still read, and announce once that their provenance is unknowable. |
| `LORRAX_DISABLE_MINIMAX_DISK_CACHE` | `""` (off) | Disables that disk cache entirely (`services/minimax/src/minimax/cache.py`). Same spelling and same effect as before the service extraction. |
| `LORRAX_MINIMAX_ALLOW_RUNTIME_SOLVE` | `1` (**on** — stage 1) | The escape hatch on lookup-and-refuse. With it on, a request no shipped table covers is solved in-process and ANNOUNCED once, naming the request, the achieved error, the measured Σ\|w\| and κ₀, and the words *uncertified, not reproducible across hosts*. Set to `0` and the same request REFUSES instead (`services/minimax/src/minimax/door.py`). It defaults on because the imaginary-axis family ships no tables at all, so arming the refusal today would refuse the default GN-PPM deck; flipping the default is gated on generating that family. |
| `LORRAX_PHDF5_STRIPE_COUNT` | **Python `clamp(nranks, 4, 128)`; C++ `16`** ⚠ | Lustre stripe count.  **Read at exactly 2 sites** (re-counted 2026-08-06): `_slab_io_ffi.py:167-205` (`_stripe_count()`) and C++ `context.cc:458-481`.  This row previously claimed 4 sites and the consistency table claimed 3; `gw/isdf_fitting.py` contains no `LORRAX_PHDF5` reference at all.  (The second Python reader, `_slab_io_mpi_host.py`, imported `_stripe_count` rather than re-reading it, and was deleted with the host tier on 2026-08-06.)  ⚠ **The two writers do not share a default.**  `e5c9618` replaced the Python literal `16` with `_stripe_policy(nranks)` — the stripe count *is* the ROMIO aggregator count (`cb_nodes = min(striping_factor, nranks)`), so it scales with the world — while `context.cc:481` still writes the literal `"16"` on the unset path.  With the variable unset the two writers request **different layouts**.  Ruling and status: [`architecture/decisions.md` 2026-08-05](../architecture/decisions.md).  **The two sites DO agree on parse since 2026-08-06**: both refuse a non-integer, and both refuse a negative count (a negative `striping_factor` means "every OST on the filesystem", the maximum-contention layout).  Until then C++ forwarded the string to `MPI_Info_set("striping_factor", …)` uninterpreted, so `sixteen` was a loud error in one writer and a silently-ignored hint in the other. |
| `LORRAX_PHDF5_STRIPE_SIZE_FS` | **Python `1M`→`4M` by rank count; C++ `1M`** ⚠ | Lustre stripe size, in the `lfs setstripe -S` spelling.  The measurements that chose it belong to `docs/architecture/slab_io.md` §Tuning and are not repeated here.  Read at `_slab_io_ffi.py:211-250` (moved there by `e5c9618` to sit beside `_stripe_count`, because the two were being resolved in two modules with two different notions of "the default") and C++ `context.cc:484-524`.  On the unset path Python takes `_stripe_policy(nranks)[1]`, which ramps the unit 1 → 4 MiB with the rank count and clamps there; C++ keeps a flat 1 MiB.  The byte-valued `LORRAX_PHDF5_STRIPE_SIZE` still works as a C++-side legacy fallback, and since 2026-08-06 also refuses a non-integer instead of silently keeping the default.  **Both sides now refuse an unknown suffix**, and both refuse `4MiB` (one suffix character only) — verified against the Python function on the same four inputs.  ⚠ **Three stale copies of the old flat `4M` default survive in source comments and sibling docs** — `context.cc:15`, `src/ffi/phdf5/ARCHITECTURE.md:316`, `src/ffi/PORTING.md:535` — while the code on both sides now resolves as described above.  Found 2026-08-06; those files are owned elsewhere, so this is reported, not edited. |
| `LORRAX_PHDF5_MPI_STACK` | `mpich` | Build+launch: which MPI the phdf5 FFI links/loads (`run_shifter.sh:42`). |
| `LORRAX_PHDF5_ALIGN_MB` | `4` | C++: `H5Pset_alignment` threshold, MiB; `0` disables. Deliberately NOT tied to `STRIPE_SIZE_FS` (it was justified that way when both were 4M). MEASURED non-load-bearing at 16×1M striping, job 56389339: `4`/`1`/`0` gave 0.830/0.809/0.813 GiB/s write at 1 node and 2.975/2.883/2.915 at 4 nodes — all inside the ±1.5 % repeat noise. Left at 4 rather than becoming a second knob to keep in sync. |
| `LORRAX_PHDF5_COLL_META` | `0` | C++: `1` re-enables collective metadata ops (default off is faster). |
| `LORRAX_PHDF5_CB_NODES` / `LORRAX_PHDF5_CB_PER_NODE` / `LORRAX_PHDF5_CB_BUFFER_SIZE` / `LORRAX_PHDF5_CB_WRITE` / `LORRAX_PHDF5_DS_WRITE` | unset (→ ROMIO auto) | ROMIO collective-buffering pass-throughs; unset means ROMIO's automatic policy in both writers (unified by AW). **Corrected 2026-08-06 — "forwarded VERBATIM by both writers" was wrong twice.** (a) `LORRAX_PHDF5_CB_PER_NODE` is **C++-only**: the Python writer's forwarding loop (`_slab_io_mpi_host.py:173-181`) covers `CB_WRITE`/`DS_WRITE`/`CB_NODES`/`CB_BUFFER_SIZE` and not this one. (b) It is **not verbatim** even in C++ — `context.cc:423-427` rewrites it as `cb_config_list = "*:" + value`. The other four are genuinely verbatim in both. The C++ writer used to FORCE Perlmutter-era defaults (`romio_cb_write=enable`, `romio_ds_write=disable`, `cb_buffer_size=64M`, `cb_nodes=world_size`); on Frontera forcing `romio_cb_write=enable` measured *slower* than ROMIO auto (wk_AI: 1826 vs 2066 MB/s) and the rest were never revalidated off Perlmutter. Keep them unset; they exist as A/B levers. |
| `LORRAX_PHDF5_INDEPENDENT` | off | Independent instead of collective MPI-IO **reads** (C++). |
| `LORRAX_PHDF5_COLLECTIVE_WRITES` | `1` (on) | Collective (two-phase) MPI-IO for the bulk **writes**.  Read TODAY at exactly one site, C++ `ffi/cpp/phdf5/context.cc:env_flag`.  (It was read in BOTH writers when this row was written — the second reader was the Python `phdf5_host` writer `file_io/_slab_io_mpi_host.py:_env_flag`, which shared the C++ boolean grammar from the fix/zq audit onward; before that audit, word spellings like `false` flipped only the Python writer.  That file was **deleted with the host tier on 2026-08-06** — see the deliberately-historical note at the top of this page — so the shared-grammar clause is now history, not a live claim about two writers.)  Default flipped 0→1 on this branch, MEASURED (wk_AI microbench, production tile geometry): `V_qmunu`'s strided 2-D tile decomposes into 4.1 M × 3.2 kB independent writes at P=144 (scorecard AF.4c's 1.7 MB/s), vs a few large aggregated writes per ROMIO aggregator under `H5FD_MPIO_COLLECTIVE` — ~3 orders of magnitude on the same transport.  `0` restores independent writes (pre-AI behaviour).  Cray caution: the `ad_cray_write_coll.c` OOM at ≥1 GB/rank aggregates predates this default — recorded in context.cc; not reproduced at 512 MiB/rank in the 2026-08-05 Perlmutter campaign. **Perlmutter revalidation done** (job 56389339): keep `1`. Independent writes are not uniformly slower — at 16×1M striping they measured 2.927 GiB/s at 4 nodes and 0.915 at 1 node, level with collective — but at 16×4M striping they collapsed to **0.068 GiB/s at 4 nodes**, a 43× spread across one unrelated knob. Collective never varied more than ~10 % from the best at its geometry. The default is chosen for bounded worst case, not mean throughput; see `docs/architecture/slab_io.md`. |
| `LORRAX_PHDF5_DEDUP_REPLICAS` | `1` (on) | One canonical writer per distinct hyperslab when a mesh axis is replicated (both writers).  `0` lets every replica rank write its identical copy — merely wasteful under independent MPI-IO, **undefined behaviour** (overlapping selections) under collective MPI-IO.  Debug-only off. |
| `LORRAX_PHDF5_REQUIRE_MPI_WORLD` | **`1` (require)** | **Added to this doc 2026-08-06; it was an undocumented fail-closed run gate.** `_slab_io_ffi.py:266`. At the first collective PHDF5_FFI open the backend asks MPI for `MPI_Comm_size(MPI_COMM_WORLD)` and compares it to `jax.process_count()`. A mismatch always refuses. This knob controls only the *unprobeable* case: `1` makes "could not determine the MPI world" a refusal, `0` downgrades it to a warning. Why it matters: `ffi.io.open_file` and `shard_index.h::validate_shard_encoding` both compare JAX to JAX and agree, so without this probe a PMI-flavour mismatch gives every rank a private singleton `MPI_COMM_WORLD` and 16 unsynchronised writers on one file — measured at job 56389339, bit-exact readback with rc=0 and no warning anywhere. **Grammar note: this is a falsy-set test** (`not in ("0","false","no","off")`), the inverse of the `_env_flag` truthy-set used by its neighbours. |
| `LORRAX_HDF5_ONE_OWNER` | `measure` | The one-owner-per-file policy (`file_io/hdf5_owner.py::policy`). **Grammar: an enum, not a boolean** — trimmed, lowercased, and exactly one of `measure` \| `strict`; **any other value REFUSES** at `policy()`, naming both, rather than falling back. `measure` counts sequential cross-stack alternation on one file and reports it per path; `strict` promotes that count to the same refusal. **The live-overlap-with-a-writer refusal is unconditional under both** and is not what this knob controls. Why the process needs it at all, what the two HDF5 library instances are, and the 1027-alternations measurement: [`architecture/slab_io.md#one-owner`](../architecture/slab_io.md#one-owner). |
| `LORRAX_PHDF5_SKIP_MPI_WORLD_CHECK` | off | **Added 2026-08-06; previously undocumented.** `_slab_io_ffi.py:262` — early-returns from `_assert_mpi_world`, disabling the guard above entirely. Debugging escape hatch, **never a remedy**: it removes the only check that distinguishes a genuine Cray collective-buffer OOM from a PMI-mismatched launch. Grammar: the shared `1/true/yes/on` set, case- and whitespace-insensitive. It was missing `.lower()` until 2026-08-06 — two lines from a sibling that had it — so `=TRUE` and `=On` did NOT enable it. It failed closed, which is why it survived unnoticed. |
| `LORRAX_SCALAPACK_MKL_THREADS` | unset (→ `auto` = cap 4) | MKL team size pinned (thread-locally) inside the ScaLAPACK FFI handlers (`eigh`, `solve_lu`). MEASURED (wk_ENV AW, pz_bench n=2448, mlx): at the production 12×12 grid pzheevd is **11.28 s/q at 14 MKL threads vs 0.463 s/q at 4** (24×); 17.0 s/q oversubscribed at 28; 8×8 at the production 2×28 placement shows the same cliff; 4×4 (P=16) is flat. `auto`/unset caps the calling thread's MKL team at 4 (only when the global setting is larger); `off`/`0` restores the pre-AW behaviour (handler inherits `MKL_NUM_THREADS`, i.e. 28 in production — do not do this at P ≥ 64); an integer pins exactly that. The global `MKL_NUM_THREADS=28` stays right for the local `zheevd_` plan-A route and is untouched.  Values are case-insensitive since the fix/zq audit, and an UNRECOGNIZED value announces loudly on stderr and falls back to `auto` — it used to fall through `atoi()` to `off`, silently restoring the 24× configuration. |
| `LORRAX_FORCE_REFIT` | `""` (off) | `1` (or any standard truthy value: `true`/`yes`/`on`, case-insensitive — `gw_config.env_bool` since P1.3; unknown tokens announce) forces the ζ fit even when `tmp/zeta_q.h5` is complete and its provenance matches (`gw_init.py`).  Ops escape hatch for the ζ-reuse cache (the reuse gate itself is provenance-checked). |
| `LORRAX_ALLOW_PARTIAL_ZETA` | `0` | Permits reading a ζ file whose `zeta_is_done` flag is unset (`services/zeta_loader/src/zeta_loader/loader.py`).  Forensics-only: a half-written ζ is otherwise indistinguishable from a complete one (QUALITY_PATTERNS #7). |

## 3. Debug / diagnostic env

None of these change results.  The print knobs that are ON by default
are on because their absence has already cost 72-node hours (the AC/AF.4c
observability failures) — cheap keeps, not clutter.

### 3a. Always-on telemetry (set to `0` to silence)

| var | default | effect |
|---|---|---|
| `LORRAX_ZETA_RANK_LOG` | `1` (on) | Per-q `n_keep / λ_max / λ_min(kept)` from the rank truncation (`isdf/core.py`, both tiers).  **The conditioning signal of the μ ladder** — leave it on. |
| `LORRAX_COLLECTIVE_CHUNK_LOG` | `1` (on) | One line per ζ-tier/W call site naming the emitted per-collective payload, q-block and cap (`isdf/core.py::_chunk_log`).  A tier that silently stopped chunking would otherwise be invisible until it took a 72-node job down again. |
| `LORRAX_RESTART_WRITE_LOG` | `1` (on) | One rank-0 line per dataset in `write_restart_state_to_h5` (name, shape, GB, s, MB/s) + an explicit `W0_qmunu placeholder ALLOCATED` line (`file_io/tagged_arrays.py`).  The instrument whose absence made AF.4c's silent 2 h 55 m undiagnosable. |
| `LORRAX_PHDF5_CLOSE_VERBOSE` | `1` | Chatter on phdf5 close (`file_io/_slab_io_ffi.py`). |
| `LORRAX_H5_JOURNAL` | `1` (on) | The per-rank HDF5 operation journal (`file_io/h5_journal.py`): one line per open/close/create/read/write/attr touch, written BEFORE the call, line-buffered, at the three existing choke points (`file_io/hdf5_owner.note_open`/`note_close`, the `SlabIO` methods, and `_slab_io_ffi`'s open/ensure/close lifecycle calls). Output `<cwd>/h5_journal.rank<R>.log`; a 256-line ring lands in `h5_journal_crash.rank<R>.txt` on a registry refusal, on any exception crossing a `SlabIO` method, and at `atexit` if either happened. **Grammar: three states, not a boolean** — `1`/`true`/`yes`/`on`/empty = on, `0`/`false`/`no`/`off` = off, `sync`/`fsync` = fsync after every line. An unrecognised token REFUSES naming the variable (a mis-set toggle must not silently lose the evidence for the run it was set to instrument). ON by default because all three failure signatures in `reports/metal_mpa_plan_2026-08-15/SLAB_IO_ROOT_CAUSE_AUDIT.md` are NATIVE deaths that leave nothing Python can inspect, and the cost is ~1-2 µs/line against a measured ~1030 HDF5 ops per SC iteration — order 1e-5 of a ~280 s iteration. `sync` costs ~1 ms/line: opt-in only, for segfault-grade capture. |
| `LORRAX_H5_JOURNAL_DIR` | unset (→ `os.getcwd()`) | Where the journal and crash-ring files land (`file_io/h5_journal.py`). The default is the process's working directory, which is the run directory for every LORRAX driver launch. A directory that cannot be created disables the journal with ONE warning and lets the run continue — an instrument, never a gate. |
| `LORRAX_HDF5_ONE_OWNER` | `measure` | One-HDF5-library-instance-per-open-file policy (`file_io/hdf5_owner.py`, audit A1 / sandbox claims/0110): `measure` counts sequential cross-stack alternation on one file and reports it in the `[hdf5-probe]` line; `strict` refuses it, and also turns `probe()`'s measured unsafe condition (two mapped libhdf5 objects AND a file written through both) into a refusal. The LIVE-overlap-with-a-writer refusal is unconditional under both. Any other token REFUSES naming the variable. **Added to this page 2026-08-15** — it was read through a module constant, which `tools/env_audit.py` and `tests/test_env_registry.py` cannot resolve, so it shipped unregistered and the gate stayed green; the read is now a literal and the gate covers it. |
| `LORRAX_PHDF5_LOG` | `1` (on) | One rank-0 banner per `phdf5_host` file open naming the EFFECTIVE collective-write / dedup / striping configuration (`file_io/_slab_io_mpi_host.py`) — the router-prints-its-decision rule (#8). |
| `LORRAX_SANITY` | unset = warn | Stage-boundary invariant checks (`common/sanity.py`): `0`/`off` skips all (escape hatch); default checks and **warns loudly but keeps running** (a false positive must never kill a 40-node job); `strict` raises `SanityError` — set in CI / regression gates. |

### 3b. Opt-in probes, dumps, test hooks

| var | default | effect |
|---|---|---|
| `LORRAX_TIMING_TRACE` / `LORRAX_TIMING_TRACE_DEPTH` | `0` / `3` | Every `timing.section` announces entry/exit on rank 0, nesting capped at the depth (`common/timing.py`).  A malformed depth REFUSES naming the variable (the parse that swallowed `ValueError` killed every rank before). |
| `LORRAX_SIGMA_TAU_TIMING` | `0` | Per-stage blocking timing rows for the staged τ kernel (`gw/ppm_tau_kernel.py:66`; the sub-rows are documented at `gw/ppm_sigma.py:377`).  Numerics identical (same primitives, same order, separate XLA modules); walltime NOT comparable to the fused path.  Scale-neutral: O(1) host work per τ stage. |
| `LORRAX_PPM_HERM_DIAG` | `0` | Deck-level ε_H measurement of the PPM amplitude's inherited hermiticity residual — `check_hermitian` over B_q and Ω_q, all q, rtol 1.0 (`gw/ppm_sigma.py:275`).  Diagnostic, not a gate (the channel merge needs no hermiticity). |
| `LORRAX_EXTRA_RANK_PAD` | `""` (→ 0) | **Test-only** extra null directions on the htransform Galerkin rank axis, on top of the mesh-lcm round-up (`bandstructure/htransform.py::resolve_extra_rank_pad`) — the pad-extent-invariance knob for this axis, exactly the role `LORRAX_EXTRA_MU_PAD` plays for μ.  Any result that moves under it at fixed P is a defect.  Negative or malformed REFUSES.  NEVER set in production.  There is deliberately no `LORRAX_EXTRA_BAND_PAD` counterpart — ruling and precondition in [`architecture/decisions.md` 2026-08-06](../architecture/decisions.md).  Exercised by `tests/test_pad_parity_gates.py` — until 2026-08-06 the ONLY test-suite mention was `test_layering.py`'s `_L1_LIBRARY_ENV_READS` registry, whose two consumers are `ast.parse` static analysis and cannot tell a working resolver from a dead one. |
| `LORRAX_FFT_FFI_LOG` | unset (presence-test) | C++ FFT-handler debug logging, ONE spelling on both platforms (`cpp/mklfft/fft_flat_k_ffi.cc::log_enabled`, `cpp/cufft/fft_flat_k_cuda_ffi.cc::log_enabled`).  Rank-scoped since the 2026-07-30 audit: rank 0 by default, `=all` for every rank (at P=1000 the old all-ranks default multiplied a 5-line answer by the process count).  Alias policy: `LORRAX_MKLFFT_LOG` and `LORRAX_CUFFT_LOG` are deprecated aliases, honored on BOTH platforms with a one-time announcement. |
| `LORRAX_SCALAPACK_PROVIDER_LOG` | unset (presence-test) | C++ (`cpp/scalapack/blacs_grid.h:369`): one stderr block, once per process, naming which library resolves each of the 11 ScaLAPACK/BLACS symbols this build calls.  Opt-in — noise in a healthy build, the first question worth asking the moment a port misbehaves. |
| `LORRAX_MEM_DEBUG` | unset | Memory high-water probes at named sites (`gw_init.py`, `isdf_fitting.py` ×3).  All four read it through `gw_config.env_bool`; they were bare presence-tests until 2026-07, under which `LORRAX_MEM_DEBUG=0` turned the probes ON. |
| `LORRAX_RCHUNK_DEBUG` | unset | Per-r-chunk shape/timing/RSS lines (`isdf_fitting.py`, `isdf/core.py`) — the instrument behind every per-chunk table on the scorecard.  Both sites read it through `gw_config.env_bool` since P1.3 (the historical presence-test split in `isdf/core` — `=0` turned one site off and the other ON — is closed; see the consistency table). |
| `LORRAX_EXIT_AFTER_ZETA` | unset | Clean `SystemExit(0)` right after the ζ fit (`gw_init.py`).  Combine with `LORRAX_MAX_RCHUNKS` + `LORRAX_RCHUNK_DEBUG` for fast fit-only sweeps (the AV chunk matrix ran exactly this way). |
| `LORRAX_W_RESIDUAL_CHECK` | `0` | Prints the direct Dyson residual `‖(1−Vχ)W − V‖/‖V‖` on the first few q after a `w_dyson_solver = distributed` W solve (`gw/w_isdf.py`) — the strict numerical contract of the distributed plan (block-cyclic LU is not bit-comparable to the local per-q LU).  Adds one diagnostic jit; leave OFF when taking collective-table probes. |
| `LORRAX_CHECK_REPLICA` | `0` (off) | Re-enables `jax.device_put`'s cross-process equality assertion inside `common/collectives.py:device_put_process_local` — i.e. deliberately pays the hidden P-linear all-gather (7.8 GB/rank at P=64, scorecard Y.5/AO) the helper exists to avoid, to verify a host table really is replica-identical. Debug only; standard falsy vocabulary (`""`/`0`/`false`/`no`/`off`, case-insensitive, since AT). |
| `LORRAX_SKIP_VQ_GATES` | `0` | Skips the V_Q interpolation self-checks (`bse/vq_interp.py`).  The gates exist because V_Q interpolation errors are silent. |
| `LORRAX_TRS_CHECK` | `1` (on) | Load-time MEASUREMENT of the WFN's symmetries against the density built from its own occupied ψ (`symmetry_maps.check_density_symmetries`, called from `wfn_loader`; contract in [`docs/services/symmetry_maps.md`](../services/symmetry_maps.md)).  `0` restores pre-2026-07 flags-only behaviour — and re-arms the scorecard-Q class of silent time-reversal; `strict` raises instead of warning.  A diagnostic whose measured verdict gates `SymMaps` TRS augmentation — default on. |
| `LORRAX_TRS_TOL` / `LORRAX_TRS_SPATIAL_TOL` / `LORRAX_TRS_MAX_K` | `1e-6` / `1e-4` / `12` | Tolerances / k-sample cap for that check.  Fixture residuals ≤ 1e-12; real magnetization is O(1e-2…1); the sample is ±-closed, so `MAX_K` is a sufficiency choice, not an approximation (`0` = all k).  `MAX_K` default is `12` (`symmetry_maps/density_symmetry_check.py::MAX_K_DEFAULT`) — this row used to say 32, which nothing ever implemented; 12 is the MEASURED choice (scorecard §U 12×12 k-ladder: verdict and residual order flat from 9 k to all 144 k, 2.5e-13 → 3.1e-14, cost linear in k). |
| `LORRAX_RHO_SYMMETRISE` | `1` (on) | Projects the valence density onto the subspace invariant under the WFN file's space group before V_H is built (`psp/get_DFT_mtxels.py::symmetrize_valence_density`, applied in `build_hartree_potential` and `compute_valence_density`).  The star average is a projector: it preserves ∫ρ exactly and moves ρ by no more than the asymmetry it removes, so it is on by default — an unsymmetrised ρ is simply wrong on a reduced k-set, and on a full-BZ sum it leaves V_H only as star-invariant as the ψ unfold that built it while every other term of H₀ is exactly star-invariant.  `0`/`off` restores the raw accumulated ρ; A/B and suspect-symmetry-block use only. |
| `LORRAX_FORCE_FULL_BZ` | off (5 sites, one grammar) | `1` disables IBZ-only ζ writes / bypasses the IBZ cascade (`gw_init.py` ×3, `screening.py`, `v_q_g_flat.py`).  Debug/test bypass: changes work done and bytes written, not physics.  All five read it through `gw_config.env_bool` (converted together, 2026-07-30) — so `on`/`true`/`yes` work and an unrecognised token is announced rather than silently resolved.  They were `bool(int(...))`, which accepted digits only. |
| `LORRAX_EXTRA_MU_PAD` | `""` (→ 0) | **Test-only** extra μ-pad rows to prove pad-extent invariance (`runtime/padding.py`).  Any result that moves under this at fixed P is a defect.  NEVER set in production. |
| `LORRAX_FFI_DEBUG_SHARDS` / `LORRAX_WRITE_NO_JIT` | unset | Slab-writer shard-descriptor dumps / un-jitted write path (`_slab_io_ffi.py`). |
| `LORRAX_FFI_PROFILE` | off | C++: per-call FFI timing. |
| `LORRAX_SCALAPACK_EIGH_LOG` | unset (presence-test) | C++ `cpp/scalapack/eigh_ffi.cc`: prints the pzheevd workspace sizes (lwork/lrwork/liwork, GiB) once per call — the workspace is malloc'd outside XLA and invisible to the JAX-side memory planner. |
| `LORRAX_MKLBLAS_LOG` | unset (presence-test) | C++ `cpp/mklblas/gemm_batch_ffi.cc`: one stderr line on the first GEMM-handler call (dtype, batch/M/N/K, threads, batched-entry vs plain-loop path). |
| `LORRAX_PHDF5_TIME` | off | C++ `read_ffi.cc` (two independent sites, same expression). **Converted 2026-08-06** to `env_flag("LORRAX_PHDF5_TIME", false)`, the shared C++ boolean grammar, so `=0` now turns timing OFF. It was `std::getenv(...) != nullptr` — a pure presence test in which `=0` turned timing **ON** — the same defect the `LORRAX_MEM_DEBUG` warning box in §"consistency audit" post-mortems. It survived because `env_flag` was `static` in `context.cc` and this file could not reach it; it now lives in `phdf5/ctx.h`. |
| `LORRAX_PHDF5_WRITE_DEBUG` | off | C++ `write_ffi.cc:137-138`: non-empty **and not exactly `"0"`**, so `=false` / `=off` / `=no` all ENABLE it. A third grammar again. |
| `LORRAX_PHDF5_DUMP_HINTS` | off | C++ `context.cc:544`: full `env_flag` (the canonical `1/true/yes/on` set), rank 0 only. The only one of the three with the shared grammar. |

The three rows above were a single row reading "off — C++-side phdf5
diagnostics" until 2026-08-06. They have **three different parses**, one of
which inverts on `=0`. (`_SKIP_DESTROY` and `_NO_COLL_META` were listed here
historically but NOTHING reads them — `_NO_COLL_META`'s live replacement is
`LORRAX_PHDF5_COLL_META` in §2b, `_SKIP_DESTROY` survives only as
ARCHITECTURE.md prose; dropped by the fix/zq audit.)
| `LORRAX_LU_DEBUG` / `LORRAX_LU_NO_PIVOT` / `LORRAX_LU_DEBUG_DUMP` | off | cuSOLVERMp LU diagnostics. |
| `KP2_DEBUG` / `STERN_DEBUG` | unset / `0` | Sternheimer debug (`psp/run_sternheimer.py`, `solvers/sternheimer_solve.py`). |
| `PF_ARTIFACTS_DIR` / `ISDF_JAX_PROFILE_DIR` | `profile` / unset | Trace output dirs (`common/jax_profile.py`, `tests/bench/test_bse.py`). |
| `ISDF_COHSEX_TEST_PLATFORM` | `auto` | Test harness: force `cpu`/`gpu` for the e2e gates (`tests/harness.py`). |

## 4. Build-time only

Read by `config/**/*.sh`, `src/ffi/cpp/common/**/build*.sh` and CMake.
Never by the running Python; setting them in a job script does nothing.

> **This section is 120 of the 234 `LORRAX` env names the tree actually
> reads — an `arch.mk` expressed as environment.** Why that is the shape
> to move away from, and which knobs defer a decision the build already
> made, is [`architecture/ffi_layout.md` §3c](../architecture/ffi_layout.md).

`LORRAX_ROOT`, `LORRAX_SRC`, `LORRAX_VENV`, `LORRAX_SITE`,
`LORRAX_SITE_PACKAGES`, `LORRAX_INSTALL_ROOT`, `LORRAX_DEPS`,
`LORRAX_IMAGE`, `LORRAX_SIF`, `LORRAX_SHIFTER*`, `LORRAX_MODULE*`,
`LORRAX_FFI_STAGE*`, `LORRAX_FFI_BUILD_DIR`, `LORRAX_FFI_SOURCES`,
`LORRAX_FFI_HOST_*`, `LORRAX_FFI_IMAGE`, `LORRAX_FFI_PYTHON`,
`LORRAX_FFI_NO_CUDA`, `LORRAX_FFI_HAVE_{PHDF5,CAL,CUBLASMP,CUFFT}`,
`LORRAX_FFI_PLATFORM`, `LORRAX_HOST_HAVE_{FFTW3,SCALAPACK}`,
`LORRAX_CBLAS_{DIR,INCLUDE_DIR,LIBRARY}`,
`LORRAX_FFTW3_{INCLUDE_DIR,LIBRARY}`,
`LORRAX_MKL_{BLACS,THREAD_LIB,SCALAPACK_LIBRARY}`,
`LORRAX_FFI_ALLOW_DEFAULT_MPI`, `LORRAX_FFI_{PHDF5,SLATE,NVHPC,FFTW}_DIR*`,
`LORRAX_FFTW_STAGE_CLOBBER`, `LORRAX_FFTW3_STAGE`,
`LORRAX_GATE_ONE_FFTW`, `LORRAX_GATE_FFTW_PY`, `CRAY_FFTW_PATH`,
`LORRAX_XLA_FFI_INCLUDE_DIR`, `LORRAX_XLA_FFI_HEADERS_DIR`,
`LORRAX_HDF5_ROOT`, `LORRAX_MPI_INCLUDE_DIR`, `LORRAX_MPICH_LIB_DIR`,
`LORRAX_MPI_LIBRARY`, `LORRAX_MPI_TYPE*`, `LORRAX_MPI_FABRICS`,
`LORRAX_IMPI_ROOT`, `LORRAX_PMI2_LIB`, `LORRAX_ICC_RUNTIME`,
`LORRAX_MKL_ROOT`, `LORRAX_SCALAPACK_LIBRARIES`, `LORRAX_NVHPC_*`,
`LORRAX_CUSOLVERMP_{STAGE,PIN}`, `LORRAX_CUBLASMP_PIN`,
`LORRAX_SLATE_*` (`REPO`, `COMMIT`, `BUILDS_DIR`, `MAKE_J`, `STACK`,
`INSTALL_DIR*`, `HOST_INSTALL_DIR`, `CUDATOOLKIT`),
`LORRAX_HAVE_SLATE`, `LORRAX_HOST_HAVE_SLATE`, `LORRAX_DARSHAN_LIB_DIR`,
`LORRAX_LUSTRE_STRIPE_*`, `LORRAX_NO_PRESTRIPE`, `LORRAX_XLA_CMDBUF`,
`LORRAX_SLURM_{ACCOUNT,QOS,CONSTRAINT}`, `LORRAX_NNODES`,
`LORRAX_NTASKS`, `LORRAX_NGPU`,
`LORRAX_GPUS_PER_NODE`, `LORRAX_SELECT_GPU`, `LORRAX_TIER2_WORKDIR`,
`LORRAX_INPUT`.

### 4a. Perlmutter launch + site config (added 2026-08-06)

Everything above §4 is Frontera-shaped. These were in source and in **no**
section of this page. Audited against `lorrax_P` @ `886139f`; the Frontera
tree at `b61c1df` does not have the first four at all.

| var | default | meaning |
|---|---|---|
| `LORRAX_NVHPC_SUBPATH` | `0.7.2_cuda12.9/math_libs/12.9/lib64` (`run_shifter.sh:171`) | The single source of truth for which cuSOLVERMp stage a run loads. ⚠ **It selects a communication path, not just a version, and every stage exports the same SONAME so a mismatch links cleanly and warns about nothing.** That is the whole of what this registry says about it; the stage/comm-path table, the CMake-default skew and the measured evidence are owned by **`docs/architecture/ffi_layout.md` §4** — read it before touching the CUDA leg. |
| `LORRAX_NVHPC_ROOT` / `LORRAX_NVHPC_MOUNT` | no default — **`build.sh:54-91` REFUSES** | The stage the `.so` is COMPILED against, and the bind-mount point (`/lorrax_nvhpc`). Refuses rather than guessing, because there is no safe default: a guess picks a comm path silently. |
| `LORRAX_PLATFORM` | `gpu` (`run_shifter.sh:55-63`) | `gpu` \| `cpu`/`host`. Decides `MPICH_GPU_SUPPORT_ENABLED` for the Shifter launch — it is per platform, not a constant; on the CPU leg it must be 0 or Cray MPICH aborts in `MPI_Init_thread`. |
| `LORRAX_MPICH_GPU_SUPPORT` | derived from the above (`run_shifter.sh:70`) | Explicit `0\|1` override; anything else refuses. Carried under a `LORRAX_` name because shifter's mpich module **unsets** `MPICH_GPU_SUPPORT_ENABLED` itself, so `in_container.sh` re-derives on the far side of that boundary. |
| `LORRAX_GATE_ONE_MPI` | `on` (`cpp/gate_one_mpi.sh:41`) | The "exactly one MPI implementation in this address space" gate (hazard S3). `=off` disables it and says so loudly on every run. |
| `LORRAX_RUN_DIR` | no default — **mandatory** (`config/frontera/templates/gw_dev.sbatch:44`) | The run directory holding the input deck. A peer of `LORRAX_ROOT`/`LORRAX_INPUT`, which were listed while this was not. |
| `LORRAX_PM_{PRGENV,CMAKE,FFTW,HDF5,HDF5_DIR,LIBSCI,LIBSCI_DIR,LIBSCI_FLAVOUR,MPICH_DIR}` | per `config/perlmutter/site_config.sh` | The Perlmutter module/prefix family: PrgEnv, cmake, cray-fftw, cray-hdf5 (+ dir), cray-libsci (+ dir, + flavour) and cray-mpich dir. Consumed by `config/perlmutter/build_ffi_host.sh`. |
| `LORRAX_FFI_{NVHPC,PHDF5,SLATE}_HOST` | `config/modulefiles/lorrax/0.1.0.lua:210-212` | Host-side counterparts of the `_DIR*` knobs above; not matched by this page's `_DIR*` globs. |
| `LORRAX_BUILD_JOBS` | `8` (`config/perlmutter/build_ffi_host.sh:246`) | `cmake --build --parallel N`. |
| `LORRAX_MPICH_CONTAINER_DIR` | `/opt/udiImage/modules/mpich` | In-container MPICH module dir, substituted into the modulefile. |
| `LORRAX_SLATE_{SRC,SCALAPACK_API_DIR}`, `LORRAX_SAPI_EXTRA_INCLUDES` | see `cpp/stage/slate_build_scalapack_api.sh:46,57,90` | SLATE ScaLAPACK-API overlay build: source tree (mandatory, `:?`), output prefix, extra `-I` flags when `nvcc` is not in the expected layout. |
| `LORRAX_{PY,VENV_DIR,SRC_DIR}`, `LORRAX_OVERLAY_BUILD_DIR`, `LORRAX_MPIWRAPPER_SO`, `LORRAX_SLATE_HOST_LIB`, `LORRAX_MKL_LIB` | `config/frontera/stage_runtime.sh`, `build_mpi_overlay.sh`, `templates/gw_dev.sbatch` | Frontera staging outputs and link-line paths; peers of the already-listed `LORRAX_STAGED`/`LORRAX_OVERLAY_DIR`. `LORRAX_MPIWRAPPER_SO` is the variable behind the `MPITRAMPOLINE_LIB` row in §5, which named the path but never the variable. |

`LORRAX_CUDA_CHECK` and `LORRAX_LIB_CHECK` are **C macros**, not env
vars — the earlier grep-based counts were misleading.  So are
`LORRAX_CFG_STR` / `LORRAX_CFG_STR2`
(`cpp/common/build_config.cc`, the two-level stringification macro
pair) and the whole `LORRAX_CFG_*` family, which are CMake vars configured
into `lorrax_config.h` rather than anything the environment can set; and so
are `LORRAX_CUSOLVERMP_CHECK` (`cpp/cusolvermp` error-check macro) and
`LORRAX_CUBLASMP_CHECK` (`cpp/cublasmp/batched_gemm_ffi.cc:44`), which
this list used to carry as if they were the STAGE/PIN staging knobs'
siblings (P1 audit, 2026-07-31).

Likewise **not variables**, and excluded on purpose so a future grep-based
diff does not re-add them: `LORRAX_ISSUES` (a substring of the filename
`KNOWN_LORRAX_ISSUES.md`), `LORRAX_SC_` / `LORRAX_JAX_CACHE_` /
`LORRAX_SLURM_` / `LORRAX_PHDF5_` / `LORRAX_FFI_` (prefix fragments from
prose globs and `startswith()` scans, whose real members are all listed
elsewhere on this page), and the `LORRAX_TEST_*` / `LORRAX_FAKE_*` /
`LORRAX_PAIR_*` / `LORRAX_T_*` family, which are string literals inside
`tests/test_env_registry.py` and `tests/test_env_grammar.py`.

Two entries removed from the list above, with reasons (P1 audit):

* **`LORRAX_FRONTERA_ADVICE` is a FILENAME, not an env var** —
  `$WORK/LORRAX_FRONTERA_ADVICE.md`, the out-of-repo machine-advice doc
  that `docs/dev/archive/HANDOFF_cpu_frontera_2026-07.md` points at.  Nothing
  reads it from the environment.
* **`LORRAX_PARTITION` has no in-repo reader** — no `config/**` or
  `src/**` file consumes it.  It is an EXTERNAL-OVERLAY variable: the
  /scratch harness generation reads it when composing sbatch headers.
  Recorded here so the name stays reserved; do not add an in-repo reader
  without moving the row to a live section.

### 4b. Launch/staging scripts (read at job-launch time, not by Python)

Read by `config/frontera/stage_runtime.sh` /
`build_cpu_runtime_bundle.sh` / `build_mpiwrapper.sh` /
`build_mpi_overlay.sh` / `ffi_env.sh`.  Same build-time rule: setting
them inside the running Python does nothing.

| var | default | effect |
|---|---|---|
| `LORRAX_BUNDLE` | unset | Path to `lorrax_cpu_bundle.tar` (required for node-local staging; unset/missing → announced fallback to the shared filesystem). |
| `LORRAX_BUNDLE_OUT` | `$SCRATCH/lorrax_bundle` | Where `build_cpu_runtime_bundle.sh` writes the bundle. |
| `LORRAX_BUNDLE_STRIPE` | script default | Lustre striping applied to the bundle output. |
| `LORRAX_STAGE` | `1` | `0` disables node-local staging entirely (announced). |
| `LORRAX_STAGE_ROOT` | `/tmp/lorrax_stage.$UID` | Node-local extraction dir. |
| `LORRAX_STAGED` / `LORRAX_STAGE_S` | outputs | Set BY `stage_runtime.sh` (staged? / seconds spent), not user knobs. |
| `LORRAX_VENV_FALLBACK` / `LORRAX_OVERLAY_FALLBACK` / `LORRAX_SRC_FALLBACK` | shared-FS paths | Where to run from when staging is off or fails. |
| `LORRAX_OVERLAY` | `$WORK/lorrax_env_mpi_overlay/site` | MPI overlay site the bundle build packs. |
| `LORRAX_OVERLAY_PREFIX` | `$WORK/lorrax_env_mpi_overlay` | `build_mpi_overlay.sh` install prefix (`…/site` is the product). |
| `LORRAX_OVERLAY_DIR` | output | Set BY `stage_runtime.sh` to the resolved (staged or fallback) overlay dir. |
| `LORRAX_MPIWRAPPER_REPO` / `LORRAX_MPIWRAPPER_COMMIT` | upstream MPIwrapper, pinned v2.11.1 commit | What `build_mpiwrapper.sh` fetches. |
| `LORRAX_MPIWRAPPER_STAGE` / `LORRAX_MPIWRAPPER_SRC` / `LORRAX_MPIWRAPPER_BUILD` / `LORRAX_MPIWRAPPER_PREFIX` | `$WORK/lorrax_mpiwrapper/…` | Stage / source / build / install dirs; the product is `$LORRAX_MPIWRAPPER_PREFIX/lib64/libmpiwrapper.so` (see `MPITRAMPOLINE_LIB`, §5). |
| `LORRAX_MPIWRAPPER_REFERENCE_SO` | unset | Optional reference `.so` for the build-note section comparison; skipped when absent. |
| `LORRAX_CMAKE` | `command -v cmake` | Which cmake `build_mpiwrapper.sh` uses. |
| `LORRAX_FFI_SO_PHDF5` | `$LORRAX_FFI_STAGE/build_phdf5/liblorrax_ffi.so` | `ffi_env.sh:41`: the phdf5-enabled CUDA FFI `.so` that `LORRAX_FFI_SO` is exported from. |

## 5. External variables LORRAX sets, reads, or that its harnesses dial

| var | LORRAX's handling |
|---|---|
| `JAX_ENABLE_X64` | `setdefault "1"` at 30 sites; canonical one is `runtime.set_default_env`.  Consistent everywhere. |
| `JAX_PLATFORMS` | `setdefault "cuda,cpu"`, or hard-set to `"cpu"` by `set_default_env(platform="cpu")`.  `ffi_loader.platform_from_env` READS it (default `""`) to pick the FFI library **without** initializing the JAX backend. |
| `JAX_PLATFORM_NAME` | jax's DEPRECATED spelling of `JAX_PLATFORMS`.  LORRAX never sets it for a run: `runtime` **pops** it at both CPU-downgrade sites (`runtime/__init__.py:565,967`) so a stale value cannot fight the downgrade; `psp/get_DFT_mtxels.py:24` presence-tests it (either spelling counts as "the caller chose a platform"); the one `setdefault "cpu"` writer is the exempt bench driver `tests/bench/benchmark_synthetic.py:22`. |
| `JAX_PROCESS_COUNT` → `JAX_NUM_PROCESSES` → `SLURM_NTASKS` → `1` | process-count resolution chain, `runtime/__init__.py`. |
| `JAX_PROCESS_INDEX` → `SLURM_PROCID` → `0` | process-index chain. |
| `JAX_COORDINATOR_ADDRESS` | overrides the `SLURM_NODELIST`-derived coordinator. |
| `FI_PROVIDER` | not read by LORRAX.  On Frontera CLX **leave it UNSET** — Intel MPI then auto-selects `mlx`: 1.07 µs / 11.4 GB/s vs the old `FI_PROVIDER=tcp` pin's 10.9 µs / 2.15 GB/s, and pzheevd n=2448 at P=144 goes 12 s/q → 0.5–0.9 s/q (AP.3/AP.4; reproduced in-container by AS.2).  Keep `I_MPI_DEBUG≥4` so rank 0 announces `libfabric provider:`; do NOT trust `fi_info`, which reports −61 for `mlx` even where it works.  In-container: apptainer's default mount already exposes the host `/dev` (uverbs included) — **never `--bind /dev[/...]`** (a nosuid,nodev shadow copy breaks every device open, AS.1); stage the RDMA userspace via the `/hostlibs` symlink pattern (AS.1 / wk_AS `as_inner.sh`), or the provider falls back to tcp, announced. |
| `CUDA_VISIBLE_DEVICES` | read to derive `local_device_ids`; `tests/conftest.py` rewrites it per xdist worker. |
| `_LORRAX_JAX_DISTRIBUTED_DONE` | LORRAX's own idempotency sentinel for `jax.distributed.initialize` — env-scoped on purpose, so it survives module re-imports.  Self-set, not a user knob.  (`_LORRAX_GLOO_PIN_DONE` went with the Gloo interface pin.) |
| `XLA_PYTHON_CLIENT_ALLOCATOR`, `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_CLIENT_MEM_FRACTION` (current spelling) / `XLA_PYTHON_CLIENT_MEM_FRACTION` (deprecated) | Registry facts only; **what the three allocators do, and which of them keeps `memory_stats()` alive, is owned by `docs/environment/overview.md` §2.1** and is no longer duplicated here. `PREALLOCATE=false` is `setdefault` in `runtime.set_default_env()`, so every driver inherits it (`psp/get_DFT_mtxels.py` keeps its own copy only because it is a standalone CLI that never calls `bootstrap()`). `ALLOCATOR` is `setdefault` **nowhere** in `src/` — deliberately left unset; `runtime._check_allocator_env()` *validates* a caller-supplied value and *removes* a blank one, which is a WRITE, not a read. The fraction is read new-spelling-first (`runtime/xla_memory.py:194`), matching jaxlib's own `generate_pjrt_gpu_plugin_options` precedence, with the deprecated spelling flagged in the startup report. |
| `TF_GPU_ALLOCATOR` | **Not a LORRAX variable and inert for JAX.** Listed only so a future grep-based diff does not re-add it: it is a TensorFlow knob, measured byte-identical with and without (job 7882442), and has no writer anywhere in `src/`. |
| `SLATE_SCALAPACK_TARGET` | SLATE's OWN dial, read by its ScaLAPACK-compat shim (`scalapack_slate.hh:170-188`), surfaced here because `blacs_grid.h:305` reads it to ANNOUNCE the demotion it controls: unset defaults to `HostTask`, so a SLATE built `gpu_backend=cuda` still runs on the **CPU** unless it is set to `devices`.  Only meaningful with `LORRAX_SCALAPACK_ALLOW_SLATE_API` (§2b). |
| `XDG_CACHE_HOME` | base for the JAX compile cache. |
| `HDF5_USE_FILE_LOCKING` | `setdefault "FALSE"` in one psp test; exported `FALSE` by every production harness. AUDITED (AW, 2026-07-27): **not load-bearing on Frontera `/scratch2`** — the mount has real `flock`, and a full 785c/P=16 e2e with the variable UNSET (HDF5 default locking) ran rc=0 with all four eqp/sigma files bit-identical (`run_800c_awlock`). The MPI-IO VFD takes no POSIX locks at all, so the variable only ever governs the serial-h5py side paths (eager reads, deferred attrs, `_introspect_dataset`). KEEP the harness export anyway: `/work2` mounts `localflock` (locks are node-LOCAL — cross-node "locking" there is silently incoherent, so honest intent is to disable), and h5py wheel HDF5s differ in lock default. Machine fact, one export per harness, never per-tool. |
| `OMP_NUM_THREADS` | `setdefault "32"` in `psp/orbital_magnetization.py` only. NOTE: on Frontera the XLA:CPU threadpool does **not** obey OMP — `taskset` pinning is the real mechanism (FRONTERA_ADVICE §10). |
| `OPENBLAS_NUM_THREADS` | Read-only, for the startup report's thread-count/oversubscription table (`runtime/__init__.py:1287`); LORRAX never sets it. |
| `SLURMD_NODENAME` / `HOSTNAME` | Coordinator-address fallback chain after `SLURM_NODELIST` (`runtime/__init__.py:768`) — machine facts, read only. |
| `MPLBACKEND` | `setdefault "Agg"` for headless plotting. |
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION` | Not read by LORRAX; jax's own config (`gloo` default \| `mpi` \| `megascale`). **LORRAX runs `mpi`.** Every jax CPU collective then goes to the MPItrampoline runtime baked into the jaxlib wheel → Intel MPI → mlx/RDMA. Two measured reasons: gloo's `reduce-scatter` **silently corrupts ~5% of executions** (plausible wrong value, no warning, rc=0) where mpi is clean in 504/504 on the identical program; and mpi is 1.18× e2e at P=16 with the collective-bound stages 1.4–8.2× (on 1.1–2.2 GB payloads, 14.99/31.11/11.98 s → 0.83/1.05/0.63 s). Requires the three rows below plus the overlay `sitecustomize`. `runtime.announce_cpu_collectives()` states the resolved value on rank 0 and warns when a multi-process CPU run has landed on gloo. The FFI side coexists by construction: `ffi/cpp/phdf5/context.cc` and `cpp/slate/context.cc` only `MPI_Init_thread(MULTIPLE)` when nothing initialized MPI first, and the phdf5 open WARNS when the granted level is below MULTIPLE (the AS.4b race signature — it means the wrapper is not on the path). |
| `MPITRAMPOLINE_LIB` | Not read by LORRAX — MPItrampoline's own dial. Must point at the LORRAX MPIwrapper built by **`config/frontera/build_mpiwrapper.sh`** (default `$WORK/lorrax_mpiwrapper/install/lib64/libmpiwrapper.so`): upstream MPIwrapper v2.11.1 plus `config/frontera/mpiwrapper/lorrax_thread.patch`. An unpatched build grants FUNNELED and measured a ~29% multi-node crash/hang rate (AS.4b: phdf5/mpi4py I/O thread vs XLA collectives thread, both inside `MPID_Progress_wait`), and it cannot satisfy the `MPI_Is_thread_main` guard at all. Deliberately NOT auto-defaulted by LORRAX (audit AW): it is a machine fact naming a build artifact outside the repo, the hazardous-vs-good choice must stay visible in the harness, and MPItrampoline already refuses loudly when it is missing. Full recipe: `docs/dev/mpi_collectives.md`. |
| `LORRAX_MPI_FORCE_THREAD_MAIN` | Read by the **MPIwrapper**, not by `src/` — **STATUS: SUPERSEDED — leave it UNSET.** `common.collectives.warm_mesh_cliques()` now creates every mesh-axis communicator (plus the world one) from the Python main thread at mesh-construction time; XLA's per-device-set clique cache then serves every later collective, so the guard is never re-evaluated and BSE runs under `impl=mpi` with this gate unset (verified on the real 785c deck). The warm-up is also safer — the override lets XLA call `MPI_Comm_split` from arbitrary pool workers with nothing serialising that across ranks. This knob is retained only as a fallback and as the positive control in the gates. Historical rationale, superseded: a jax/XLA knob forcing the sequential thunk executor (one negative already measured — no such `set_xla_cpu_*` option exists in this jaxlib and `jax_cpu_enable_async_dispatch=0` is not the lever); restructuring BSE so its collectives are issued from the main thread (its matvec payload is only 0.41 MB, which permits options Σ cannot afford); or an upstream fix, since `MPI_Is_thread_main` is a STRICTER condition than `MPI_THREAD_MULTIPLE` requires. NOTE the wrapper's always-on `MPI_THREAD_MULTIPLE` upgrade is NOT superseded — it is about concurrent MPI progress during execution (the AS.4b ~29% class) and retires only via an upstream jaxlib change. See docs/dev/mpi_collectives.md. the opt-in half of `config/frontera/mpiwrapper/lorrax_thread.patch`. Set (`1`) makes the wrapper's `MPI_Is_thread_main` report true to every caller. jaxlib's `xla::cpu::MpiCollectives::CreateCommunicators` refuses to `MPI_Comm_split` a new clique unless `MPI_Is_thread_main()` is true, and XLA:CPU dispatches collective thunks to intra-op pool workers whenever the `ThunkExecutor` takes its PARALLEL path — so **any clique whose first use is inside a real jitted program is refused**, which is what killed BSE under `impl=mpi`. This is NOT a thread-LEVEL test: `MPI_Is_thread_main` is false on a non-initialising thread even under `MPI_THREAD_MULTIPLE`, so no `MPI_THREAD_*` setting and no collective warm-up ordering can substitute. Legal only in combination with the same patch's always-on THREAD_MULTIPLE upgrade, and only because XLA creates cliques in a rank-identical deterministic order. **Default OFF** — unset, the wrapper is byte-for-byte the THREAD_MULTIPLE-only build. Blast radius is exactly XLA's CPU collectives; mpi4py/h5py/the FFI host `.so` link Intel `libmpi.so.12` directly and never see it. No-op under gloo and with an unpatched wrapper. |
| `LORRAX_MPI_FINALIZE_FIX` | Read by the OVERLAY `sitecustomize.py` (`$WORK/lorrax_env_mpi_overlay/site`), not by `src/`. **Mandatory for `impl=mpi`.** `skip_atexit` (recommended) suppresses jax's atexit `collectives.Finalize` so the C++ destructor finalizes exactly once — without it every impl=mpi run exits **rc=1 after SUCCESS** (double MPI_Finalize). `hard_exit` is the fallback; `LORRAX_MPI_INIT_FIRST=mpi4py` remains a documented DO-NOT-USE (hangs the trampoline on a pre-initialized MPI). No-op under gloo. |
| `LORRAX_MPI_PROVIDER` | The harness/`ffi_env.sh` dial over `FI_PROVIDER` (scorecard AP.7/AS.5). `auto` (default) **unsets** `FI_PROVIDER`+`FI_TCP_IFACE` so Intel MPI picks the native provider (`mlx` on CLX); `tcp` restores the IPoIB pin with `FI_TCP_IFACE=ib0` (the rtx/mlx4 escape hatch); any other value force-requests that provider (never `verbs` at P≥144 one-block — 68 s/q pathology, AP.4). Read by the sbatch env blocks and by `config/frontera/ffi_env.sh`; not read by Python. The `auto` unset is load-bearing, not hygiene: TACC's default impi module exports `FI_PROVIDER=mlx` into every login/compute shell and sbatch/ssh inherit it, so "leave it unset" requires actively unsetting. |
| `I_MPI_FABRICS` | Harnesses export `shm:ofi` — which is Intel MPI 2019+'s own default, so this is *documentation of intent* plus a guard against a stray inherited value, not a behavior change (AU: pingpong identical with it unset). `ffi_env.sh` defaults it to `shm` (single-node rtx bring-up: skips OFI init entirely); override with `LORRAX_MPI_FABRICS=shm:ofi` for multi-node. |
| `I_MPI_PMI_LIBRARY` | Needed ONLY under `srun --mpi=pmi2` (the Intel-MPI-under-slurm bootstrap; harness cells that run no MPI code don't need it, and nothing loads it unless MPI inits). MUST be set unconditionally where used: TACC's login env exports `/usr/lib64/libpmi.so` — a PMI-**1** library, wrong protocol for `--mpi=pmi2` AND absent inside the container. The staged PMI2 lib is `$WORK/host_pmi/libpmi2.so.0` (`LORRAX_PMI2_LIB` overrides in `ffi_env.sh`). |
| `I_MPI_DEBUG` | Default 4 in every harness + `ffi_env.sh`. Init-time-only output (provider banner + pinning table); AU measured steady-state pingpong identical at `I_MPI_DEBUG=0`, so the banner is free. It is MANDATORY telemetry — the `libfabric provider:` line is the only trustworthy provider observable (`fi_info` false-negatives on mlx), and a silent transport is how the em1/tcp era happened. Keep ≥4. |
| `FI_PROVIDER_PATH` | Harnesses + `ffi_env.sh` pin it to `$IMPI/libfabric/lib/prov` (the bundled provider .so dir). **REQUIRED in-container, not belt-and-braces**: `mpivars.sh` (which normally sets it) is not sourced there, and AU measured that with it unset PMPI_Init aborts outright — `MPIDI_OFI_mpi_init_hook ... addrinfo() failed ... No data available`, i.e. libfabric finds NO providers at all. (Note this is the exact error string of the rtx-era "tcp/mlx4 fails in-container" archaeology — some of that history may have been a missing FI_PROVIDER_PATH, not the fabric.) |
| `UCX_*` (`UCX_TLS`, retry/timeout tunings) | TACC's default impi module exports `UCX_TLS=knem,dc_x,rc` + `UCX_{RC,DC,UD}_MLX5_{TIMEOUT,RETRY_COUNT}` bumps into every shell, and sbatch/ssh launches inherit them — so every AP/AS `mlx` number (1.07 µs / 11.4 GB/s, pzheevd 0.52 s/q) was measured UNDER those tunings, not under bare UCX defaults. AU A/B (in-container, provider auto): stripping every `UCX_*` leaves 8 B pingpong/allreduce unchanged (1.07 µs / 3.38 µs) but **doubles the 1 MiB 32-rank Allreduce (419 → 799 µs)** — the tunings are load-bearing for large-message collectives. The harness blocks therefore SETDEFAULT the six module values (`${UCX_TLS:-knem,dc_x,rc}` etc.): inherited values always win, but a stripped launch environment (cron, clean ssh) no longer silently loses 2×.  Since the fix/zq audit the TRACKED `config/frontera/ffi_env.sh` carries the same six setdefaults (guarded to the non-`tcp` provider cases), so the mitigation is auditable in-repo rather than only in the /scratch harnesses. Do not hard-pin them (rtx/mlx4 has no `dc_x`), and do not add other UCX knobs — the mlx TPN=4 × n=5024 anomaly (AP.9.4) is the only open UCX question and is not production-relevant (2×28 layout is clean). |
| `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` | `GLOO_SOCKET_IFNAME` is **INERT with jax** — the string appears nowhere in the shipped jax/jaxlib (scorecard AF.5/AK.4); every job script that exports it is exporting a no-op. There is no LORRAX Gloo dial any more either — LORRAX's CPU collectives run on MPI, whose transport is selected by `LORRAX_MPI_PROVIDER` / `FI_PROVIDER`.  (`LORRAX_GLOO_IFNAME`, the AL-era interface pin for `runtime::pin_gloo_interface`, is **HISTORICAL**: the pin was removed with the mpi migration and the name survives only in a comment at `runtime/__init__.py:56` and in `docs/dev/archive/HANDOFF_2026-07-28.md`.  Exporting it does nothing.) `NCCL_SOCKET_IFNAME` *is* read by NCCL and still matters on GPU runs. |

---

## Consistency audit

Every LORRAX-owned variable read at more than one site was checked for
default drift (re-checked 2026-07-27).

> **WHAT THIS AUDIT DOES NOT CHECK, and what that cost (2026-07-30).**  It
> compares the **default** each site falls back to.  It does not look at the
> PARSE.  So a knob read the same wrong way at every site scores a ✓: the
> row below used to read `LORRAX_MEM_DEBUG | 4 | all presence-test ✓`,
> where "presence-test" means `if os.environ.get(...)` — under which
> `LORRAX_MEM_DEBUG=0` turns the probes **on**.  Four sites agreeing on a
> broken parse is not consistency; it is one defect copied four times, and
> the ✓ is what stopped anyone looking.  A green "consistency" column
> means only *"these sites agree"*, never *"these sites are right"*.
> `tests/test_env_grammar.py` is the check that speaks to correctness.

| var | sites | defaults | parse |
|---|---|---|---|
| `LORRAX_FORCE_FULL_BZ` | 5 | all default-off ✓ | all `gw_config.env_bool` ✓ — converted together 2026-07-30; previously all five were `bool(int(os.environ.get(...)))`, which accepts decimal digits ONLY, so `=true`/`=on`/`=yes` raised `invalid literal for int()` from inside the ISDF/V_q/W paths and `=2` silently meant "on" |
| `LORRAX_PHDF5_STRIPE_COUNT` | **2** (was miscounted as 3 here and 4 in §2b; re-counted 2026-08-06) | **✗ SPLIT since `e5c9618`** — Python `clamp(nranks, 4, 128)`, C++ literal `16`.  This row read "both `16` ✓" until 2026-08-06 | **✓ agreed (2026-08-06)** — both refuse a non-integer and both refuse a negative count.  Was SPLIT: Python refused loudly, C++ forwarded the raw string to `MPI_Info_set` with no validation at all. |
| `LORRAX_PHDF5_STRIPE_SIZE_FS` | 2 | **✗ SPLIT since `e5c9618`** — Python ramps 1 → 4 MiB with the rank count, C++ is flat `1M`.  This row read "both `1M` ✓" until 2026-08-06 | **✓ agreed (2026-08-06)** — both refuse an unknown suffix, and both refuse `4MiB`.  Was SPLIT: C++ computed `mult=0` for an unknown suffix and **silently kept 1 MiB**, which is what the warning box above says the audit cannot catch — identical defaults, different parses. |
| `LORRAX_MEM_DEBUG` | 4 | all default-off ✓ | all `gw_config.env_bool` ✓ (was: presence-test at all 4, i.e. `=0` turned the probes ON — the row the note above is about) |
| `LORRAX_RCHUNK_DEBUG` | 2 | both default-off ✓ | one grammar ✓ — `gw/isdf_fitting.py` and `isdf/core.py` both read it through `gw_config.env_bool` (P1.3 unification, 2026-07-31; the historical split — a bare presence test in `isdf/core` under which `=0` meant **on** — is closed, and `tests/test_env_grammar.py` now scans `isdf/core.py` as an owned file so the class cannot recur silently) |
| `LORRAX_ZETA_RCOND` | 2 factor sites + 1 provenance echo | ONE shared non-empty-env-wins rule (`isdf/core._env_override_raw`) ✓ | that one rule serves both the factor sites (`_deprecated_env_float`) and the provenance record (`deprecated_env_record`); the inline mirror in `gw_init` was deleted by the fix/zq audit ✓ |
| `LORRAX_ZETA_RANK_LOG` | 2 | both default-**ON** ✓ (`isdf/core.py:1665,2019` pass `default=True`; the "default-off" this row used to claim contradicted the §3a row AND the code — resolved to code truth, P1 audit 2026-07-31) | one shared helper (canonical `1/true/yes/on` grammar — `gw_config.env_bool` since P1.3; was `isdf/core._env_bool`) ✓ — the old `not in ('0','','false')` off-set left the log ON under `=False`/`=off`; fixed by the fix/zq audit |

The scanner also flags `JAX_PLATFORMS`, `CUDA_VISIBLE_DEVICES`,
`XLA_PYTHON_CLIENT_*` and `TF_GPU_ALLOCATOR` as having "multiple
defaults".  Each is a `setdefault` (writer) paired with a plain `get`
(reader) — the intended pattern, not drift.  (`TF_GPU_ALLOCATOR` no longer
has a writer at all: it is inert for JAX and was deleted, not corrected.
See the `XLA_PYTHON_CLIENT_ALLOCATOR` row above.)

Re-run the audit with:

```bash
python3 tools/env_audit.py src        # AST walk; flags "MULTIPLE DEFAULTS".
                                      # Sees helper-mediated reads (env_bool /
                                      # env_float / _env_falsy / Gate(env=...))
                                      # and carries a --selftest that fails
                                      # loudly on interpreters whose AST it
                                      # cannot read (py3.7 ast.Str), instead of
                                      # printing a FALSE-CLEAN empty report.
grep -rn 'getenv(' src/ffi            # the C++ side, which the AST walk can't see
python3 tests/test_env_registry.py    # ENFORCEMENT: every LORRAX read site
                                      # under src/ (py AND C++) must have a row
                                      # on this page, or the gate fails.
```

The registry gate is what stops this page decaying again (it had gone
stale twice before 2026-07-31, both times because the audit tool was a
silent no-op on the login python).  C++ knob spellings registered here are
the CURRENT ones (`LORRAX_FFT_FFI_{THREADS,CHUNK,LOG}`,
`LORRAX_SCALAPACK_*`); the P1-wave renames landed 2026-07-31 with
`LORRAX_MKLFFT_*` / `LORRAX_CUFFT_LOG` kept as deprecated warned aliases,
and the rows above carry the new spellings.  Known gate gap: the C++ scan
matches `getenv`/`log_here`/`env_flag` literals only, so reads funneled
through `mklpin::knob_value(...)` are invisible to it — those rows are
maintained by hand.

## Why CPU collectives run on `impl=mpi`

Not on this page. The mechanism, the gloo silent-corruption evidence, the
`MPI_Is_thread_main` gate and the launch recipe are owned by
**`docs/dev/mpi_collectives.md`**, with the measured transport verdicts in
**`docs/environment/transports.md`**.

A ~35-line retelling stood here until 2026-08-06 and is deleted rather than
trimmed. It had already gone wrong twice in ways the owner page had not — it
carried a superseded remedy (`LORRAX_MPI_FORCE_THREAD_MAIN=1`) as the current
one, and before that a mechanism ("collectives inside a `lax.scan` inside a
`shard_map`") that a clean-room probe refuted. Both errors are exactly what a
second copy is for. The registry rows for
`JAX_CPU_COLLECTIVES_IMPLEMENTATION`, `MPITRAMPOLINE_LIB`,
`LORRAX_MPI_FORCE_THREAD_MAIN` and `LORRAX_MPI_FINALIZE_FIX` stay in §5,
where they belong; their *explanations* now live in one place.
