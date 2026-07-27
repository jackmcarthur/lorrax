# LORRAX environment variables — the registry

*Every environment variable LORRAX reads, where it is read, what it
defaults to, and whether it should still be an env var at all.*

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
3. `LORRAX_WFN_BACKEND` (`""` → config/auto; forces `eager` \| `phdf5` \|
   `phdf5_host`, `file_io/wfn_loader.py`) → the `slab_io`/backend config
   section, so the read path is recorded alongside the write path.

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
| `LORRAX_GLOO_IFNAME` | unset (→ auto-detect) | Which NIC carries JAX's **Gloo CPU collectives** (`runtime/__init__.py:pin_gloo_interface`, from `bootstrap()`).  Unset: auto-detect the first UP `ib*`/`hsn*` IPv4 interface and re-register the CPU backend factory; jax's own default binds the coordinator-route NIC — on Frontera the **1 GbE `em1`**.  A name forces that interface (skipped loudly if not UP with an IPv4); `off`/`none`/`0` disables.  Decision announced on rank 0; every failure path degrades to stock transport with a printed reason.  No-op for P=1, for `JAX_PLATFORMS != cpu`, and under `cpu_collectives_implementation=mpi` (announced as a no-op there). | ib0 vs em1: 3.3× whole-pipeline at 785c/P=16, bit-identical outputs (AK.10/AL); 606c/P=80 e2e 544 s (AL). |
| `LORRAX_COLLECTIVE_CHUNK_MB` | `128` | Upper bound on ONE emitted collective's payload in the ζ `distributed` tier (C⁺ formation + back-solve GEMM, `isdf/core.py`) and the distributed W Dyson A-build (`gw/w_isdf.py`), enforced as a host-level q-block loop XLA cannot fuse back (AF).  `0`/negative = unbounded — reproduction escape hatch only.  A per-instruction TRANSPORT cap, deliberately orthogonal to the LIVE-bytes memory cap `LORRAX_ZETA_GATHER_CAP_GIB`. | Bracketed at P=144 on **em1-era Gloo**: 1.15 GB single-shot AllGather fatal, 0.104 GB good (AF).  Re-priced 2026-07-27 (scorecard AV matrix), caps {64,128,256,512,∞}: at 785c/P=16 impl=mpi/mlx the cap is indistinguishable from unbounded (596-714 ms back-solve, no trend); at P=64 ib0-Gloo the chunk loop costs +35-70 ms/r-chunk (~2-3 % of the ζ-fit) vs a healthy 192 MB single-shot, and at P=144 the chunked tier measured 1.9× FASTER than its per_q control (AF.4).  Default stays 128 = protective at the one measured fatality point, near-free everywhere else; on ib0/mlx it is not known to be *necessary* below ~370 MB payloads. |
| `LORRAX_ZETA_GATHER_CAP_GIB` | `4` | Byte budget for the ζ back-solve's replicated-factor all-gather transient (LIVE bytes; also bounds the distributed tier's eager q-batch, `isdf/core.py`). | |
| `LORRAX_ZETA_REPLICATE_CAP_GIB` | `4` | See §1b(1) — promotion candidate. | production 12×12 runs raise to 16 (overnight A/B) |
| `LORRAX_MAX_RCHUNKS` | unset | Ceiling on the r-chunk count the planner may pick (`gw/isdf_fitting.py`).  Memory/perf, not numerics — but chunking is load-bearing at large μ (scale ladder). | |
| `LORRAX_GALERKIN_CHUNK_GIB` | `6` | htransform's Galerkin accumulation chunk budget (`bandstructure/htransform.py`).  Perf only. | |
| `LORRAX_FAILFAST` | `1` (on) | CLI failure propagation (`runtime/__init__.py::bootstrap`): an uncaught exception aborts the whole step instead of leaving P−1 ranks hanging (QUALITY_PATTERNS #7).  `0` disables.  `SystemExit(0)` (e.g. `LORRAX_EXIT_AFTER_ZETA`) stays a clean exit. | |
| `LORRAX_MALLOC_TUNE` / `LORRAX_MALLOC_MMAP_MB` (`1`) / `LORRAX_MALLOC_TRIM_MB` (`128`) | on | glibc malloc tuning at bootstrap — the arena-retention cure (scorecard T).  `LORRAX_MALLOC_TUNE=0` disables. | RSS ramp root-caused + cured at 12×12/P=80 (T.2) |
| `LORRAX_MALLOC_TRIM` | `1` (on) | Per-r-chunk `malloc_trim` in the ζ-fit loop (`gw/isdf_fitting.py`); glibc-only, no-op elsewhere. | same (T.2) |

### 2b. Libraries, backends, caches, I/O placement

| var | default | effect |
|---|---|---|
| `LORRAX_GLOO_IFNAME` | unset (→ auto-detect) | Which NIC carries JAX's **Gloo CPU collectives** (`runtime/__init__.py:pin_gloo_interface`, called from `bootstrap()`). Unset: auto-detect the first UP `ib*`/`hsn*` interface with an IPv4 and re-register the CPU backend factory so `make_gloo_tcp_collectives` gets `interface=` — on Frontera that is `ib0` (InfiniBand); jax's default binds the NIC that routes to the coordinator, which is the **1 GbE management NIC `em1`** (measured 3.3× whole-pipeline on the 4×4 deck, bit-identical outputs — scorecard AK.10/AL). A name forces that interface (skipped loudly if it is not UP with an IPv4); `off`/`none`/`0` disables the pin. Machine capability, not policy — but the decision is **always announced on rank 0**, and every failure path degrades to stock jax transport with a printed reason rather than crashing. No-op for single-process runs and for `JAX_PLATFORMS != cpu` (GPU collectives are NCCL's business). |
| `LORRAX_FFI_SO` | in-tree `src/ffi/common/cpp/build/liblorrax_ffi.so` | Path to the **CUDA** FFI library (`ffi/common/ffi_loader.py:96`). |
| `LORRAX_FFI_HOST_SO` | in-tree `src/ffi/common/cpp/host/build/liblorrax_ffi_host.so` | Path to the **host** FFI library (`ffi_loader.py:103`). Point this at `$WORK/lorrax_ffi_unified/build_host/…` for the SLATE+ScaLAPACK build. |
| `LORRAX_WFN_BACKEND` | `""` (→ config/auto) | Forces the WFN read backend: `eager` \| `phdf5` \| `phdf5_host` (`file_io/wfn_loader.py:277`). |
| `ISDF_JAX_CACHE_DIR` | `$XDG_CACHE_HOME/isdf_jax_compilation` (→ `~/.cache/...`) | JAX persistent compile-cache dir (`common/jax_compile_cache.py`). `""` opts out entirely. **Entries live in ONE shared `{base}/np{P}/` per world size and the cache is ON at every P** (scorecard AH). It used to be refused at `process_count() > 1` because the old per-rank `rank{i}/` layout combined with JAX's process-0-only write (`jax/_src/compiler.py::_cache_write`) guaranteed a divergent hit/miss pattern, and XLA:GPU compilation is a collective (`xla::gpu::AutotunerPass` → `MultiProcessKeyValueStore` → `CoordinationServiceAgent::GetKeyValue`) → permanent silent hang (scorecard AG). AH replaces the refusal with a real fix: a process-invariant cache key, a coordination-service agreement on the usable entry set taken before any compile, atomic writes, and JAX's rank-asymmetric XLA sub-caches disabled. |
| `LORRAX_JAX_CACHE_MULTIPROCESS` | `1` (**on** — was `0`) | Historically the escape hatch that "re-armed the deadlock"; now the DEFAULT path. Set it to `0` to restore the scorecard-AG refusal (no persistent cache at all when P > 1) if you ever need to bisect against it (`common/jax_compile_cache.py`). |
| `LORRAX_JAX_CACHE_INVARIANT_KEY` | `1` (on, P>1 only) | Makes jax's persistent-cache key process-invariant (forces the GPU-only device-assignment strip on every platform and canonicalises the accelerator-config hash). MEASURED without it (4 CPU ranks): rank 0 hits 7/7 while ranks 1-3 hit 0/7 and compile — which is the AG divergence, so `0` does not merely lose the win, it switches the cache OFF at P > 1 rather than let process 0 hit alone. The per-rank component is `accelerator_config` (the serialized topology), not the device assignment. |
| `LORRAX_JAX_CACHE_AGREE_TIMEOUT_S` | `300` | Timeout for the P>1 hit/miss agreement. On expiry the run degrades to cache-off with a printed reason; it never hangs. |
| `LORRAX_JAX_CACHE_STRICT` | `1` | An entry all ranks agreed on but this rank cannot load is a divergence hazard → abort loudly. `0` downgrades to a warning (**unsafe on GPU: can hang**). |
| `LORRAX_JAX_CACHE_FORCE_DIVERGE` | `0` | TEST HOOK / positive control: every rank != 0 pretends its N alphabetically-last cache entries are missing. The agreement must drop them and say so; the run must NOT hang. |
| `LORRAX_JAX_CACHE_NO_AGREE` | `0` | TEST HOOK: shared dir with the agreement layer OFF — the naive design, i.e. the deadlock reproducer. Never use in production. |
| `LORRAX_JAX_CACHE_EXPLAIN` | `0` | Turns on `jax_explain_cache_misses` (JAX logs why each entry was not written/read). |
| `LORRAX_JAX_CACHE_PREFETCH` | `1` (on, P>1) | After the agreement, pulls the agreed entries into the page cache from a thread pool. Cache entries are tiny (876 kB for 140 of them) so reading them is pure per-file Lustre latency: measured **29 s serial** at 606c/P=16, against ~4.5 s of XLA compile saved — without this the cache is a net loss on a cold-read CPU run. `LORRAX_JAX_CACHE_PREFETCH_THREADS` (16) sets the pool size. |
| `LORRAX_MINIMAX_CACHE_DIR` | package dir | Where minimax grid tables are cached (`gw/minimax_screening.py:69`). |
| `LORRAX_DISABLE_MINIMAX_DISK_CACHE` | `""` (off) | Disables that disk cache (`minimax_screening.py:67`). |
| `LORRAX_PHDF5_STRIPE_COUNT` | `16` | Lustre stripe count. Read at 4 sites (`_slab_io_ffi.py`, `_slab_io_mpi_host.py`, `isdf_fitting.py`, C++ `context.cc`) — all `'16'`, consistent. |
| `LORRAX_PHDF5_STRIPE_SIZE_FS` | `4M` | Lustre stripe size in the `lfs setstripe -S` spelling. Read at the same 4 sites since workstream AW — previously the C++ writer read only the undocumented byte-valued `LORRAX_PHDF5_STRIPE_SIZE`, so THE documented knob silently did not reach the `PHDF5_FFI` writer (the default CPU route since AM). The byte spelling still works as a C++-side legacy fallback. |
| `LORRAX_PHDF5_MPI_STACK` | `mpich` | Build+launch: which MPI the phdf5 FFI links/loads (`run_shifter.sh:42`). |
| `LORRAX_PHDF5_ALIGN_MB` | `4` | C++: `H5Pset_alignment` threshold, MiB; `0` disables. |
| `LORRAX_PHDF5_COLL_META` | `0` | C++: `1` re-enables collective metadata ops (default off is faster). |
| `LORRAX_PHDF5_CB_NODES` / `_CB_PER_NODE` / `_CB_BUFFER_SIZE` / `_CB_WRITE` / `_DS_WRITE` | unset (→ ROMIO auto) | ROMIO collective-buffering pass-throughs, forwarded VERBATIM by **both** writers when non-empty; unset means ROMIO's automatic policy in both (unified by AW). The C++ writer used to FORCE Perlmutter-era defaults (`romio_cb_write=enable`, `romio_ds_write=disable`, `cb_buffer_size=64M`, `cb_nodes=world_size`); on Frontera forcing `romio_cb_write=enable` measured *slower* than ROMIO auto (wk_AI: 1826 vs 2066 MB/s) and the rest were never revalidated off Perlmutter. Keep them unset; they exist as A/B levers. |
| `LORRAX_PHDF5_INDEPENDENT` | off | Independent instead of collective MPI-IO **reads** (C++). |
| `LORRAX_SCALAPACK_MKL_THREADS` | unset (→ `auto` = cap 4) | MKL team size pinned (thread-locally) inside the ScaLAPACK FFI handlers (`eigh`, `solve_lu`). MEASURED (wk_ENV AW, pz_bench n=2448, mlx): at the production 12×12 grid pzheevd is **11.28 s/q at 14 MKL threads vs 0.463 s/q at 4** (24×); 17.0 s/q oversubscribed at 28; 8×8 at the production 2×28 placement shows the same cliff; 4×4 (P=16) is flat. `auto`/unset caps the calling thread's MKL team at 4 (only when the global setting is larger); `off`/`0` restores the pre-AW behaviour (handler inherits `MKL_NUM_THREADS`, i.e. 28 in production — do not do this at P ≥ 64); an integer pins exactly that. The global `MKL_NUM_THREADS=28` stays right for the local `zheevd_` plan-A route and is untouched. |
| `LORRAX_FFI_PROFILE` | off | C++: per-call FFI timing. |
| `LORRAX_FORCE_REFIT` | `""` (off) | `1` forces the ζ fit even when `tmp/zeta_q.h5` is complete and its provenance matches (`gw_init.py`).  Ops escape hatch for the ζ-reuse cache (the reuse gate itself is provenance-checked). |
| `LORRAX_ALLOW_PARTIAL_ZETA` | `0` | Permits reading a ζ file whose `zeta_is_done` flag is unset (`file_io/zeta_loader.py`).  Forensics-only: a half-written ζ is otherwise indistinguishable from a complete one (QUALITY_PATTERNS #7). |

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
| `LORRAX_SANITY` | unset = warn | Stage-boundary invariant checks (`common/sanity.py`): `0`/`off` skips all (escape hatch); default checks and **warns loudly but keeps running** (a false positive must never kill a 40-node job); `strict` raises `SanityError` — set in CI / regression gates. |

### 3b. Opt-in probes, dumps, test hooks

| var | default | effect |
|---|---|---|
| `LORRAX_TIMING_TRACE` / `_DEPTH` | `0` / `3` | Every `timing.section` announces entry/exit on rank 0, nesting capped at `_DEPTH` (`common/timing.py`). |
| `LORRAX_MEM_DEBUG` | unset | Memory high-water probes at named sites (`gw_init.py`, `isdf_fitting.py` ×3; presence-test at all 4). |
| `LORRAX_RCHUNK_DEBUG` | unset | Per-r-chunk shape/timing/RSS lines (`isdf_fitting.py`, `isdf/core.py`) — the instrument behind every per-chunk table on the scorecard. |
| `LORRAX_EXIT_AFTER_ZETA` | unset | Clean `SystemExit(0)` right after the ζ fit (`gw_init.py`).  Combine with `LORRAX_MAX_RCHUNKS` + `LORRAX_RCHUNK_DEBUG` for fast fit-only sweeps (the AV chunk matrix ran exactly this way). |
| `LORRAX_W_RESIDUAL_CHECK` | `0` | Prints the direct Dyson residual `‖(1−Vχ)W − V‖/‖V‖` on the first few q after a `w_dyson_solver = distributed` W solve (`gw/w_isdf.py`) — the strict numerical contract of the distributed plan (block-cyclic LU is not bit-comparable to the local per-q LU).  Adds one diagnostic jit; leave OFF when taking collective-table probes. |
| `LORRAX_CHECK_REPLICA` | `0` | Re-arms the replica-consistency `assert_equal` that `common/collectives.py::device_put_process_local` suppresses (JAX's hidden `process_allgather`, AA.1).  Debug-only: O(P × tensor). |
| `LORRAX_SKIP_VQ_GATES` | `0` | Skips the V_Q interpolation self-checks (`bse/vq_interp.py`).  The gates exist because V_Q interpolation errors are silent. |
| `LORRAX_TRS_CHECK` | `1` (on) | Load-time MEASUREMENT of the WFN's symmetries against the density built from its own occupied ψ (`common/density_symmetry_check.py`, called from `wfn_loader`).  `0` restores pre-2026-07 flags-only behaviour — and re-arms the scorecard-Q class of silent time-reversal; `strict` raises instead of warning.  A diagnostic whose measured verdict gates `SymMaps` TRS augmentation — default on. |
| `LORRAX_TRS_TOL` / `LORRAX_TRS_SPATIAL_TOL` / `LORRAX_TRS_MAX_K` | `1e-6` / `1e-4` / `32` | Tolerances / k-sample cap for that check.  Fixture residuals ≤ 1e-12; real magnetization is O(1e-2…1); the sample is ±-closed, so `MAX_K` is a sufficiency choice, not an approximation (`0` = all k). |
| `LORRAX_FORCE_FULL_BZ` | `0` (5 sites, consistent) | `1` disables IBZ-only ζ writes / bypasses the IBZ cascade (`gw_init.py` ×3, `screening.py`, `v_q_g_flat.py`).  Debug/test bypass: changes work done and bytes written, not physics. |
| `LORRAX_EXTRA_MU_PAD` | `""` (→ 0) | **Test-only** extra μ-pad rows to prove pad-extent invariance (`runtime/padding.py`).  Any result that moves under this at fixed P is a defect.  NEVER set in production. |
| `LORRAX_PER_PROC_RESTART` | `0` (off) | Per-rank restart shard dump (`file_io/tagged_arrays.py::save_restart_state_per_proc`).  No in-tree reader; measured 4 m 43 s + 72 GB at 12×12/c2406/P=144 immediately after the canonical collective write of the same data finished in 2.2 s.  Forensics-only; **candidate for deletion** next time `file_io/` is open (owner: I/O sibling). |
| `LORRAX_FFI_DEBUG_SHARDS` / `LORRAX_WRITE_NO_JIT` | unset | Slab-writer shard-descriptor dumps / un-jitted write path (`_slab_io_ffi.py`). |
| `LORRAX_FFI_PROFILE` | off | C++: per-call FFI timing. |
| `LORRAX_PHDF5_TIME` / `_WRITE_DEBUG` / `_DUMP_HINTS` / `_SKIP_DESTROY` / `_NO_COLL_META` | off | C++-side phdf5 diagnostics. |
| `LORRAX_LU_DEBUG` / `LORRAX_LU_NO_PIVOT` / `LORRAX_LU_DEBUG_DUMP` | off | cuSOLVERMp LU diagnostics. |
| `LORRAX_JAX_CACHE_EXPLAIN` | `0` | Turns on `jax_explain_cache_misses` logging. |
| `LORRAX_JAX_CACHE_FORCE_DIVERGE` / `LORRAX_JAX_CACHE_NO_AGREE` | `0` | Compile-cache TEST HOOKS: positive control / the deadlock reproducer.  Never in production. |
| `KP2_DEBUG` / `STERN_DEBUG` | unset / `0` | Sternheimer debug (`psp/run_sternheimer.py`, `solvers/sternheimer_solve.py`). |
| `PF_ARTIFACTS_DIR` / `ISDF_JAX_PROFILE_DIR` | `profile` / unset | Trace output dirs (`common/jax_profile.py`, `bse/test_bse.py`). |
| `ISDF_COHSEX_TEST_PLATFORM` | `auto` | Test harness: force `cpu`/`gpu` for the e2e gates (`tests/harness.py`). |

## 4. Build-time only

Read by `config/**/*.sh`, `src/ffi/common/cpp/**/build*.sh` and CMake.
Never by the running Python; setting them in a job script does nothing.

`LORRAX_ROOT`, `LORRAX_SRC`, `LORRAX_VENV`, `LORRAX_SITE`,
`LORRAX_SITE_PACKAGES`, `LORRAX_INSTALL_ROOT`, `LORRAX_DEPS`,
`LORRAX_IMAGE`, `LORRAX_SIF`, `LORRAX_SHIFTER*`, `LORRAX_MODULE*`,
`LORRAX_FFI_STAGE*`, `LORRAX_FFI_BUILD_DIR`, `LORRAX_FFI_SOURCES`,
`LORRAX_FFI_HOST_*`, `LORRAX_FFI_IMAGE`, `LORRAX_FFI_PYTHON`,
`LORRAX_FFI_NO_CUDA`, `LORRAX_FFI_HAVE_{PHDF5,CAL,CUBLASMP}`,
`LORRAX_FFI_ALLOW_DEFAULT_MPI`, `LORRAX_FFI_{PHDF5,SLATE,NVHPC}_DIR*`,
`LORRAX_XLA_FFI_INCLUDE_DIR`, `LORRAX_XLA_FFI_HEADERS_DIR`,
`LORRAX_HDF5_ROOT`, `LORRAX_MPI_INCLUDE_DIR`, `LORRAX_MPICH_LIB_DIR`,
`LORRAX_MPI_LIBRARY`, `LORRAX_MPI_TYPE*`, `LORRAX_MPI_FABRICS`,
`LORRAX_IMPI_ROOT`, `LORRAX_PMI2_LIB`, `LORRAX_ICC_RUNTIME`,
`LORRAX_MKL_ROOT`, `LORRAX_SCALAPACK_LIBRARIES`, `LORRAX_NVHPC_*`,
`LORRAX_CUSOLVERMP_{STAGE,PIN,CHECK}`, `LORRAX_CUBLASMP_{PIN,CHECK}`,
`LORRAX_SLATE_*` (`REPO`, `COMMIT`, `BUILDS_DIR`, `MAKE_J`, `STACK`,
`INSTALL_DIR*`, `HOST_INSTALL_DIR`, `CUDATOOLKIT`),
`LORRAX_HAVE_SLATE`, `LORRAX_HOST_HAVE_SLATE`, `LORRAX_DARSHAN_LIB_DIR`,
`LORRAX_LUSTRE_STRIPE_*`, `LORRAX_NO_PRESTRIPE`, `LORRAX_XLA_CMDBUF`,
`LORRAX_SLURM_{ACCOUNT,QOS,CONSTRAINT}`, `LORRAX_NNODES`,
`LORRAX_NTASKS`, `LORRAX_PARTITION`, `LORRAX_NGPU`,
`LORRAX_GPUS_PER_NODE`, `LORRAX_SELECT_GPU`, `LORRAX_TIER2_WORKDIR`,
`LORRAX_FRONTERA_ADVICE`, `LORRAX_INPUT`.

`LORRAX_CUDA_CHECK` and `LORRAX_LIB_CHECK` are **C macros**, not env
vars — the earlier grep-based counts were misleading.

## 5. External variables LORRAX sets, reads, or that its harnesses dial

| var | LORRAX's handling |
|---|---|
| `JAX_ENABLE_X64` | `setdefault "1"` at 30 sites; canonical one is `runtime.set_default_env`.  Consistent everywhere. |
| `JAX_PLATFORMS` | `setdefault "cuda,cpu"`, or hard-set to `"cpu"` by `set_default_env(platform="cpu")`.  `ffi_loader.platform_from_env` READS it (default `""`) to pick the FFI library **without** initializing the JAX backend. |
| `JAX_PROCESS_COUNT` → `JAX_NUM_PROCESSES` → `SLURM_NTASKS` → `1` | process-count resolution chain, `runtime/__init__.py`. |
| `JAX_PROCESS_INDEX` → `SLURM_PROCID` → `0` | process-index chain. |
| `JAX_COORDINATOR_ADDRESS` | overrides the `SLURM_NODELIST`-derived coordinator. |
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION` | jax's own dial (`gloo` default \| `mpi` \| megascale).  `gloo`: the `LORRAX_GLOO_IFNAME` pin applies (certified default).  `mpi`: the AS.4c-certified experimental stack — REQUIRES the thread-MULTIPLE MPIwrapper (`MPITRAMPOLINE_LIB=…/mpiw_thr_install/lib64/libmpiwrapper.so`) and `LORRAX_MPI_FINALIZE_FIX=skip_atexit`; the unpatched wrapper multi-node has a measured ~21–29 % segv/hang rate (FUNNELED thread level vs concurrent mpi4py/h5py progress). |
| `MPITRAMPOLINE_LIB` / `LORRAX_MPI_FINALIZE_FIX` / `LORRAX_MPI_INIT_FIRST` | Harness-level (wk_AS `sitecustomize.py`, NOT read by `src/`): the MPIwrapper shim path; the double-`MPI_Finalize` fix (`skip_atexit` recommended); mpi4py-first init — documented DO-NOT-USE (hangs the trampoline on a pre-initialized runtime). |
| `LORRAX_MPI_PROVIDER` | Harness-level dial (runAC.sbatch / gw800_merged.sbatch case-block, NOT read by `src/`): `auto` = `FI_PROVIDER` unset ⇒ Intel MPI picks the native `mlx` (UCX) provider on CLX; `tcp` = the rtx/mlx4 escape hatch; any other value force-requests that provider.  Never `verbs` at P≥144 with the one-block eigh layout (68 s/q pathology, AP.4). |
| `FI_PROVIDER` | not read by LORRAX.  On Frontera CLX **leave it UNSET** — Intel MPI then auto-selects `mlx`: 1.07 µs / 11.4 GB/s vs the old `FI_PROVIDER=tcp` pin's 10.9 µs / 2.15 GB/s, and pzheevd n=2448 at P=144 goes 12 s/q → 0.5–0.9 s/q (AP.3/AP.4; reproduced in-container by AS.2).  Keep `I_MPI_DEBUG≥4` so rank 0 announces `libfabric provider:`; do NOT trust `fi_info`, which reports −61 for `mlx` even where it works.  In-container: apptainer's default mount already exposes the host `/dev` (uverbs included) — **never `--bind /dev[/...]`** (a nosuid,nodev shadow copy breaks every device open, AS.1); stage the RDMA userspace via the `/hostlibs` symlink pattern (AS.1 / wk_AS `as_inner.sh`), or the provider falls back to tcp, announced. |
| `CUDA_VISIBLE_DEVICES` | read to derive `local_device_ids`; `tests/conftest.py` rewrites it per xdist worker. |
| `_LORRAX_JAX_DISTRIBUTED_DONE` / `_LORRAX_GLOO_PIN_DONE` | LORRAX's own idempotency sentinels (for `jax.distributed.initialize` and `pin_gloo_interface`) — env-scoped on purpose, so they survive module re-imports.  Self-set, not user knobs. |
| `XLA_PYTHON_CLIENT_ALLOCATOR`, `TF_GPU_ALLOCATOR`, `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_MEM_FRACTION` | `setdefault` in the psp CLIs; `gw_init`/`gw_output` only *read* them, to caveat the high-water-mark report (cuda_async under-reports).  Not an inconsistency. |
| `XDG_CACHE_HOME` | base for the JAX compile cache. |
| `HDF5_USE_FILE_LOCKING` | `setdefault "FALSE"` in one psp test; exported `FALSE` by every production harness. AUDITED (AW, 2026-07-27): **not load-bearing on Frontera `/scratch2`** — the mount has real `flock`, and a full 785c/P=16 e2e with the variable UNSET (HDF5 default locking) ran rc=0 with all four eqp/sigma files bit-identical (`run_800c_awlock`). The MPI-IO VFD takes no POSIX locks at all, so the variable only ever governs the serial-h5py side paths (eager reads, deferred attrs, `_introspect_dataset`). KEEP the harness export anyway: `/work2` mounts `localflock` (locks are node-LOCAL — cross-node "locking" there is silently incoherent, so honest intent is to disable), and h5py wheel HDF5s differ in lock default. Machine fact, one export per harness, never per-tool. |
| `OMP_NUM_THREADS` | `setdefault "32"` in `psp/orbital_magnetization.py` only. NOTE: on Frontera the XLA:CPU threadpool does **not** obey OMP — `taskset` pinning is the real mechanism (FRONTERA_ADVICE §10). |
| `MPLBACKEND` | `setdefault "Agg"` for headless plotting. |
| `FI_PROVIDER` | not read by LORRAX. On Frontera CLX **leave it UNSET** — Intel MPI then auto-selects the native `mlx` (UCX) provider: 1.07 µs / 11.4 GB/s vs the old `FI_PROVIDER=tcp` pin's 10.9 µs / 2.15 GB/s, and pzheevd n=2448 at P=144 goes 12 s/q → 0.5–0.9 s/q (scorecard AP.3/AP.4; reproduced in-container by AS.2). `tcp` is an rtx/mlx4 workaround; the harness dial is `LORRAX_MPI_PROVIDER` (`auto` = unset ⇒ mlx; `tcp` = the rtx escape hatch; anything else force-requests that provider — never `verbs` at P≥144 one-block, 68 s/q pathology). Keep `I_MPI_DEBUG≥4` so rank 0 announces `libfabric provider:`; do NOT trust `fi_info`, which reports −61 for `mlx` even where it works. In-container note: apptainer's default mount already exposes the host `/dev` (uverbs included) — never add `--bind /dev[/...]`, which shadows it with a `nosuid,nodev` copy that breaks every device open (AS.1). |
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION` | Not read by LORRAX; jax's own config (`gloo` default \| `mpi` \| `megascale`). `mpi` puts EVERY jax CPU collective on the MPItrampoline runtime baked into the jaxlib wheel → Intel MPI → mlx/RDMA (measured 1.18× e2e at P=16, collective-bound stages 1.4–8×, scorecard AS.4). EXPERIMENTAL until the AS.4c rep ledger extends at scale; requires the two rows below plus the overlay `sitecustomize`. The FFI side coexists by construction: `ffi/phdf5/cpp/context.cc` and `slate/cpp/context.cc` only `MPI_Init_thread(MULTIPLE)` when nothing initialized MPI first, and the phdf5 open now WARNS when the granted level is below MULTIPLE (the AS.4b race signature). |
| `MPITRAMPOLINE_LIB` | Not read by LORRAX — MPItrampoline's own dial. Must point at the **THREAD_MULTIPLE-patched** MPIwrapper (`wk_AS/mpiw_thr_install/lib64/libmpiwrapper.so`); the unpatched build grants FUNNELED and measured a ~29% multi-node crash/hang rate (AS.4b: phdf5/mpi4py I/O thread vs XLA collectives thread, both inside `MPID_Progress_wait`). Deliberately NOT auto-defaulted by LORRAX (audit AW): it is a machine fact naming a build artifact outside the repo, the hazardous-vs-certified choice must stay visible in the harness, and MPItrampoline already refuses loudly when it is missing. Harness block: scorecard AS.7. |
| `LORRAX_MPI_FINALIZE_FIX` | Read by the OVERLAY `sitecustomize.py` (wk_AS), not by `src/`. `skip_atexit` (recommended) suppresses jax's atexit `collectives.Finalize` so the C++ destructor finalizes exactly once — without it every impl=mpi run exits rc=1 after SUCCESS (double MPI_Finalize). `hard_exit` is the fallback; `LORRAX_MPI_INIT_FIRST=mpi4py` remains a documented DO-NOT-USE (hangs the trampoline on a pre-initialized MPI). No-op under gloo. |
| `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` | `GLOO_SOCKET_IFNAME` is **INERT with jax** — the string appears nowhere in the shipped jax/jaxlib (scorecard AF.5/AK.4); every job script that exports it is exporting a no-op. The working dial is `LORRAX_GLOO_IFNAME` above (§2). `NCCL_SOCKET_IFNAME` *is* read by NCCL and still matters on GPU runs. |

---

## Consistency audit

Every LORRAX-owned variable read at more than one site was checked for
default drift (re-checked 2026-07-27).  **No inconsistencies found:**

| var | sites | defaults |
|---|---|---|
| `LORRAX_FORCE_FULL_BZ` | 5 | all `'0'` ✓ |
| `LORRAX_PHDF5_STRIPE_COUNT` | 3 | all `'16'` ✓ |
| `LORRAX_PHDF5_STRIPE_SIZE_FS` | 3 | all `'4M'` ✓ |
| `LORRAX_MEM_DEBUG` | 4 | all presence-test ✓ |
| `LORRAX_RCHUNK_DEBUG` | 2 | all presence-test ✓ |
| `LORRAX_ZETA_RCOND` | 2 factor sites + 1 provenance echo | one shared helper (`isdf/core._deprecated_env_float`); provenance mirrors its empty-is-unset semantics ✓ |
| `LORRAX_ZETA_RANK_LOG` | 2 | same off-set `('0','','false')` ✓ |

The scanner also flags `JAX_PLATFORMS`, `CUDA_VISIBLE_DEVICES`,
`XLA_PYTHON_CLIENT_*` and `TF_GPU_ALLOCATOR` as having "multiple
defaults".  Each is a `setdefault` (writer) paired with a plain `get`
(reader) — the intended pattern, not drift.

Re-run the audit with:

```bash
python3 tools/env_audit.py src        # AST walk; flags "MULTIPLE DEFAULTS"
grep -rn 'getenv(' src/ffi            # the C++ side, which the tool can't see
```
