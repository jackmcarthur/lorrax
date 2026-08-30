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
* the CPU collectives transport and **why** it resolved that way; live
  multi-process CPU startup refuses gloo because its measured failure is
  silent corruption (the pure formatter retains the hypothetical diagnosis);
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
2. `ISDF_CHUNK_TARGET_UTILIZATION` → a memory-section key, for symmetry
   with `band_chunk_size` / `r_chunk_size` / `gflat_chunk_size`, which are
   already keys (the planner clamps a positive override to [0.85, 1.0]).
3. `LORRAX_WFN_BACKEND` (`""` → config/auto; forces `eager` \| `phdf5`,
   `services/wfn_loader/src/wfn_loader/loader.py`) → the `slab_io`/backend config
   section, so the read path is recorded alongside the write path.
4. **`LORRAX_SIGMA_PLAN` → a Sigma planning key.** Default `panes` preserves
   the incumbent GN-PPM and MPA pane/window planners exactly; `delivered`
   selects the shared measure-apportioned hybrid planner for both routes
   (`gw.sigma_plan.resolve_sigma_plan`). Grammar is the exact, case-insensitive
   enum `panes` \| `delivered` after stripping; blank means `panes`, and every
   other value REFUSES naming both choices. This changes the quadrature and is
   therefore policy, not machine capability; the env form is the initial
   opt-in and must be promoted before it becomes a default candidate. See
   [the delivered-plan contract](delivered_plan.md).
5. *(removed 2026-08-31)* `LORRAX_DELIVERED_TAU_GRID` no longer exists:
   lookup-served rules carry their own nodes and there is one grid mode.
