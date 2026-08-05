# Distributed linalg at large μ — route map, capacity, and the speed trade

Workstream **DLM** (owner-directed, 2026-07-29). Harnesses: `wk_REL/dlm_*`.
Run dirs: `/scratch2/08271/jackmc/mos2_4x4_test/run_DLM*`.

**STATUS (2026-07-29 ~19:45 CDT, successor agent #3): the ROUTE MAP is COMPLETE
(§1) and the P=64 campaign is now FULLY HARVESTED (§2, §6). All 8 jobs reached
a terminal state; every number below is from disk with sacct clock
cross-checks. §6 carries the harvest and supersedes the predictions in §4.2.**

---

> ## ⚠ RANK-DEFICIENT DECK — CAPACITY ONLY, NO PHYSICS
>
> Every P=64 cell here runs on the `_b1024` centroid family
> (`centroids_b1024_c*.txt`). Release blocker `d79141e` showed
> `kmeans_cli._resolve_sigma_window` clamped centroid selection to a
> (0,26)×(0,52) window, so those sets span at most 26·52 = 1352 independent
> pair densities and the achievable ISDF rank is **mathematically capped
> around 1352 regardless of μ**.
>
> **No eqp / gap / Σ / QP number from ANY b1024 run in this file is quotable
> as physics, in any context, ever.** What IS valid from these runs: wall
> time, VmHWM/rank, collective payloads, route resolution strings, allocation
> sizes, and failure objects. Those are unaffected by which subspace the
> centroids span. Every claim below is one of those.

---

## 0. Source pin (REUSE THIS — do not rebuild)

    SRCSNAP = /scratch2/08271/jackmc/lorrax_setup/wk_REL/srcsnap_dlm_20260729_091352_ec96ba9
    pointer = wk_REL/snapshots/pointers/LAST_SNAP_DLM   (convenience only; jobs take SRCSNAP EXPLICITLY)
    git HEAD  ec96ba96df1e2def8767883bd20fbb186497ad88   (= wsREL-isdf-window tip)
    worktree  /work2/08271/jackmc/frontera/wt-DLM   DETACHED at ec96ba9 — NOTHING COMMITTED
    manifest  v2 (srcpin_snapshot_v2), 343 files, rooted at the snapshot
    verified  MANIFEST VERIFIED at job START in every job; coverage 343/343, no .pyc
              PYTHONDONTWRITEBYTECODE=1 exported by srcpin_resolve AND by every inner.sh

**Why a new base was needed.** The three fixes the owner named live on two
DIVERGENT branches (merge-base `1a52d51`), so neither branch alone carries all
three:

| fix | where it lived | how it is in this snapshot |
|---|---|---|
| capacity refusal removal (`f1957ab`) — removes the premature μ≤16,384 refusal on the distributed ζ route | `wsREL-isdf-window` ancestry | inherited from HEAD `ec96ba9` |
| PPM-fit q-chunking (`src/gw/minimax_screening.py`, the 74.27 GiB `_gn_ppm_fit_kernel` arena) | **UNCOMMITTED** in `wt-RELC` | applied as `git diff` patch |
| centroid active-mask (`d58bad5`), process-local centroid mesh (`2cbd824`), BSE h5py write race (`e63bc8a`) | `wt-REL10k-scale` only | applied as `git diff 1a52d51 d58bad5 -- src/` |

Both patches applied `--check` CLEAN — the REL10k-only commits touch only
`src/bse/**` and `src/centroid/**`, disjoint from the `wsREL-isdf-window`
changes in `src/gw/**` + `src/isdf/**`. The centroid fixes are *not* on this
workstream's hot path (deck centroid sets are pre-generated); they are carried
so the snapshot is a strict superset. `PROVENANCE.txt` lists the six modified
files verbatim, so the pin is auditable from the artifact alone.

> Observation on the PPM q-chunking patch, for whoever lands it:
> `fit_gn_ppm_from_wc_pair` sizes `_per_q` from `_W0.shape[1:]`, which is the
> **GLOBAL** `(μ,μ)` extent, while the comment and the arena it is bounding are
> **LOCAL** (per-rank `(μ/Px, μ/Py)`). It therefore overestimates by ~P and is
> conservative, not wrong. At μ_pad=24960/P=64 both spellings give
> `q_block = 1`, so it makes no measured difference here — but the basis
> mismatch should be fixed before the constant is tuned.

---

## 1. THE ROUTE MAP — answered definitively from source

### 1.1 The ζ least-squares solve (`distributed_zeta_solve`)

Resolver `isdf/core.py:1451 _resolve_zeta_gather`; consumed at
`gw/isdf_fitting.py:442`. Vocabulary `auto | replicated | per_q | distributed`
(default `auto`, `gw_config.py:616`; validated `gw_config.py:1752-1757`).

| tier | factorisation | back-solve | LARGEST per-rank *gathered* object | q-chunks gathered? | FFI handler |
|---|---|---|---|---|---|
| `replicated` | dense `eigh`/`cholesky` per q, **redundantly on every rank** (`core.py:1560`) | `_solve_all_at_once` all-gathers the WHOLE `(q_batch,μ,μ)` factor (`core.py:2816-2819`) | `nq·μ²·16 B` | **YES — a whole q-batch** | none (pure JAX) |
| `per_q` | same replicated dense factor | `_per_q_block` all-gathers ONE `(μ,μ)` tile inside a `shard_map` (`core.py:2859-2866`) | `μ²·(1+1/Py)·16 B`, ×nq executions per r-chunk | **YES — one q at a time** | none |
| `distributed` | ScaLAPACK `pzheevd`, ONE FFI call for the whole `(nq,μ,μ)` stack (`core.py:1966-1968` → `ffi/linalg/plan.py:134` → `ffi/scalapack/eigh.py:74`) | stacked 2-D block GEMM `C⁺@Z`, BOTH operands `P(None,'x','y')` (`core.py:2060 _distributed_pinv_apply`) | **NO `(μ,μ)` object at all** — only `max(μ·μ/Px, μ·r/Py)·16 B` per q-block | **NO** | `ScalapackEighHostFfi` |
| `auto` | — | `replicated` while `nq·μ²·16 ≤ LORRAX_ZETA_GATHER_CAP_GIB` (4 GiB), `per_q` above (`core.py:1550-1554`) | — | yes (whichever it picked) | none |

Two structural facts:

* `distributed` is the **only** tier whose factorisation cost divides by P.
  The other two run one dense `eigh` per q on *every* rank — `O(nq·μ³)` with no
  P-scaling at all (`core.py:1482-1484`; ~5.5 h at μ=4k, ~86 h at μ=10k on 28
  cores, per that comment).
* `λ` is replicated by ScaLAPACK's own contract, so the rank-truncation mask is
  computed **locally**, identically on every rank, with **no collective**
  (`core.py:1944-1947`). That is what lets the spectral cut stay
  mesh-invariant while nothing `(μ,μ)`-shaped is ever gathered.

Guards on `distributed`, all at resolve time (`core.py:1515-1543`): requires
`charge_zeta_solve = 'rank_truncate'`; requires the mesh; then delegates
platform / compiled-capability / process-coverage / geometry / divisibility to
`ffi.linalg.resolve_backend('eigh','distributed', mesh, n=μ_pad)`.
On **transverse** channels (`vertex_mu_L != 0`) it deliberately resolves to
`per_q` (`core.py:1526`) — the transverse CCT is Hermitian *indefinite*, so no
eigh-based truncation applies; its distributed route is a different key
(`distributed_lu = scalapack`, `pXgetrf`/`pXgetrs`, `core.py:2615`).

**TRAP worth naming:** when the tier resolves to `distributed`,
`gw/isdf_fitting.py:463-469` **RAISES** unless `distributed_cholesky` resolved
to `replicated_rank_truncate`. So the fully-distributed ζ route requires
`distributed_cholesky = auto`. Pinning `slate`/`cusolvermp` alongside it is a
hard refusal, not a combination.

### 1.2 The W Dyson solve (`w_dyson_solver`)

Exactly TWO plans (`gw/w_isdf.py:6-21`). Vocabulary
`gw_config.py:356 normalize_w_dyson_solver`: `local`/`auto`/None → `local`;
`distributed` → `distributed`; `lu` → `local` + DeprecationWarning;
**`lstsq` → `ValueError`** (removed in the two-plan cleanup, `gw_config.py:369-372`).
*This is the key the owner's word "lstsq" refers to.*

| plan | how | LARGEST per-rank object | FFI handler |
|---|---|---|---|
| `local` (**DEFAULT**) | pads nq up to P, reshards to q-parallel `P(('x','y'),None,None)`, one dense pivoted `lu_factor`/`lu_solve` per owned q (`w_isdf.py:245`, `299-302`, `313-315`) | **a WHOLE `(μ,μ)` tile per owned q** — `μ²·16 B` each for V, χ, W | none (`jax.scipy.linalg`) |
| `distributed` | `A = I − V·(pref·χ)` by 2-D block GEMM inside `shard_map` (`w_isdf.py:459-474`), then ONE `plan('solve_lu', mesh, backend='distributed').batched(A,B)` for the whole stack (`w_isdf.py:443`, `528`) | **no `(μ,μ)` tile ever** — `max(μ²/Px, μ²/Py)·16 B` per q-block | `ScalapackBatchedSolveLuHostFfi` (`ffi/scalapack/solve_lu.py:45`) |

Both ScaLAPACK wrappers are **pure `shard_map` with zero collectives in the
JAX layer**: `in_specs=(P(None,'x','y'), P(None,'x','y'))`,
`out_specs=P(None,'x','y')`, one `ffi_call` that loops q *inside the handler*
(`ffi/scalapack/solve_lu.py:103-116`; same shape at `ffi/scalapack/eigh.py:119-134`).
There is no `all_gather` anywhere in either wrapper.

### 1.3 ⇒ THE OWNER'S QUESTION, ANSWERED

> *"isn't there an env var for a second linalg backend that does distributed
> solves for all q and never allgathers chunks of q?"*

**It exists, but it is NOT an env var — it is two INPUT-FILE keys, and there is
deliberately no env form** (`auto` never picks either route, because a
block-cyclic factorisation changes the gauge; `core.py:1485-1487`).
The fully-distributed, never-allgather combination is exactly:

```
[cohsex]
charge_zeta_solve      = rank_truncate    # already the production default; REQUIRED
distributed_zeta_solve = distributed      # ζ lstsq: ScaLAPACK pzheevd + 2-D sharded C⁺@Z
w_dyson_solver         = distributed      # W  lstsq: ScaLAPACK pzgetrf/pzgetrs
distributed_cholesky   = auto             # MUST stay auto or the ζ tier RAISES
distributed_lu         = auto             # transverse/bispinor only — inert on this path
eigh_backend           = auto             # BSE/htransform only — NOT this path
```

Requires a **host mesh, one process per device, square-or-1-D, μ_pad divisible
by both axes** — all checked at resolve time, all raising with the failed guard
named. 8×8 (P=64) qualifies; a production 8×10 would NOT (`linalg_ffi.md:500-504`).

Three corrections to the brief's working assumptions, each verified from source:

1. **`eigh_backend` does not reach either route.** It is consumed only by
   `bse_setup` / `vq_interp` (`docs/dev/linalg_ffi.md:248`). The ζ tier
   hard-selects `backend='distributed'` at `core.py:1966` and the W plan does
   the same at `w_isdf.py:443`, precisely so the platform default is chosen in
   ONE place. Neither reads `eigh_backend`.
2. **`distributed_lu` / `distributed_cholesky` are not the dials for this.**
   `distributed_lu` drives the *transverse* (bispinor) ζ channels;
   `distributed_cholesky` picks the charge factor *library* and must be left at
   `auto` here (see the raise above).
3. **"never allgathers chunks of q" is satisfied in a STRONGER sense than
   asked.** Neither distributed route gathers along q at all — q is not a
   sharded axis in either (both hold `P(None,'x','y')`: the q index is local,
   the `(μ,μ)` face is sharded). Both loop over q **blocks** at host level
   (`core.py:2054`, `core.py:2155`, `w_isdf.py:522`) — one XLA execution per
   block — purely to bound the per-instruction collective payload. *A
   host-level loop is not a gather.* What they never do is materialise a full
   `(μ,μ)` object on any rank, which is exactly what `replicated` (whole
   q-batch), `per_q` (one q's tile) and `local` W (one tile per owned q) each do.

### 1.4 Payload arithmetic and scaling laws (verified against a logged run)

Job **7879487** (`run_CAPFIX_rebase_c25000/gw.log`), μ=24933 → μ_pad=24960,
nq=10 IBZ, P=64 (8×8), r_chunk=35520, n_rtot=46080:

| site | formula | predicted | LOGGED |
|---|---|---|---|
| ζ C⁺ formation (pinv) | `max(μ²/Py, μ²/Px)·16` = 24960·3120·16 | 1246.0 MB | `1246.0 MB` |
| ζ C⁺ back-solve (GEMM) | `max(μ·μ/Px, μ·r_chunk/Py)·16` = 24960·4440·16 | 1773.2 MB | `1773.2 MB` |
| W Dyson A-build (GEMM) | `max(μ²/Px, μ²/Py)·16` | 1246.0 MB | `1246.0 MB` |
| ζ `per_q` gathered tile | `μ²·(1+1/Py)·16` | **11.21 GB** | banner `11.214 GB` |
| ζ `replicated` gather | `nq·μ²·16` | **99.68 GB** | banner `99.68 GB/rank` |
| W `local` per owned q | `3·μ²·16` (V, χ, W) | **29.90 GB** | *to be confirmed by 7879689* |

At **μ_pad = 10048** (μ=10015), P=64, logged live by 7879686:
`replicated` = 16.15 GB/rank, `per_q` tile = 1.817 GB — both match the formulas.

**The whole capacity claim in one line:** at fixed P the distributed sites grow
as **μ²/P** (formation, A-build) and **μ·r_chunk/P** (back-solve), while the
replicated / per_q / local-W sites grow as **μ²** with no P in the denominator
at all.

Honest floor: `_chunk_q` splits **only** the q axis, so once ONE q exceeds
`LORRAX_COLLECTIVE_CHUNK_MB` (128 MB) the bound is abandoned, not enforced —
`q_block=1` and the payload grows as μ² unchecked. `ec96ba9` added the loud
warning for exactly this (`core.py:1851-1893`). At μ=24933 all three sites are
9–13× over the advertised cap. **No failure has ever been attributed to that**;
the earlier "fatal collective ceiling bracketed 1386–1773 MB" rested on false
hangs and is VOID (ladder notes R20.1).

---

## 2. WHAT WAS RUN (all states from `sacct`, all numbers from disk)

Snapshot for every job: `srcsnap_dlm_20260729_091352_ec96ba9`.
Deck: MoS2 4×4, 30 Ry, nb=1024 (`WFN_b1024.h5` family) except the P=4 gate.

| jobid | harness | μ | P | route | `sigma_omega_layout` | state @ handoff | outcome |
|---|---|---|---|---|---|---|---|
| 7879683 | `dlm_smoke4.sbatch` | 785 (c800, nb=128) | 4 | 6 cells | both | COMPLETED | **all 6 cells rc=0**: def 251 s, dist 161 s, zeta 162 s, wonly 165 s, perq 182 s, distsh 181 s |
| 7879686 | `dlm_route64.sbatch` | 10015 | 64 | `def` | sharded | COMPLETED | **rc=0, 6009 s**, VmHWM 36.03 GiB |
| 7879687 | `dlm_route64.sbatch` | 10015 | 64 | `dist` | sharded | COMPLETED | **rc=0, 1520 s**, VmHWM 36.02 GiB — **3.95× faster** |
| 7879688 | `dlm_route64.sbatch` | 10015 | 64 | `dist`, WRES=1 | **replicated** | COMPLETED (step) | **rc=124 TIMEOUT, 13501 s — HANGS**, see §6.3 |
| 7879698 | `dlm_pair_dev.sbatch` | 10015 | 64 | `zeta` then `wonly` | sharded | COMPLETED | zeta rc=0 1543 s; wonly rc=124 at its 3000 s TCELL (still in the replicated factor) |
| 7879700 | `dlm_route64.sbatch` | 19991 | 64 | `def` | sharded | COMPLETED (step) | **rc=1, 127 s — RESOLVE-TIME REFUSAL**, §6.1 |
| 7879701 | `dlm_route64.sbatch` | 19991 | 64 | `dist` | sharded | COMPLETED | **rc=0, 5277 s** — full artifacts, 22.55 GB `sigma_mnk.h5` |
| 7879689 | `dlm_route64.sbatch` | 24933 | 64 | `def` | sharded | COMPLETED (step) | **rc=1, 153 s — RESOLVE-TIME REFUSAL**, §6.1 |
| 7879690 | `dlm_route64.sbatch` | 24933 | 64 | `dist` | sharded | COMPLETED (step) | **rc=134, 4450 s — OOM in the σ τ-kernel arena**, §6.2 |
| 7879695 | `dlm_pair_dev.sbatch` | — | 64 | def+dist | — | CANCELLED by me at 0:38 | deliberately cancelled as redundant once 7879686/7879687 started; **my own job, no foreign job was ever touched** |

### 2.1 RESULTS IN HAND — P=4 route gate (job 7879683), μ=785, 2×2 mesh

Sequential cells in ONE allocation. All numbers from
`/scratch2/08271/jackmc/mos2_4x4_test/dlm_smoke4.7879683.out` and the per-cell
`run_DLMS_*/gw.log`.

| cell | rc | wall | resolved route | VmHWM/rank |
|---|---|---|---|---|
| `def` | **0** | **251 s** | `replicated_rank_truncate` factor + **`replicated`** back-solve tier (`auto` picks it: 0.10 GB gather ≪ 4 GiB cap) | 32.53 GiB |
| `dist` | **0** | **161 s** | `distributed_rank_truncate` (2-D-sharded C⁺) + `solve_lu: 'distributed' -> scalapack (ONE tile over the 2x2 mesh at P('x','y'), n=788)` | 32.51 GiB |
| `zeta` | **0** | **162 s** | `distributed_rank_truncate` + `distributed` tier; W left on the `local` plan | — |
| `wonly` / `perq` / `distsh` | (cells were still running at handoff — read them out of `dlm_smoke4.7879683.out`, and the cross-cell `eqp0.dat` / `sigma_diag.dat` comparison the job prints at the end) | | | |

The `zeta` cell landing at 162 s against `dist`'s 161 s says the ζ tier is
where essentially all of the difference vs `def` sits at this μ — but see the
cache-warmth caveat immediately below before treating that as a measurement.

**Numerical gate PASSED — the distributed W Dyson residual:**

    [W solve] Dyson residual |(1-Vchi)W - V|/|V| (4 q):
        static: q0=2.632e-15 q1=2.828e-15 q2=1.684e-15 q3=2.655e-15  max=2.828e-15
        probe : q0=5.696e-16 q1=6.649e-16 q2=5.963e-16 q3=6.380e-16  max=6.649e-16

That is the documented strict contract of the distributed plan (`w_isdf.py:546`)
and it is at the f64 noise floor. The block-cyclic `pzgetrf`/`pzgetrs` solve is
correct.

**ζ rank-truncation gate:** `def` cell logs
`n_keep/q=[785]×10, lam_max/q≈1.305e-2, lam_min_kept/q≈2.7e-10` — full rank
kept, so the truncation is not the discriminator at this μ.

**Do NOT over-read the 161 s vs 251 s.** The `def` cell ran FIRST in the
allocation and paid the cold Lustre page cache on `WFN.h5`; `dist` ran second
and inherited it. That confound is exactly why the authoritative wall-time
comparison is the P=64 pair (7879686 vs 7879687), which are **separate
allocations, both cache-cold** (`ISDF_JAX_CACHE_DIR=""` in every inner.sh).

### 2.2 PARTIAL — P=64, μ=10015 (jobs 7879686 / 7879687 / 7879688)

Route resolution confirmed live from each run's own banner:

* **7879686 `def`** → `Computing L_q = rank-truncated pinv [path=replicated_rank_truncate]`
  and `Zeta back-solve tier: per_q (distributed_zeta_solve=auto) — replicated
  (nq,μ,μ) gather would be 16.15 GB/rank; per-q tile 1.817 GB (×nq executions/r-chunk)`.
  So at nb=1024/μ=10015/P=64 the DEFAULT is **`per_q` + local W**, not `replicated`.
* **7879687 `dist`** → ScaLAPACK eigh running:
  `[scalapack.eigh] n=10048 g=1256 grid=8x8 loc=1256x1256 lwork≈11.7-12.0e6 (0.175-0.179 GiB) lrwork≈4.8e6 (0.036 GiB) liwork=70402`.
  μ_pad = **10048** (round-up to the 64-device count, not to max(Px,Py)).
* At ~7 min in, `dist` was already past the ζ factorisation into the q-IBZ
  stage while `def` was still in the replicated factor — **an early, not-yet-
  quantified indication that the distributed ζ factorisation is FASTER here**,
  consistent with `core.py:1482-1484` (the replicated tiers run one dense eigh
  per q redundantly on every rank, with no P-scaling). *This is an observation
  from a partial log, NOT a measurement. It must be replaced by the stage
  tables when the jobs land.*

### 2.3 Instrument caveat found and recorded

`dlm_route64.sbatch` / `dlm_smoke4.sbatch` print `ERR Traceback (most recent
call last):` lines that are the **benign CUDA-plugin probe** (`cuInit(0) failed:
Unknown CUDA error 303`) — the filter `grep -vaE "cuInit|CUDA"` drops the body
lines but not the bare `Traceback` header. **rc is authoritative, not that
grep.** `dlm_harvest.sh` has a corrected extractor (awk block-filter) that
suppresses CUDA-probe tracebacks properly; use it rather than the in-job grep.

---

## 3. TASK STATUS

| task | status |
|---|---|
| 1. Route map, cited, with the definitive never-allgather answer | **DONE** (§1) |
| 2. Test at large μ, escalate, characterise failures | **DONE** (§6.1, §6.2) |
| 3. Capacity gain in μ + speed trade, quantified | **DONE** (§6.4) |
| 4. `sigma_omega_layout=replicated` at nb≥1024 | **DONE — it HANGS** (§6.3) |

---

## 4. RESUME PROCEDURE — exact commands

### 4.0 First: get every state from `sacct`, never from a notification

```bash
sacct -j 7879683,7879686,7879687,7879688,7879689,7879690,7879698,7879700,7879701 \
      -X --format=JobID,JobName%18,State,Elapsed,ExitCode,NNodes
squeue -u jackmc -o "%.10i %.14P %.20j %.9T %.9M"
```
Job list also in `wk_REL/results/dlm_jobs.txt`. **Never `scancel` a job that is not in
that list** — 7879672 (`L7_b1024_bigmu_p64`) is the ladder workstream's.

### 4.1 Harvest (everything from disk)

```bash
/scratch2/08271/jackmc/lorrax_setup/wk_REL/harness/dlm_harvest.sh \
  run_DLM_def_c10000_sh run_DLM_dist_c10000_sh run_DLM_dist_c10000_rep \
  run_DLMP_zeta_c10000_sh run_DLMP_wonly_c10000_sh \
  run_DLM_def_c20000_sh run_DLM_dist_c20000_sh \
  run_DLM_def_c25000_sh run_DLM_dist_c25000_sh
```
Job `.out` files: `/scratch2/08271/jackmc/mos2_4x4_test/dlm_route64.<jobid>.out`,
`dlm_pair_dev.<jobid>.out`, `dlm_smoke4.7879683.out`.

### 4.2 What to look for in each output

**The speed trade** — `--- Timing ---` stage table in each `.out`. Compare
7879686 (`def`) vs 7879687 (`dist`) at μ=10015 and 7879700 vs 7879701 at
μ=19991. The rows that must be attributed separately are
`zeta_fit.cholesky` (the ζ factor: replicated dense eigh vs `pzheevd`),
the ζ back-solve inside `fit_one_rchunk`, and `W.exec`. The `[gw rc= wall=]`
line is the composed wall.

**The memory trade** — `MAX VmHWM across ranks` in each `.out` (label-anchored
`/proc` parse, sampled every 20 s by every rank; `sacct` MaxRSS undersamples
and must not be used).

**The collective tables** — each `.out` ends with a `COLLECTIVE TABLE` section
from `wk_AN/colltable.py` over `run_*/hlo_dump`. **Only valid cache-cold**,
which every cell is (`ISDF_JAX_CACHE_DIR=""`). NB 7879688 runs with `WRES=1`,
which adds the residual modules to its table — do not compare its table to
7879687's; compare *values* between those two, not tables.

**The capacity answer** — ⚠ **THESE PREDICTIONS WERE ALL WRONG. See §6 for what
actually happened.** Kept here only so the reasoning that produced them is
auditable:

* ~~`local` W plan: a whole `(μ,μ)` c128 tile per owned q — `μ²·16` = 9.97 GB
  at μ_pad=24960, ×3 live (V, χ, W) ⇒ ~29.9 GB/rank~~ — **NEVER REACHED.**
  `def` refuses at resolve time (§6.1), ~150 s in, before any allocation.
* ~~`per_q` ζ tier gathered tile 11.21 GB/rank~~ — also never reached at
  μ≥19991, same reason.
* ~~a 74.27 GiB `jit__gn_ppm_fit_kernel` temp would mean the PPM patch did not
  engage~~ — **the patch DID engage**: measured **4.64 GiB**, prediction met
  exactly, and the 74.27 GiB allocation is absent from every dump (§6.2).
  Note the degeneracy trap: that module still contains a `74.27 **MiB**`
  live-out. Do not match on the number alone.

**What to look for instead**, from §6: `def` fails with a `ValueError` from
`isdf/core.py:1194` and rc=1 in ~2 min with a flat VmHWM; `dist` fails (if at
all) with `RESOURCE_EXHAUSTED` deep in σ, and the object is named in
`hlo_dump/*jit__tau_kernel*-memory-usage-report.txt`, not in `gw.log`.

**Task 4 (`sigma_omega_layout=replicated` at nb=1024)** — job **7879688**.
It is the same config family as the two runs the withdrawn D1 claim rested on
(7879357, 7879382: nb=1024, μ=10015, P=64, SHARDED=off), which were **killed
mid-`sigma_mnk.h5` write, not hung** (ladder notes R20.1 — D1 is VOID).
**The settling criterion is: does 7879688 reach `rc=0` with `eqp0.dat`,
`sigma_diag.dat` and `sigma_mnk.h5` on disk?** Give it room — the replicated
omega cube adds `n_omega·nk·nb²·16` ≈ **11.0 GB/rank** at nb=1024 (41 ω × 16 k)
and the post-σ phase writes a ~22 GB h5; a long silence there is expected, and
reading that silence as death is precisely the mistake that produced D1. If it
completes, ALSO diff its `eqp0.dat`/`sigma_diag.dat` against 7879687's — the
two layouts are documented **byte-identical** (712a866), so that is a genuine
value-parity gate, and a difference would be a real finding.

### 4.3 Gates — which applies where (do not demand bit-parity where it is wrong)

| comparison | gate | why |
|---|---|---|
| `w_dyson_solver` local vs distributed | **Dyson residual** `‖(1−Vχ)W−V‖/‖V‖`, `LORRAX_W_RESIDUAL_CHECK=1` | block-cyclic LU is NOT bit-comparable to per-q dense LU (`w_isdf.py:546-554`). **Already PASSED at P=4: max 2.83e-15.** |
| `distributed_zeta_solve` distributed vs replicated/per_q | **gauge-invariant** `C⁺ = Z diag(1/λ) Zᴴ` (~1e-14, `linalg_ffi.md:229-233`) + matching `n_keep/q`, `λ_max/q` in the `[zeta rank_truncate]` log | block-cyclic eigh picks a different, equally valid eigenvector gauge (`core.py:1485-1487`, `1939-1942`) |
| `per_q` vs `replicated` | **bit-identical** — value parity | same shapes, same operand values (`core.py:2867-2869`) |
| `sigma_omega_layout` sharded vs replicated | **byte-identical** — value parity | 712a866 |

### 4.4 Runs still worth adding

```bash
cd /scratch2/08271/jackmc/lorrax_setup/wk_REL
SNAP=$(cat LAST_SNAP_DLM)

# (a) explicit `replicated` ζ tier at mu=10015 — bounds the low end of the
#     tier axis (auto picks per_q there, so `replicated` is otherwise untested
#     at nb=1024). 16.15 GB/rank gather; expect it to fit, and to be slow.
sbatch -t 04:00:00 --export=ALL,SRCSNAP=$SNAP,ROUTE=repl,\
CENTFILE=centroids_b1024_c10000.txt,SHARDED=on,WRES=0 dlm_route64.sbatch

# (b) the frontier, if `dist` c25000 completes: push to c30000 / c32000
#     (centroids_b1024_c30000.txt = 30011, _c32000.txt = 32059 exist on the deck)
sbatch -t 06:00:00 --export=ALL,SRCSNAP=$SNAP,ROUTE=dist,\
CENTFILE=centroids_b1024_c30000.txt,SHARDED=on,WRES=0 dlm_route64.sbatch

# (c) same-allocation matched pair (removes node-set variation from the wall
#     comparison) — the DEVELOPMENT queue was near-empty and started a 32-node
#     job in ~30 s, so this is the fast lane:
sbatch --export=ALL,SRCSNAP=$SNAP,CENTFILE=centroids_b1024_c20000.txt,\
CELLS="def:on dist:on",TCELL=3000,WRES=0 dlm_pair_dev.sbatch
```

Queue notes measured today: `normal` reported an 18:09 backfill estimate but
actually started three 32-node jobs within 2 min. `development` had 394 nodes
with 13 in use and started a 32-node job in ~30 s — **use it for anything that
fits in 2 h** (a μ=10015 cell is ~30 min).

### 4.5 Harnesses (all syntax-checked, all assert their own gw.in)

| file | what |
|---|---|
| `wk_REL/harness/dlm_route64.sbatch` | one P=64 cell. `ROUTE=def\|dist\|zeta\|wonly\|repl\|perq`, `CENTFILE`, `SHARDED=on\|off`, `WRES`, `RCHUNK`, `PPMARENA`. Asserts every route key it claims BEFORE launch and echoes the resolution from the run's own banner, so a mislabelled cell cannot survive. |
| `wk_REL/harness/dlm_pair_dev.sbatch` | N cells sequentially in ONE 32-node dev allocation. `CELLS="def:on dist:on"`, `TCELL` bounds each cell. Ends with a cross-cell value comparison and per-cell collective tables. |
| `wk_REL/harness/dlm_smoke4.sbatch` | P=4 route gate, 6 cells + value comparison. Run this FIRST after any source change — it is what keeps a 32-node job from being the first thing to execute new code. |
| `wk_REL/harness/dlm_harvest.sh` | disk-only extractor, corrected traceback filter. |

All take **`SRCSNAP` explicitly**; no pointer file is read (wk_REL is shared and
a pointer is a repointable shared mutable).

---

## 5. CONCLUSIONS THAT STAND AT HANDOFF

1. **The fully-distributed, never-allgather combination is
   `distributed_zeta_solve = distributed` + `w_dyson_solver = distributed`,
   with `charge_zeta_solve = rank_truncate` and `distributed_cholesky = auto`.**
   It is not an env var and there is no env form; `auto` never selects it.
   Both routes land on the ScaLAPACK host handlers (`ScalapackEighHostFfi`,
   `ScalapackBatchedSolveLuHostFfi`) via `ffi.linalg.plan(..., backend='distributed')`.
   `eigh_backend`, `distributed_lu` and `distributed_cholesky` are **not** the
   dials for this path. Neither route gathers a q-chunk, and neither ever
   materialises a full `(μ,μ)` object on any rank. (§1.3, source-cited.)
2. **The distributed W solve is numerically sound**: Dyson residual
   max 2.83e-15 (static) / 6.65e-16 (probe) at P=4 — its documented contract,
   passed (§2.1).
3. **At nb=1024 / μ=10015 / P=64 the DEFAULT ζ tier is `per_q`, not
   `replicated`** (16.15 GB/rank would be needed; the 4 GiB `auto` cap forbids
   it). So the "default route" being compared against is already the
   memory-conscious one — the capacity claim must be made against `per_q` +
   local-W, and that is how the cells are configured.
4. **Nothing about capacity or the speed trade is measured yet.** The early
   stage-progress difference at μ=10015 is suggestive, not a measurement, and
   is labelled as such.

---

## 6. THE HARVEST (successor agent #3, 2026-07-29 19:00-20:00 CDT)

All 8 campaign jobs terminal. Every wall cross-checked: sacct window vs the
`[gw rc= wall=]` line vs the run dir's own file mtimes. All agree.

> Still **CAPACITY-ONLY**: the `_b1024` centroid family is rank-deficient
> (achievable ISDF rank capped ~1352 by the old (0,26)x(0,52) window). No eqp /
> gap / Sigma / QP number from ANY rung here is quotable, **including from the
> mu=19991 run that completed with full artifacts**.

### 6.1 `def` does not DIE at large mu — it REFUSES, before allocating

| mu | job | rc | wall | VmHWM |
|---|---|---|---|---|
| 19991 | 7879700 | 1 | 127 s | 27.49 GiB |
| 24933 | 7879689 | 1 | 153 s | 39.54 GiB |

Both die on the same `ValueError` from `isdf/core.py:1194` (`_auto_pre`):

    charge_zeta_solve='rank_truncate' needs the replicated route, but the CCT
    stack (nq=10, n_mu=24933) is 92.63 GiB > the 4.00 GiB cap.

Mechanism, from source: `gw/isdf_fitting.py:453` sets
`replicated_factor_used=(_resolved_zeta_gather != 'distributed')`, which is
**True on the default route**, so the f1957ab capacity-fix branch
(`core.py:1178`) is skipped and control falls through to the raise.

**The exact ceiling.** The binding gate is `_replicate_rank_truncate_ok`
(`core.py:1000-1031`) with `_replicated_factor_q_chunk` (`core.py:1703-1706`).
Both caps are 4 GiB, so the criterion collapses to *one `(mu,mu)` c128 matrix
<= 4 GiB*:

    mu_max = isqrt(4 GiB / 16 B) = 16,384   EXACTLY

Verified numerically: mu=16384 resolves, mu=16385 refuses. Bracketed by
measurement: 15,007 completes (ladder R36), 19,991 refuses.

**Defect in the message (FIXED, uncommitted, `wt-DLM/src/isdf/core.py`).** It
quoted the *stack* (`nq*mu^2*16`) and advised `LORRAX_ZETA_REPLICATE_CAP_GIB=`
61 / 94, when the gate that actually failed is the *per-q-batch* figure and
7 / 11 clears it — advice ~8.5x too large. The rewritten message names the
per-batch quantity, records the `mu <= 16,384` ceiling, and warns that raising
the cap makes the route RESOLVE, not finish (below).

**`def` above mu=16,384 is time-bound as well as guard-bound.** The replicated
factor is a dense eigh per q run redundantly on every rank, O(nq*mu^3), no
P-scaling. Measured **4712.4 s at mu=10015 on 64 ranks** => ~10.4 h at 19,991
and ~20.2 h at 24,933 for the factor alone. Removing the guard does not make
the route usable; it makes it slow. This is why no `def` run was resubmitted
with a raised cap.

### 6.2 `dist` at mu=24933: the PPM wall MOVED, and the next object is named

**The PPM demonstration RAN** — job 7879672, `run_PPMFIT_demo_c25000`, snapshot
`srcsnap_ppmfit_20260729_044438_ec96ba9` (pin 4f77842), 12:07:36->13:21:47.
(R36 open item O3 said it "never left PENDING"; that is superseded — it was
scheduled ~2.5 h after that handoff was written.)

    module_0914.jit__gn_ppm_fit_kernel  allocation 36: size 4.64 GiB
                                        (was: allocation 35: size 74.27 GiB)

Predicted 4.64 GiB, measured 4.64 GiB. **Zero** 74.27 GiB allocations anywhere
in the dump. Parameters `c128[1,3120,3120]` confirm q_block=1. R37's reading is
also confirmed arithmetically: 74.27/4.64 = 16.00 = nq, i.e. the object was the
leading-axis-scaled replicated global mask `f64[nq,24960,24960]`, and
`1*24960^2*8 = 4.6417 GiB` is exactly the residue at q_block=1.

**THE NEW WALL.** Both mu=24933 runs then died at the same place with the same
bytes — an independent double on two different snapshots:

| job | snapshot | module | rc |
|---|---|---|---|
| 7879690 | `srcsnap_dlm_...091352` | `module_0936.jit__tau_kernel` | 134 |
| 7879672 | `srcsnap_ppmfit_...044438` | `module_0958.jit__tau_kernel` | 134 |

    allocation 21: size 25,699,614,720 B = 23.93 GiB, preallocated-temp

Traceback: `ppm_sigma.py:1125 -> :710 -> :489 -> :396 ->
ppm_accumulators.py:294 _drain_one -> np.asarray(hr)` (the OOM surfaces at the
async D2H drain; the buffer is the tau kernel's).

**The object, to its shape** (`ppm_tau_kernel.py:113-116` names the axes):
dominant slot 9.28 GiB = `c128[16,2,3120,2,3120]` = sigma `(nk, s, mu_X, s', mu_Y)`
with nk=16, s=s'=2, mu_X=mu_Y=mu_pad/8=3120. Two such live (G(k) and sigma(k)),
plus their f64 Re/Im split channels at 4.64 GiB each.

    exact:  nk * s * s' * mu_pad^2 * 16 / P = 16*4*24960^2/64*16 = 9,968,025,600 B
    arena:  ~160 * nk * mu_pad^2 / P bytes

**MEASURED at three mu, all P=64** (this is the scaling law, not a fit):

| mu | mu_pad | arena `alloc 21` | law accounts for |
|---|---|---|---|
| 10015 | 10048 | **4.92 GiB** | 76% |
| 19991 | 20032 | **15.90 GiB** | 94% |
| 24933 | 24960 | **23.93 GiB** | 97% |

(remainder is a linear `mu*nb` tail, relatively larger at small mu.)

**Why it OOMs at 24933 and not at 19991** — closed account: at 19991 VmHWM
peaked at 70.90 GiB *including* the 15.90 GiB arena; at 24933 the request for
23.93 GiB arrived on ranks already at 71.54 GiB, i.e. 95.5 GiB needed against
~93 GiB/rank. Short by ~2.5 GiB.

**It divides by P.** P=256 (16x16) => ~6.5 GiB, which clears. See §7.

### 6.3 TASK 4 — `sigma_omega_layout=replicated` at nb=1024: **IT HANGS**

Job 7879688 (mu=10015, P=64, SHARDED=off, WRES=1): rc=124, 13501 s.

It completed **all four sigma branches** — last progress line **09:45:18** — then
produced nothing for **3 h 21 m 22 s** until the wall clock killed it.

**The object:** `module_0978.jit__identity_fn`,
`ENTRY %main.0_spmd (param: c128[41,16,128,128]) -> c128[41,16,1024,1024]` —
the two-stage all-gather rebuilding the full replicated Sigma_c(omega,k,m,n) cube:

    %all-gather   = c128[41,16,128,1024]   1,375.73 MB
    %all-gather.1 = c128[41,16,1024,1024]  11,005.85 MB   <-- never returns

`n_omega*nk*nb^2*16` = 41*16*1024^2*16 = 11,005,853,696 B, matching the 11.0 GB/rank
prediction in §4.2. The *size* was predicted correctly; the *behaviour* was
unknown, and it is: the collective does not complete.

**Why this is a hang and not D1's mistake** (D1 rested on runs killed
mid-`sigma_mnk.h5` write and was rightly voided; this run is different, and the
conclusion must be re-established on THIS evidence, not D1's wording):

* `module_0978` is the **last** module XLA compiled, at 09:45:18. **Zero** HLO
  files written afterwards — no compilation for 3 h 21 m.
* **No `sigma_mnk.h5` was ever created.** It is not a slow write; there is no
  write.
* All 64 ranks alive: the `/proc` sampler kept writing every 20 s through
  13:06:32, 8 s before the kill.
* VmHWM frozen at **36.00 GiB** per rank from 09:35 to the kill — no growth, so
  not an OOM and not thrash.
* Its `sharded` twin 7879687 — same mu, same routes, same cache-cold setup —
  finished the **entire run** in 1520 s.

**Mechanism hypothesis, NOT asserted:** 11,005,853,696 B exceeds 2^31 while the
first-stage gather (1,442,840,576 B) does not; an int32 byte count in the CPU
collectives layer would fit the signature exactly. Collectives here are `mpi`,
not gloo, so this is not the AQ gloo/ib0 timeout. Testing this needs a
one-collective repro, not a GW run.

**Verdict: do not use `sigma_omega_layout=replicated` at nb>=1024.** `sharded`
is the default and completes.

### 6.4 The speed trade, mu=10015, P=64, both cache-cold, separate allocations

| stage | `def` 7879686 | `dist` 7879687 | |
|---|---|---|---|
| `zeta_fit.cholesky` (the factor) | **4712.36 s** | **235.99 s** | **20.0x** |
| zeta back-solve (`chunk.solve`) | 40.64 | 20.34 | 2.0x |
| `chunk.z_q_build` | 58.30 | 57.86 | 1.0x |
| `W.exec` static / probe | 33.65 / 39.80 | 40.43 / 64.28 | **dist SLOWER (+31 s)** |
| `sigma.exec` | 810.65 | 806.93 | 1.00 |
| **wall** | **6009 s** | **1520 s** | **3.95x** |
| VmHWM | 36.03 GiB | 36.02 GiB | equal |

**The entire win is the zeta factorisation**: 4476 s of the 4489 s saved.
Confirmed by the attribution pair 7879698 — the `zeta` cell (distributed zeta +
local W) hit 1543 s with `cholesky` 230.9 s, matching `dist`; the `wonly` cell
(per_q zeta + distributed W) blew its 3000 s TCELL still inside the replicated
factor. **The distributed W plan buys capacity, not speed.**

**New numerical gate — distributed W Dyson residual at P=64** (7879688, WRES=1,
mu=10015): max **1.887e-14** static / **4.680e-15** probe. Previously gated only
at P=4 (2.83e-15). The block-cyclic pzgetrf/pzgetrs solve is sound at scale.

**Capacity summary.** At fixed P the distributed sites grow as mu^2/P; the
`def` route's replicated factor is capped at **mu <= 16,384** by construction and
is O(nq*mu^3) besides. Certified this campaign at P=64, nb=1024:
`dist` completes at **mu=19,991**; `def` cannot resolve above 16,384.

---

## 7. RECORDED, NOT RUN — the one job that would move the frontier

**mu=24,933, `dist`, P=256 (16x16 = 128 nodes).** Owner's call; not launched
unilaterally.

* Predicted tau arena **~6.5 GiB** (23.93 GiB * 64/256 + the linear tail),
  against the ~93 GiB/rank budget. Everything else already fits at P=64.
* **P=128 is NOT an option** — `ffi.linalg.resolve_backend` requires a
  square-or-1-D mesh, and 8x16 is neither (`linalg_ffi.md:500-504`).
* mu_pad at P=256 rounds to 25,088; 25088/16 = 1568, divisible — geometry OK.
* Command shape: the §4.4 `dlm_route64.sbatch` line with `ROUTE=dist`,
  `CENTFILE=centroids_b1024_c25000.txt`, on 128 nodes.
* Expected next binder if it clears: the `f64[nk, 2*mu_X, nb]` family
  (780 MiB/slot at P=64) and the h5 write, neither of which has ever bound.

**Do NOT bother with `dist` at c30000** — the tau arena scales as mu^2, giving
~34.6 GiB at P=64: a certain OOM on the same object, no new information.
