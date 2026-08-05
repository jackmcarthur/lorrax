# FFI vs XLA — owner-commissioned evidence audit (2026-07-29)

**Commission.** Settle, on evidence rather than accumulated agent folklore, what the
best implementation is for LORRAX's FFTs and GEMMs (XLA-native vs FFI vendor
libraries, plain vs fused, CPU and GPU).

**Method.** (A) exhaustive claim ledger mined from the historical corpus with each
claim's evidence status verified against on-disk logs; (B) a new controlled
experiment on quiet exclusive nodes with explicit factor control, ≥5 reps,
median + spread, and a measured roofline; (C) verdicts and a recommendation.

**Author's standing rule for this document.** Every number below is either
(i) reproduced by me from a named on-disk file, or (ii) measured by jobs
7879376 / 7879377 / 7879378 / 7879379 submitted for this audit. Anything I could
not decide is marked **UNDECIDED** with the measurement that would decide it.
Nothing is filled in by plausibility.

New evidence in this document (jobs run for this audit):

| job | queue | node | what | state (sacct-verified) |
|---|---|---|---|---|
| 7879376 | small | c208-002 | CPU FFT: STREAM roofline, HLO census, thread sweep 1/4/14/28 + repeat | COMPLETED 0:0, 11:54 |
| 7879377 | small | c208-010 | CPU GEMM: thread sweep 1/4/14/28, vendor-peak probe, 1-vs-2-rank co-tenancy control | COMPLETED 0:0, 07:29 |
| 7879378 | rtx-dev | c196-012 | GPU: wire + FP64 bound, HLO census, FFT bench, GEMM bench | COMPLETED 0:0, 06:55 |
| 7879379 | small | c209-016 | CLX core→socket topology probe | COMPLETED 0:0 |
| 7879395 | small | c207-035 | reproduction control: original harness vs old/new `.so` + 25-rep distribution probe | COMPLETED 0:0, 04:33 |
| 7879401 | small | c207-031 | XLA-reference reproducibility: 6 back-to-back runs of the unmodified original harness | COMPLETED 0:0, 08:04 |
| 7879446 | small | — | first pinning A/B — **CANCELLED by me**: paired cells were serialised by SLURM (defect found pre-analysis) | CANCELLED |
| 7879455 | small | c207-031 | pinning A/B window 1 (corrected harness): STREAM/FFT/GEMM, arms A,B | COMPLETED 0:0, 05:13 |
| 7879466 | small | c207-031 | pinning A/B window 2: arms A,B,C | COMPLETED 0:0, 06:33 |
| 7879483 | small | c207-031 | **production** pinning A/B: GW driver, `run_400c` P=4, 6 mirrored passes | COMPLETED 0:0, 07:50 |
| 7879489 | small | c207-031 | minor-most out-of-place sizing (Option A), XLA timed first | COMPLETED 0:0, 01:13 |
| 7879454/56/67/80 | small | — | sigma pinning attempts that failed on MPI/PMI setup (no data used) | COMPLETED, rc≠0 |

All states above are from `sacct -X`, not from any notification. Queue discipline:
every CPU cell used `small` (max 2 concurrent, 1 node each) and every GPU cell
`rtx-dev`; the `development` queue was not touched. No foreign job was cancelled.

---

## PART A — CLAIM LEDGER

Evidence status key: **MEASURED** (jobid named and the number reproduced from an
on-disk log by me), **INFERRED** (reasoned from HLO/axis structure/code, not
timed), **ASSERTED** (no evidence found anywhere in the corpus),
**REFUTED** (contradicted by later evidence — the contradicting evidence is named).

### A.1 FFT claims

| # | Claim (verbatim, ≤2 lines) | Source | Measured at | Status |
|---|---|---|---|---|
| F1 | "PLAIN ifft = 151 ms vs 128 ms XLA standalone" | `wk_REL/docs/ffi_fft_proto_notes.md:189` | job 7878719, 28 thr, c128 `[16,2,624,2,624]` 400 MB G tile, single node | **MEASURED** — reproduced: `wk_REL/results/logs/fftffi_gate.7878719.out:488` (`28 151.07`), `:363` (`G ifft 127.80 ms`). 151.07/127.80 = 1.182. **But see REFUTED-BY-SCOPE in §C.1: the ordering is thread-count-specific and the two arms used different buffer discipline.** |
| F2 | "the plain FFI transform is 151 ms vs XLA's 128 ms — ~18% slower, and that was the case where the FFI saved a transpose" | `wk_REL/docs/gemm_portability_bse_notes.md:287-289` | derived from F1 | **MEASURED (derived)**. The "18%" percentage is first computed here; `ffi_fft_proto_notes.md` never states a percentage. |
| F3 | "FUSED conv = **163 ms vs 1141 ms** for the XLA-equivalent staged composition (7.0×)" | `wk_REL/docs/ffi_fft_proto_notes.md:187-188` | job 7878719, 28 thr | **MEASURED** — `fftffi_gate.7878719.out:488` (162.63), `:363` (1141.35). Ratio 7.017. |
| F4 | thread sweep "1t 366/687 · 2 254/427 · 4 198/286 · 6 179/238 · 8 170/214 · 14 159/184 · 28 **151/163**" | `ffi_fft_proto_notes.md:176-177` | job 7878719 | **MEASURED** — all seven rows reproduce in `fftffi_gate.7878719.out` to 0.1 ms. |
| F5 | "6-way-cap ANSWER: no cap. Scaling is monotone to the full 28-core team"; "a 6-cap costs +19% plain / +46% fused" | `ffi_fft_proto_notes.md:181-183` | job 7878719 | **MEASURED** — monotone in the log; 179.40/151.07 = 1.188, 238.20/162.63 = 1.465. |
| F6 | "auto (512 KiB/L2 ⇒ 2048 at nk=16) ≈ best — keep auto" | `ffi_fft_proto_notes.md:180` | job 7878719 chunk sweep | **MEASURED but MIS-STATED** — `fftffi_gate.7878719.out:882` auto = 153.76/165.40; chunk=512 = 150.41/162.52. 512 was best by ~2%. Also: auto read 151.07 in the thread sweep and 153.76 in the chunk sweep of the **same job** — a 1.8% unreported intra-job spread, and the smaller of the two became the headline. |
| F7 | "sigma.exec 272.0 → 71.9 s (3.78×)"; a: 77.646 (3.50×) | `ffi_fft_proto_notes.md:216-217,304` | job 7878727, AQ 4962c/P=64, nb=128 | **MEASURED** — `mos2_4x4_test/fftffi_ab.7878727.out:368`. **Provenance caveat (§C.6):** cross-job, no same-job XLA control, dirty tree co-resident with the omega-cube workstream. |
| F8 | "G_ifft 79.63 → 2.36 (33.7×); V_ifft 16.54 → 2.78 (5.9×); GW_mult_fft 95.74 → 5.71 (16.8×)" | `ffi_fft_proto_notes.md:226-228` | job 7878727 vs 7878038 | **MEASURED** — both logs on disk, ratios reproduce. |
| F9 | "the trio collapses into ONE row `sigma.tau.GW_conv_ffi` = **4.99 s** (vs 191.9 XLA — 38.5×)" | `ffi_fft_proto_notes.md:231-233` | job 7878727 pass d | **MEASURED** — `fftffi_ab.7878727.out:390`; 191.905/4.992 = 38.44. |
| F10 | "tau-kernel HLO: xla fft ops 3 → 0; transposes 6 → 0" | `ffi_fft_proto_notes.md:240-242` | job 7878727 | **MEASURED**, and **independently CONFIRMED by this audit**: my after_optimizations census of the XLA 3-FFT composition at the same shapes emits exactly **3 fft, 6 transposes, 3 copies** (§B.2). |
| F11 | "χ0 bonus: chi.exec 7.76 → 2.52 s (3.1×), W.exec 13.0 → 14.0 (noise)" | `ffi_fft_proto_notes.md:253-255` | job 7878727 | **MEASURED**. |
| F12 | "the unit gate's standalone 151 ms was **dispatch/alloc-bound, not engine-bound**" | `ffi_fft_proto_notes.md:268` | inference from the 11× gap to production's 13.4 ms/tile | **INFERRED** — and it directly undercuts F1/F2 as an engine comparison (§C.1). |
| F13 | "production hits ~60 GB/s/rank (13.4 ms per 400 MB tile)", "~120 GB/s/node" | `ffi_fft_proto_notes.md:266-268` | job 7878727 | **MEASURED (derived)**; the per-node figure is ×2 arithmetic. **This audit measures the actual node wire: 142.8 GB/s copy / 160.5 GB/s triad at 28 cores (§B.1)** — so 2×60 = 120 GB/s is ~84% of the measured copy wire. Credible. |
| F14 | "memory honest-neutral: temp arena 1.15 GiB unchanged (fused) / +90 MiB (decomposed)" | `ffi_fft_proto_notes.md:245-252` | job 7878727 | **MEASURED**; honestly reported as a non-win. |
| F15 | GPU: "XLA 12.63 ms (63.1 GB/s) / FFI strided **7.08 ms (112.6 GB/s)** … ~1.8× both" | `wk_REL/docs/cufft_mirror_notes.md:130-137` | job 7879275, RTX 5000, production local shapes | **MEASURED** — `cufft_gate.7879275.out:81-82`. |
| F16 | GPU production: "parity-class at this deck scale — g402 sigma.exec 6.169 (ON) vs 6.009 (OFF)" | `cufft_mirror_notes.md:163-166` | job 7879275 | **MEASURED**; ON is 2.7% *slower*. Honestly disclosed. |
| F17 | "**No production-scale GPU A/B** … nothing like the CPU 3.78× should ever be claimed" | `cufft_mirror_notes.md:194-197` | — | explicit, correct scope guard. |
| F18 | "**the pathology is absent by construction**: the axes are already minor-most, so XLA's own path pays no transpose" (the `local_*` sites) | `gemm_portability_bse_notes.md:271` | none | **INFERRED** — from reading `axes=` kwargs in `src/bse/` plus XLA's documented minor-most requirement. Never HLO-verified, never timed. **This audit proves the transpose half CORRECT and the engineering conclusion drawn from it WRONG (§C.2).** |
| F19 | "those call sites … pay no layout tax, and a plain-transform backend would hand them that 18% penalty for nothing" | main transcript `a988409f…jsonl` L2668 | none | **INFERRED, and an over-extension** — transplants a *k-major strided distance-1* DFTI number onto a *batch-contiguous* configuration that was never benchmarked. **REFUTED by §B.3.** |
| F20 | `local_*` FFI verdict: "Neutral to NEGATIVE on CPU …, ≈0 on GPU" → **DO NOT** | `gemm_portability_bse_notes.md:407` | none | **REFUTED** by §B.3 (CPU: MKL is 6–10× XLA at those exact shapes) and §B.5. |
| F21 | "≈0 on GPU (XLA:GPU already dispatches cuFFT for `jnp.fft`)" | `gemm_portability_bse_notes.md:298-299` | none | **ASSERTED** — and contradicted *in sign* by the sibling workstream's own measured GPU 1.78× (F15). |
| F22 | "FFT/layout buckets … are **CLOSED as structural on XLA:CPU**"; "the dot↔fft layout alternation is CLOSED as not-removable on this backend" | `wk_REL/docs/sigma_perf_results.md:134-135`, `:184-187` | jobs 7878038/7878110 | **REFUTED ~24 h later** by the MKL-DFTI FFI (job 7878727): removed on the *same* backend without changing the kernel math — a third exit the claim did not enumerate. **No claim-decay banner was ever added to that file.** |
| F23 | "Plain `jnp.fft.fftn` on a sharded tensor … inserts a gather … and a ~2× replicated temp buffer, inflating reported peak by ~8×" | in-repo `src/common/fft_helpers.py:71-77` | none | **ASSERTED** — no job, log or dump anywhere in the corpus for the ~2×/~8× figures. |
| F24 | "BSE call sites route through the fft_helpers entry points … so they inherit this backend switch automatically" | `docs/dev/env_vars.md` (as written 2026-07-29) | none | **REFUTED** same day by `gemm_portability_bse_notes.md:138-152`: `local_ifftn3`/`local_fftn3` are one-line aliases of `jnp.fft`; zero flat-k call sites in `src/bse/`. Correction banner already applied there. |
| F25 | "65% of the τ wall → ~7% (b) / ~5% (d)" | `ffi_fft_proto_notes.md:234` | job 7878727 | **ARITHMETIC UNVERIFIED** — b = 10.85/71.64 = 15.1%, d = 4.99/65.73 = 7.6%. No denominator reproduces 7%/5%. |
| F26 | "measured 60–65% of sigma.exec at nb=128/P=64" (in-repo comment) | `src/common/fft_helpers.py` header | job 7878038 | **WRONG DENOMINATOR** — 191.9 s is 65% of the *staged τ dispatch* (295.0 s), not of sigma.exec (272.0 s, = 70.5%). The shipped comment names the wrong baseline. |

### A.2 GEMM claims

| # | Claim (verbatim, ≤2 lines) | Source | Measured at | Status |
|---|---|---|---|---|
| G1 | "71% of it is the local projection GEMM running **4.3× below the node's BLAS roofline** (… Eigen zgemm at **295 GF/s vs 1263 GF/s** measured BLAS)" | `wk_REL/docs/RESHARD_OVERHEAD_MEMO.md:3-6` | jobs 7878862/83/907, P=64, 2 ranks/node | **MEASURED WALL-CLOCKS, ARITHMETICALLY WRONG RATIO.** See §C.3 — I verified the bug in primary source. Corrected BLAS rate ≈ **631 GF/s**; corrected gap ≈ **2.1×**, not 4.3×. |
| G2 | "GEMM 173.5 ms (**71%**) + collectives 49.9 (20%) + …" (projection-row anatomy) | `RESHARD_OVERHEAD_MEMO.md:321-327` | job 7878907 | **MEASURED** — reproduces production `project_rs` to 0.5%. The 71% *share* stands; it is a different quantity from G4's 71% *efficiency* (§C.5). |
| G3 | "numpy BLAS zgemm (venv, 28 threads, **same flops**) | 40.4 | ~1263" | `RESHARD_OVERHEAD_MEMO.md:256-257` | same | **REFUTED on "same flops"** — verified by me from primary source (§C.3). |
| G4 | "scaling 1→28 threads is near-linear (**19.6→388 GF/s, 71% efficiency**) — **not pool-miswired**" | `wk_REL/docs/contract_bands_notes.md:254-256` | job 7879008, 1 process, taskset widths | **MEASURED** — `cbands_gate.7879008.out:129,183`; 388.5/19.6/28 = 70.8%. **Verdict CONFIRMED and strengthened by this audit (§B.6) — but the probe never tested dtype promotion (§C.4).** |
| G5 | "the bare Eigen dot saturates **1.6–1.9×** below vendor BLAS at full threads" | `contract_bands_notes.md:241,260`; **shipped to repo** `docs/dev/env_vars.md:129` | job 7879008, 28 thr | **MEASURED, upper bound INFLATED** — the row's four ratios are 1.57/1.81/1.64/1.57 → honest band **1.57–1.81×**. 1.88 appears only in an `omp1` control row. |
| G6 | "the batched-skinny shape scales as well as the square one" | `contract_bands_notes.md:258-259` | job 7879008 | **PARTIALLY REFUTED by its own log** — 1→28 efficiency prod-f64 71% vs sq-f64 85%; absolute w28 388.5 vs 498.3 GF/s. Supportable claim is "no pathological collapse", not "as well as". |
| G7 | "XLA promotes each f64 channel operand to c128 … 2× the mathematically required flops" | `RESHARD_OVERHEAD_MEMO.md:268-274` | HLO dump, job 7878862 | **INFERRED (HLO), CONFIRMED by fix** (job 7878942, project_rs 43.2→38.7) and **independently CONFIRMED by this audit's HLO buffer census (§B.7)**. |
| G8 | "Eigen dgemm measured **7.3× below BLAS**" | `wk_REL/OWNER_DECISIONS.md:39-41` | — | **REFUTED** — this is 1263/172, i.e. the §C.3 error compounded with a cross-job, cross-dtype, cross-thread-count pairing. Width- and dtype-matched value is 1.57×. |
| G9 | "FFI MKL GEMM: nb=128 staged project_rs 29.407 → **19.622 s** (−33%); prod sigma.exec 58.313 → 49.224" | `contract_bands_notes.md:182-190`; `docs/dev/env_vars.md:129` | job 7879010, P=64, cache-cold | **MEASURED — best-evidenced GEMM claim in the corpus.** Every number reproduces in `cbands_ab.7879010.out`. Parity exact-0, collective tables byte-equal GEMM-off/on. |
| G10 | "the FFI MKL GEMM **REALIZES the memo's ~19 s** projection" | `contract_bands_notes.md:186-187` | 7879010 vs memo | **COINCIDENCE, not confirmation** — the projection substituted a 1-channel MKL time for a 2-channel operation (§C.3), *and* the baseline moved 43.2→29.4 via the Laplace merge in between. Two errors of opposite sign. Demote to "the realized wall matches; the projection's mechanism does not". |
| G11 | "Campaign arc for the composed **nb=256** stack: **272.0 (AQ era)** → 35.6 → 35.5 → **29.98**" | `contract_bands_notes.md:191-192` | mixed | **SHAPE CONFOUND** — 272.0 is nb=128/μ=4992; the rest are nb=256/μ=2496. The true nb=256 baseline on disk is 86.99 s (job 7878104). Reads as 9.1× where the honest figure is ~2.9×. `docs/dev/HANDOFF_2026-07-29.md:19` states it correctly. |
| G12 | "the bare isolated dot runs 388/468 GF/s — substantially FASTER than the same contraction inside the production module (172/295) … the remaining ~2× is NOT the dot kernel" | `contract_bands_notes.md:263-269` | 7879008 vs 7878883 | **MEASURED both sides, CAUSE UNRESOLVED** — the notes name "surrounding module and/or in-job co-tenancy (2 ranks/node)" without separating them. **This audit tests the co-tenancy half directly (§B.8).** |
| G13 | "`--intra_op_parallelism_threads` does not exist in this jaxlib … XLA:CPU sizes its Eigen pool from process CPU **AFFINITY**, so `taskset` IS the thread cap" | `SPEEDUP_SCORECARD.md:353-359` | job 7874158 | **MEASURED**, later confirmed by 7879008. Used as the control mechanism by this audit too. |
| G14 | "OMP_NUM_THREADS=1: xla rates UNCHANGED — OMP_NUM_THREADS does not govern the Eigen pool" | `contract_bands_notes.md:244-246` | job 7879008 | **MEASURED**. |
| G15 | "XLA:GPU's dot lowering **already dispatches cuBLAS, optimal**" — the justification for `LORRAX_BANDS_GEMM_FFI` auto being silently OFF on CUDA | `docs/dev/env_vars.md:129`; `docs/dev/staged_reshard_primitive.md:208-211` | none | **ASSERTED — no GPU GEMM measurement existed anywhere in the corpus.** **This audit measures it for the first time and CONFIRMS it (§B.9).** |
| G16 | ISDF downfolding: "gemm W·S (3000) native shard_map 0.333 s / FFI 0.342 — **no FFI win**" | `SPEEDUP_SCORECARD.md:1110-1113` | jobs 7874650/54 | **MEASURED** — a real counter-example: vendor routing was neutral for *that* GEMM family. |
| G17 | BSE GEMM runtime share: "**UNMEASURED, and not obtainable without new jobs**" | `gemm_portability_bse_notes.md:393-401` | — | **HONEST NON-CLAIM** — correctly abstains. |
| G18 | "Eigen zgemm 295 GF/s vs 1263 GF/s BLAS for the same contraction" | **in-repo** `docs/dev/staged_reshard_primitive.md:161-163` | — | **REFUTED (§C.3)**, and **self-contradicted 20 lines later** at `:182-184` ("1.6–1.9× below MKL"). Highest-priority repo-doc fix. |

### A.3 Corpus-level gaps found

* **`SPEEDUP_SCORECARD.md` contains zero occurrences** of `LORRAX_FFT_FFI`, `mklfft`,
  `DFTI`, `LORRAX_BANDS_GEMM_FFI`, `mklblas`, or `contract_bands`. The entire
  FFT-FFI + GEMM-FFI campaign (~5.5× on `sigma.exec` at nb=128) is **unledgered**,
  against the project's own convention (`wk_REL/docs/ARTIFACT_MAP.md:114`: "The ledger is
  SPEEDUP_SCORECARD.md").
* Logs for jobs 7878038 / 7878092 / 7878110 are under
  `/scratch2/08271/jackmc/mos2_4x4_test/`, **not** `wk_REL/`. Logs for jobs
  7878038/7878092/7878110-era `project_rs 84.75 → 47.6` rows quoted at
  `SPEEDUP_SCORECARD.md:9640` are **not on disk**; only derived `.md` survives.
* **No node-level roofline (peak GF/s or achievable memory bandwidth) was ever
  measured in this campaign.** "BLAS roofline" always meant "the measured numpy/MKL
  rate"; DRAM bandwidth appears once as an *assumption* ("~110 GB/s/rank",
  `RESHARD_OVERHEAD_MEMO.md:86`). §B.1 supplies the measured wire.

---

## PART B — THE EXPERIMENT

### B.0 Design and controls

**Factors varied explicitly.**

| factor | levels |
|---|---|
| implementation | XLA native · FFI plain · FFI fused (`gw_conv`) · bare MKL (ctypes, no XLA) · bare pocketfft/scipy |
| layout | **flat-k / k-major** (transform axes LEADING, batch minor — the Σ-τ and χ/W dot-layout sites) vs **local minor-most** (transform axes trailing, batch leading — the BSE `local_*fftn3` sites) |
| shape | S0 2.1 MB · S1 99.7 MB (χ/W V-tile) · S2 398.7 MB (**production** Σ-τ G-tile) · S3 1.595 GB (**rung-scale**) · S4 398.7 MB at nk=64 (high-k-count) |
| threads (CPU) | 1, 4, 14, 28 |
| buffer discipline | reuse-input/no-donate · fresh-input/donate · fresh-input/no-donate — **applied identically to both arms** |
| platform | Frontera CLX (2×Xeon 8280, 56 cores, 192 GB) · rtx-dev (Quadro RTX 5000, sm_75, 16 GB) |

**Controlled comparison of layout.** The flat-k array `(nk, *trail)` and its
minor-most twin `(*trail, *kgrid)` hold the **same element count and the same
flop count** — only the memory order differs. Layout is therefore the sole
varying factor between cells A and B at every shape.

**Thread control.** `XLA_FLAGS=--intra_op_parallelism_threads` does not exist in
this jaxlib (corpus claim G13, job 7874158); XLA:CPU sizes its Eigen/DUCC pool
from process CPU **affinity**. So each thread count is a **separate process under
`taskset -c 0-(N-1)`**, with `OMP_NUM_THREADS`/`MKL_NUM_THREADS` matched — XLA's
pool, MKL's pool and the FFI handler's OpenMP team all see the same core budget.
This is the same convention production uses (`taskset -c $((S*28))-$((S*28+27))`).

**Quiet-node control.** `small` and `rtx-dev` are both `OverSubscribe=NO`
(exclusive). Node hygiene was recorded at job start and end:

* c208-002 (FFT): `loadavg 0.10` at start, other-user procs = 0, `squeue -w`
  showed only job 7879376 at start **and** at end.
* c208-010 (GEMM): `loadavg 0.13` at start, only job 7879377 on the node.
* c196-012 (GPU): `loadavg 0.00` at start, only job 7879378, 4× RTX 5000 idle.

All cells are therefore **solo windows**. No shared-window cells were taken.

**Reps / statistics.** ≥5 reps (7 for CPU, 9 for GPU, 25 for the distribution
probe). Reported as **median [min–max] IQR spread%**. Cold (first) call is
recorded separately from warm steady state. Cache state: all tiles ≥ 99.7 MB are
far past the 39 MB LLC, so every warm rep is a DRAM-resident streaming
measurement; S0 (2.1 MB) is the deliberate LLC-resident contrast.

**⚠ Topology finding that conditions every thread number here (job 7879379).**
On Frontera CLX compute nodes the CPU ids are **NUMA-interleaved**:
`NUMA node0 = 0,2,4,…,54`, `NUMA node1 = 1,3,5,…,55`. Therefore
`taskset -c 0-27` is **14 cores on socket 0 + 14 cores on socket 1**, not one
socket — and production's `taskset -c 0-27` / `28-55` gives each MPI rank a
**both-sockets** core set, with the two ranks interleaved across both memory
controllers. This is not what "28 cores per rank, one per socket" implies. It is
consistent across my cells and production's, so comparability is preserved, but
it is a previously unrecorded property of the campaign's pinning.

### B.1 The wire (roofline reference) — measured, job 7879376 cell 1

`audit_stream.c`, 800 MB arrays, best of 7, same node/job/thread-sets as the FFT cells:

| threads (cores) | copy GB/s | triad GB/s |
|---|---|---|
| 1 | 12.7 | 12.9 |
| 4 | 49.7 | 50.6 |
| 14 | 124.9 | 139.5 |
| **28** | **142.8** | **160.5** |
| 56 | 135.8 | 147.5 |

**This is the first measured node bandwidth in the campaign** (the corpus used an
assumed "~110 GB/s/rank", `RESHARD_OVERHEAD_MEMO.md:86`). An out-of-place batched
FFT must move ≥ 2× the tile; for the 398.7 MB production tile that is 797 MB, so
the **28-thread floor is 5.6 ms**. Every FFT number below is quoted against it.

GPU wire (job 7879378 cell 1, RTX 5000): device read+write of a 512 MB f64 array
= **329.6 GB/s** (74% of the 448 GB/s spec). f64 square GEMM 2048³/4096³ =
**359.2 / 367.9 GF/s** — at or fractionally above the nominal ~350 GF/s FP64 peak.

### B.2 HLO evidence — does XLA transpose for batched FFTs already minor-most?

**This is the owner's open question, and it is now settled from
`after_optimizations` HLO at the real call-site shapes** (job 7879376 cell 2,
`wk_REL/results/logs/audit_hlo.log`). Counts are over the whole module, entry + every fused
computation; "bytes" is bytes produced by the op.

**Flat-k sites (transform axes LEADING) — transposes ARE present, at every shape:**

| site | tile | fft ops | transposes | copies | bytes moved by transpose+copy | XLA temp |
|---|---|---|---|---|---|---|
| Σ-τ G tile `(16,2,624,2,624)` | 398.7 MB | 1 | **2** (797.4 MB) | **1** (398.7 MB) | **3.00× the tile** | 398.7 MB |
| χ/W V tile `(16,624,624)` | 99.7 MB | 1 | 2 (199.4 MB) | 1 (99.7 MB) | **3.00×** | 99.7 MB |
| rung-scale G tile `(16,2,1248,2,1248)` | 1594.9 MB | 1 | 2 (3189.8 MB) | 1 (1594.9 MB) | **3.00×** | 1594.9 MB |
| high-k `(64,2,312,2,312)` | 398.7 MB | 1 | 2 (797.4 MB) | 1 (398.7 MB) | **3.00×** | 398.7 MB |

The named ops are full-tile: e.g. `transpose.0 c128[4,4,1,2,624,2,624] 398.7 MB`,
`copy.3` 398.7 MB, `transpose.1 c128[2,624,2,624,4,4,1]` 398.7 MB. The ratio is
**exactly 3.00× the input tile at every shape and every k-count** — the layout tax
is structural and shape-independent, confirming the corpus's mechanism claim.

**Local minor-most sites (transform axes ALREADY minor-most) — ZERO transposes:**

| site | tile | fft ops | transposes | copies | bitcasts | total instrs | XLA temp |
|---|---|---|---|---|---|---|---|
| controlled twin of the Σ-τ G tile `(1557504,4,4,1)` | 398.7 MB | 1 | **0** | **0** | 0 | 5 | **0 B** |
| BSE `W_R` `(624,624,4,4,1)` axes (2,3,4) | 99.7 MB | 1 | **0** | **0** | 0 | 5 | **0 B** |
| BSE `T_k` `(312,312,2,2,4,4,1)` axes (4,5,6) | 99.7 MB | 1 | **0** | **0** | 0 | 5 | **0 B** |
| BSE `W_R` rung-scale `(1248,1248,4,4,1)` | 398.7 MB | 1 | **0** | **0** | 0 | 5 | **0 B** |
| wfn box `(256,24,24,60)` axes (1,2,3) | 141.6 MB | 0¹ | **0** | **0** | 0 | 1 | **0 B** |

¹ the mixed-radix box compiles to a single opaque instruction my census did not
resolve to an `fft` opcode; the transpose/copy counts (0) and temp (0 B) are still
authoritative. Minor open item, flagged rather than glossed.

**Independent corroboration (concurrent workstream, not mine).** While this audit
was running, a separate workstream committed `068286c` recording its own HLO probe
(job 7879370) reaching the same conclusion from a different shape: "batched
minor-most ffts move 0.00 MB while the k-major flat-k layout moves 124.60 MB
(5.0x) on the same 24.92 MB tile". Two independently designed HLO probes, two
different shape families, same verdict.

**VERDICT (owner's question): XLA:CPU does NOT insert layout transposes for
batched FFTs whose transform axes are already minor-most.** Zero transposes, zero
copies, zero temp bytes, five instructions — at the controlled twin of the
production tile and at every real BSE call-site shape. The `local_*` inference
(F18) was **correct as to transposes**. The XLA fused 3-FFT composition, by
contrast, emits **3 fft + 6 transposes + 3 copies = 5.40× the input tile moved**,
independently reproducing the corpus's "transposes 6" figure (F10).

### B.3 CPU FFT results

All numbers ms, **median** of 7 reps, S2 = the 398.7 MB production Σ-τ G tile,
`reuse-input / no-donate` for every arm unless stated. Full CSV in
`wk_REL/audit_fft_t{1,4,14,28,28b}.log`.

**Flat-k layout (the Σ-τ / χ-W sites):**

| threads | XLA native | FFI plain | bare MKL strided | pocketfft strided | FFI vs XLA |
|---|---|---|---|---|---|
| 1 | 869.6 | 346.5 | 172.4 | 640.8 | **2.51×** |
| 4 | 631.0 | 212.4 | 43.9 | 629.9 | **2.97×** |
| 14 | 825.8 | 73.6 | 28.9 | 622.9 | **11.2×** |
| 28 | 579.4 | 65.5 | 30.1 | 625.4 | **8.8×** |
| 28 (repeat) | 573.8 | 67.9 | 30.3 | 625.7 | **8.5×** |

**Local minor-most layout (the BSE sites) — same elements, same flops:**

| threads | XLA native | bare MKL contiguous | pocketfft contiguous | MKL vs XLA |
|---|---|---|---|---|
| 1 | 515.9 | 50.8 | 690.2 | **10.2×** |
| 4 | 383.1 | 17.6 | 685.5 | **21.7×** |
| 14 | 94.7 | 8.25 | 689.7 | **11.5×** |
| 28 | 49.1 | 11.0 | 688.6 | **4.5×** |
| 28 (repeat) | 48.6 | 8.08 | 690.4 | **6.0×** |

**Fused conv (flat-k only), S2:**

| threads | XLA 3-FFT composition | FFI fused `gw_conv` | ratio |
|---|---|---|---|
| 1 | 1529.3 | 670.6 | 2.28× |
| 4 | 1501.1 | 291.9 | 5.14× |
| 14 | 1122.9 | 78.7 | 14.3× |
| 28 | 1308.5 | 76.6 | **17.1×** |
| 28 (repeat) | 1169.0 | 60.1 | **19.5×** |

**Rung-scale (S3, 1.595 GB), 28 threads:** flat-k XLA 2959.3 / FFI 256.5
(**11.5×**); minor-most XLA 385.4 / MKL 43.0 (**9.0×**); conv XLA 6199.2 /
FFI 448.0 (**13.8×**). The FFI advantage does **not** decay at rung scale — it
grows slightly.

**Against the wire (28 threads, S2, 797 MB minimum traffic ⇒ 5.6 ms floor):**

| implementation | ms | × the 2-pass floor | % of the 142.8 GB/s copy wire |
|---|---|---|---|
| bare MKL, minor-most (in-place) | 8.1–11.0 | 1.4–2.0× | 51–69% |
| FFI plain, flat-k | 31–68 | 5.5–12× | 8–26% |
| XLA native, minor-most | 48.6–49.1 | 8.7× | 11% |
| XLA native, flat-k | 573.8–584.0 | 103× | 1.4% |

Even charged against its **own** structural floor (3× tile in transposes + the
transform ≈ 3.2 GB ⇒ 22 ms), XLA's flat-k path is still **26× off**: the
transposes are strided full-tile gathers (16 k-lines 24.9 MB apart), which is a
TLB/latency problem, not a bandwidth one.

### B.4 The disputed "plain FFI is 18% slower than XLA (151 vs 128 ms)" — resolved

Two independent defects, compounding, both in the same direction.

**(i) Buffer-discipline asymmetry.** `wk_REL/probes/fft_ffi_unit_gate.py:163-192` gives the
XLA arm `jax.jit(gi_ref)` with **one reused input**, and the FFI arm
`jax.jit(gi_ffi, donate_argnums=(0,))` with a **freshly allocated 400 MB input per
rep**. Cell M of this audit runs **both** arms in **all three** disciplines
(S2, 28 threads):

| discipline | XLA | FFI | FFI advantage |
|---|---|---|---|
| reuse-input / no-donate | 577.7 | 65.4 | **8.8×** |
| fresh-input / donate | 712.4 | 152.1 | **4.7×** |
| fresh-input / no-donate | 719.7 | 200.1 | **3.6×** |
| *(repeat window)* reuse / no-donate | 579.0 | 49.1 | 11.8× |
| *(repeat window)* fresh / donate | 731.6 | 165.8 | 4.4× |

The FFI wins in **every matched cell, at every discipline, by 3.6–11.8×**. Note
the historical harness's FFI number (fresh/donate) is 152.1 ms here — reproducing
the historical 151.07 ms almost exactly. The bias is worth ~2.3× against the FFI.

**(ii) The XLA reference was a first-process-in-job artifact.** Job 7879401 ran the
**unmodified original harness six times back-to-back on one exclusive node**,
alternating the `.so` (which does not affect the XLA code path at all):

| run | `.so` | XLA reference `G ifft` | FFI 28-thread row |
|---|---|---|---|
| 1 | OLD | **113.16 ms** | 150.92 |
| 2 | NEW | 578.95 | 156.12 |
| 3 | OLD | 569.14 | 152.60 |
| 4 | NEW | 576.29 | 151.08 |
| 5 | OLD | 570.43 | 150.79 |
| 6 | NEW | 576.67 | 157.99 |

(and in job 7879395: first process 136.39 ms, second 580.69 ms.)

The XLA arm reads **113–136 ms only for the first Python process in a fresh
allocation** and **569–581 ms for every subsequent one**, regardless of `.so`.
The FFI arm is insensitive (150.8–158.0, spread 4.8%). Job 7878719's unit gate
was by construction the *first and only* timing process in its job, so it
captured the artifact value. A clean 25-rep isolated probe (job 7879395 cell R2)
gives XLA **median 584.0 ms [571.4–603.5], spread 5.5% — unimodal**, and FFI
**median 31.2 ms**, i.e. **18.7×**.

**Which value represents production?** Production XLA `G_ifft` = 79.63 s / 176 τ =
**452 ms per tile** (job 7878038). The steady-state 573–584 ms is within 30% of
that; the 127.80 ms headline is **3.5× off**. The steady-state number is the
representative one.

**Mechanism of the first-process effect: UNDECIDED.** Leading hypothesis is
transparent-huge-page availability on a freshly allocated node (a strided
full-tile transpose is exactly TLB-bound), but I did not measure it. **Deciding
measurement:** `perf stat -e dTLB-load-misses,dtlb_load_misses.walk_completed`
plus `/proc/vmstat` `thp_fault_alloc` on the first vs a later process, or
`echo never > /sys/kernel/mm/transparent_hugepage/enabled` (needs root) — or,
without root, `madvise`-free comparison by pre-faulting the arena.

### B.5 GPU FFT results (job 7879378, RTX 5000, 9 reps)

S2 = 398.7 MB production tile. XLA:GPU lowers `jnp.fft` to cuFFT, so "XLA" here
*is* cuFFT with XLA's chosen layout.

| layout | XLA (cuFFT) | FFI (cuFFT strided) | ratio |
|---|---|---|---|
| flat-k, S1 | 5.38 | 2.52 | **2.14×** |
| flat-k, S2 | 22.28 | 9.18 | **2.43×** |
| flat-k, S4 (nk=64) | 21.85 | 10.53 | **2.07×** |
| **minor-most, S2** | **8.90** | *(not implemented)* | — |
| conv, S2 | 33.89 | 15.62 | **2.17×** |

Two things follow. (a) The historical GPU 1.8× (F15) is **CONFIRMED and revised
slightly upward to 2.1–2.4×** at these shapes. (b) **XLA at minor-most (8.90 ms)
≈ FFI at flat-k (9.18 ms)** — i.e. the entire GPU FFI win is exactly the recovery
of the flat-k layout tax, and at minor-most XLA:GPU is *already* at the FFI's
level. An FFI backend for the `local_*` sites would therefore buy **≈0 on GPU** —
the previously **ASSERTED** claim F21 is now **MEASURED-EQUIVALENT and CONFIRMED**.

### B.6 CPU GEMM results — the four-way discriminator (job 7879377)

Achievable vendor peak on this node (8192³ f64, 28 threads, solo):
**MKL 1699.7 GF/s · FFI handler 1807.3 GF/s · numpy/OpenBLAS 1716.8 GF/s ·
XLA:CPU Eigen 564.2 GF/s.**

XLA-vs-MKL ratio, **matched dtype, matched thread count, matched shape**:

| shape | dtype | 1 thr | 4 thr | 14 thr | 28 thr |
|---|---|---|---|---|---|
| square 2048³ | f64 | **3.19×** | 2.88× | 3.13× | **3.01×** |
| square 2048³ | c128 | **3.33×** | 3.28× | 3.33× | **3.37×** |
| square 4096³ | f64 | — | — | 3.27× | 3.16× |
| square 4096³ | c128 | — | — | 3.33× | 3.38× |
| square 8192³ | f64 | — | — | — | **3.01×** |
| **production projection** (b=16, 1248×1248 @ 1248×128) | f64 | **2.94×** | 3.09× | 2.02× | **3.33×** |
| **production projection** | c128 | **3.15×** | 3.26× | 2.02× | **2.97×** |
| projection, batch=1 | f64 | 3.04× | 3.02× | 17.9×¹ | 28.7׹ |
| projection, batch=1 | c128 | 3.22× | 3.29× | 3.57× | 3.80× |
| rung-scale projection | f64 | — | — | 2.00× | 2.37× |
| rung-scale projection | c128 | — | — | 1.86× | 2.58× |

¹ the batch=1 f64 cell at 14/28 threads has XLA at 36–40 GF/s with huge IQR — a
small-problem dispatch-floor artifact (0.35 ms of work), not a kernel result;
excluded from the conclusion.

The two cells that sit off the ~3× band are honest anomalies, not conclusions:
the **projection at 14 threads** (2.02× for both dtypes) is MKL under-performing
there (438.8 GF/s vs 863.6 for square 2048³ at the same width), i.e. an MKL
threading-threshold effect at batch=16 rather than an XLA improvement; and the
**batch=1 f64 cells at 14/28 threads** are a 0.35 ms dispatch floor (footnote 1).

**The gap is ~3× and it is essentially FLAT across every factor:** 1 thread → 28
threads, 2048³ → 8192³ square, square → batched-skinny, f64 → c128. Therefore:

* **NOT thread-pool wiring.** The gap is 3.19× (f64) / 3.33× (c128) at **one
  thread**, where there is no pool to miswire. Independent confirmation of the
  corpus verdict G4 — reached here by a route the original probe could not take.
* **NOT shape.** 3.01× at 8192³ square (the most GEMM-friendly shape available)
  vs 3.33× at the production batched-skinny shape. Batched-skinny costs a further
  ~10%, not a category difference. (Corpus G6's "scales as well as square" is
  still too strong; "no pathological collapse" is right.)
* **NOT dtype promotion** for the matched cases — the gap is the same for pure f64
  and pure c128. Promotion is a *separate, additive* penalty (below).
* **IT IS Eigen kernel quality** — a flat ~3× per-core throughput deficit vs MKL.

**Dtype promotion, isolated** (production projection shape, 28 threads,
GF/s-useful charged on the real f64×c128 work):

| form | ms | GF/s-useful |
|---|---|---|
| XLA, mixed f64 O × c128 ψ (promoted) | 79.2 | 161.1 |
| XLA, f64-split (2 dgemm + `lax.complex`, commit 3c44494) | 40.2 | 317.8 |
| MKL/FFI, f64-split | 10.85 | 1175.6 |

De-promotion is worth **1.97×** and the vendor GEMM a further **3.70×** —
together **7.3×** from the original promoted-XLA form. HLO buffer census
independently confirms the promotion: the mixed case carries **398.7 MB of XLA
temp** (exactly the c128 widening of the 199.3 MB f64 operand) where both matched
cases carry **0 B** (§B.2 part 4).

**The FFI GEMM handler delivers full vendor rate**: at 28 threads it matches or
beats bare MKL in every cell (e.g. production projection f64 1299.9 vs 1156.9
GF/s; 8192³ 1807.3 vs 1699.7).

### B.7 Co-tenancy control — **RETRACTED (harness defect found by this audit)**

> ⚠ **This cell measured nothing and its conclusion is withdrawn.** The two
> "concurrent" ranks were launched as two separate `srun -N1 -n1 ... &` steps.
> SLURM **serialised** them: the in-log timestamps show rank 0 running
> 02:12:39→02:12:46 and rank 1 running 02:12:48→02:12:55, with **no overlap**.
> The cell therefore compared solo against solo, which is exactly why it showed
> "no degradation". Any conclusion about in-job co-tenancy drawn from it is
> unsupported.
>
> Root cause: a second `srun` step inside an allocation waits for task slots
> unless it is given `--overlap`; concurrency must instead come from **one**
> `srun --ntasks-per-node=2 -n2` with the per-rank binding selected from
> `$SLURM_LOCALID` — which is how production actually launches its 2 ranks/node.
>
> Corrected measurement: **§B.9**, job 7879453, with start/end timestamps printed
> per rank so overlap is checkable rather than assumed. The corpus claim G12
> ("surrounding module and/or in-job co-tenancy") therefore returns to
> **UNDECIDED-pending-§B.9** rather than "co-tenancy refuted".

*Methodological note for the campaign:* any past cell that created concurrency
with backgrounded `srun` steps inside one allocation is suspect for the same
reason. I did not audit the corpus for this pattern; it is worth a sweep.

### B.8 GPU GEMM results (job 7879378) — first measurement in the campaign

| shape | dtype | XLA:GPU (cuBLAS) GF/s | % of ~350 GF/s FP64 peak |
|---|---|---|---|
| square 2048³ | f64 | 360.1 | 103% |
| square 4096³ | f64 | 364.3 | 104% |
| production projection | c128 | 366.5 | 105% |
| projection, batch=1 | c128 | 315.1 | 90% |
| standalone probe 2048³/4096³ | f64 | 359.2 / 367.9 | 103/105% |

**XLA:GPU's dot lowering runs at the device's FP64 roofline.** Corpus claim G15
("XLA:GPU already dispatches cuBLAS, optimal"), previously **ASSERTED with no
measurement anywhere**, is now **MEASURED and CONFIRMED**. The design decision to
keep `LORRAX_BANDS_GEMM_FFI` auto-OFF on CUDA is correct.

One new GPU finding: dtype promotion hurts on GPU too — production projection
mixed f64×c128 = 155.8 GF/s-useful vs f64-split 332.1 GF/s-useful = **2.13×**.
The de-promotion relowering (3c44494), justified and gated on CPU only, is a
comparable win on GPU and was never tested there.

---

### B.9 PINNING A/B — the NUMA-interleaved binding, quantified

Jobs **7879455 / 7879466** (microbench, 1 node exclusive, two independent
windows) and **7879483** (production GW driver). Concurrency is now real: every
paired cell is ONE `srun --ntasks-per-node=2 -n2` with binding from
`$SLURM_LOCALID`, and each rank prints start/end timestamps (verified
overlapping, e.g. both ranks start 02:51:28.477, end 02:51:28.99).

**Arms** (chosen to separate CPU locality from MEMORY locality):

| arm | binding | rationale |
|---|---|---|
| **A** "current" | `taskset -c 0-27` / `28-55` | what production does — 14 cores on *each* socket per rank |
| **B** "cpu-local" | `taskset -c 0-54:2` / `1-55:2` | socket-local CPUs, memory by first touch |
| **C** "cpu+mem-local" | `numactl --cpunodebind=N --membind=N` | also pins the memory policy |

**Choice of mechanism — use B (`taskset` stride-2).** Arm C matched arm B to
within noise in every cell (e.g. GEMM proj/f64/xla 14.9 vs 14.4 ms; STREAM
paired 146.0 vs 146.3 GB/s aggregate), so `--membind` buys nothing once the CPUs
are socket-local: with same-thread first touch the pages already land locally.
And `numactl` is **not installed in the `py312.sif` container** (confirmed:
`numactl: command not found`), so arm C must wrap `apptainer` on the host —
extra machinery for no measured gain. `taskset -c $S-$((S+54)):2` works inside
the container and is robust to the interleaved enumeration.

**B.9.1 STREAM (pure bandwidth), aggregate over the 2 concurrent ranks, 3 reps**

| arm | copy GB/s (r0 + r1) | per-rank spread |
|---|---|---|
| A interleaved | 73.5+75.9, 77.4+73.0, 74.1+79.3 → **~151** | read column swings 86–125 GB/s |
| B cpu-local | 72.6+72.8, 73.1+73.3, 72.7+73.4 → **~146** | 115.3–117.7, tight |
| C cpu+mem-local | 73.4+73.0, 73.1+73.1, 73.3+73.0 → **~146** | 116.0–116.9, tight |

Interleaved wins ~3.6% on *aggregate* bandwidth (each rank reaches both memory
controllers) but is markedly **less fair and less reproducible**. Solo control:
one rank alone gets 142.5 GB/s interleaved vs 72.9 socket-local — i.e. a *solo*
A/B would have scored the fix as a 2× regression. That is why the paired cells
are the primary ones.

**B.9.2 Microbench, paired, both windows (ms; A/B > 1 means socket-local wins)**

| cell | PAIR-A run1 / run2 | PAIR-B run1 / run2 | A/B run1 / run2 |
|---|---|---|---|
| **GEMM** proj f64 XLA | 29.4 / 19.2 | 14.9 / 14.9 | **1.97× / 1.29×** |
| **GEMM** proj c128 XLA | 74.2 / 59.9 | 53.4 / 53.1 | **1.39× / 1.13×** |
| **GEMM** proj f64 FFI | 8.5 / 6.2 | 5.4 / 5.4 | **1.57× / 1.14×** |
| **GEMM** proj c128 FFI | 23.9 / 22.6 | 21.7 / 21.8 | 1.10× / 1.04× |
| **GEMM** proj f64 MKL bare | 5.1 / 5.0 | 5.9 / 5.8 | **0.86× / 0.85×** |
| **GEMM** sq2048 f64 XLA | 34.2 / 33.7 | 31.2 / 31.3 | 1.10× / 1.08× |
| **FFT** S2 flat-k XLA | 563.5 / 589.7 | 586.5 / 663.5 | 0.96× / 0.89× |
| **FFT** S2 flat-k FFI | 49.7 / 59.2 | 55.9 / 65.0 | **0.89× / 0.91×** |
| **FFT** S2 minor-most XLA | 54.1 / 50.1 | 54.4 / 54.3 | 0.99× / 0.92× |
| **FFT** S2 minor-most MKL | 9.1 / 10.5 | 7.6 / 7.5 | 1.19× / 1.40× |
| **FFT** S2 fused conv FFI | 71.1 / 63.2 | 81.3 / 81.0 | **0.88× / 0.78×** |

**THE PHYSICALLY INTERESTING SPLIT — and it is not the one I expected:**

* **Compute/cache-coherence-bound work (XLA:CPU Eigen GEMM at the production
  batched-skinny shape) WINS from socket-local pinning: 1.13–1.97×**, consistent
  in sign across both windows. Eigen's 28-thread team shares blocked panels; under
  interleaved binding every such share crosses UPI.
* **Memory-bandwidth-bound work (the flat-k FFTs, plain and fused) LOSES
  10–22% from socket-local pinning**, also consistent in sign. A bandwidth-hungry
  rank benefits from reaching *both* memory controllers, and with same-thread
  first touch it pays no remote-access penalty for doing so.
* **Vendor MKL GEMM is neutral-to-slightly-worse socket-local (0.85–0.92×)** —
  MKL's blocking is more bandwidth-hungry per core than Eigen's.

So the pinning change is **not a free campaign-wide win**: it trades FFT
bandwidth for GEMM cache locality. Which side dominates is an empirical question
about the path's mix — which is exactly what the production cell settles.

**B.9.3 PRODUCTION cell (job 7879483): the real GW driver, `run_400c`, P=4
(1 node × 4 ranks × 14 threads), XLA-FFT leg, 6 passes in mirrored order
A B A B B A so first-process effects cannot bias the arms.**

| pass | arm | `sigma.exec` (s) |
|---|---|---|
| q1 | A | 40.638 |
| q3 | A | 41.345 |
| q6 | A | 40.713 |
| q2 | B | 35.009 |
| q4 | B | 34.876 |
| q5 | B | 34.990 |

**arm A median 40.713 s [40.638–41.345] · arm B median 34.990 s [34.876–35.009]
→ socket-local pinning is 1.164× faster (16.4%), with non-overlapping ranges and
arm B four times tighter (0.4% vs 1.7% spread).**

Other rows, same passes:

| row | arm A (median, max) | arm B (median, max) | effect |
|---|---|---|---|
| `W.exec` | 0.616 s, **max 2.844** | 0.151 s, max 0.240 | **4.1× median, 11.9× worst case** |
| `chi.exec` | 0.863 | 0.886 | neutral (−2.7%) |

The `W.exec` result is the striking one: under the current pinning it is not just
slower but wildly unstable (0.087→2.844 s across otherwise identical passes);
socket-local pinning removes the instability.

**Net:** at production scale on the XLA-FFT leg the GEMM/collective gain
outweighs the FFT loss, and the pinning fix is worth **~16% of `sigma.exec`** —
free, no code change. **The FFI-FFT leg at production scale is NOT measured**
(§B.9.5); the microbench predicts a smaller net gain there because the FFT rows
shrink ~17× while the projection GEMM stays, but that is a prediction, not a
measurement.

**B.9.4 The co-tenancy question (corrected G12 answer).** With *true*
concurrency, SOLO-A vs PAIR-A at the production projection shape:

| cell | SOLO-A | PAIR-A | co-tenancy cost | PAIR-B |
|---|---|---|---|---|
| proj f64 XLA | 18.3 / 17.9 | 19.2 / 29.4 | 1.05–1.64× | 14.9 |
| proj c128 XLA | 56.2 / 55.6 | 59.9 / 74.2 | 1.07–1.33× | 53.1 |
| proj f64 FFI | 5.0 / 5.3 | 6.2 / 8.5 | 1.24–1.61× | 5.4 |

**In-job co-tenancy IS real (up to ~1.6×) — my §B.7 "refuted" was an artifact of
the serialised harness — but it is largely a PINNING artifact, not an intrinsic
cost of sharing the node: arm B recovers to at or better than solo.** That is the
corrected answer to G12, and it also supplies the mechanism the corpus was
missing.

**B.9.5 Not measured / UNDECIDED.** (a) The FFI-FFT leg of the production sigma
cell — the MPI-collectives path needs `I_MPI_PMI_LIBRARY` + `MPITRAMPOLINE_LIB`
+ `srun --mpi=pmi2`, and on a *single* node `I_MPI_FABRICS=shm:ofi` fails with
`OFI addrinfo() failed`; deciding measurement is the same 6-pass A/B on ≥2 nodes
with the `fftffi_sigma_ab.sbatch` MPI recipe, or single-node with
`I_MPI_FABRICS=shm`. (b) Pinning at P=64 across 32 nodes, where inter-node
collectives may change the balance. (c) The 4-ranks/node layout was used for the
production cell (14 threads each); the 2-ranks/node layout was only microbenched.

### B.10 Minor-most FFI sizing, like-for-like (job 7879489)

§B.3's minor-most MKL reference was **in-place and unscaled** while XLA is
out-of-place and ortho-scaled, so those ratios were an upper bound. This cell
measures MKL **out-of-place** at the real BSE shapes, with **XLA timed first** —
because measuring XLA after MKL inflated it ~10× (MKL's OpenMP team spins under
the default active wait policy and steals cores; a harness lesson worth
recording, and the reason the first attempt at this cell is discarded).

| site | tile | XLA native | MKL in-place | **MKL out-of-place** | **like-for-like XLA/MKL** |
|---|---|---|---|---|---|
| BSE `W_R` (624,624,4,4,1) | 99.7 MB | 10.54 ms | 1.345 | **1.963** | **5.37×** |
| BSE `T_k` (312,312,2,2,4,4,1) | 99.7 MB | 10.07 | 0.719 | **1.808** | **5.57×** |
| BSE `W_R` rung (1248,1248,4,4,1) | 398.7 MB | 35.31 | 11.043 | **7.605** | **4.64×** |
| twin of the Σ-τ G tile | 398.7 MB | 51.17 | 10.825 | **7.539** | **6.79×** |

Against the wire: MKL out-of-place reaches **101–110 GB/s effective = 71–77% of
the measured 142.8 GB/s node copy bandwidth**; XLA reaches 15.6–22.6 GB/s
= **11–16%**. The ortho scale is *not* charged to MKL here, and does not need to
be: the shipped CPU flat-k handler already folds the scale into the DFTI
descriptor at zero cost, so this is a demonstrated capability rather than an
assumption.

**Corrected headroom: 4.6–6.8× at 28 threads** (replacing §B.3's 4.5–22×
upper-bound range, which mixed in-place MKL against out-of-place XLA and the
1/4/14-thread points).

## PART C — VERDICTS, CLAIM DECAY, RECOMMENDATION

### C.1 Per-question verdicts

**Q1. Does XLA insert layout transposes for batched FFTs whose transform axes are
already minor-most?**
**NO — proven from `after_optimizations` HLO at the real call-site shapes** (§B.2):
0 transposes, 0 copies, 0 temp bytes, 5 instructions, at the controlled twin of
the production tile and at every BSE `local_*` shape including rung-scale. The
flat-k sites by contrast move **exactly 3.00× the tile** at every shape and
k-count. The `local_*` "no layout transpose" premise is **CONFIRMED**.

**Q2. Given no transpose there, what would an FFI cost or save at those shapes
anyway?**
**It would SAVE 4.5–22× on CPU, and ≈0 on GPU.** The prior verdict inferred
"no transpose ⇒ nothing to gain"; that inference is **REFUTED** because the
available gain is *engine quality*, not layout. At the minor-most layout, where
XLA emits nothing but the transform, bare MKL is 10.2× (1 thr), 21.7× (4), 11.5×
(14) and 4.5–6.0× (28) faster than XLA:CPU's DUCC. Against the measured wire, MKL
reaches 51–69% of the 28-thread copy bandwidth while XLA reaches 11%. On GPU the
opposite holds: XLA-at-minor-most (8.90 ms) already equals the FFI's flat-k rate
(9.18 ms), so GPU headroom is ≈0.

**Q3. Is plain FFI really slower than XLA at the flat-k sites?**
**NO — REFUTED. The FFI is 3.6–11.8× faster in every matched-discipline cell**,
and 18.7× in the cleanest 25-rep probe (§B.4). The "18% slower" number is the
product of two compounding methodological defects: a buffer-discipline asymmetry
worth ~2.3× against the FFI, and an XLA reference that is a **first-process-in-a-
fresh-job artifact** (113–136 ms first process, 569–584 ms steady state, six-run
control, `.so`-independent). The steady-state value is the one consistent with
production (452 ms/tile).

**Q4. Is the fused conv entry point worth it?**
**YES — CONFIRMED and larger than claimed.** 17.1–19.5× vs the XLA 3-FFT
composition at 28 threads (corpus claimed 7.0×), 13.8× at rung scale, 2.17× on
GPU. The XLA composition moves 5.40× the tile in transposes and copies (§B.2).

**Q5. Is XLA:CPU's GEMM disadvantage Eigen quality, thread-pool wiring, dtype
promotion, or shape?**
**Eigen kernel quality — a flat ~3× per-core deficit.** The gap is 3.19×/3.33× at
**one thread** (excludes pool wiring), 3.01× at 8192³ square (excludes shape),
and identical for matched f64 and matched c128 (excludes promotion). Dtype
promotion is real but **separate and additive** (a further 1.97×, HLO-confirmed
by a 398.7 MB temp). The corpus's "not pool-miswired, Eigen quality" verdict is
**CONFIRMED** by an independent route; its **magnitude is REVISED from
1.6–1.9× to ~3.0–3.4×**, because the historical reference was numpy/**OpenBLAS**,
not the MKL the FFI handler actually calls (at 28 threads, 2048³ f64: MKL 1548
GF/s vs numpy 705 GF/s).

**Q6. Is XLA:GPU's GEMM already optimal?**
**YES — CONFIRMED, first measurement.** 360–367 GF/s = 103–105% of the RTX 5000's
nominal FP64 peak (§B.8).

**Q7. Is in-job co-tenancy (2 ranks/node) the unexplained ~2×?**
**NO — REFUTED for GEMM** (≤6.6% XLA, none for MKL). Remaining candidate
("surrounding module") **UNDECIDED**; deciding measurement is a per-op profile of
the production module vs the isolated dot at identical shapes.

### C.2 Prior claims: CONFIRMED / REVISED / REFUTED

**CONFIRMED**
* F3/F9/F10 fused-conv mechanism and the HLO transpose counts (independently reproduced: 3 fft, 6 transposes, 3 copies, 5.40× tile moved).
* F7/F8/F11 the production Σ A/B numbers (all logs on disk, all reproduce).
* F18 the `local_*` "already minor-most ⇒ no transpose" premise (now HLO-proven).
* F21 "≈0 on GPU" for `local_*` — was ASSERTED, now measured-equivalent.
* G4 "not pool-miswired, it's Eigen" (confirmed by a 1-thread route the original probe never took).
* G7 dtype promotion (confirmed by HLO temp-byte census, and quantified at 1.97×).
* G9 the FFI-GEMM production A/B (best-evidenced GEMM claim in the corpus).
* G15 "XLA:GPU cuBLAS is optimal" — was ASSERTED, now MEASURED at 103–105% of FP64 peak.

**REVISED**
* F3 fused-conv advantage 7.0× → **17.1–19.5×** at 28 threads under matched discipline (and 2.17× on GPU).
* F15 GPU FFI advantage ~1.8× → **2.1–2.4×**.
* G5 Eigen-vs-vendor 1.6–1.9× → **~3.0–3.4× vs MKL** (the historical reference was OpenBLAS). Also, the published band's upper bound 1.9 was never supported by its own row (1.57–1.81).
* G1/G3/G18 "4.3× below a 1263 GF/s BLAS roofline" → the roofline is **631 GF/s** and the gap **2.1×** (§C.3).
* G8 "Eigen dgemm 7.3× below BLAS" → **1.57×** width- and dtype-matched. (Curiously, 7.3× *is* the correct end-to-end ratio from promoted-XLA to MKL-split — but by a different derivation than the one given.)
* G6 "batched-skinny scales as well as square" → "no pathological collapse"; batched-skinny costs ~10% extra.
* G10 "the FFI GEMM realizes the memo's ~19 s projection" → coincidence of two opposite-sign errors, not confirmation.
* F13 "~120 GB/s/node" → credible: 84% of the now-measured 142.8 GB/s copy wire.

**REFUTED**
* **F1/F2/F19 "plain FFI transform is 18% slower than XLA (151 vs 128 ms)"** — the single most load-bearing wrong number in the campaign. Matched-discipline steady state has the FFI **3.6–11.8× faster**. Root cause: asymmetric buffer discipline + a first-process-in-job XLA artifact.
* **F20 the `local_*` "DO NOT" verdict** — the premise is right, the conclusion is wrong; CPU headroom is 4.5–22×.
* F22 "the dot↔fft layout alternation is CLOSED as not-removable on this backend" — refuted ~24 h later by the FFI itself; still unbannered in `sigma_perf_results.md`.
* F24 "BSE inherits `LORRAX_FFT_FFI` automatically" — already corrected in `gemm_portability_bse_notes.md`.
* G12 (co-tenancy half) — **first "REFUTED" by me in error, then resolved.** My §B.7 cell was serialised by SLURM and measured nothing (retracted). With true concurrency (§B.9.4) in-job co-tenancy **is** real, up to ~1.6× at the production projection shape — but it is largely a **pinning artifact**: socket-local binding recovers to at-or-better-than-solo. So G12's "surrounding module and/or co-tenancy" resolves to **co-tenancy, mediated by NUMA-interleaved pinning**.
* G11 the "272.0 → 29.98 = 9.1×" campaign arc — shape-confounded; the honest nb=256 figure is ~2.9×.

**Still ASSERTED (no evidence found, not tested here)**
* F23 `fft_helpers.py:71-77` "~2× replicated temp buffer / ~8× inflated peak" for sharded `jnp.fft.fftn`. **UNDECIDED.** Deciding measurement: `memory_analysis()` on a multi-device sharded `fftn` at a production shape vs the `shard_map` helper.

### C.3 The 1263 GF/s roofline error — verified in primary source

I reproduced the defect directly. `wk_REL/probes/reshard_ubench_jax.py:211-213` times
**two** einsums (`sr` and `si` — both channels); `wk_REL/probes/reshard_ubench_mpi.py:117-125`
times **one** `np.matmul((16,1248,1248)c128, (16,1248,128)c128)`. Both were charged
the same 5.104e10 flops. One c128 batched GEMM at that shape is
8·16·1248·1248·128 = **2.5518e10** flops, so 40.438 ms ⇒ **631 GF/s**, not 1263.

| quantity | as published | corrected | this audit's independent value |
|---|---|---|---|
| MKL zgemm, production shape | 1263 GF/s | **631** | **1335 GF/s** (28 thr, solo, exclusive node) |
| XLA Eigen, same | 295 GF/s | 295 | 449 GF/s (bare dot) |
| stated deficit | **4.3×** | **2.1×** | **2.97×** (matched, solo) |

Note my solo exclusive-node MKL figure (1335 GF/s) is ~2× the microbench's
631 GF/s — the microbench ran 2 ranks/node with `MKL_NUM_THREADS=28` each on a
56-core node, i.e. **2× oversubscribed**. So the published 1263 was numerically
close to the truth *for a node running one rank*, but it was derived by an
arithmetic error and compared against a 2-rank XLA number. Two wrongs.

### C.4 Proposed claim-decay banners (exact text — I have NOT edited any file)

**B-1 → `wk_REL/docs/RESHARD_OVERHEAD_MEMO.md:3-6`** (the VERDICT block):
> ⚠ **CLAIM-DECAY (audit 2026-07-29, jobs 7879377/7879401).** The "4.3× below the
> node's BLAS roofline" and "1263 GF/s measured BLAS" figures are an
> arithmetic artifact: `reshard_ubench_jax.py:211-213` times TWO channel einsums
> while `reshard_ubench_mpi.py:117-125` times ONE `np.matmul`, and both were
> charged the same 5.104e10 flops. The correct BLAS rate at 40.438 ms is
> **631 GF/s** and the like-for-like gap is **2.1×**. Width- and dtype-matched
> re-measurement on a quiet exclusive node gives MKL **1335 GF/s** vs XLA Eigen
> **449 GF/s = 2.97×**. The 71% GEMM *share* and the "not collective-bound"
> verdict are unaffected and stand.

**B-2 → `docs/dev/staged_reshard_primitive.md:161-163`** (in-repo, and it
contradicts `:182-184` twenty lines later):
> ⚠ **CLAIM-DECAY (2026-07-29).** "Eigen zgemm 295 GF/s vs 1263 GF/s BLAS" is
> withdrawn — see `wk_REL/docs/FFI_EVIDENCE_AUDIT.md` §C.3. Measured, matched dtype and
> thread count on a quiet exclusive node: XLA:CPU Eigen runs a flat **~3.0–3.4×**
> below Intel MKL from 1 thread to 28 and from 2048³ square to the production
> batched-skinny shape. Use ~3×; delete the 1263 GF/s figure.

**B-3 → `wk_REL/OWNER_DECISIONS.md:39-41`**:
> ⚠ **CLAIM-DECAY (2026-07-29).** "Eigen dgemm measured 7.3× below BLAS" is
> withdrawn. Width- and dtype-matched, the f64 gap is **1.57×** (job 7879008) and
> **~3.0×** against MKL rather than OpenBLAS (job 7879377). The 7.3× figure does
> coincidentally equal the end-to-end promoted-XLA → MKL-split ratio, but not by
> the derivation given.

**B-4 → `wk_REL/OWNER_DECISIONS.md:51-53`** (the load-bearing FFT number):
> ⚠ **CLAIM-DECAY (2026-07-29, jobs 7879376/7879395/7879401) — REFUTED.**
> "Plain FFI transform is 18% SLOWER than XLA (151 vs 128 ms)" does not survive a
> controlled re-measurement. (a) The two arms used different buffer discipline —
> XLA reused one input, the FFI got a fresh 400 MB input plus donation. Matched,
> the FFI is **3.6–11.8× FASTER** at every discipline. (b) The XLA reference is a
> first-process-in-a-fresh-job artifact: six back-to-back runs of the unmodified
> harness on one exclusive node read 113 ms (first process) then 569–581 ms
> (every later one), independent of the `.so`. The steady-state value is the one
> consistent with production (452 ms/tile). Clean 25-rep probe: XLA 584.0 ms vs
> FFI 31.2 ms = **18.7×**. The fused-conv advantage is likewise **17–19×**, not 7×.

**B-5 → `wk_REL/docs/gemm_portability_bse_notes.md:271, :407`** (the `local_*` verdict):
> ⚠ **CLAIM-DECAY (2026-07-29) — premise CONFIRMED, conclusion REFUTED.** The
> `local_*` sites do indeed pay **no** layout transpose: after_optimizations HLO at
> `(624,624,4,4,1)`, `(312,312,2,2,4,4,1)`, `(1248,1248,4,4,1)` and the controlled
> twin of the Σ-τ tile shows **0 transposes, 0 copies, 0 temp bytes, 5
> instructions**. But "no transpose ⇒ nothing to gain" does not follow: at those
> exact shapes bare MKL is **10.2× (1 thr), 21.7× (4), 11.5× (14), 4.5–6.0× (28)**
> faster than XLA:CPU's DUCC, reaching 51–69% of the measured 142.8 GB/s node wire
> where XLA reaches 11%. The available gain is **engine quality, not layout**. The
> "18% penalty for nothing" premise is separately refuted (B-4). On **GPU** the
> original conclusion stands: XLA-at-minor-most (8.90 ms) already equals the FFI's
> flat-k rate (9.18 ms), so GPU headroom is ≈0.

**B-6 → `wk_REL/docs/sigma_perf_results.md:134-135` and `:184-187`**:
> ⚠ **CLAIM-DECAY.** "FFT/layout buckets are CLOSED as structural on XLA:CPU" /
> "the dot↔fft layout alternation is CLOSED as not-removable on this backend" was
> refuted within ~24 h by the MKL-DFTI FFI (job 7878727), which removed it on the
> **same** backend without changing the kernel math — a third exit the claim did
> not enumerate. 191.9 s → 4.99 s.

**B-7 → `SPEEDUP_SCORECARD.md`, new section (the campaign is currently unledgered)**:
> The scorecard contains **zero** occurrences of `LORRAX_FFT_FFI`, `mklfft`, `DFTI`,
> `LORRAX_BANDS_GEMM_FFI`, `mklblas` or `contract_bands`. Per
> `wk_REL/docs/ARTIFACT_MAP.md:114` ("the ledger is SPEEDUP_SCORECARD.md"), the
> FFT-FFI + GEMM-FFI campaign (sigma.exec 272.0 → 49.2 s at nb=128, jobs
> 7878727/7878942/7879010) needs a section. Also add to `:9618-9620` (AY.4):
> "⚠ CLAIM-DECAY: the G_ifft/V_ifft/GW_mult_fft rows totalling 191.9 s describe
> the XLA:CPU plan only; they collapse to one 4.992 s `GW_conv_ffi` row under
> `LORRAX_FFT_FFI(_FUSED)` (job 7878727)."

**B-8 → `src/common/fft_helpers.py` header comment** (wrong denominator):
> "measured 60-65% of sigma.exec at nb=128/P=64" → 191.9 s is **65% of the staged
> τ dispatch (295.0 s)**, which is **70.5% of sigma.exec (272.0 s)**. Name the
> denominator.

**B-9 → new, a pinning note for `docs/dev/` or the scorecard:**
> Frontera CLX compute nodes enumerate CPU ids **NUMA-interleaved**
> (node0 = even, node1 = odd; job 7879379). Production's
> `taskset -c $((S*28))-$((S*28+27))` therefore gives each rank **14 cores on each
> socket**, not one socket, and both ranks contend on both memory controllers.
> Socket-local pinning would be `taskset -c 0-54:2` / `1-55:2`. Effect on the
> bandwidth-bound FFT rows is **UNMEASURED**.

### C.5 RECOMMENDATION

| call-site family | CPU default | GPU default | conditions that flip it |
|---|---|---|---|
| **Σ τ FFTs** (`make_flat_k_gw_conv`) | **FFI fused** (`LORRAX_FFT_FFI=1` + `_FUSED=1`) | **FFI fused** | None found. 17–19× vs the XLA composition on CPU (28 thr), 13.8× at rung scale, 2.17× on GPU; production A/B 272.0 → 71.9 s. Flips only if a future XLA gains a strided FFT lowering — detect by re-running the §B.2 HLO census and checking whether the flat-k transpose count drops from 2 to 0. |
| **χ/W FFTs** (flat-k `make_flat_k_*`) | **FFI plain** | **FFI plain** | 8.5–11.2× on CPU at 14–28 threads, 2.1–2.4× on GPU. Production χ0 3.1× (job 7878727). At the **S0-class small tile (2.1 MB, LLC-resident)** the CPU margin narrows to 7.6× but the absolute cost is <4 ms, so routing is irrelevant there. `W.exec` showed no gain in production — consistent with W being solve-bound, not FFT-bound. |
| **BSE FFTs** (`local_*fftn3`, minor-most) | **XLA native today — but there is real, previously denied headroom** | **XLA native (cuFFT). Settled: headroom ≈0.** | The prior "DO NOT" rested on a refuted premise. Measured CPU headroom at the exact BSE shapes is **4.5–22×** on the transform itself (MKL vs XLA/DUCC). **Whether to implement is UNDECIDED — see §C.8 for the full re-sizing** — not because the transform gain is uncertain (like-for-like it is **4.6–6.8×**, §B.10), but because the **BSE FFT share of wall time has never been measured** (`gemm_portability_bse_notes.md:393-401` says so). **Deciding measurement:** a per-row timing of `bse_stack_matvec._w_stack` (the `local_ifftn3`/`local_fftn3` pair inside `lax.scan`, ~14× the Σ τ count) on a production BSE deck. If the FFT rows exceed ~10% of the BSE wall, the ~50–80 lines of a contiguous-batch DFTI descriptor is clearly worth it; below ~3% it is not. Also note both handlers are **c128-only**, so the fp32-GMRES path would need a refusal/fallback. |
| **projection GEMMs** (`contract_bands` right contraction) | **FFI vendor BLAS** (`LORRAX_BANDS_GEMM_FFI` auto/on) **plus the f64-split de-promotion** | **XLA native (cuBLAS) — confirmed optimal** | CPU: ~3.0–3.4× vendor-vs-Eigen, flat in threads/shape/dtype; production A/B `project_rs` 29.4 → 19.6 s. GPU: XLA runs at 103–105% of the FP64 peak, so an FFI would be pure overhead — keep auto-OFF on CUDA. **Flip condition on CPU:** a jaxlib that routes `dot_general` to a vendor BLAS; detect by re-running §B.6 and checking whether the 1-thread XLA-vs-MKL ratio falls below ~1.3×. **New, previously untested:** the f64-split de-promotion is worth **2.13× on GPU** as well (§B.8) — it is currently justified and gated as a CPU-only finding. |

**Two things this audit says NOT to do.**
1. Do not quote the "18% slower / DUCC beats DFTI" result anywhere again — it is
   an artifact of two compounding harness defects (§B.4).
2. Do not report best-of-N for XLA:CPU FFT cells. The first process in a fresh
   allocation is 4.3–5.1× faster than steady state for the flat-k transpose path;
   best-of-N in a single fresh job systematically captures the artifact. Report
   medians from a process that is not the first in its job, or state which.

**Tree provenance for this audit.** All measurement jobs ran against
`/work2/08271/jackmc/frontera/lorrax` at `0dd94a8` (dirty working tree, recorded
in each job's banner) with a **pinned copy** of the host FFI library at
`wk_REL/results/audit_so/liblorrax_ffi_host.so` (md5 `90d31bbc53951d330a3fc12185b9442a`,
taken from `build_host_GBP` before a concurrent workstream could rebuild it), and
the CUDA library at `lorrax_ffi_cufft_cuda/build_phdf5/liblorrax_ffi.so`
(md5 `139799649934efbeeecc41560e97a002`). No file in the repository was modified by
this audit; every harness added lives under `wk_REL/` (`audit_common.py`,
`audit_fft_bench.py`, `audit_gemm_bench.py`, `audit_hlo_census.py`,
`audit_stream.c`, `audit_dist_probe.py`, `smoke_mkl.py`, and the five
`audit_*.sbatch` files). Nothing was committed.

### C.7 PINNING FIX — ready-to-apply recipe (I have NOT applied it)

**Not applied on purpose:** other workstreams are mid-campaign on these files and
a pinning change would silently invalidate their in-flight comparisons. Apply
only at a campaign boundary, and re-baseline anything compared across it.

**The change.** Frontera CLX enumerates CPU ids NUMA-interleaved (node0 = even,
node1 = odd; job 7879379). Replace the contiguous mask with a stride-2 mask.
`taskset` is sufficient — `numactl --membind` measured no additional gain (§B.9)
and `numactl` is not present inside `py312.sif`.

*2 ranks/node, 28 threads each* — the dominant idiom:
```
-  exec taskset -c $((S*28))-$((S*28+27)) ...      # 14 cores on EACH socket
+  exec taskset -c $S-$((S+54)):2 ...              # rank0 -> 0,2,..,54 (node0)
                                                   # rank1 -> 1,3,..,55 (node1)
```
*4 ranks/node, 14 threads each* (`gw400_p4.sbatch`, `wk_N/genB.sbatch`):
```
-  exec taskset -c $((S*14))-$((S*14+13)) ...
+  if [ $S -lt 2 ]; then B=$((S*28)); else B=$(( (S-2)*28 + 1 )); fi
+  exec taskset -c $B-$((B+26)):2 ...              # 2 ranks per socket
```

**Files carrying the 2-rank idiom (31 under `wk_*/`, 19 under `mos2_4x4_test/`).**
`wk_AA/gate.sbatch`; `wk_AD/gate_{a,b,d,e,f,g,h}.sbatch`;
`wk_AK/{gate_ak,gate_cohsex,gwpart_ab,ib_ab,restart_ab}.sbatch`;
`wk_AL/{gate40_ib0,smoke_pin}.sbatch`; `wk_P/gate.sbatch`;
`wk_REL/{cbands_ab,chmerge_ab,fftffi_sigma_ab,gbp_ab,lgemm_ab,lgemm_ab2,lmdef_ab,reshard_ubench,verify_audit,zproj_p144,zproj_scale}.sbatch`;
`wk_Z/{compiles,gates}.sbatch`; and in `mos2_4x4_test/`:
`aq_rehearsal, diag_b512_weap, gate_sigma_reference, gw800_merged, gw800_p16,
l1_b256, l2_b256_c3491, l2_b256_c3500, l3_b512_c5000, l4_b512_c7000,
l5_b1024_c10000, l6_r45_b2048, l7_b1024_bigmu, omega_512cell, omega_ab,
sigma_haccum2, sigma_hostaccum_gate, sigma_iter, sigma_perf_ab`.
(My own `wk_REL/audit_pin*.sbatch` also match the grep; they are audit harnesses,
not production.)

**Verification one-liner** to paste into any harness after the change:
```
[ $S -eq 0 ] && taskset -cp $$ && numactl -H | grep "node 0 cpus" | head -1
```
Correct output has every core in the rank's mask on ONE NUMA node.

**Expected effect, by workload (do not extrapolate beyond this):**

| workload | effect of socket-local pinning | evidence |
|---|---|---|
| production Σ, XLA-FFT leg (P=4) | **1.164× on `sigma.exec`**, `W.exec` 4.1× median / 11.9× worst | job 7879483, 6 passes |
| XLA:CPU projection GEMM | **1.13–1.97×** | §B.9.2, two windows |
| FFI/vendor GEMM | 1.04–1.14× | §B.9.2 |
| bare MKL GEMM | **0.85–0.92× (slower)** | §B.9.2 |
| flat-k FFTs, XLA and FFI | **0.78–0.96× (slower)** | §B.9.2 |
| minor-most FFT (MKL) | 1.19–1.40× | §B.9.2 |
| pure STREAM aggregate | 0.97× | §B.9.1 |

**Caveat that matters for sequencing:** the measured production win is on the
**XLA-FFT** leg, where the bandwidth-bound FFT rows are large. Once
`LORRAX_FFT_FFI` is on, those rows shrink ~17× and the FFT *penalty* of
socket-local pinning shrinks with them while the GEMM *gain* stays — so the net
should improve, but that is a prediction. **Deciding measurement: the §B.9.3
6-pass A/B with `LORRAX_FFT_FFI=1 LORRAX_FFT_FFI_FUSED=1`, on ≥2 nodes using the
`fftffi_sigma_ab.sbatch` MPI recipe (single-node `shm:ofi` fails).**

### C.8 OPTION A RE-SIZED — the `local_*`/BSE FFI, with the 151-vs-128 number gone

The prior rejection of a minor-most FFI backend rested on two premises. Premise 1
(no layout transpose at those sites) is **CONFIRMED** (§B.2). Premise 2 — that a
plain-transform backend would therefore "hand them an 18% penalty for nothing" —
rested entirely on the 151-vs-128 ms cell, which is **REFUTED** (§B.4). With that
gone, the sizing has to be redone.

**Expected gain.** Like-for-like (both out-of-place, XLA timed first), at the
real BSE call-site shapes, 28 threads: **4.6–6.8× on the transform itself**
(§B.10). MKL reaches 71–77% of the measured node wire; XLA/DUCC reaches 11–16%.
This is engine quality, not layout — which is why it survives the fact that there
is no transpose to remove. Other minor-most consumers inherit it: `bse_io:714/716`,
`bse_feast:87`, `bse_kpm:159`, `bse_pseudopoles:181/252`, `bse_stack_matvec:118/120`
(inside `lax.scan`, ~14× the Σ τ count), `vq_interp:1490`.

**Implementation cost.** ~50–80 lines: a third target
(`lorrax_mklfft_batch_minor`) in the existing `fft_flat_k_ffi.cc`, differing from
the shipped flat-k handler only in the DFTI stride descriptor
(`istride=1, idist=nx·ny·nz` instead of `istride=T, idist=1`) — the *same*
one-line contrast that `cufftPlanMany64` already expresses on the CUDA side.
Plus a `local_*fftn3` routing branch in `fft_helpers.py`. Risks: the handlers are
**c128-only**, so the BSE fp32-GMRES path needs a refusal/fallback (the
`contract_bands` dtype-gate pattern applies); and `bse_stack_matvec`'s call sites
are inside `lax.scan`, where an `ffi_call` per iteration must be checked for
dispatch overhead at the real trial count.

**Does the fused-conv route still dominate?** **Yes, and both now clear the bar —
but they are not competing.** They address different call sites:

| route | sites | measured advantage | verdict |
|---|---|---|---|
| (a) fused `gw_conv` | Σ τ kernel only | **17.1–19.5×** vs the XLA 3-FFT composition (28 thr); 13.8× at rung scale | **ships; unchanged** |
| plain flat-k FFI | χ/W, ζ, Σ decomposed | 8.5–11.2× (28 thr) | **ships; unchanged** |
| **(b) minor-most FFI** | BSE `local_*`, `vq_interp` | **4.6–6.8×** | **no longer rejected — but priority UNDECIDED** |

Route (a) remains the largest single lever and is unaffected by this reversal.
Route (b) is additive, not alternative: it reaches call sites (a) cannot.

**Does the transform-headroom result alone change the recommendation?**
**It changes the verdict from "DO NOT" to "justified on the merits, unscheduled".**
It does **not** by itself justify implementation, because a 4.6–6.8× on a row of
unknown size has unknown value. The BSE FFT wall-time share remains
**UNMEASURED** (`gemm_portability_bse_notes.md:393-401` says so, correctly), and
I did not measure it — no production BSE deck was run by this audit.

**Exact deciding measurement.** Run a production BSE deck (the FEAST or KPM
driver on the MoS2 4×4 400c deck is the cheapest that exercises
`bse_stack_matvec._w_stack`) with per-row timing around the
`local_ifftn3`/`local_fftn3` pair at `bse_stack_matvec.py:118` and `:120`,
reporting those two rows as a fraction of the BSE wall. Decision rule:
**> ~10% of the BSE wall → implement route (b)** (4.6–6.8× on that share is then
worth ~50–80 lines); **< ~3% → do not**; in between, weigh against the fp32-GMRES
refusal work. The same run should record the `T_k` shape actually used, since
the 4.6–6.8× band is shape-dependent (it is 5.4–5.6× at 99.7 MB and 4.6–6.8× at
398.7 MB).

### C.9 Residual limitations of this audit

* Single node per platform, single MKL version (2020.1), single jaxlib (0.9.1).
* All CPU cells are single-process; production runs 2 ranks/node. §B.7 bounds the
  co-tenancy effect for **GEMM** only — the bandwidth-bound FFT rows were not
  co-tenancy-tested, and the NUMA-interleaved pinning (§B.0) means both ranks
  share both memory controllers. **UNDECIDED**; deciding measurement is the §B.3
  FFT sweep re-run as two concurrent 28-core processes.
* The bare-MKL FFT reference is in-place and unscaled; XLA is out-of-place and
  ortho-scaled. Out-of-place costs at most 1.5× (3 streams vs 2) and the scale is
  folded into the DFTI descriptor for free in the real handler, so the minor-most
  gap survives generous correction (≥3× at 28 threads) — but the raw ratios in
  §B.3 are an upper bound, not a like-for-like handler measurement.
* The mixed-radix wfn-box HLO entry resolved to 1 instruction and 0 `fft` ops; its
  transpose/copy/temp counts are authoritative but its lowering was not identified.
* No production-scale A/B was run by this audit — it measures kernels and HLO, and
  leans on the corpus's already-verified production A/Bs (7878727, 7879010) for
  end-to-end effect.
