# The GN-PPM sigma performance record, adjudicated claim by claim — and the deck that decides it

2026-08-11. Lane `lane/gnppm-perf-decomp-2026-08-11`, agent `gnppmdecomp_0811`,
base `dc766220`.

Earlier the same day, a lane measured the GN-PPM tau kernel's FFT at **32 ms of a
44.9 s driver wall — 0.07 %** — and concluded that there was no FFT lever and
never had been (`2026-08-11-gnppm-fft-is-already-on-the-ffi.md`). That
measurement is correct. Its generalisation is not. On the only deck it ran, the
FFT is a sixth of the tau dispatch; on a Si 6x6x6 deck at the same four
processes and the same allocator, **the same FFT is 85 % of the tau dispatch and
about 28 % of the driver wall.** The claim that the share "is not a small-deck
artifact" is exactly backwards: it is a small-deck artifact, and the deck it was
measured on is a fixture whose driver wall is 56 % bring-up.

This page adjudicates that claim and every other live performance claim about
the sigma / tau / GN-PPM path that a lane could reach today, and it does it
against measurements taken at three system sizes rather than one. The rule
applied throughout is the one the record already contains but does not obey:
**a share is meaningless until its denominator and its deck are both named.**

---

## 1. What was measured, and under what

**Seven legs** — two `lx batch` invocations plus one A/B leg, each at
`-N 1 -G 4 -n 4 -P 1` — own allocation **56662933** (the shared pool 56647203 was
fragmented at 2/4 GPUs free per node, and a fragmented pool means waiting, not
reshaping — `SMALL_ISSUES` 46). Every leg asserted **HEAD `dc766220`,
dirty-count 0, `device_count=4`, `device mesh is 2x2`** from its own startup
block and refused to bank otherwise. Allocator **BFC@0.85** in all seven
(`XLA_PYTHON_CLIENT_ALLOCATOR=default`, `MEM_FRACTION=0.85`,
`PREALLOCATE=false`). **7/7 exit 0.**

Three system sizes, each run twice — once in production form and once with
`LORRAX_SIGMA_TAU_TIMING=1`:

| rung | deck | bands | k (full BZ) | centroids | FFT grid |
|---|---|---|---|---|---|
| **A** | `tests/regression/gnppm_debug` — MoS2 3x3, nspinor=2 | 46 | 9 | 399 | 24x24x80 |
| **B** | Si 4x4x4 big-continuum (`symgate444_0810/armA` deck) | 100 | 64 | 1128 | 24x24x24 |
| **C** | Si 6x6x6 (`si666_ref_0810` deck), noncollinear | 60 | 216 | 1104 | 36x36x36 |

Rung C's **Sigma values are embargoed** by the RED IBZ-cascade gate
(`2026-08-10-ibz-cascade-vs-full-bz-sigma-6x6x6.md`). Nothing on this page cites
a Sigma value from it. Wall-clock decomposition is unaffected by that embargo
and is all this page takes.

**Why both forms of every rung.** `LORRAX_SIGMA_TAU_TIMING=1` makes the five tau
stages separate blocking jits, so the staged wall is not the production wall —
`ppm_tau_kernel.py:341` says so in the tree, and this lane confirms it (rung C:
119.1 s staged against 95.9 s production). The staged rows are used **only for
ratio attribution inside the tau dispatch**; every share-of-wall number below
comes from the production leg.

### The production decomposition, all three rungs

Phases are the driver's own timing tree, grouped: **bring-up** =
`runtime_stack.*` + `imports` + `startup`; **load + ISDF** = `gw_jax.isdf`;
**W/eps** = `minimax_quadrature` + `screening` + `static_head`; **W writes** =
`persist_w0`; **sigma** = `gw_jax.sigma`; **QP + output** = `kin_ion_load` +
`solve_qp` + `qp_eigh` + `output`.

| phase | A: MoS2 3x3 | B: Si 4x4x4 | C: Si 6x6x6 |
|---|---|---|---|
| driver wall | **21.715 s** | **37.870 s** | **95.919 s** |
| bring-up | 12.200 s / **56.2 %** | 12.611 s / 33.3 % | 14.647 s / 15.3 % |
| load + ISDF | 5.148 s / 23.7 % | 9.198 s / 24.3 % | 26.555 s / 27.7 % |
| W / eps assembly | 1.144 s / 5.3 % | 2.082 s / 5.5 % | 5.450 s / 5.7 % |
| W writes (`persist_w0`) | 1.345 s / 6.2 % | 2.552 s / 6.7 % | 7.926 s / 8.3 % |
| **sigma** | **1.125 s / 5.2 %** | **10.828 s / 28.6 %** | **40.128 s / 41.8 %** |
| QP + output | 0.638 s / 2.9 % | 0.375 s / 1.0 % | 0.874 s / 0.9 % |
| untimed | 0.118 s | 0.224 s | 0.340 s |

**Top three sinks.** Rung A: bring-up **56.2 %**, ISDF 23.7 %, W writes 6.2 % —
sigma is sixth at 5.2 %. Rung B: bring-up 33.3 %, sigma 28.6 %, ISDF 24.3 %.
Rung C: **sigma 41.8 %**, ISDF 27.7 %, bring-up 15.3 %.

The six phases plus `(untimed)` close to the driver wall within 3 ms at every
rung, so nothing is hiding between them.

The owner's standing claim — that on bigger calculations GN-PPM sigma is the
longest-running part outside the ISDF fit — is confirmed and can be
strengthened: **at Si 6x6x6 sigma is the longest-running part, full stop**,
larger than the ISDF stage (41.8 % against 27.7 %), and the crossover happens
between rungs A and B.

### Inside sigma, and the row-semantics trap

The production leg attributes sigma's time to `sigma.tau.host_accum` and its
`d2h_wait` child, because on the fused path `sigma.tau.dispatch` is an async
submit that costs no host time and the wait for the device lands downstream.
`ppm_accumulators.py:425-435` already warns that `d2h_wait` "absorbs the kernel
time... that is the DEVICE wall surfacing here, not host work". At rung C:

| row | rung C production |
|---|---|
| `sigma.tau.dispatch` (168 nodes) | 0.189 s — async submit |
| `sigma.tau.host_accum` | 32.999 s |
| ↳ `sigma.tau.d2h_wait` (+ finalize's) | **31.466 s = 32.8 % of the driver wall** |
| ↳ `sigma.tau.omega_project` | 2.242 s = 2.3 % |
| `sigma.host_gather` | 0.485 s = 0.5 % |

So **78 % of the sigma stage at rung C is one number: the device wall of the tau
kernel**, and the host-side omega projection that the record has argued about
for two weeks is 2.3 % of the run.

The staged leg opens that device wall up. Ratio attribution inside
`sigma.tau.dispatch`, all three rungs:

| staged tau row | A | B | C |
|---|---|---|---|
| dispatch total | 0.193 s | 6.628 s | 36.751 s |
| `sigma.tau.GW_conv_ffi` (the fused IFFT·multiply·FFT) | 0.031 s — **16.1 %** | 4.007 s — **60.5 %** | **31.189 s — 84.9 %** |
| `sigma.tau.project_rs` | 0.070 s — 36.3 % | 1.614 s — 24.4 % | 3.338 s — 9.1 % |
| `sigma.tau.G_build` | 0.031 s | 0.782 s | 1.728 s |
| `sigma.tau.w_phase` | 0.023 s | 0.143 s | 0.402 s |

Applying rung C's 84.9 % to the 31.466 s device wall the production leg
measures: **the flat-k FFT convolution is ~26.7 s of a 95.9 s driver wall, about
28 %.** Not 0.07 %.

The two attributions agree independently. The staged FFT row (31.189 s) and the
production device wall (31.466 s) are the same object measured two ways, and
`project_rs` — which the Frontera record calls "the sigma wall" — falls from
36 % of the dispatch at rung A to **9 %** at rung C while the FFT rises to 85 %.

A third, independent decomposition confirms it. Leg `L3d` re-ran rung C with
`LORRAX_FFT_FFI_FUSED=0`, which splits the fused entry into its three
transforms:

| rung C, decomposed (`L3d`) | seconds | share of the 39.091 s dispatch |
|---|---|---|
| `sigma.tau.G_ifft` | 14.315 | 36.6 % |
| `sigma.tau.V_ifft` | 3.614 | 9.2 % |
| `sigma.tau.GW_mult_fft` | 15.582 | 39.9 % |
| **the three together** | **33.511** | **85.7 %** |
| `sigma.tau.project_rs` | 3.345 | 8.6 % |

**84.9 % fused, 85.7 % decomposed** — two different code paths, the same
answer. The fused entry is also the faster one here (31.189 s against
33.511 s, a 7 % edge), so the shipping default is not the thing to change.

---

## 2. The claims, adjudicated

Verdicts: **TRUE-AS-MEASURED** (keep, measurement cited beside it),
**STALE-BUT-CORRECTABLE** (correct, denominator named), **DEAD** (describes
deleted code or unmeasurable context — deleted, per the owner's standing
preference for deletion over patched half-truths).

### 2.1 The FFT claims — the ones that produced tonight's error

| # | claim | site | verdict |
|---|---|---|---|
| 1 | "the FFTs of the whole self-energy integration are 32 ms of a 44.9 s run, which is 0.07 %" | `2026-08-11-gnppm-fft-is-already-on-the-ffi.md` | **TRUE-AS-MEASURED, on `gnppm_debug` only.** Reproduced here: 0.031 s of 21.7 s. The deck is a fixture whose wall is 56.2 % bring-up and whose sigma stage is 5.2 %; it cannot carry a statement about the sigma path's cost. |
| 2 | "The share is not a small-deck artifact... Two decks three orders of magnitude apart in size agree that the FFT is roughly a sixth of the tau dispatch" | same file | **STALE-BUT-CORRECTABLE — this is the false claim.** The two decks agreed by coincidence: one is a fixture, the other is a CPU run at nb=128. At Si 4x4x4 the FFT is **60.5 %** of the tau dispatch and at Si 6x6x6 **84.9 %**, both at P=4 on A100s under BFC@0.85, HEAD `dc766220`. The share **rises steeply with k-point count**, which is the axis neither prior measurement varied. |
| 3 | "the tau dispatch is a tenth of `sigma.exec`" | same file | **STALE-BUT-CORRECTABLE.** True on `gnppm_debug` (0.193/0.666 = 29 %, and 0.077/0.552 = 14 % in production form). At rung C the tau kernel's device wall is **83 % of `sigma.exec`** (31.466/38.043). |
| 4 | "Deleting the FFT outright would return seven parts in ten thousand" | same file | **STALE-BUT-CORRECTABLE.** Deck-conditional. At rung C it would return ~28 % of the driver wall. |
| 5 | "65 % of the staged tau dispatch (191.9 s of 295.0 s) at nb=128/P=64" + the MIND THE DENOMINATOR note | `src/ffi/fft.py:30-48`, `src/common/fft_helpers.py:363-372`, `docs/dev/flat_k_fft_service.md:87-106` | **TRUE-AS-MEASURED.** The denominator correction those three sites received this morning is right and stays. What they need is the second half: the *post*-FFI numbers they quote (7.6-16.5 %) are themselves deck-conditional, and on a 216-k deck the post-FFI share is 85 %. Corrected in place. |
| 6 | "the service inside the production Σ driver on GPU — measured 2026-08-11: `GW_conv_ffi` 0.032 s over 155 τ dispatches (16.5 % of `sigma.tau.dispatch`, 0.07 % of the driver wall)" | `docs/dev/flat_k_fft_service.md:279` | **STALE-BUT-CORRECTABLE.** Row now carries all three rungs. |
| 7 | "#1: replace the 16-pt flat-k FFT round-trip in `ppm_tau_kernel` with a DFT-matrix GEMM, est. 25-45 % of `sigma.exec` at μ=4962/P=64" | `wk_REL/ARTIFACT_MAP.md:93`, `SIGMA_PPM_CAMPAIGN.md:102-105` | **TRUE-AS-MEASURED as an estimate, and better than the record credits.** Both sites label it "estimated / untested on GPU", which is honest. This lane measures the object it was estimating at **83 % of `sigma.exec`** at rung C — the estimate was low, not high. The *proposed remedy* (DFT-as-matmul) remains owner-vetoed and this lane does not reopen it. |

### 2.2 The allocator claim — true on one deck, false on another

| # | claim | site | verdict |
|---|---|---|---|
| 8 | "`sigma.tau.dispatch`, 6.87-6.92 s over 170 tau nodes, collapses to 0.171 s [under BFC] — allocator churn masquerading as device work"; "that setting costs 40 % of the Sigma stage" | `SIGMA_PPM_CAMPAIGN.md:27-30, 171, 410-412` | **TRUE-AS-MEASURED on its deck (MoS2 6x6, μ=1496, nb=200), and it does NOT generalise.** The row behaviour reproduces exactly at rung C: under `platform` (`si666_ref_0810/gw_666.log`) `sigma.tau.dispatch` reads **41.277 s** and `d2h_wait` 0.004 s; under BFC (this lane) dispatch reads **0.189 s** and `d2h_wait` **29.902 s**. But **the wall does not move**: 96.317 s under `platform` against 95.919 s under BFC, a 0.4 % difference on a run whose own wall noise is larger. At rung C the allocator relocates 41 s of attribution and buys nothing, because the device is genuinely busy. "Masquerading as device work" is true where the device is idle and **false where it is not**. |
| 9 | "every number in this document moves once [the allocator] changes" | `SIGMA_PPM_CAMPAIGN.md:416-420, 502-504` | **TRUE-AS-MEASURED**, and this lane is a second instance of it. Every timing table in `symgate444_0810`, `si666_ref_0810` and the `si_gnppm_0809` family was taken under `XLA allocator: platform`; their `sigma.tau.dispatch` rows are not comparable to any BFC number, including the ones on this page. Flagged at those sites. |

### 2.3 The host-accumulate claims

| # | claim | site | verdict |
|---|---|---|---|
| 10 | "`sigma.tau.omega_project` 1.62-1.65 s, 2.9 % — host accumulate, not overlapped; 'host thread exonerated' true at Frontera nb=128/256, false here" | `SIGMA_PPM_CAMPAIGN.md:173` | **STALE-BUT-CORRECTABLE — the label is right, the denominator is wrong, and the conclusion does not survive scale.** 2.9 % is of TOTAL wall, a denominator the same document (L152-156) reports swinging 52.9-58.8 s across identical repeats. Measured here across the ladder, `omega_project` is 0.6 % (A), 10.1 % (B) and **2.3 % (C)** of the driver wall — it does not grow with system size, and at rung C it is a twelfth of the tau kernel's device wall. The row is real and it is not the lever. |
| 11 | "D2 ceiling: ~1.6 s, 30 % of the post-fix tau wall" | `SIGMA_PPM_CAMPAIGN.md:422` | **DEAD as written.** "The post-fix tau wall" is not a timing row anywhere in the tree; back-solving gives ~5.3 s, which matches no named row (BFC `sigma.exec` 5.58 s, `host_accum` 4.316 s, `d2h_wait`+`omega_project` 4.45 s). A ceiling stated against an object that does not exist cannot be checked. |
| 12 | "`sigma.tau.host_accum` is 84-89 % of `sigma.exec`, superlinear in μ (73/136/329/610 s)" | `wk_REL/ladder_rung1_notes.md`, `notes/SPEEDUP_SCORECARD.md:9705-9707, 9809-9811`, `wk_REL/SIZE_CAMPAIGN_BRIEF.md:98-100` | **STALE-BUT-CORRECTABLE, and already retracted elsewhere in the same corpus without a banner at these sites.** `sigma_perf_results.md:100-127` and `scale10k_notes.md:120` re-attribute the row as device wait; `ppm_accumulators.py:425-435` documents it in the tree. This lane confirms the corrected reading directly: at rung C `host_accum` is 32.999 s of which **29.902 s is `d2h_wait`** and only 2.147 s is the host projector. The series is real and it measures the DEVICE, not the host thread. Banners added. |

### 2.4 Claims that describe code that no longer exists

| # | claim | site | verdict |
|---|---|---|---|
| 13 | "All τ nodes of a single window run inside one `jax.lax.scan` (`_get_sigma_tau_scan_kernel`) so XLA can pipeline NCCL across iterations — this is what makes the GN-PPM path competitive with static COHSEX wall-time for small ω grids" | `docs/theory/physics.md:696` | **DEAD, twice over.** `_get_sigma_tau_scan_kernel` does not exist; `ppm_sigma.py:320` records that "sigma stays a Python τ loop because its per-τ body emits NCCL and a monolithic scan regressed MoS2 3x3 by ~80 %". So the page attributes the path's competitiveness to a mechanism that was tried, measured as an 80 % regression, and reverted. The comparative ("competitive with static COHSEX wall-time") has no measurement behind it anywhere in the corpus. Deleted. |
| 14 | "~410 s eager tau-loop compilation"; "Actual GPU sigma work — tau-loop einsums + FFTs — ~5 s — Healthy"; "Multi-GPU (4x A100) gave zero speedup because compilation dominated" | `wk_REL/reference/perlmutter/XPROF_TRACE_GUIDE.md:306-308` | **DEAD.** The table is presented as "Known LORRAX cost centers" and **carries no date**. Its numbers are from `ppm_sigma_profiling_2026-04-05.md`, four months and an architecture ago; `LORRAX_FRONTERA_ADVICE.md:185` ("compilation is not a bottleneck in this code") contradicts it, and this lane measures `sigma.compile` at 0.14-0.80 s across all three rungs. Deleted from the guide, with a pointer to the dated source. |
| 15 | "MKL FFT (DFTI API) on cpu meshes" as the description of the service the τ kernel uses | `docs/architecture/services.md:571`, `src/gw/ppm_tau_kernel.py:72, 74, 254, 345`, `src/common/fft_helpers.py:361` | **STALE-BUT-CORRECTABLE.** `mklfft/fft_flat_k_ffi.cc` has contained zero `DftiCreateDescriptor` calls since 2026-08-05 and four `fftw_plan_many_dft`; `ffi/fft.py:138-145` and `env_vars.md:54` were corrected, these six sites were missed. Corrected to "FFTW3-ABI host handler". |

### 2.5 Refuted claims still standing unbannered

| # | claim | site | verdict |
|---|---|---|---|
| 16 | "FFT/layout buckets scale ∝ μ² and are CLOSED as structural on XLA:CPU"; "the dot↔fft layout alternation is CLOSED as not-removable on this backend" | `wk_REL/sigma_perf_results.md:134-135, 184-187` | **STALE-BUT-CORRECTABLE.** Refuted ~24 h later by the MKL-DFTI FFI (job 7878727); `FFI_EVIDENCE_AUDIT.md:729` (F22) notes the refutation is "still unbannered in `sigma_perf_results.md`" — still true at `dc766220`. Banner added. |
| 17 | "Eigen zgemm 295 GF/s vs 1263 GF/s BLAS"; "4.3-7.3x below the node's BLAS roofline" | `docs/dev/staged_reshard_primitive.md:161-165`, `docs/dev/vendor_gemm_service.md:88-91`, `src/common/contract_bands.py:86-95` | **STALE-BUT-CORRECTABLE.** `FFI_EVIDENCE_AUDIT.md` G1/G18 rules the ratio arithmetically wrong (corrected ≈631 GF/s, gap ≈2.1x) and names `staged_reshard_primitive.md` the "highest-priority repo-doc fix". Uncorrected at three sites, one of them a live source docstring. Corrected. |
| 18 | "the 60-65 % of the Σ τ kernel" | `wk_REL/gemm_portability_bse_notes.md:460, 669` | **STALE-BUT-CORRECTABLE.** This morning's commit says the bad range "shipped in three places" and that all three were fixed. These are the fourth and fifth, and they are the two nearest to the shape of tonight's error (a bare "60-65 %" with a loose denominator). Corrected. |

### 2.6 Live claims that were bare and are now measured or gone

| # | claim | site | verdict |
|---|---|---|---|
| 19 | "stride descriptors... deleting XLA's layout transposes (sigma.exec 272 → 71.9 s at nb=128/P=64)" | `docs/architecture/services.md:571-574` | **STALE-BUT-CORRECTABLE.** The number is real but is CPU-only and pre-dates the GPU evidence; the sibling page carries the "on CPU" qualifier and this one does not. `cufft_mirror_notes.md:194-197` says explicitly that "nothing like the CPU 3.78x should ever be claimed" on GPU. Qualified in place. |
| 20 | "The ISDF quadrature runs regardless; it is cheap next to Σ" | `src/gw/sigma_dispatch.py:459-461` | **TRUE-AS-MEASURED, now with a number.** At the two production rungs `gw_jax.isdf` is 24.4 % and 27.7 % of wall against sigma's 28.6 % and 41.8 % — so "cheap next to Σ" holds at scale and inverts on the fixture deck (23.7 % against 5.2 %). Scoped. |
| 21 | "the ζ-fit and V_q build — the dominant cost — are not redone" | `tests/conftest.py:211-214` | **TRUE-AS-MEASURED for the fixture it describes.** On `gnppm_debug`, ISDF is 23.7 % of wall and sigma 5.2 %, so ζ-fit *is* the dominant physics cost there. Left alone; it is a statement about the fixture, not about the pipeline. |
| 22 | "6x6 → 12x12 costs ~4.4x runtime" | `manual/03_tutorial/3.3_2d_soc_teaser.md:6` | **STALE-BUT-CORRECTABLE — asserts a measured ratio and names no run.** The harvest for this lane found **no hierarchical timing table anywhere** for the 12x12/400-band MoS2 reference (`lorrax_sandbox_pre_august/runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21`); it predates the timer. The ratio is unverifiable from anything on disk. Marked as unsourced rather than deleted, because the tutorial's qualitative point (O(N_k log N_k), not N_k²) is separately supported. |

---

## 3. The verdict on the owner's question

> *"is there any real path to accelerating sigma gn-ppm"*

**The lever is the flat-k FFT convolution inside the tau kernel —
`sigma.tau.GW_conv_ffi` — at ~28 % of the driver wall at Si 6x6x6, and rising
with k-point count.** Acting on it is a separate owner decision and this lane
did not touch a line of it.

The mechanism, stated so it can be attacked or dismissed on its merits: the
fused entry does one IFFT, one real-space multiply and one FFT per tau node,
over a `(nk, μ_local, N1, N2, N3)` tile. Its cost goes as
`n_tau · nk · μ_local · N_grid log N_grid`, and **`nk` is the axis that moved**
between the decks that disagree: 9 k-points at rung A, 64 at rung B, 216 at rung
C, against μ that barely changed between B and C (1128 → 1104). That is why the
share tracks k-point count and not centroid count, and it is why every prior
measurement — all of them at 9-64 k on GPU, or at nb=128 on CPU — missed it.
The owner's real workloads are on the far end of that axis.

What this does **not** say. It does not say the FFI is slow: the FFI is what
made this affordable at all, and `LORRAX_FFT_FFI_FUSED=0` measures the
decomposed chain against it on the same deck. It does not name a fix — a
DFT-as-matmul replacement is owner-vetoed and is not being reopened here. And it
does not say the other rows are free: the ISDF stage at 27.7 % and `persist_w0`
at 8.3 % are both larger than anything the FFT lane was looking at.

The three sinks worth an owner's eye at production scale, in order:
**sigma's tau device wall (32.8 % of wall, ~85 % of it the FFT), the ISDF stage
(27.7 %), and the W0 restart write (8.3 %, and `write_restart_tensors = false`
already exists for one-shot runs).**

---

## 4. Evidence

`/pscratch/sd/j/jackm/gnppmdecomp_0811/` — `EVIDENCE.md`, seven leg logs under
`_logs/`, run directories `L1`/`L2`/`L3` (staged), `U1`/`U2`/`U3` (production),
`L3d` (decomposed-FFT A/B at rung C). Two `lx batch` invocations plus one A/B
leg, all `-N 1 -G 4 -n 4 -P 1`, allocation **56662933** (own; created because
the shared pool was fragmented). Every leg asserts HEAD `dc766220`,
dirty-count 0, `device_count=4` and `device mesh is 2x2` on its own startup
block, and refuses to bank otherwise. No source file changed, no reference
re-frozen, no `.so` built, no tolerance moved, nothing optimised.

Inputs for rungs B and C are **symlinked read-only** from
`symgate444_0810/armA` and `si666_ref_0810`; neither workspace was written to.
