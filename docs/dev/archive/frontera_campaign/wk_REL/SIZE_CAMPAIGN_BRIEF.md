# Size-escalation campaign brief (owner-directed 2026-07-28)

**Goal**: on FIXED architecture (32 nodes / P=64, 8x8 mesh), push the 4x4 MoS2
**30 Ry** deck as large as possible in (nband, N_mu), N_mu ~ 6-14x nband.
Find where runs die; diagnose (OOM-killer vs XLA alloc vs FAIL-FAST vs other);
fix when low-hanging (workspace/movement, not fundamental); record the frontier
and the fundamental limits. SEQUENCED AFTER the sigma-perf work (owner order).

## Ladder (draft rungs, same 30 Ry cutoff, sym on)
(nband=128, 4962) DONE green 514s | (256, ~2.5k) | (256, ~3.5k) | (512, ~5k) |
(512, ~7k) | (1024, ~10k) | ... until death; bisect the dying rung.

## Per-rung prerequisites
1. WFN regen at higher nband — AJ recipe `mos2_4x4_test/deck_complete.sbatch`
   (30 Ry unchanged); NSCF cost grows with bands.
2. **dipole.h5 MUST be regenerated on any band-window change** (open-ledger:
   the one artifact without a provenance guard).
3. kin_ion.h5 — check band dependence; regen if needed.
4. Centroid sets per N_mu: `kmeans_cli N --orbit --weight-bands 0:<nband>`
   (aq_c5000_driver.sh pattern; orbit-closed count %8 must exercise pads).
5. gw.in: nval/ncond/nband, memory_per_device_gb (96 GB/rank avail at 2/node).

## Harness
`mos2_4x4_test/aq_rehearsal.sbatch` family: distributed tiers forced, mpi
collectives (gloo is DEAD at P=64 — reproducible ReduceScatter timeout),
cache-cold when a collective table is wanted, LORRAX_FFI_HOST_SO required.
Add per-stage RSS high-water telemetry to compare vs memory model.

## Memory model (task #6)
gflat_memory_model.py + planner: record predicted chunking vs measured RSS per
rung; fix obvious errors (missing terms / wrong exponents); phenomenological
tweaks to max out memory are OWNER-APPROVED (updates the earlier "don't push
planner work" stance — tweaks in service of max size yes, recalibration
campaigns no).

## Known size-relevant hazards (from AQ P=64)
- zeta-apply full-mu gather [1, mu_pad, N_r/8] (460 MB at mu=4962) — grows
  linearly in mu; candidate first OOM/perf wall in the zeta path.
- sigma stage = 69% of runtime pre-optimization; sigma-perf scout results
  (wf_5ec71915-466) land first and may change the sigma memory profile.
- Restart tensors ~13.5 GB at mu=4962 — writer is fast (AI) but scratch
  footprint grows ~mu^2.

## FRONTIER LEDGER (appended 2026-07-28, ladder agent; full detail:
## wk_REL/docs/ladder_rung1_notes.md, all numbers disk-verified)

Rungs executed (30 Ry, P=64 8x8, budget 90 GB/dev, cache-cold, coll=mpi,
build_host_AUDIT FFI; deck artifact families _b256/_b512 built without
touching rung-0 artifacts):


> ⚠⚠ CLAIM-DECAY (2026-07-29, R30.3) — **THE "ratio" COLUMN IN EVERY TABLE
> BELOW/ABOVE IS WRONG: it divides GB by GiB.** The planner prints
> `bytes/1e9` (true GB); the harness printed `kB/1048576` (GiB) but LABELLED it
> "GB". So every predicted-vs-MEASURED ratio quoted in this campaign understates
> the measurement by 7.37%. Corrected:
>
> | rung | pred GB | meas GiB | meas GB | quoted ratio | TRUE ratio |
> |---|---|---|---|---|---|
> | 1 | 8.61 | 8.91 | 9.57 | 1.035 | **1.111** |
> | 2 | 11.98 | 12.06 | 12.95 | 1.006 | **1.081** |
> | 3 | 17.51 | 18.61 | 19.98 | 1.063 | **1.141** |
> | 4 | 24.36 | 23.60 | 25.34 | 0.969 | **1.040** |
> | 5 | 37.67 | 36.03 | 38.69 | 0.956 | **1.027** |
> | 6 | 56.49 | 53.40 | 57.34 | 0.945 | **1.015** |
>
> **The planner does NOT bracket reality — it systematically UNDER-predicts by
> 1.5-14%.** A planner that reads LOW is worse than one that reads high: it
> promises headroom that is not there. The "model requires NO fixes / NO tweaks"
> verdict inherited from R5 rests on this arithmetic and is WITHDRAWN pending a
> recalibration against correctly-united measurements.
> Source of the error FIXED in all 10 ladder harnesses: the summary line now
> prints `%.2f GiB = %.2f GB`. The two PRE-REGISTERED predictions (R8.2 rung 7
> 0.07%, R10.3 rung 8 0.25%) compared my GB against the planner's printed GB
> and are UNAFFECTED.

| rung | (nb, mu) | job | gw wall | HWM pred/meas GiB | ratio | eqp0/eqp1 gap eV |
|---|---|---|---|---|---|---|
| 0 (AQ) | (128, 4962) | 7877789 | 514 s | 16.53@40GB / n.m. | — | 3.5788 / 3.2526 |
| 1 | (256, 2475) | 7878104 | 434 s | 8.61 / 8.91 | 1.035 | 3.5819 / 3.2516 |
| 2 | (256, 3491) | 7878225 | 495 s | 11.98 / 12.06 | 1.006 | 3.6290 / 3.2895 |
| 3 | (512, 4951) | 7878263 | 795 s | 17.51 / 18.61 | 1.063 | 3.4594 / 3.1603 |
| 4 | (512, 6947) | 7878363 | 1217 s | 24.36 / 23.60 | 0.969 | 3.2194 / 2.9778 |

All four rungs GREEN (rc=0, no OOM, W residual <=1.7e-14, H0/route gates).
Memory-model verdict (task 6): NO fixes, NO tweaks needed — |error| <=6.3%
across 2.7x HWM, nb x2 twice, mu x2.8. GW memory wall extrapolates to
mu ~ 26k at this architecture — NOT the frontier.

RANKED WALLS (fundamental-vs-fixable):
1. FIXED (movement, gate-verified same session): single-node centroid
   Gram build — pair_density materialized 2x (nk,2,2,M,M) = 2x98 GB at
   M~9.8k (kmeans N=7000); killed job 7878309 on 192 GB nodes; bridged
   via nvdimm (job 7878358). Column-blocked Gram implemented in wt-REL
   (branch wsREL-gramfix, uncommitted); gate job 7878488: forced-block
   c2475 regen byte-identical to control + original. Standard nodes
   suffice again below ~mu 20k. Details: ladder notes R6.
2. FIXABLE (design in hand): sigma.tau.host_accum time — superlinear in mu
   (73/136/329/610 s at mu 2475/3491/4951/6947), 89% of sigma.exec at
   rung 4; sharded-consumer memo (DESIGN_MEMO_omega_cube_sharding.md).
3. FIXABLE (same memo): omega-cube nb^2 replicated residency —
   2751.46 MB/rank at 512b (exactly 4x the 256b 687.87 MB), ~11 GB at
   1024b; collides with the thousands-of-low-memory-ranks target.
4. FUNDAMENTAL-ish (writer is fast, disk is the cost): restart scratch
   ~mu^2 — 3.47 / 6.70 / 13.86 / 26.54 GB along the ladder.
5. Zeta-apply gather stays linear in mu (230/324/460/640 MB); colltable's
   (mu,mu)+ FLAG went formally clean at rung 4 (5760 < mu).

RUNG-5 BLOCKERS (1024b, ~10k): true NSCF at nbnd=1024 (WFN ~610 MiB, QE
chain trivial via the staged-density b512 recipe); omega-cube fix strongly
advised BEFORE 1024b (2x ~11 GB/rank replicated residency); centroid set
needs the blocked-Gram fix or nvdimm; restart scratch ~54 GB/run.
Physics note for the convergence workstream: mu-convergence at fixed
window is NOT monotone-from-below on this deck (256b: +47 meV 2475->3491;
512b: -240 meV 4951->6947).

## OWNER DIRECTIVE 2026-07-29 — lift the basis ceiling with cutoff, not just mu

R7.1 found the 30 Ry 4x4 deck's PW basis caps the band axis (1024 bands
succeeded only after the npol patch; beyond that "more bands than PWs").
Owner ruling: escalate by raising the cutoff to ~45 Ry rather than riding
mu alone.  PW count ~ E_cut^(3/2): 30 -> 45 Ry ~ 1.84x PWs (~3900 -> ~7200
per spinor component) — lifts the band ceiling past 1024 and grows the
genuine per-rank workload (a physically meaningful route to the OOM
frontier).

Requirements for the 45 Ry lineage (distinct suffix; 30 Ry artifacts
NEVER overwritten): FULL regen — SCF density recomputed AT the new cutoff
(the staged-density shortcut is valid only at fixed cutoff), then NSCF,
pw2bgw -> WFN/vxc/kih/RHO, kin_ion, dipole (MANDATORY), and centroid sets
generated on the NEW FFT grid (no 30 Ry centroid reuse).  Gates are
INTERNAL only (H0/implied-Vxc at the new cutoff, W residual, Kramers,
collective table) — NO cross-cutoff eqp parity; the 30-vs-45 Ry gap
difference is recorded as a cutoff-convergence physics observation.
Estimate SCF+NSCF node-hours from the b1024 timings BEFORE submitting and
stage across jobs if it exceeds the 2 h dev-queue limit.

## FRONTIER LEDGER — successor agent, 2026-07-29 (all numbers disk-verified)

Handover hazard neutralised first: the live repo tree was being edited by the
concurrent GEMM/BSE workstream (`src/common/contract_bands.py`, on sigma's hot
path, mtime 00:39).  All runs below import a READ-ONLY pin,
`git archive 4f77842 -> wk_REL/snapshots/srcpin_4f77842` (228 py files, verified == HEAD).
Nothing was committed to the lorrax repo.

### Rungs (30 Ry, P=64 8x8, budget 90 GB/dev, cache-cold, coll=mpi)


> ⚠⚠ CLAIM-DECAY (2026-07-29, R30.3) — **THE "ratio" COLUMN IN EVERY TABLE
> BELOW/ABOVE IS WRONG: it divides GB by GiB.** The planner prints
> `bytes/1e9` (true GB); the harness printed `kB/1048576` (GiB) but LABELLED it
> "GB". So every predicted-vs-MEASURED ratio quoted in this campaign understates
> the measurement by 7.37%. Corrected:
>
> | rung | pred GB | meas GiB | meas GB | quoted ratio | TRUE ratio |
> |---|---|---|---|---|---|
> | 1 | 8.61 | 8.91 | 9.57 | 1.035 | **1.111** |
> | 2 | 11.98 | 12.06 | 12.95 | 1.006 | **1.081** |
> | 3 | 17.51 | 18.61 | 19.98 | 1.063 | **1.141** |
> | 4 | 24.36 | 23.60 | 25.34 | 0.969 | **1.040** |
> | 5 | 37.67 | 36.03 | 38.69 | 0.956 | **1.027** |
> | 6 | 56.49 | 53.40 | 57.34 | 0.945 | **1.015** |
>
> **The planner does NOT bracket reality — it systematically UNDER-predicts by
> 1.5-14%.** A planner that reads LOW is worse than one that reads high: it
> promises headroom that is not there. The "model requires NO fixes / NO tweaks"
> verdict inherited from R5 rests on this arithmetic and is WITHDRAWN pending a
> recalibration against correctly-united measurements.
> Source of the error FIXED in all 10 ladder harnesses: the summary line now
> prints `%.2f GiB = %.2f GB`. The two PRE-REGISTERED predictions (R8.2 rung 7
> 0.07%, R10.3 rung 8 0.25%) compared my GB against the planner's printed GB
> and are UNAFFECTED.

| rung | (nb, mu) | job | gw wall | HWM pred/meas GiB | ratio | eqp0/eqp1 gap eV |
|---|---|---|---|---|---|---|
| 0 (AQ) | (128, 4962) | 7877789 | 514 s | 16.53@40GB / n.m. | — | 3.5788 / 3.2526 |
| 1 | (256, 2475) | 7878104 | 434 s | 8.61 / 8.91 | 1.035 | 3.5819 / 3.2516 |
| 2 | (256, 3491) | 7878225 | 495 s | 11.98 / 12.06 | 1.006 | 3.6290 / 3.2895 |
| 3 | (512, 4951) | 7878263 | 795 s | 17.51 / 18.61 | 1.063 | 3.4594 / 3.1603 |
| 4 | (512, 6947) | 7878363 | 1217 s | 24.36 / 23.60 | 0.969 | 3.2194 / 2.9778 |
| 5 | (1024, 10015) | 7879295 | 1770 s | 37.67 / **36.03** | 0.956 | **0.3645 / -0.3639 RED** |

**Rungs 0-4 were a NO-WEAPONS baseline** (disk-verified from
`run_L4_b512_c7000/inner.sh`: only `LORRAX_FFI_HOST_SO=.../build_host_AUDIT`,
and no `sigma_omega_layout` in gw.in).  Rung 5 is the FIRST weaponised run at
any size and it is **physics-RED**: the QP correction at k1/band 26 is -8.12 eV
(rung 4: -0.77 eV) and the eqp1 fundamental gap is NEGATIVE.  Memory, timing,
W-Dyson residual (1.9e-14), H0 gate and rc=0 are all clean — the failure is
confined to Sigma consumption / the QP solve.  Gap convention certified by
reproducing rungs 3 and 4 from their own eqp files
(`wk_REL/harness/qpgap.sh`).  **Rung 5's MEMORY row stands; its PHYSICS row is
withdrawn.**  2x2 in flight: job 7879357 = (1024,10015) weapons OFF,
job 7879359 = (512,6947) weapons ON.

### Rung-5 prerequisites — ALL GREEN (jobs 7879046 / 7879287 / 7879286)

- NSCF nbnd=1024 on the npol-patched pw.x: ethr 3.85e-13, 30.7 avg iterations.
- **el_compare vs b512: max |d| = 1.859e-11 eV** over bands 1..512, all k.
  This MEASURES the memory_report patch's numerical innocence.
- kin_ion_b1024.h5 (537 MB), **dipole_b1024.h5 (940 MB) MANDATORY regen done**,
  gate_h0 rms 3.9e-5 eV, implied-Vxc vs QE 3.9e-5 eV, TRS holds 1.27e-14.
- **Blocked-Gram ACCEPTANCE TEST: PASS on a STANDARD 192 GB node, no nvdimm.**
  M=13872, auto col_block=1480 (10 blocks), 10015 orbit-closed centroids,
  **VmHWM 66.2 GB** of 186.  Ladder wall #1 is CLOSED at the committed code.
  Scaling correction: centroid generation does NOT scale with nband (the Gram
  uses the prune window v x (v+c) = 26 x 52, not nband) — the mu axis is open
  on this node class far past anything the GW side survives.

### CERTIFIED WALL #1 (QE-side, arithmetic): the 30 Ry band axis ends at 1024

`memory_report.f90:91` `npwx_g = NINT(4pi/3*sqrt(ecutwfc)^3/(2pi^3/omega))` is
exactly ecut^1.5 and npol-BLIND.  Measured `ngkmax` = 1964 @30 Ry, 3597 @45 Ry
(ratio 1.831 vs analytic 1.837).  Ceilings (david, ndim=2 => nbndx=2*nbnd):
stock `npwx_g < nbndx` -> nbnd <= 974 @30 Ry, <= 1789 @45 Ry;
patched `npwx_g*npol < nbndx` -> nbnd <= 1948 @30 Ry, <= 3578 @45 Ry.

### CERTIFIED WALL #2 (predicted, mechanism identified): r-chunk PERFORMANCE FLOOR

`gflat_memory_model.py:366` sets `r_lo = min(mu, n_rtot)` and
`r_chunk = max(r_lo, min(n_rtot, headroom_C/C_slope))`.  While the budget term
wins, growth in mu is absorbed by shrinking r_chunk and HWM PINS to the 76.5 GB
target.  Once `r_lo` wins, HWM goes QUADRATIC in mu with no automatic knob left.
Calibrated on rungs 4/5/6 planner blocks:

    C_slope = 68.05*mu + 10,330   B per r-point   (band_chunk=64, P=64)
      -> predicted 1,031,556 vs rung-6 actual 1,031,467 B  (0.01% error)
    persistent(nb=1024) = 540,244*mu + 4*mu^2  B

    mu < 20,250  : 1 r-chunk, HWM rising
    20,250-29,000: planner r-chunks, HWM pinned at target -- SURVIVES
    mu > 29,000  : r_lo binds, HWM quadratic
    mu ~ 31,700  : 90 GB budget exceeded  <-- PREDICTED TERMINAL OOM

Hard ceiling on this axis: centroids are grid points, mu <= n_rtot = 46080
@30 Ry (max feasible target N ~ 33,000 at the 1.35x candidate oversample).
The predicted OOM sits just inside it.  Centroid sets built and accepted for
the approach: c15000 (15007), c20000 (19991); c25000/c30000/c32000/c33000 in
flight (jobs 7879344, 7879354).
**The ONE named escape lever** is `r_chunk_size` (gw_config.py:644 ->
`r_chunk_override`), wired into the ladder template as `RCHUNK=`.  Behind it
the next object is `persistent`'s `psi_copies`, divided by **sqrt(P) only**
(2 copies on 'x' + 2 on 'y', never /P) — the true architectural floor at fixed
P and the term the low-memory-ranks target ultimately collides with.

### DECK LINEAGE 2: 45 Ry (owner directive) — established and gated

FULL regen (the staged-density trick is valid only at fixed cutoff):
`scf_r45.in -> out_r45/` (12 s wall, 11 iterations, E = -178.22961446 Ry) ->
staged into per-nbnd NSCF outdirs.  Measured basis: dense G 28615, **FFT
(25,25,100) = 62500 points vs 46080 @30 Ry (x1.356)** — that factor is the
GW-side content of the cutoff raise, since Stage C's transient is
`r_chunk * C_slope` with r_chunk capped by n_rtot.
`WFN_r45_b1024.h5` (1.176 GB) built and in place; nbnd=2048 NSCF in flight
(job 7879353).  Gates are INTERNAL per owner ruling — NO cross-cutoff eqp
parity; the 30-vs-45 Ry gap difference will be reported as a cutoff-
convergence physics observation only.

### DEATH #1 (job 7879348, QE/tool FAIL-FAST) and its escape

45 Ry nbnd=2048 died in 18 s: `diag_bands (1): too many bands, or too few
plane waves` — `c_bands.f90:308`, `IF (nbndx > ipw)` with
`ipw = mp_sum(npwx)`.  A SECOND instance of the same npol-blindness already
corrected in memory_report: it compares nbndx=4096 against the SPATIAL PW count
(~3600) rather than the spinor dimension `ipw*npol` (~7200).  ONE escape spent:
extend the identical correction (`nbndx > ipw*npol`); `ipw` occurs at exactly 3
lines in the file and nothing is computed from it.  Empirical warrant: the
30 Ry nbnd=1024 run ALREADY ran at nbndx=2048 above its spatial npwx=1964 and
matched b512 to 1.86e-11 eV.  pw.x rebuilt 01:40; gate = el_compare 2048-vs-1024
INSIDE the 45 Ry family (the b1024 reference was built at 01:36, BEFORE the
rebuild, so it is genuinely independent).  Fallback if it fails: nbnd=1536
(nbndx=3072 < 3578) needs neither patch.

### QE PARALLELIZATION — measured, per owner directive (jobs 7879300, 7879346)

Same physical NSCF (45 Ry, nbnd=1024) five ways.  **This deck has 10 k-points
in the IBZ, not 16** (`/mf_header/kpoints/ngk` has 10 entries), so one-k-per-
pool is `-nk 10`.

| probe | layout | wall | node-h |
|---|---|---|---|
| a | 2 nodes, -nk 10 -ndiag 1 | 234 s | 0.130 |
| b | 2 nodes, -nk 10 -ndiag 9 | 252 s | 0.140 |
| c | 2 nodes, -nk  4 -ndiag 16 | 350 s | 0.194 |
| **d** | **1 node, -nk 10 -ndiag 1** | **228 s** | **0.063** |
| e | 1 node, -nk  8 -ndiag 1 | 421 s | 0.117 |

Optimal = **1 node, `-n 50 -nk 10 -ndiag 1`**.  Three measured findings:
(1) the pool count must DIVIDE the k-count and equal it if possible — `-nk 8`
on 10 k costs 1.85x; (2) `-ndiag > 1` LOSES (cdiaghg 202.1 s vs 149.8 s) because
this QE is built `-D__DFTI -D__MPI` with **no `-D__SCALAPACK`**, so it falls to
LAXlib's own MPI ortho, communication-bound at nbndx=2048 on 3x3 ranks;
(3) since the diag is then serial on ONE rank per pool and is 64-87% of the
NSCF wall, ranks/pool are mostly idle — halving the nodes costs nothing in wall
and halves the bill.  QE moved to `small` (48 h, independent cap); GW ladder
runs moved to `normal` (the dev 2 h wall and its 2-job cap, shared with the
concurrent workstream, both bind at these sizes).  Architecture unchanged:
P=64, 8x8, 32 nodes x 2.

### UPDATE 02:00 — weapon bisect done; 45 Ry lineage complete through the WFN

**The weapons are EXONERATED at rung-4 size, and are a 1.84x speedup.**
Three configurations at (512, 6947), same deck, same centroid set:

| config | job | eqp0 | eqp1 | wall | VmHWM |
|---|---|---|---|---|---|
| no weapons (rung 4) | 7878363 | 3.2194 | 2.9778 | 1217 s | 23.60 GB |
| ALL weapons ON | 7879359 | 3.2194 | 2.9778 | **661 s** | 23.60 GB |
| all weapons minus sharded | 7879361 | 3.2194 | 2.9778 | — | — |

Bit-identical physics, unchanged peak memory, **1.84x end-to-end** — an
independent ladder-scale validation of 712a866 / 5918cf6 / 0225b5f / 5894dcd
at 4x the band window and 1.4x the mu of the AQ deck they were gated on.

So the rung-5 RED is a SIZE effect or a WEAPON x SIZE interaction above
nb=512, NOT a broken weapon.  Every deck artifact is already cleared by
independent evidence (Σ_X matches rung 4 to 0.03%; head R_h to 0.026% with an
IDENTICAL on-shell shift of -1.0802 eV; el_compare 1.86e-11 eV; gate_h0
3.9e-5 eV).  Discriminator in flight: job 7879357 = (1024, 10015) weapons OFF.

**GATE-SUITE GAP (worth fixing regardless):** no rung compares eqp against an
independent baseline, so a Σ_c-only corruption passes every gate the campaign
runs — which is exactly what happened.  Recommend carrying one fixed
(nb, mu) reference point and asserting its gap on every rung.

### MODEL GAP: the gflat planner does not model the Σ omega cube

Weapons-ON and weapons-OFF at (1024, 10015) print the IDENTICAL planner block
(37.67 GB), yet `sigma_omega_layout=sharded` is precisely what removes the
replicated `(nω,nk,nb,nb)` cube (2.75 GB/rank at nb=512, ~11 GB at 1024,
~44 GB projected at 2048).  The planner covers only the ISDF chunk plan; the
Σ-stage cube is outside it.  The rungs 1-5 agreement (0.956-1.063) is
therefore a statement about the ISDF stage, valid only while C_fit is the
global binder.  **At nb=2048 the replicated cube would BIND before C_fit and
the planner would not see it coming** — so on the 45 Ry lineage the sharded
layout is load-bearing, which makes the rung-5 regression in that code path
critical-path, not a side quest.

### 45 Ry lineage: NSCF nbnd=2048 SUCCEEDED; both QE patches now certified

| stage | job | result |
|---|---|---|
| SCF 45 Ry | 7879300 | 12 s, 11 iterations, E=-178.22961446 Ry |
| NSCF 2048 | 7879353 | **rc=0, 995 s, 0.276 node-h**, ethr 3.85e-13, 22.4 it |
| WFN_r45_b2048.h5 | | 2,351,256,564 B |
| el_compare 2048 vs 1024 | 7879367 | **max 1.405e-10 eV — PASS** |
| RHO provenance (r45 internal) | 7879353 | 4 bytes, all header timestamp |

Both LOCAL QE patches are measured, not argued:
`memory_report.f90:486` -> 1.859e-11 eV (30 Ry b1024 vs b512);
`c_bands.f90:308` -> 1.405e-10 eV (45 Ry b2048 vs b1024).

**CUTOFF-CONVERGENCE PHYSICS OBSERVATION** (explicitly NOT a gate): the DFT
fundamental gap moves **1.7 meV** for a 50% cutoff raise —
30 Ry: -5.3602/-3.1481 = 2.2121 eV;  45 Ry: -5.3575/-3.1471 = 2.2104 eV.
The cutoff raise buys band-axis headroom and per-rank work (n_rtot x1.356),
not a different answer.

### Centroid ladder toward the predicted wall (all on STANDARD 192 GB nodes)

| target N | accepted | M (Gram) | col_block | VmHWM |
|---|---|---|---|---|
| 10000 | 10015 | 13872 | 1480 | 66.2 GB |
| 15000 | 15007 | 20287 | 1012 | 70.4 GB |
| 20000 | 19991 | — | — | — |
| 25000 | 24933 | 30919 | 664 | 79.4 GB |
| 32000 | (in flight) | 36163 | 567 | 85.6 GB |

The blocked-Gram fix holds to M=36163 on a standard node — nvdimm is not
needed anywhere on this ladder.  Rungs submitted: 6 = (1024, 15007)
job 7879349; 7 = (1024, 24933) job 7879369, planner-checked at a predicted
76.50 GB = 85% of budget with r_chunk 35467 (2 chunks) and `r_lo` NOT yet
binding — i.e. still in the regime where the planner absorbs growth.

## CERTIFIED WALL #3 — **mu = 16,384 EXACTLY** (the first ARCHITECTURAL wall)

**Job 7879369**, rung 7 = (nb 1024, mu 24933), died in 2m51s.

CLASSIFICATION: **LORRAX FAIL-FAST at resolve time.**  Not OOM-killer, not XLA
allocator, not QE.  All 64 ranks raised the same ValueError; peak
`VmHWM = 39.64 GB` against a 76.45 GB planner estimate (sacct step MaxRSS
42,225,676 kB concurs) — **memory was never the binding constraint.**

    ValueError: charge_zeta_solve='rank_truncate' needs the replicated route,
    but the CCT stack (nq=10, n_mu=24933) is 92.63 GiB > the 4.00 GiB cap.

### The object and its scaling law (`isdf/core.py`, source-read)

    _replicated_factor_q_chunk(nq, mu) = max(1, min(nq, 4GiB // (mu^2*16)))
    resolve OK  <=>  batch * mu^2 * 16 <= max(LORRAX_ZETA_REPLICATE_CAP_GIB,
                                              _REPLICATED_FACTOR_MAX_BATCH_BYTES = 4 GiB)

`batch` has already collapsed to 1 at every ladder size, so the criterion is
`mu^2 * 16 <= 4 GiB`:

    **mu_max = sqrt(4 GiB / 16 B) = 16,384 EXACTLY — independent of nq, of the
    cutoff, of the grid, and of P.**

The object is ONE q-slice of the replicated rank-truncating charge factor:
`mu^2` complex128, QUADRATIC in mu and REPLICATED on every rank with **no P
scaling at all**.  Confirmed against the ladder:

| mu | q-slice | default cap | observed |
|---|---|---|---|
| 10015 (rung 5) | 1.49 GiB | OK | ran; factor 240.6 s |
| 15007 (rung 6) | 3.36 GiB | OK | ran |
| 19991 | 5.96 GiB | REFUSED | not attempted |
| 24933 (rung 7) | 9.26 GiB | **REFUSED** | **died 2m51s** |
| 32059 (rung 8) | 15.32 GiB | REFUSED | needs ZCAP |

The refusal is deliberate and correct: core.py:1157-1164 records that a silent
fallback here on 2026-07-21 returned zeta 4.5x too large and rebuilt V_q to
relF 16-32 instead of 1.8e-15.  `rank_truncate` is the only route carrying the
rank-truncation physics cure, so above the cap the code refuses rather than
silently lose it.

### Escape SPENT and CONFIRMED WORKING

`LORRAX_ZETA_REPLICATE_CAP_GIB=16`, chosen from the CORRECT quantity (the
9.26 GiB q-slice), not from the message's suggestion.  Job **7879380** cleared
the resolve gate and reached the planner (76.45 GB) — escape effective, now
wired into the ladder template as `ZCAP=`.

**MESSAGE DEFECT (worth a fix):** the error quotes the WHOLE STACK
`nq*mu^2*16 = 92.63 GiB` and advises `CAP=94`, but the branch that refused
tests ONE q-batch (`mu^2*16 = 9.26 GiB`) — that difference is the entire point
of `_replicate_rank_truncate_ok` being separate from `_replicate_charge_ok`.
Following the message reserves 94 GiB where 10 suffices, turning a workable
escape into an apparently impossible one.

### Why this is ARCHITECTURAL, not budgetary

The lever buys headroom, not scalability.  The object is replicated `mu^2` and
the work is a dense `eigh` per q run **REDUNDANTLY on every rank, O(nq*mu^3),
with ZERO P-scaling** — core.py:1019-1024 says so explicitly and concludes the
route "needs a genuinely distributed eigh (SLATE/ScaLAPACK via ffi.linalg;
cuSOLVERMp is out on a rectangular mesh), not a bigger cap."  At fixed P=64
this cannot be escaped by buffer donation, comm reduction, or sharding.  It is
exactly the collision with the thousands-of-low-memory-ranks target that this
campaign exists to find, and it is the first wall on the ladder that is
architectural rather than budgetary.
Calibration for planning: the factor measured **240.6 s at mu=10015** (rung 5
`zeta_fit.cholesky`), and it scales as mu^3 — 62 min at mu=24933, ~2.6 h at
mu=32059.  It also applies UNCHANGED to the 45 Ry lineage (cutoff-independent),
where only the mu=13000 rung clears the default cap.

## STATE AT END OF SESSION (02:22, successor agent #2)

### What is CERTIFIED

1. **mu = 16,384 EXACTLY** — the FAIL-FAST wall on the mu axis at fixed P=64.
   Object: one q-slice of the replicated rank-truncating charge factor,
   `mu^2 * 16 B`, quadratic in mu and replicated per rank with NO P-scaling.
   Bracketed on BOTH sides by four runs (15007 ran / 24933 refused / 30011
   refused / escaped at 24933 and 32059).  Behind the cap the work is a dense
   `eigh` per q run REDUNDANTLY on every rank, `O(nq*mu^3)` — the code's own
   comment says the route "needs a genuinely distributed eigh, not a bigger
   cap".  **This cannot be escaped by donation, comm reduction, or sharding.**
2. **The 30 Ry band axis ends at nbnd=1024**; 45 Ry lifts it to 3578 (patched)
   / 1789 (stock).  Both QE npol patches numerically validated (1.86e-11 and
   1.405e-10 eV against independently-built references).
3. **Ladder wall #1 (centroid Gram) is CLOSED** — the committed blocked-Gram
   path carried M=13872..36871 on STANDARD 192 GB nodes, peak 66-86 GB.
   nvdimm is not needed anywhere on this ladder.
4. **The weapons are correct and are a 1.84x speedup at ladder scale**
   (bit-identical eqp at (512,6947), 1217 -> 661 s).

### What is OPEN (harvest these first)

- **Rung-5 physics RED at (1024, 10015)** — gap 0.36 eV vs 3.2-3.6 expected.
  Weapons exonerated at nb=512; deck artifacts exonerated by Sigma_X and the
  head fit.  Jobs 7879357 (WEAPONS=off) and 7879382 (SHARDED=off) decide
  between a pure size effect and a weapon x size interaction above nb=512.
- Rungs 6/7/8 memory (jobs 7879349 / 7879380 / 7879381) — rung 8 is
  pre-registered at 91.47 GB = 102% of budget, the terminal probe.
- 45 Ry GW has NOT been run yet; its deck is complete and gated, centroids in
  flight (job 7879368).  **R10 applies unchanged there** (cutoff-independent),
  so only its mu=13000 rung clears the default cap.

### Pre-registered, unverified: the MEMORY wall behind R10

With `ZCAP` raised, the next wall is the r-chunk performance floor:
`HWM(r_lo-bound) = 72.05*mu^2 + 550,574*mu` bytes, reaching the 96 GB/rank
physical limit at **mu ~ 34,600** (predicted-vs-actual has run 0.945-0.956, and
the model has been pre-registered correct to 0.07% and 0.25% at rungs 7 and 8).
Hard ceiling of the axis: `mu <= n_rtot = 46,080` at 30 Ry, 62,500 at 45 Ry.

## PRIORITY REDIRECT 02:35 — rung-5 correctness outranks size escalation

### The mu=16,384 wall: FIX LINKAGE (coordinator, 2026-07-29)

The certified wall's fix is already designed.  `isdf/core.py:1019-1024` says the
replicated rank-truncate route "needs a genuinely distributed eigh
(SLATE/ScaLAPACK via `ffi.linalg`; cuSOLVERMp is out on a rectangular mesh),
not a bigger cap."  **That is exactly wk_AP's two-plan distributed-eigh memo:
plan A local `zheevd`, plan B subgrid `pzheevd`.**  Implementing either lifts
`mu_max` off 16,384 and removes the O(nq*mu^3) redundant-per-rank cost with it.
The `ZCAP` lever is a stopgap that buys headroom, not scalability.

### NAMED MECHANISM for the rung-5 RED (full detail: ladder notes R12)

**`src/centroid/kmeans_cli.py:_resolve_sigma_window()`**

    n_cond = int(args.prune_n_cond) if given else min(n_val, nb_total - n_val)
                                                  -> min(26, 998) = 26

`n_cond` defaults to `n_val`, so the pivoted-Cholesky prune window is
`left=(0,26) right=(0,52)` for **every centroid set this campaign has built** —
disk-verified identical in b256 c2500, b512 c7000, b1024 c10000 and c15000 —
while the deck's real sigma window at rung 5 is `nval 26 + ncond 998 = 1024`.
**The ISDF basis is selected to resolve a 26x52 pair-density block while
Sigma_c consumes a 1024x1024 one.**  Aggravating: the default prune MODE
`v_x_vc` keeps `left=(0,n_val)`, so cond x cond pair densities never enter the
selection at ANY n_cond, though they are the bulk of the Sigma sum for
conduction QP states at ncond=998.  `--prune-window vc_x_vc` exists for this.

The docstring calls this function *"the sigma window — the bands the ISDF must
span"* and *"single source of truth for both consumers"*; that is false
whenever `ncond != nval`.  The `--prune-n-cond` help ("default = n_val") is
literally correct — the CLI is honest, the docstring is not.

**Why it fits every observation**: mis-targeting has existed since rung 1, is
harmless while the basis is incidentally rich enough (nb<=512), becomes fatal
at nb=1024 (the observed threshold), and is partially rescued by more mu
(observed: eqp0 0.3645 -> 1.4296 for mu 10015 -> 15007 at fixed nb=1024).
A logic/indexing bug would not heal with more mu.

**STATUS: identified and consistent, NOT YET PROVEN.**  Decisive dose-response
submitted — jobs 7879432 (`--prune-n-cond 230`) and 7879433 (`--prune-n-cond
998`), same N/WFN/orbit closure as the rung-5 set, only the prune window
differs.  If the gap returns toward 3.2-3.6 eV the release-blocking statement
is: *"ISDF centroid selection silently ignores the deck's conduction window, so
every GW result with ncond >> nval rests on a basis chosen for a different,
much smaller problem."*  Next suspect if exonerated: `ppm_windows.py`
`_SigmaBranch` E_ref_A/E_ref_B (`np.min` over band-range-masked E_A) — genuinely
band-range-derived and NOT yet cleared.

## ADOPTED: the permanent Sigma reference gate

`mos2_4x4_test/gate_sigma_reference.sbatch` (submitted 7879436 to pin it at the
current build).

- **Reference**: 30 Ry b256 deck, `nval=26 / ncond=230 / nband=256`,
  `centroids_b256_c2500.txt` (2475 orbit-closed), P=64 8x8, weapons ON —
  ladder rung 1, the cheapest configuration that exercises the whole Sigma_c
  path (434 s, 8.91 GB/rank, **3.9 node-hours**).
- **Quantity**: the eqp0 and eqp1 INDIRECT QP gaps, extracted by
  `wk_REL/harness/qpgap.sh` whose convention is certified by reproducing rungs 3 and 4
  from their own eqp files.
- **Pinned values** (job 7878104): **eqp0 = 3.5819 eV, eqp1 = 3.2516 eV**.
- **Tolerance 1.0e-3 eV.**  The same (deck, mu, window) recomputed is
  deterministic so the honest expectation is ~1e-6 eV; 1e-3 leaves room for
  compile/threading nondeterminism while still catching the rung-5 class of
  corruption (2.9 eV = 2900x tol) by ~3.5 decades.
- **When**: before accepting any ladder rung, and on every change to the
  Sigma/ISDF path.  Negligible against a rung.
- **Why a reference run and not an internal invariant**: the rung-5 corruption
  passed el_compare (1.86e-11 eV), gate_h0 (3.9e-5), W-Dyson residual
  (1.9e-14), TRS/density symmetry, route checks, rc=0, AND the two Sigma-side
  quantities that are cheap to check — bare Sigma_X (0.03% vs rung 4) and the
  q->0 head fit (0.026%, identical on-shell shift).  No internal witness
  existed; only the output itself is diagnostic.
- **Designed, not built** (the better long-term gate): an ISDF **sigma-window
  fit-residual** assertion — sample K random (n,m) pairs from the FULL sigma
  window and assert the relative pair-density interpolation error.  That
  measures the thing R12 says is broken, per-rung and without a reference run.

## SIGMA REFERENCE GATE: WIRED AND **PASSING** (job 7879439)

    eqp0  3.5819 vs pinned 3.5819   |d| = 0.00e+00 eV
    eqp1  3.2516 vs pinned 3.2516   |d| = 0.00e+00 eV
    === SIGMA REFERENCE GATE: PASS (tol 1.0e-03 eV) ===

**Bit-exact**, and note what it spans: the pinned values come from job 7878104
(rung 1) at a PRE-WEAPONS commit; the gate re-ran at pinned 4f77842 with all
four weapons ON.  Identical to the last printed digit.  So the current build
reproduces the campaign's oldest Sigma_c result exactly — **the codebase is not
broken in general, and the rung-5 defect is size-specific.**  The gate is now
`mos2_4x4_test/gate_sigma_reference.sbatch`, 3.9 node-hours, run it before
accepting any rung and on every Sigma/ISDF change.

## MECHANISM: two MEASURED conditioning failures, both in the ISDF

**(A) Selection Gram — the prune-window clamp** (ladder notes R12).
`kmeans_cli._resolve_sigma_window` defaults `n_cond = min(n_val, nb-n_val)` =
26, so every centroid set the campaign owns was selected on a 26x52
pair-density block while Sigma_c consumes 1024x1024.  Rebuilt at identical
N/WFN/M with a corrected window:

| prune window | Gram diag min | achieved rank / requested |
|---|---|---|
| (0,52) default | 7.632e-17 | **630 / 897** |
| (0,256) | 7.189e-13 | **897 / 897** |
| (0,1024) full | 4.996e-12 | **897 / 897** |

The default is **rank-deficient by 30%** and the shortfall was printed in every
centroid log the campaign ever produced.

**(B) Fit Gram — rank truncation discards real content** (ladder notes R13,
the owner's hypothesis, confirmed from EXISTING artifacts with no new jobs).
`isdf/core.py` already prints `[zeta rank_truncate/distributed]` every run:

| rung | nb | mu | n_pad | n_keep | keep % |
|---|---|---|---|---|---|
| 3 | 512 | 4951 | 4992 | 4183 | 83.8% |
| 4 | 512 | 6947 | 6976 | 4570 | 65.5% |
| 5 | 1024 | 10015 | 10048 | 6700 | 66.7% |
| 6 | 1024 | 15007 | 15040 | 7108 | **47.3%** |

- **The retained rank SATURATES**: +49.8% mu buys +6.1% rank at nb=1024
  (+40.3% -> +9.3% at nb=512).  **The fit cannot be repaired by adding mu** —
  which is exactly why rung 6 is still RED at 15007 centroids, and why the
  mu-axis escalation was chasing something it could never catch.
- **The truncation discards real content by six decades**: at rung 6 the cut
  is `lam_max*rcond = 5.12e-9` and it throws away 7,932 of 15,040 modes, while
  the f64 Gram noise floor is `~6.3e-15`.  `zeta_rcond = 1e-8` is a default
  sized for much smaller fits.
- The trend is already inside the GREEN rungs (83.8% -> 65.5% at nb=512), which
  fits the known non-monotone mu-convergence there (+47 meV at 256b, -240 meV
  at 512b).  **The defect did not switch on at nb=1024; it crossed a severity
  threshold.**

**2x2 IN FLIGHT** (positions vs truncation, cleanly separated):
  7879438 prune (0,256), rcond default | 7879441 prune (0,1024), rcond default
  7879450 prune default, rcond 1e-10   | 7879451 prune default, rcond 1e-12

## NEW FAILURE MODE: sigma_omega_layout=replicated HANGS at nb=1024

Both weapons-off discriminators (7879357, 7879382) reached
`Finished sigma[w<E_F val]` cleanly, then produced NOTHING for 20+ minutes;
all 64 ranks showed `VmRSS = 0` (processes exited) while srun never returned.
No sigma_mnk.h5, no eqp files. Cancelled. Rungs 5/6 with `sharded` ON completed
normally at the same size.  **`sigma_omega_layout=sharded` is therefore
effectively MANDATORY at nb>=1024**, which compounds R8.1 (the planner does not
model the replicated omega cube at all).  This also means the
"weapons-off at nb=1024" control is not obtainable by that route — but it is
moot: the b512 bisect showed the weapons bit-identical, and the reference gate
above reproduces rung 1 exactly with weapons ON.

## OWNER-APPROVED FIXES + GATES (2026-07-29) — code in wt-RELC, NOT committed

Worktree **/work2/08271/jackmc/frontera/wt-RELC**, branch **wsREL-isdf-window**,
base 4f77842, py_compile PASS, +80/-2 lines in one file plus one new doc.

| # | item | status |
|---|---|---|
| 1 | **Rank assertion as HARD REFUSAL** in `kmeans_cli` acceptance + harness twin `wk_REL/harness/centroid_rank_gate.sh`, wired into 4 centroid harnesses | DONE, self-tested FAIL(630/897)/PASS(897/897) |
| 2 | **`_resolve_sigma_window` fixed**: `min(n_val, nb-n_val)` -> `max(0, nb-n_val)`, framed as a BUG with the pre-fix sets named | DONE |
| 3 | **`zeta_rcond` rule** proposed, NOT adopted pending the 2x2 | PROPOSED |
| 4 | **`docs/dev/isdf_rank_saturation_and_max_usable_nb.md`** | DONE |
| 5 | **D1/D2 defects recorded** with evidence | DONE |

### THE RULE (now documented for released-code users)

> **There is a maximum usable `nband` per plane-wave cutoff, set by ISDF rank
> saturation — not by memory and not by `N_mu`.**

Recipe, using the diagnostic the code ALREADY prints
(`[zeta rank_truncate/distributed]`): run at your `nband`; record
`n_keep/n_pad`; re-run at ~1.5x `N_mu`; **if `n_keep` barely moves you are at
the ceiling and that `nband` is not usable at this cutoff**; and compare the
truncation floor `lambda_max*zeta_rcond` against `~eps*lambda_max*sqrt(n_pad)`.
On MoS2 4x4 at 30 Ry the retained rank saturates near **~7,100**; `nb=1024` is
past the point where `Sigma_c` can be trusted, `nb=512` is comfortable.

**The `N_mu ~ 6-14x nband` heuristic is RETIRED above saturation** — all three
failing points sit INSIDE the band (9.8x, 14.7x), so membership tells you
nothing. Use retained rank.

### D1 — `sigma_omega_layout=replicated` HANGS at nb>=1024 (release-note level)

Jobs 7879357 / 7879382, both (1024, 10015): reached `Finished sigma` cleanly
(622 s / 338 s), then nothing for 20+ min — no `sigma_mnk.h5`, no eqp, gw.log
mtime frozen, and **VmRSS = 0 on ranks 0/1/17/33/63 (all processes exited)**
while `srun` never returned and Slurm still showed RUNNING. Rungs 5/6 at the
same size with `sharded` ON completed normally. **`sharded` is MANDATORY at
nb>=1024.**

### D2 — the chunk planner is blind to the sigma omega cube

Replicated and sharded runs at (1024, 10015) print an identical
`HWM estimate = 37.67 GB/dev`, though replicated additionally holds
`n_omega*nk*nb^2*16 B`/rank (~11 GB at nb=1024, ~44 GB projected at nb=2048,
where it would BIND before C_fit). **The 0.945-1.063 planner agreement quoted
across rungs 1-8 is an ISDF-stage statement only** and must be quoted that way.

## CAPACITY PASS (owner redirect 2026-07-29) — physics parked, caps being removed

> ⚠ CLAIM-DECAY on CERTIFIED WALL #3 and on my proposed rcond rule. Both were
> wrong in their conclusions and are corrected below. The measurements stand.

### CAP 1 — mu = 16,384 is a PREMATURE REFUSAL, not an architectural wall

`gw/isdf_fitting.py:434-465` resolves in this order:

    1. _resolved_solver_kind = _resolve_solver_kind(...)     # RAISES above the cap
    2. _resolved_zeta_gather = _resolve_zeta_gather(...)
    3. if tier == 'distributed': _resolved_solver_kind = 'distributed_rank_truncate'

Step 1 enforces the REPLICATED q-slice cap (`mu^2*16 <= max(ZCAP, 4 GiB)`).
Step 3 then **discards that resolution** and substitutes the distributed route,
whose layout contract never replicates an O(mu^2) object (C_q/C+/V all
`P(None,'x','y')`; only lambda (nq,mu) replicated). **The check refuses a run on
the size of a buffer that route never allocates.**

PROOF from this campaign's own logs: rung 7 (mu=24933, ZCAP=16, job 7879380) and
rung 8 (mu=32059, ZCAP=24, job 7879381) BOTH print
`path=distributed_rank_truncate` — the replicated factor was never used in
either. `ZCAP` "worked" only by letting the resolver past a check for a
discarded route.

**So the earlier conclusion — "architectural, escapable only by implementing a
genuinely distributed eigh" — is WITHDRAWN.** The distributed eigh already
exists, is already the default at `distributed_zeta_solve=distributed`, and was
already running. wk_AP plan A/B remain right for decks that DO take the
replicated route, and for mu ~50k where one (mu,mu) tile stops fitting a rank —
but they are not needed to move THIS frontier.

**FIX** (wt-RELC, branch `wsREL-isdf-window`, NOT committed, py_compile PASS):
resolve the tier FIRST, pass `replicated_factor_used=(tier != 'distributed')`
into the charge resolver, and skip the replication capacity check when False.
The refusal is PRESERVED when the replicated factor really is used (the
2026-07-21 protection: silent fallback gave zeta 4.5x too large). Expected
effect: the mu ceiling for the distributed tier moves off 16,384 to hardware.
GATE job **7879469**: mu=24933, **no ZCAP**, patched source — the identical
config refused in 2m51s as job 7879369.

### CAP 2 — the restart tensor is written unconditionally and scales as mu^2

MEASURED: `tmp/isdf_tensors_<mu>.h5` = **26.5 GB** (mu 6947), **56.6 GB**
(10015), **123.2 GB** (15007) — clean mu^2. Projected **341 GB** at 24933,
**564 GB** at 32059, **1.17 TB** at the axis ceiling mu=46080.
`gw_init.py:926` branches on `if not cfg.restart:`, i.e. `restart=false` means
*compute fresh AND WRITE*; there is **no key that computes fresh without
writing** (`write_restart|save_restart|skip_restart` absent from gw_config).
RECOMMENDED (not implemented — new input key, touches the restart contract):
add `write_restart` (default true). Pure win for scaling runs; the file is only
ever an input to a LATER run.

### CAP 3 — planner blindness to the omega cube: scoped with arithmetic

`gflat_memory_model.py` receives neither `n_omega` nor the sigma layout, so it
cannot see the cube. Add `G_sigma_omega = n_omega*nk*nb^2*16` (replicated) as
its OWN stage, and REFUSE at resolve time naming `sigma_omega_layout=sharded`
when it alone busts the budget — which converts D1's 20-minute silent hang into
an instant actionable error. This deck (n_omega=41, nk=16): 2.75 GB at nb=512,
**11.0 GB** at nb=1024, **44.0 GB** at nb=2048 (would bind before C_fit).

### rcond: my proposed rule is WITHDRAWN

`gw_config.py:617-628` documents that `zeta_rcond=1e-8` was chosen from a
measured sweep and is the LOW end of an over-complete recovery PLATEAU
(1e-8..1e-4): at MoS2 4x4/1204c **1e-10 only partially recovers, MAE 1.4 eV vs
BGW**, and bulk Si 4x4x4/960c genuinely has eigenvalues below the cut.
**Lowering rcond is documented to be WORSE.** My noise-floor arithmetic stands,
but the inference that retaining those modes would help does not — they are the
over-complete directions the truncation exists to remove. No default changed.

### 2x2 recorded and parked (retained rank per cell, nb=1024, n_pad=10048)

| cell | prune window | rcond | n_keep | retained |
|---|---|---|---|---|
| baseline | (0,52) | 1e-8 | 6700 | 66.7% |
| prune fix | (0,256) | 1e-8 | 6793 | 67.6% |
| prune fix full | (0,1024) | 1e-8 | 6788 | 67.6% |
| rcond 1e-10 | (0,52) | 1e-10 | 8290 | 82.5% |
| rcond 1e-12 | (0,52) | 1e-12 | 9461 | 94.2% |

Structural result: the prune window barely moves the FIT rank (+1.4%) while
rcond moves it a lot — **the selection defect and the fit truncation are
independent knobs**, which is why the selection fix is kept as a capacity
enabler (better-conditioned basis at identical mu, selection rank 630 -> 897)
while the rcond question goes to the numerical-stability pass.

## HANDOFF: the code deliverable (NOT committed — orchestrator merges)

Worktree **/work2/08271/jackmc/frontera/wt-RELC**, branch
**wsREL-isdf-window**, base **4f77842** (HEAD unmoved — nothing committed).
Patch also written out as **`wk_REL/docs/patches/wsREL-isdf-window.patch`** (22.7 KB).

    docs/dev/isdf_rank_saturation_and_max_usable_nb.md | 203 +++   (new)
    src/centroid/kmeans_cli.py                         |  82 +-    (2 fixes)
    src/gw/isdf_fitting.py                             |  25 +-    (reorder)
    src/isdf/core.py                                   |  18 +-    (flag)
    4 files changed, 315 insertions(+), 13 deletions(-)   py_compile PASS

| change | gate | status |
|---|---|---|
| capacity: replicated check no longer gates the distributed route | job 7879469, mu=24933 **no ZCAP** — identical config refused at 2m51s as 7879369; patched run passed it, `path=distributed_rank_truncate`, planner identical | **(a)(b) PASS**, (c) eqp parity pending |
| `_resolve_sigma_window` clamp -> full WFN conduction window | rank recovery 630/897 -> 897/897 at identical N/WFN/M (jobs 7879286 vs 7879432/7879433) | **PASS** |
| centroid rank gate (hard refusal) | self-test both directions: FAIL on 7879286 (630/897, 70.2%), PASS on 7879432 (897/897) | **PASS** |
| Sigma reference gate (separate, in mos2_4x4_test/) | job 7879439: \|d\| = 0.00e+00 eV vs pinned 3.5819/3.2516 | **PASS** |

**CAPACITY DELTA**: mu ceiling for `distributed_zeta_solve=distributed` moves
from the spurious **16,384** to the next real constraint — the R8 r-chunk
performance floor at **mu ~ 34,600** — with the hard axis ceiling
`mu <= n_rtot = 46,080` behind it. **~2.1x usable mu from a 3-file ordering
fix, no new numerics.**

Not implemented, flagged with arithmetic: `write_restart` key (CAP 2, saves
564 GB/run at mu=32059) and the planner's omega-cube stage + refusal (CAP 3,
converts D1's silent hang into an instant error).

## CAP SWEEP (item 4) — verified against this campaign's own logs

Full detail + "checked and harmless" list: ladder notes R16.

| binds at | cap | file | loud? | verified |
|---|---|---|---|---|
| mu ~8,192 | collective-chunk floor `max(1,…)`, 128 MB bound | isdf/core.py:1790,1819 | **SILENT** | **YES — measured** |
| nb 1024 | HDF5 4 GiB chunk on Sigma(omega) (10.25 GiB requested) | file_io/sigma_output.py:310 | loud, **backend-dependent** | **YES — measured** |
| mu ~13,258 | H5PY_ALLGATHER replicates whole V_qmunu, 2 copies | _slab_io_allgather.py:72 | SILENT OOM | static |
| mu ~23,177 | >2 GB single H5Dwrite, no payload chunking | v_q_bispinor.py:336 + _slab_io_mpi_host.py:472 | deep failure | static |
| nb ~2,929 | full Sigma_c host buffer `n_om*nk*nb^2*16` | gw/ppm_sigma.py:1023 | SILENT OOM | static (= D2) |
| deck-dep | pivoted-Cholesky 50%-of-basis guard | pivoted_cholesky.py:348 | loud | static |

**Two verified findings worth acting on:**

1. **The 128 MB collective bound is STALE, and its floor is silent.** Measured
   `max collective/exec` on my runs: 97.3 MB (mu 6947, under) -> 926.0 MB
   (10015) -> 452.4 MB (15007) -> **1773.2 MB (24933) = 13.2x the bound**, all
   surviving. core.py:1783 records 1.15 GB as *measured-fatal* — but that was
   job 7876062 on **Gloo**; this campaign runs MPI/mlx. So the bound was
   calibrated for a transport production no longer uses. Fix = re-derive it for
   MPI/mlx and make the floor LOUD when it cannot honour the bound; do NOT
   lower mu on account of it.
2. **Sigma(omega) already requests a 10.25 GiB HDF5 chunk at nb=1024** — 2.6x
   the hard 4 GiB `H5Pset_chunk` limit. My runs survive ONLY because
   `_slab_io_ffi.py:620` no-ops `chunks` (128 such warnings in rung 5's log
   while writing a 22.6 GB sigma_mnk.h5). `_slab_io_mpi_host` and
   `_slab_io_allgather` honour it. **Same run, same shape: succeeds under
   PHDF5_FFI, refuses under PHDF5_HOST.** Release-relevant.

**Behaviour change my R15.2 fix introduces** (must be in the merge note): the
prune window is now the full WFN window, so `max_band = nb_total` instead of 52,
and a deck with `nb_total > 0.5*ngkmax*nspinor` (e.g. nb_total=2000 at 30 Ry,
ceiling 1964) will now hit the pivoted-Cholesky 50%-of-basis refusal where it
previously — wrongly — succeeded. The refusal is CORRECT; it is still a change.

### Revised mu-axis picture after the R15.2 capacity fix

    mu  8,192   collective-chunk floor crossed   SILENT, non-fatal on mpi/mlx (stale bound)
    mu 13,258   H5PY_ALLGATHER 90 GB/rank        only if that backend is selected
    mu 16,384   replicated-factor refusal        **REMOVED**
    mu 23,177   >2 GB single H5Dwrite            mpi_host writer only
    mu ~34,600  r-chunk performance floor (R8)   the real memory wall
    mu  46,080  n_rtot                           hard axis ceiling

## ***RESOLVED: the prune-window clamp WAS the rung-5 defect*** (job 7879438)

(nb=1024, mu=10015) rebuilt with ONLY the ISDF prune window widened
(`--prune-n-cond 230`: `right=(0,52)` -> `(0,256)`), everything else identical:

| | eqp0 | eqp1 |
|---|---|---|
| rung 5 baseline | **0.3645** | **-0.3639** |
| **prune window widened** | **3.1350** | **3.0710** |
| healthy family (rungs 1-4) | 3.22 - 3.63 | 2.98 - 3.29 |

Controls: same WFN, same 10,015 centroids, same weapons, same `rcond=1.0e-08`
(read from the run's own `[zeta rank_truncate]` line), same `n_pad=10048`;
`gw.in` differs only in `centroids_file`. Bare Sigma_X unchanged (-40.5368 vs
-40.5358). **COMPLETED 27:51.**

### This CORRECTS my earlier emphasis — and the correction is important

I reported rank saturation (the owner's conditioning hypothesis) as the
mechanism. **It was not.** The two numbers, read correctly:

    prune fix moves the FIT-Gram retained rank by only +1.4% (6700 -> 6793)
    prune fix moves the QP gap from 0.3645 eV to 3.1350 eV

**The operative quantity is WHICH centroids were selected, not HOW MANY
directions the fit Gram retains.** Rank COUNT is not basis QUALITY. Rank
saturation is a real, separate method limit — but the saturation table was
taken entirely on rank-deficient bases and must be re-measured with a correct
prune window before it is published as a "max usable nb per cutoff" rule.
`docs/dev/isdf_rank_saturation_and_max_usable_nb.md` needs its §2.1/§2.2 roles
swapped and its central claim softened accordingly — flagged, not yet done.

### Release-blocker statement, now earned

> **ISDF centroid selection silently ignored the deck's conduction window, so
> every GW result with `ncond >> nval` rests on a basis chosen for a different,
> much smaller problem. At nb=1024 that produced a QP gap of 0.36 eV against a
> true ~3.1 eV — a ~2.8 eV error — while passing every gate in the suite.**

Fix + the rank refusal that makes it un-repeatable: wt-RELC /
`wk_REL/docs/patches/wsREL-isdf-window.patch` (NOT committed). Dose-response completion in
flight: job 7879441 (`--prune-n-cond 998`, full window).

## 2x2 COMPLETE — axis A fixes it, axis B is catastrophic

nb=1024, mu 10015-10037, n_pad=10048, same WFN, weapons ON:

| cell | prune window | rcond | fit n_keep | eqp0 | eqp1 |
|---|---|---|---|---|---|
| baseline | (0,52) | 1e-8 | 6700 | 0.3645 | -0.3639 |
| **A1** | (0,256) | 1e-8 | 6793 | **3.1350** | **3.0710** |
| **A2** | (0,1024) | 1e-8 | 6788 | **3.7227** | **3.4551** |
| B1 | (0,52) | 1e-10 | 8290 | **-206.83** | **-1039.84** |
| B2 | (0,52) | 1e-12 | 9461 | **-5049.59** | **-304.20** |

**Axis B is not "worse", it is destroyed** — hundreds to thousands of eV on a
2.2 eV DFT gap. My earlier reasoning (that truncation was discarding real
content six decades above the f64 noise floor, so rcond should be sized against
that floor) was **exactly backwards**: retaining +41% more rank moves the answer
from wrong-by-2.8-eV to wrong-by-5000-eV. Those modes are the over-complete
near-null directions whose pseudo-inverse amplifies noise by 1/lambda;
truncating them is the cure. `zeta_rcond = 1e-8` is doing its measured job.

> **Retained rank is not basis quality — in EITHER direction.**
> Axis A: +1.4% rank -> +2.8 eV of correctness.
> Axis B: +41% rank -> -5000 eV of correctness.
> What matters is WHICH directions the basis spans, never how many.

Had `zeta_rcond` been lowered on my noise-floor argument, a 5000 eV error would
have shipped behind a plausible rationale. The rule against silently changing
physics defaults, plus reading the config's own recorded measurement history,
are what stopped it.

## CONSOLIDATED SET — replayed onto 1a52d51, ready to commit

Worktree **/work2/08271/jackmc/frontera/wt-RELC**, branch
**wsREL-isdf-window**, now based on **1a52d51**. Replay was clean (the three new
commits touch a disjoint file set — verified with `comm`). HEAD unmoved;
nothing committed. `py_compile` PASS on all three sources on the new base.

    docs/dev/isdf_basis_adequacy_at_large_nband.md | 236 +++   (new, RENAMED)
    src/centroid/kmeans_cli.py                     |  82 ++-
    src/gw/isdf_fitting.py                         |  25 +-
    src/isdf/core.py                               |  18 +-
    4 files changed, 348 insertions(+), 13 deletions(-)

### Commit split as requested

**(a) RELEASE BLOCKER — centroid selection fix + rank gate**
    src/centroid/kmeans_cli.py   (only)
Evidence: 0.3645 -> 3.1350 -> 3.7227 eV monotone in prune width at identical
WFN/centroid-count/weapons/rcond (jobs 7879295 / 7879438 / 7879441); selection
rank 630/897 -> 897/897 (jobs 7879286 / 7879432 / 7879433); rank gate
self-tested FAIL(630/897)/PASS(897/897).
**Behaviour change for the commit note**: the wider default prune window makes
`max_band = nbands`, so decks with `nbands > 0.5*ngkmax*nspinor` will now hit
the pivoted-Cholesky 50%-of-basis refusal where they previously (wrongly)
succeeded. The refusal is correct; it is still a change. This deck: ceiling
1964 at 30 Ry, 3597 at 45 Ry — nb=1024 and 2048 both clear it.

**(b) CAPACITY — premature-refusal fix, ~2.1x usable mu**
    src/gw/isdf_fitting.py  +  src/isdf/core.py
Evidence: the replicated-factor capacity check gated a route the next statement
discards (rungs 7/8 both print `path=distributed_rank_truncate`, so the
replicated factor was never used). Gate job 7879469: mu=24933 with NO ZCAP ran
past the 2m51s point where the identical config refused (job 7879369), same
planner block. Re-gate on the rebased base in flight (job 7879487).
Ceiling moves 16,384 -> the next real constraint.

**(c) DOC**
    docs/dev/isdf_basis_adequacy_at_large_nband.md
Rewritten per my own correction and **renamed** (the old title asserted the
withdrawn conclusion). Selection bug is now the mechanism (Sec 2); rank
saturation is demoted to Sec 3 behind an explicit caveat that **every row was
measured on a rank-deficient basis**; the "max usable nb per cutoff" rule is
**withdrawn in the text**, not merely softened; the `N_mu ~ 6-14x` heuristic is
described as "not a check" rather than refuted.

> **CAVEAT on the mu frontier, and it downgrades an earlier claim of mine.**
> Rungs 7 (mu=24933) and 8 (32059) HUNG — all ranks exited, srun never
> returned, logs frozen 10 and 59 min — and BOTH had `sharded` ON. So D1 is not
> specific to `replicated`, and my "the 128 MB collective bound is stale and
> non-fatal" verdict was wrong: 1386.1 MB completes (mu 15007), 1773.2 MB hangs
> (mu 24933). The usable mu ceiling on the distributed path is bracketed
> **15,007 - 24,933**, BELOW the memory wall. The capacity fix is still real and
> still worth landing, but "~2.1x usable mu" is a statement about the removed
> REFUSAL, not a demonstrated end-to-end run at mu>16,384.

## ⚠⚠ RETRACTION — THE COLLECTIVE-PAYLOAD WALL DOES NOT EXIST (my parsing bug)

**Do not pursue the collective-payload target.** It was created entirely by a
field-index error in my own liveness check. The sampler writes

    1785312831 VmHWM: 74961688 kB VmRSS: 25533656 kB
    $1         $2     $3       $4 $5     $6       $7

and I read `$5` as the VmRSS VALUE. `$5` is the string `"VmRSS:"`, which awk
evaluates as **0**. Every run reported "VmRSS = 0.00 GB" and I called them dead.
Re-read with `$6`: DIAG noweap **16.8 GB**, DIAG noshard **25.3 GB**, rung 7
**24.4 GB**, rung 8 **28.1 GB** resident. All four alive. **I cancelled four
healthy 32-node runs.**

Positive proof for rung 7, from its own log AFTER my scancel landed at 03:14:04:

    [stage 03:15:01]  <- W[probe] Dyson solve (16 q, mu=24933, full BZ)  735.6 s
    [stage 03:15:01]  -> W[probe] finiteness + hermiticity gate

It finished a 735.6 s solve and advanced while being torn down. That stage emits
nothing for ~12 min; I read silence as death.

### VOID

- **D1** (`sigma_omega_layout=replicated` hangs at nb>=1024) — VOID. "Sharded is
  mandatory at nb>=1024" is **WITHDRAWN**; never demonstrated.
- **The 1386/1773 MB fatal ceiling** and **"usable mu bracketed 15,007-24,933"**
  — VOID.
- My "correction" to CAP A was itself wrong. Standing truth: the 128 MB bound IS
  exceeded (up to 13x, `q_block=1` already, chunker out of axis) with **no
  observed ill effect**. Mis-calibration + silent floor, not a wall.

### SURVIVES (all from COMPLETED runs or static code reading)

Prune-window bug + fix (fe00242) · rcond 2x2 · capacity/premature-refusal fix ·
Sigma reference gate · the static cap findings (HDF5 4 GiB chunk, allgather
replication, >2 GB H5Dwrite, planner blindness, restart mu^2).

### Lesson recorded

`VmRSS` printing as exactly `0.00 GB` for every rank of four independent jobs
should itself have been the tell. **A diagnostic reporting a suspiciously clean
value everywhere is more likely broken than the system it measures.** Correct
fields: `$3` VmHWM, `$6` VmRSS.

## Capacity: (b) and (c) ready; item 4 in flight

- **(b) re-gate on 1a52d51 PASSED** — job 7879487, mu=24933, no ZCAP: no
  refusal, `path=distributed_rank_truncate`. Commit-ready.
- **(c) doc corrected** — the hang bullet is deleted from Sec 7 and replaced
  with an honest statement (bound exceeded up to 13x at `q_block=1`, **no
  failure attributed**). Commit-ready.
- **Item 4 needs no new job**: 7879469 and 7879487 ARE mu=24933 > 16,384, no
  ZCAP, patched. Both healthy and progressing (71.5 / 40.2 GB VmHWM;
  31.1 / 19.1 GB VmRSS). Watcher armed on PROGRESSION only — it never infers
  death from silence, and it will not cancel anything.

## LOUD FLOOR (approved) + MEASUREMENT-INFRASTRUCTURE AUDIT

### The loud floor is in, and building its gate corrected the finding

`isdf/core.py::_chunk_log`, placed BEFORE the `LORRAX_COLLECTIVE_CHUNK_LOG`
early-return (a logging knob must not silence a bound violation), own dedup key.
py_compile PASS. Gate: b256 reference, mu=2475, only the budget moves —
**silent side (537 MB budget) warnings=0; firing side (134 MB) warnings=1.**

**The first gate design was wrong and the correction ENLARGES the finding.** The
two call sites have different payload laws:

| site | per-q payload | breaches 134.2 MB above |
|---|---|---|
| C+ formation (pinv) | `mu^2*16/Px` | mu ~ **8,192** |
| C+ back-solve (GEMM) | `(mu^2/Px + mu*r_chunk/Py)*16` — **linear in mu** | mu ~ **1,456** |

At the campaign's smallest deck (mu=2475) pinv sits at 12.5 MB/q while the
back-solve already emits **230.0 MB/q**. So the advertised bound has been
silently abandoned at that site by **essentially every run in this project's
history** — including AQ rung 0 and the b256 config that is now the pinned Sigma
reference — not merely above mu>8192 as I reported. The message no longer quotes
one global threshold. Still: no failure attributed to the violation.

### Audit findings so far — 4 defects, 3 of them in gates I wrote

| # | file | category | failure mode | loud? | blast |
|---|---|---|---|---|---|
| 1 | `b256_verify.py:18` | (a)(c) | `GRID=(24,24,80)` hard-coded; rejected 100% of a 45 Ry run for 95 min | SILENT (looks like a real reject) | **HIGH** |
| 2 | rank-gate wiring in 4 `deck_*.sbatch` | (d) | `$0.$SLURM_JOB_ID.out` never exists → primary path always failed under `2>/dev/null`; rc echoed then discarded by `exit 0` | SILENT | **HIGH** |
| 3 | `centroid_rank_gate.sh` | (b)(c) | `tail -1` of target and of rank taken INDEPENDENTLY → on multi-attempt logs it paired target 1630 (in-progress) with rank 597 (completed) | SILENT | **HIGH** |
| 4 | `qpgap.sh` | (a) | `nval` defaults to 26 unannounced; wrong nval yields a plausible gap between the wrong band pair | SILENT | MEDIUM |

All four FIXED and regression-tested. #1 confirmed decisively: the *identical
file* `centroids_frac_12979_r45_b2048_c13000.txt` that was REJECTED is now
ACCEPTED with the grid passed — the sets were always correct, the gate was the
defect. #3 verified three ways (r45 multi-attempt pairs correctly now; 30 Ry
defective still FAIL 630/897; 30 Ry fixed still PASS 897/897).

**Counter-example worth recording**: the l5/l7 harnesses' own VmHWM summary is
LABEL-keyed (`if($i=="VmHWM:") v=$(i+1)`), immune to the column shift that broke
my ad-hoc check. So every VmHWM number reported from a job `.out` this campaign
is sound, and the fragile parser was one I hand-rolled *next to a correct one I
could have reused*. Lesson: **when a harness already parses a log correctly,
call it rather than writing a second parser.**

### A physics finding fell out of the fixed gate: 45 Ry is WORSE

`c13000` on the 45 Ry b2048 deck (pinned src, old clamped window):
**rank 597 / 1166 = 51.2%**, against 70.2% (630/897) on 30 Ry b1024. Same
`26 x 52` clamped window on both, but the 45 Ry sigma window is 2048 wide vs
1024 — half as representative, half the certified directions. **An independent
deck confirming the release-blocker at a different cutoff, band count and FFT
grid.** Every 45 Ry centroid set built so far is rank-deficient; none has been
used in a GW run, so nothing is invalidated. Rebuild with the fixed
`kmeans_cli` is in flight (job 7879527, via a new `SRCDIR` hook).

## HARNESS AUDIT — 21 HIGH findings; the campaign's gates lied more than the code

Full ranked list: ladder notes R24-R27 + the audit output. Headline: **across
this campaign the measurement infrastructure produced more false results than
the system under test.** Every HIGH finding fails SILENTLY.

### THE DEFECT IN ONE LINE (found by fixing audit #4)

Rewriting the rank gate to check EVERY attempt gated the whole `cbig` log:

| set | requested | achieved | % of request |
|---|---|---|---|
| c10000 | 897 | 630 | 70.2% |
| c15000 | 1357 | 860 | 63.4% |
| c20000 | 1824 | 1009 | 55.3% |
| c25000 | 2289 | 1089 | 47.6% |
| c30000 | 2769 | 1138 | 41.1% |

**A `(0,26) x (0,52)` prune window contains at most `26*52 = 1352` independent
pair densities, so the achievable rank is MATHEMATICALLY CAPPED at ~1352** — the
measured ranks asymptote to it (630/860/1009/1089/1138) and the deficiency
deepens with mu because the request grows while the ceiling does not. That one
sentence explains the nb-dependence, the mu-dependence that made it look like a
conditioning problem, and why the fix works instantly (`(0,256)` offers 6,656).

**SCOPE: every big-mu set on the 30 Ry b1024 deck is rank-deficient.** Rungs 6-8
and both capacity runs use them. Capacity/memory results are unaffected (VmHWM,
planner agreement, the premature-refusal fix do not depend on basis quality) but
**no PHYSICS number from any big-mu rung is usable. Rung 6's 1.4296 eV is
withdrawn as a datum** — it is a symptom, not a measurement.

### Fixed this pass (all mine, all regression-tested)

| # | file | defect | was |
|---|---|---|---|
| 3 | `centroid_rank_gate.sh` | hard-coded char offset `RSTART+7`; a wording change → REQ non-numeric → FLOOR 0 → **PASS forever** | **FAIL-OPEN** |
| 4 | same | gated only the LAST attempt; harnesses ship every set → 3 of 4 ungated | SILENT |
| 17 | same | reads the job's own still-flushing stdout | SILENT |
| 9 | `l7`, `diag_b512_weap`, `gate_sigma_reference` | weapons banner was an unconditional literal: a `WEAPONS=off` run PRINTED "sharded FFT_FFI=1 …" — in the scripts written to attribute a regression to a weapon | SILENT |
| 7 | `gate_sigma_reference` | `timeout 27000` under `-t 01:00:00`; SLURM killed first so **the reference assertion never ran** on a slow job | SILENT |
| 6 | `gate_sigma_reference` | `CENTFILE` overridable but `REF0/REF1` pinned to c2475 → a different mu asserted against the wrong reference and reported as "Sigma_c is corrupted" | SILENT |
| 21 | `gate_sigma_reference` | ended on `echo`; SLURM recorded COMPLETED for a failed run, and the last line advertised the **gw** rc as if it were the gate's | SILENT |
| 1 | `b256_verify.py` | hard-coded `GRID` (fixed earlier) | SILENT |
| 19 | `qpgap.sh` | unannounced `nval=26` default (fixed earlier) | SILENT |

Rank gate regression-tested 4 ways: defective→FAIL rc=2; fixed→PASS rc=0;
multi-attempt→all 4 gated; corrupted target→**FAIL CLOSED rc=4** (the old gate
would have said PASS).

### Ranked, NOT yet fixed — for the orchestrator

**HIGH, silent:** `audit_pin.sbatch:78-83` and `audit_cpu_gemm.sbatch:99-106`
still carry the backgrounded-srun "concurrency" defect with **no retraction
marker** (retracted only in `audit_pin2.sbatch`'s header) and their logs are on
disk readable as good data · `h5_sigma_compare.py:57` uses `or` not `and`, so
the parity gate is the LOOSER of abs/rel and a file compared against itself
reports BIT-IDENTICAL · `colltable.py:34,79` treats an EMPTY dump dir as a clean
bill of health (rc=0), and `l7:262` discards its rc through a pipe —
`run_L7_b1024_10000/hlo_dump` has 0 modules today · `colltable.py:64` only flags
a trailing adjacent `(mu,mu)` pair · `l7:38` TAG collision (my prune-fix legs
were safe only because I passed `TAG=` explicitly) · `l7:128-142` injects
`sigma_omega_layout`/`zeta_rcond`/`zeta_ridge`/`r_chunk_size` **without**
asserting them, though `l5`/`l6` do assert · `qe_b1024.sbatch` has no stop
condition at all · all ladder harnesses `exit 0` · `gbp_inner_par{xla,auto}.sh`
read the LIVE tree the l7 header warns about.

**MEDIUM:** `el_compare_b512/b1024.py` hard-code the reference band count and
**do not check `ngk`** (the r45 version does both, correctly) and neither
verifies the two paths differ — in a symlink-heavy directory that makes a
provenance gate compare a file with itself and report `max|d| = 0.000e+00 PASS`
· `cbands_ab.sbatch:204` takes performance numbers from `awk $3` on an
unanchored `head -1` · `check_channel_hermiticity.py:79` returns 0 for an
all-zero kernel (vacuous PASS) · `zproj_refuse.py:26` accepts a refusal that
fired for the wrong reason (`"multiple of"` is a substring of another message).

**Clean, on the record:** `dlsym_control.py` (explicit negative AND positive
controls, ANDed, real exit code) · `el_compare_r45_b2048.py` · `audit_pin2.sbatch`
· `l6` assertion list · `qe_r45_b2048.sbatch:96` hard stop · **and the VmHWM
parsing campaign-wide is label-anchored at all 17 call sites** — no survivors of
the positional bug, so every reported memory number stands.

# ============================================================

> ⚠⚠ CLAIM-DECAY (2026-07-29, R32) — **THE "object" AND "scaling law" SECTIONS
> BELOW ARE WRONG.** The object was located in the run's own HLO dump:
> `module_0914.jit__gn_ppm_fit_kernel`, `allocation 35: size 74.27GiB,
> preallocated-temp` = 79,744,204,800 bytes exactly.
>
> * It is a **TEMP ARENA**, not a single array, and **not** `(n_tau, mu, mu)`.
>   The byte count was degenerate across three shapes and I picked the wrong one.
> * The kernel's parameters and outputs are **PROPERLY SHARDED**:
>   `c128[16,3120,3120]` = 2.32 GiB, with `3120 = 24960/8 = mu_pad/p_x`.
>   The global object would be 159.5 GB and is never allocated.
> * arena = **32.0 x one sharded tile** = `32*nq*mu^2*16 / P` — **it DOES scale
>   with P** (74.3 GiB at P=64, 37.1 at P=128, 18.6 at P=256).
> * Cause: ~111 full-tile elementwise instructions with only 6 fusions — an
>   UNFUSED chain, i.e. a code-generation/placement problem.
>
> So "whole-object materialisation on one rank", "NOT divided by P" and
> "adding RANKS will not move it" are all WITHDRAWN. **The measured wall
> (mu=24,933 dies at P=64, reproduced byte-identically by job 7879487) STANDS;
> only my explanation of it was wrong, and the fix space is much larger than
> reported.**

# CERTIFIED FRONTIER — 30 Ry MoS2 4x4, nb=1024, P=64 (8x8), 32 nodes
# ============================================================
# Job 7879469. This is a CAPACITY result only — see "Physics" below.

## The wall

    mu = 24,933  DIES.   mu = 15,007 completes.   Usable mu < 24,933 at P=64.

    [gw rc=134 wall=4387s]                134 = 128+6 = SIGABRT
    sacct 7879469.0  FAILED  ExitCode 6:0  MaxRSS 103,894,092 kB
    MAX VmHWM across ranks  99.05 GiB = 106.35 GB
    available               186 GiB/node / 2 ranks = 93.0 GiB/rank

**CLASSIFICATION: XLA allocator OOM.** Verbatim on 60+ of 64 ranks:
`INTERNAL: Buffer Definition Event: ... Out of memory allocating 79744204800
bytes.` Not the OOM-killer (no cgroup/slurmstepd kill line). Not a LORRAX
refusal — the `LORRAX FAIL-FAST` lines are LORRAX *reporting* the JAX error,
which is its designed behaviour. Not QE.

## The object, exactly

    79,744,204,800 B = 74.27 GiB = 79.74 GB in ONE allocation
      / 16 (complex128)          = 4,984,012,800 elements
      solved: 8 x 24,960^2 x 16  = 79,744,204,800     EXACT

    shape = (n_tau, mu_pad, mu_pad) complex128
      n_tau   = 8       the W[probe] tau-node count, confirmed in this run's
                        own log: "W[probe] chi0 build (8 tau nodes, 16 q,
                        mu=24933)"  (W[static] uses 11)
      mu_pad  = 24,960 = 390 x 64   (mu = 24,933 padded to the mesh)

Died between `[stage 04:09:55] <- W[probe] finiteness + hermiticity gate` and
04:14:08 — i.e. in the CONSUMER of the probe W, not in its construction (the
build itself completed in 123.9 s and the Dyson solve in 724.7 s).

## Scaling law and what is / is not P-scalable

    bytes_per_rank = n_tau * mu_pad^2 * 16        NOT divided by P

That is the whole point: **79.74 GB is the FULL object, not a shard.** Sharded
over P=64 it would be 1.25 GB/rank. So this allocation is a whole-object
materialisation on one rank, exactly the class of D2 (the sigma omega cube) and
of the pre-fix centroid Gram — an object that should be sharded or streamed and
is not.

    mu ceiling at n_tau=8, 93 GiB/rank, ignoring the rest of the working set:
        mu_max = sqrt(93*2^30 / (8*16)) = 26,900
    measured death at mu=24,933 -> consistent once the ~20 GiB of other
    resident state is counted.

## The knob — and a correction to the proposed one

**It is NOT `q_block`, and `_chunk_q` is not involved.** Checked directly: at
mu_pad=24,960 one `(mu,mu)` slice is 9.28 GiB, so EVERY chunker in the module
returns 1, not 8 —

    _collective_chunk_bytes (128 MB)     -> max(1, 134217728 // 9968025600) = 1
    _ZETA_GATHER_MAX_BYTES (4 GiB)       -> 1
    _REPLICATED_FACTOR_MAX_BATCH (4 GiB) -> 1

The 8 is the minimax tau-node count of the W probe. So the hoped-for
"q_block=1 gives 8x headroom, mu ~ 76,000" does not apply: there is no q_block
here to force down.

Knobs that DO shrink this object, and why neither is a capacity lever:
- `minimax_target_error` (1e-6) / `minimax_max_nodes` — fewer tau nodes means a
  less accurate screening quadrature. **Physics-affecting; not to be changed
  for capacity**, per the standing rule that has already saved this campaign
  once (the zeta_rcond episode, where "more retained rank" cost 5000 eV).
- Nothing else exposed. The real fix is STRUCTURAL and movement-only: keep the
  probe W sharded through its consumer, or stream the consumer over tau. That
  is a code change, not a knob, and I am not building it on speculation.

**Honest status of the frontier**: `usable mu < 24,933 at P=64, nb=1024`, set by
a whole-object `(n_tau, mu, mu)` materialisation with no P scaling. It is a
budgetary wall in the sense that more memory per rank would move it, and an
architectural one in the sense that adding RANKS will not.

## Physics: this is a capacity result ONLY

**No physics number from any big-mu rung is usable.** Every big-mu centroid set
on this deck is rank-deficient (c15000 63%, c20000 55%, c25000 48%, c30000 41%)
because the ISDF prune window was clamped — the release blocker, fixed but not
re-run at these sizes. The OOM is unaffected by that (memory does not depend on
basis quality), but nothing about eqp/gaps from mu > 10,015 should be quoted.

## Independent control

Job 7879487 — same configuration, same mu, rebased source — was still running
at 1:01 when this was written. If it OOMs at the same allocation size, the wall
is reproducible; if it does not, this entry needs revisiting.
