# Agent D — HLO calibration of `pair_density_slots` and `fft_box_factor_D`

**Date:** 2026-05-17
**Allocation:** JID 53075115 (4 nodes, 16 GPUs hbm80g)
**Config:** CrI3 6×6×1 80 Ry SOC bispinor, 16 GPUs, 4×4 mesh.
- `nk=nq=36, ns=4 (bispinor), nb=150, n_rmu_charge=1520 (padded), n_rmu_transverse=1504, ngkmax=59990, n_rtot=1,125,000, fft_grid=(75,75,200), p_x=p_y=4, p_xy=16`.

Per-rank shapes derived from the buffer layout: `mu_loc = mu_padded / p_x`, `r_loc = r_chunk / p_y` (NOT `/ p_xy`).

Run dirs (HLO dumps preserved):
- M1 / M2: `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_D_bispinor_hlo_2026-05-17/`
- M3: `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_D_bispinor_hlo_gflat1_2026-05-17/`
- M4: `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_2026-05-13/` (pre-existing).

---

## M1 — `pair_density_slots`, bispinor 80Ry, r=24576, b=32

**Cohsex.in:** `bispinor=true, r_chunk=24576, band_chunk=32, gflat_chunk_size=360, mu_charge=1520, mu_transverse=1504, memory_per_device_gb=70.0`.

**Planner pick at run-time:**
- `band_chunk = 32, r_chunk = 24576 (46 chunks), gflat_chunk_size = 1409` (planner-suggested; runtime uses cohsex.in override = 360).
- HWM estimate = **65.99 GB/dev (94% of budget) [bottleneck: C_fit_one_rchunk]**.

### Per-channel fit_one_rchunk fused kernels

Four bispinor channels each produce a `module_NNNN.jit_fn.*-memory-usage-report.txt`. All have the same slot structure (only mu differs: charge=1520, transverse=1504).

**Charge channel — `module_0438.jit_fn.*-memory-usage-report.txt`:**

```
Total: 61.77 GiB
  allocation 12: 60.12 GiB, preallocated-temp
    offset       1536  →  20.04 GiB:  4×c128[36,4,6144,380,4]  + c128[36,24576,1520] + c128[4,6144,380,4,6,6,1] + c128[6144,380,6,6,1]
    offset 21516781056  →  20.04 GiB:  4×c128[36,4,6144,380,4]  + c128[36,24576,1520] + c128[4,6144,380,4,6,6,1]
    offset 43033560576  →  20.04 GiB:  2×c128[4,2334720,4,36]   + c128[36,2,4,75,75,200] + c128[36,32,4,24576] + ...
    offset 48217560576  →   4.83 GiB:  c128[36,8,1125000]
    + smaller (FFT phase tables, indices, etc.)
  allocation 0: 1.25 GiB, output c128[36,380,6144]  (= zeta_chunk per-rank)
  allocation 1-3: 154+133+125 MiB, parameters (sphere indices, psi_l/r centroids)
```

**Pair-density buffer shape:** `c128[36, 4, 6144, 380, 4]` = `c128[nk, ns, r_loc, mu_loc, ns]` (bispinor ns=4).

Per-buffer bytes: `36 × 4 × 6144 × 380 × 4 × 16 = 21,495,705,600 B = 20.02 GiB` ✓ matches slot size.

Equivalently: `nk × ns² × r_chunk × mu_padded × 16 / p_xy = 36 × 16 × 24576 × 1520 × 16 / 16 = 21.50 GB`.

**Slot count of pair-density-shaped buffers = 3** (offsets 1536, 21516781056, 43033560576). Each slot also aliases the rank-7 P_l_3d (post-IFFT reshape), the bispinor FFT box `c128[36,2,4,75,75,200]`, and the zeta_chunk output `c128[36,24576,1520]` — but the slot size is set by the LARGEST aliased variant, which is the pair-density at 20.04 GiB.

**Transverse channels — `module_{0521,0523,...0969}.jit_fn`:** identical structure, slot size 19.83 GiB (mu=1504, mu_loc=376, otherwise identical).

### **M1 RESULT**

`pair_density_slots = 3` (empirical, exact match to planner default). Per-slot 20.04 GiB (charge) / 19.83 GiB (transverse), exactly matching `_bytes_c128(nk, ns, ns, mu, r_chunk, shard=p_xy)`. Slot lifetimes alias the bispinor FFT box, zeta_chunk, and post-IFFT P_l_3d at the same offset — matches the docstring's "fit in same lifetime slots" claim. Planner formula `3 × 36 × 16 × 1520 × 24576 / 16 × 16 = 64.5 GB` matches `3 × 20.04 GiB = 60.12 GiB` total preallocated-temp.

---

## M2 — `fft_box_factor_D`, same run, accumulate_rchunk_to_gflat

**Module 0474** = `accumulate_rchunk_to_gflat._kernel` at `gflat_chunk_size = 360` (user override). 

```
Total: 20.92 GiB
  allocation 7: 16.61 GiB, preallocated-temp
    offset          384  →   6.03 GiB:  c128[360, 75, 75, 200]                    (FFT box, spatial)
    offset 6480000384  →   6.03 GiB:  2×c128[360, 1125000]                       (FFT box, flat in+out aliased)
    offset 12960000384  →   3.22 GiB:  5×c128[3600, 59990]                        (gflat_acc + scan carry, padded N=3600)
    offset 16415424384  →   1.32 GiB:  c128[3600, 24576]                          (zeta_chunk slab input, padded N=3600)
    + indices, sphere tables (≤ 0.5 MiB)
  allocation 0: 3.06 GiB, parameter 1 / output c128[36,95,59990]                  (gflat_acc reshape)
  allocation 1: 1.25 GiB, parameter 0 c128[36,95,24576]                           (zeta_chunk input)
```

Bare 1× FFT box at cs=360: `360 × n_rtot × 16 B = 360 × 1,125,000 × 16 = 6,480,000,000 B = 6.04 GiB` ✓.

**Two FFT-box-shaped slots:** spatial reshape (`c128[cs, nx, ny, nz]`) + flat (`c128[cs, n_rtot]`). cuFFT's out-of-place 3D FFT needs both. No separate "cuFFT scratch" slot is allocated — XLA's planner folds the workspace into the two box-sized slots.

### **M2 RESULT**

| Quantity | Empirical | Agent A audit | Planner default (B's branch) | Match |
|---|---|---|---|---|
| FFT-box slot count | **2** (spatial + flat) | "needs ~2×, not 4×" | `fft_box_factor_D = 2.0` (lorrax_B unstaged edit) | ✓ exact |
| Per-slot bytes | 6.03 GiB | 6.48 GB (predicted) | 6.48 GB | ✓ exact |
| **Empirical factor_D** | **2.0** | recommend 2.0 | 2.0 | ✓ exact |
| Slot 3 (gflat_acc + scan carry) | 3.22 GiB | persistent term, 3.28 GB | persistent | ✓ |
| Slot 4 (zeta_chunk slab input) | 1.32 GiB | transient term, 1.16 GB | transient | ✓ |

Total Peak D = 2 × 6.03 + 3.22 + 1.32 + small ≈ **16.6 GiB** at cs=360. Planner's old formula with factor=4 predicted 4 × 6.48 + 3.28 + 1.16 = 30.4 GB. **Old planner over-predicted by 13.8 GB**; the refit with `factor_D=2` lands within ~5% of empirical.

---

## M3 — gflat_chunk_size = 1 sanity check

**Cohsex.in:** same as M1 but `gflat_chunk_size = 1`.

**Module 0363** = accumulate_rchunk_to_gflat at cs=1:

```
Total: 4.35 GiB
  allocation 7: 34.74 MiB, preallocated-temp     ← collapses by factor 480× vs cs=360
    offset       256  →  17.17 MiB:  2×c128[1, 1125000]                          (FFT flat in+out, aliased)
    offset 18000256  →  17.17 MiB:  c128[1, 75, 75, 200]                         (FFT spatial)
    + tiny indices
  allocation 0: 3.06 GiB, parameter 1 / output c128[36,95,59990]                  (gflat_acc — DONATED in place)
  allocation 1: 1.25 GiB, parameter 0 c128[36,95,24576]                           (zeta_chunk input)
```

Bare 1× FFT box at cs=1: `1 × 1,125,000 × 16 = 18 MB = 17.17 MiB` ✓.

**Two FFT-box slots × 17.17 MiB = 34.33 MiB. factor_D = 2.0** confirmed at cs=1 — same factor as cs=360.

### Empirical Peak D at gflat=1
- Persistent gflat_acc + zeta_chunk: 3.06 + 1.25 = **4.31 GiB** (now in parameter slots, no preallocated-temp)
- Transient (FFT-box family + indices): **34.74 MiB**
- **Total Peak D ≈ 4.35 GiB.**

Planner prediction at gflat_chunk_size=1 (with `factor_D = 2`):
- gflat_acc persistent: `36 × 1520 × 59990 × 16 / 16 = 3.28 GB`
- zeta_chunk transient: `36 × 1520 × 24576 × 16 / 16 = 1.34 GB`
- accumulate_fft_box: `1 × 1,125,000 × 16 × 2 = 36 MB`
- **Predicted total ≈ 4.66 GB ≈ 4.34 GiB**.

**Discrepancy: < 1%.** Confirms `factor_D=2.0` is the right model for accumulate FFT and the "one FFT-box minimum" is a real lower bound: at cs=1 you cannot shrink Peak D below ~4.3 GiB.

### Did the run finish?
**Yes — all 4 channels completed.** Per-channel zeta-fit wall at gflat=1 with `LORRAX_MAX_RCHUNKS=2`:
- charge: 24 s
- transverse 1: 21 s
- transverse 2: 19 s
- transverse 3: 19 s

Per-r-chunk wall is 19–26 s (vs M1's 20–25 s) — gflat=1 does NOT increase per-r-chunk wall (the bulk is `z_q_build + solve`, not the accumulate FFT). cuFFT plan amortizes over the 3420 scan iters per channel. **No OOM, no remat-blocked compile failure.** All 4 channels' accumulate jits have identical structure (2 FFT-box slots × 17.17 MiB at all four).

---

## M4 — non-bispinor 80Ry cross-check

**Existing dump:** `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_2026-05-13/xla_dump/module_0408.jit__kernel.*-memory-usage-report.txt`.

Cohsex: `bispinor=false, ns=2 (SOC), r=73328, b=16, gflat=64, mu=1504`.

Three pair-density slots at offsets 10112, 15881095040, 31762179968 — each 14.79 GiB — holding `c128[2, 6892832, 2, 36] = c128[ns, mu_loc·r_loc, ns, nk]` with **ns=2**. Per-buffer: `2 × 6892832 × 2 × 36 × 16 = 14.79 GiB ✓`. Equivalently `nk × ns² × r × mu × 16 / p_xy = 15.88 GB` matches.

**Total pair-density bytes = 3 × 14.79 = 44.37 GiB**, matching the earlier killed agent's claim exactly.

**M4 RESULT: `pair_density_slots = 3`, per-slot = 14.79 GiB** — same slot count as bispinor (M1) despite different ns. The planner's byte formula is correct because `meta.nspinor = 4 if bispinor else 2` (`wfn_loader.py:774`).

---

## Calibrated values + verdict

| Constant | Empirical (HLO) | Planner refit default (lorrax_B unstaged) | Old planner default | Verdict |
|---|---|---|---|---|
| `pair_density_slots_charge` | **3** | 3 | 3 | **right** |
| `pair_density_slots_transverse` | **3** | 3 | 3 | **right** |
| `fft_box_factor_D` | **2.0** | 2.0 | 4.0 | **refit is right; old default 2× too conservative** |
| `fft_box_factor_A` (not measured here) | — | 4.0 | 4.0 | unchanged |

The four numbers:

```
pair_density_slots_empirical_bispinor = 3
pair_density_slots_empirical_charge   = 3
fft_box_factor_D_empirical            = 2.0
fft_box_factor_A_empirical            = 4.0   (carry-over; not re-measured)
```

### One-line verdict

**The refit's `pair_density_slots=3` and `fft_box_factor_D=2.0` defaults match HLO reality exactly on production-scale bispinor 80Ry CrI3 — slot counts at both ns=4 (bispinor) and ns=2 (non-bispinor) are 3 with no discrepancy, and the accumulate FFT scratch is 2× the bare box across cs=1 → cs=360.**

---

## Notes / discrepancies vs Agent A audit

1. **No ns² bug**: planner formula `_bytes_c128(nk, ns, ns, mu, r_chunk, shard=p_xy)` is already correct for bispinor because `meta.nspinor=4` for bispinor (see `wfn_loader.py:774`).
2. **Slot aliasing breadth**: each pair-density slot also holds the rank-7 P_l_3d, the bispinor FFT-box `c128[36,2,4,75,75,200]`, the zeta_chunk output `c128[36,24576,1520]`, and the rank-4 reshape. Slot size is set by the largest (pair-density at 20 GiB) — confirming the docstring's "pair-density dominates" claim.
3. **Slot count is invariant under `bispinor`, `gflat_chunk_size`, `band_chunk`**: charge + transverse channels (M1), gflat=360 vs gflat=1 (M3), bispinor vs non-bispinor (M4) all show 3 pair-density slots. The slot count depends only on the algebraic structure of fit_one_rchunk (Round-6 scan-INSIDE-shard_map).

Slot lifetimes (per `sources/lorrax_B/src/common/isdf_fitting.py`): P_l_acc (line 625), P_r_acc (line 627), post-scan reshape work (lines 713–720, `P_l_3d → P_l_R → P_l_R_conj`).
