# Agent C — FFI / all-P-shard audit (4-GPU vs 16-GPU device invariance)

**Date:** 2026-07-08 · **Checkout:** `sources/lorrax_D` @ `agent/memplanner-cleanup` (bb95bc3, unmodified)
**Scope:** every P-dependent code path OTHER than μ-padding: cuSolverMp FFI (2×2 vs 4×4 grid),
solver-kind dispatch, staged reshards / W-solve / all-P q-k shards, phdf5 valid_shape, mesh-shape branches.

## Executive verdict

**The FFI paths are exonerated — measured, not just audited.** No P-dependent algorithm switch
exists between P=4 and P=16 anywhere in the audited surface. The eV-scale eqp divergence was
**localized in both systems** as a by-product of the exoneration measurements:

| System | Where the divergence enters | Where it does NOT |
|---|---|---|
| A_charge (GN-PPM) | **PPM Σ_c stage only**: pad modes enter the PPM mode census at P=16 (n_total_modes = 16·1216² vs 16·1204²), 2 real modes flip valid↔invalid, adaptive minimax node counts change (15→13, 15→14) | Σ_X, V_H **identical to 1e-6 eV** across P → ζ(charge), V_q, unfold all clean |
| B_bispinor (COHSEX) | **Transverse ζ solve output**: ζ^{γ̃1,2,3} at 16 GPU differ from 4 GPU by rel. 0.9–546 (γ̃² worst, max abs 3.3e6); Σ^B tile (2,2) trace −0.153 eV (4g) vs **−117.9 eV** (16g) | Σ_SX, Σ_COH, V_H diag **identical to 1e-6 eV**; charge ζ rel diff 2.4e-7 |

Both loci correlate exactly with the μ-pad asymmetry (charge 1204→1216 only in the PPM-side
arrays; transverse 668→672) — **consistent with, and handing off to, the padding lead.**

## Decisive measurements

### 1. Solver-swap discriminator (bispinor, strongest evidence)
Reran `head_16g` with `cusolvermp_lu = off` (legacy per-q `jnp.linalg.solve`, identical padding;
run dir `runs/MoS2/Z_memplanner_validation_2026-07-06/B_bispinor/head_16g_luoff`, `memory_per_device_gb`
lowered 30→12 because the planner sizes r_chunk for the cuSolverMp path and the legacy path OOMs).
All 4 ζ channels completed; compared ζ on disk (`reports/device_invariance_2026-07-08/zeta_compare.py`):

| pair | zeta_q (charge) | zeta_q_mu1 | zeta_q_mu2 | zeta_q_mu3 |
|---|---|---|---|---|
| 4g vs 16g (both cuSolverMp) | 2.4e-7 | **8.98e-1** | **5.46e+2** | **8.95e-1** |
| 4g vs 16g-luoff (different solvers) | 2.4e-7 | 8.98e-1 | 5.46e+2 | 8.95e-1 |
| **16g vs 16g-luoff (same P, different solver)** | **2.1e-16** | **2.6e-10** | **3.5e-7** | **2.5e-10** |

Two completely different solvers (cuSolverMp getrf/getrs on a 4×4 grid vs single-device
`jnp.linalg.solve`) produce the **same** wrong transverse ζ at 16 GPU. The corruption is in the
**padded transverse system being solved** (C_q^μ / Z_q^μ content, identity-pad + ridge at padded
extent), not in the solver. Note the transverse channels are the ones that acquire pad rows at
P=16 (668→672) while charge (640 % 16 = 0, pad-free) stays clean — exact correlation.

### 2. cuSolverMp 4×4-grid unit tests (first-ever 4×4 validation)
The library (0.7.2) had a documented history of grid-shape-dependent silent corruption
(0.6.0: getrf/getrs garbage on Px>1∧Py>1; potrs wrong when NRHS≤N on 2D grids; two
empirically-found non-determinism triggers in our handlers — shared getrf/getrs workspace,
cudaMallocAsync-pooled ipiv — see `src/ffi/cusolvermp/cpp/batched_solve_lu_ffi.cc` header).
Prior validation covered only 2×2 / 1×4 / 4×1. Ran on 16 GPUs (4 nodes):

- `common.cusolvermp_solve_lu_test --nbatch 9 -n 672 --nrhs 672 --mesh 4x4 --dtype c128`
  → residual **7.3e-16, PASS**.
- `potrs_4x4_check.py` (this report dir; standalone because the in-tree
  `cusolvermp_batched_test.py` crashes — it allgathers the donated A array):
  `-n 1216 --mrhs 608 --mesh 4x4` (the historically buggy NRHS<N regime)
  → residual **1.1e-15, PASS**.

### 3. Charge-system Σ decomposition across P (from existing artifacts)
`A_charge/head_{4g,16g}/sigma_diag.dat`, 1280 (k,n) records:
max|ΔΣ_X| = 0.0, max|ΔV_H| = 1e-6 eV, max|ΔRe Σ_c| = **5.665 eV**. Every record with
|ΔRe Σ_c| > 0.01 eV has |Im Σ_c| ≥ 2 eV (near-pole). ζ→V_q→Σ_X chain is P-invariant at
print precision; everything downstream of the PPM pole fit is not.

### 4. Charge-system PPM stage census (run logs)
| quantity | 4g (P=4, no μ-pad) | 16g (P=16, 1204→1216) |
|---|---|---|
| n_total_modes | 23,193,856 = 16·1204² | 23,658,496 = 16·**1216²** |
| GN invalid modes | 255,980 (1.10%) | 720,622 (3.05%) = 255,980 + 464,640 pad-modes **+ 2 flipped real modes** |
| minimax nodes, "b_slab" windows | 15 / 15 | **13 / 13** |
| minimax nodes, "single" windows | 15 / 15 | **14 / 14** |

The padded μ extent flows into the PPM B_q/Ω_q mode space at P=16 only. Even with pad modes
masked invalid, the **masked-Ω statistics feeding the adaptive minimax window fit change**
(`ppm_windows._build_windows_for_branch` → `_masked_stats_device` min/max/count →
different node counts) — a P-dependent *algorithm* change affecting Σ_c at every (k,n),
hugely amplified on near-pole bands. The 2 flipped real modes additionally change Σ_c
discretely (`ppm_invalid_mode` handling).

### 5. Bispinor Σ^B tile traces (run logs)
| tile | 4g | 16g |
|---|---|---|
| (1,1) | −0.152598 | −0.152495 |
| (2,2) | −0.152608 | **−117.914143** |
| (3,3) | −0.143666 | −0.143666 |
| (2,3)/(3,2) | −0.012362 | −0.010001 |

The bispinor eqp divergence (max 2.53 eV, eqp0) enters entirely through Σ^B: all
sigma_diag columns (Σ_SX/Σ_COH/V_H) are P-invariant to 1e-6 eV; all QP levels shift
systematically (degeneracies preserved) once the corrupted Σ^B is added to H_qp.

## Audit results (per directive item)

1. **cuSolverMp 2×2 vs 4×4 numerics** — Same solver kind selected at both P:
   `_resolve_solver_kind_{charge,transverse}` (`isdf/core.py:821-901`) gate only on
   "true 2D mesh" (px≥2 ∧ py≥2), true for both 2×2 and 4×4; confirmed in all 4 run logs
   (`path=cusolvermp_cholesky` / `path=cusolvermp_lu`, `grid: 2x2` / `4x4`, lib 0.7.2 both).
   Descriptor blocking is pure block distribution (mb=N/Px, nb=N/Py — no lcm/block-size
   branch on the FFI path; `cholesky_2d.py`'s J%Px/J%Py + lcm constraints are on the
   in-tree fallback, not taken here). NRHS pads to multiples of Py with zero columns,
   trimmed on return (`core.py:1190-1196,1220-1226`) — exact. Grid-order roundoff differs
   between 2×2 and 4×4 but measured end-to-end effect on charge ζ is 2.4e-7 rel. **Clean.**
   Unit tests at 4×4 pass at machine precision (measurement #2).
2. **Staged reshards / all-P shards** — W-solve backend is static (`auto` →
   `ScreeningSolver.JAX_NATIVE`, `gw_config.py:137-140` — legacy `low_mem` string is the only
   route to CUBLASMP_FFI; not used in these runs). JAX_NATIVE q-pad slices (nq=9→12 at P=4,
   →16 at P=16) solve A=I, B=0 → exact 0, sliced off (`w_isdf.py:239-263`). Per-q LU runs on
   full replicated (μ,μ) blocks — per-q bitwise P-independent given equal inputs. SC k-shard
   eigh not active (one-shot runs); Σ band psum_scatter divisibility satisfied at both P
   (nb=80, 30: %2 = %4 = 0). **No shard-count-dependent branch found.**
3. **phdf5 valid_shape** — write clips per-rank hyperslabs to the logical prefix
   (`write_ffi.cc:282-388`); read `memset`s the pinned buffer to zero before reading only
   the valid region (`read_ffi.cc:76`) → pad regions read back as exact zeros. **Clean.**
4. **Mesh-shape algorithm branches** — swept `process_count()/device_count()/mesh.shape`
   uses across `src/{gw,isdf,common,file_io}`: all are layout/padding math, P>1 gates
   (same at 4 and 16), or prints. `_select_accum_mode` (`ppm_accumulators.py:45-78`) gates
   on n_proc≠1 → KIJ_HOST at both. Both meshes square (2×2, 4×4) so any latent px↔py-swap
   bug is invisible in BOTH configs (flagged, not the differentiator). **No algorithm switch.**

## Findings ranked by plausibility of producing the eV-scale eqp shifts

1. **CONFIRMED-LOCUS (bispinor): padded transverse ζ system at P=16.** Solver-independent
   O(1)–O(500) corruption of ζ^{γ̃i} when n_rmu_T pads 668→672; γ̃² catastrophic
   (Σ^B tile trace −117.9 eV). Root is in C_q^μ/Z_q^μ pad content, the identity-pad +
   1e-12·|tr|/n ridge interaction with the indefinite near-singular transverse CCT, or the
   padded-extent reshards feeding the solve — i.e. Agent-padding territory, now with the
   stage pinned and a solver-swap control run on disk.
2. **CONFIRMED-LOCUS (charge): PPM mode census + adaptive minimax windows over the padded
   μ extent at P=16.** Pad modes change masked-Ω stats → different node counts (a discrete
   algorithm change) + 2 real modes flip validity. Everything upstream of the PPM fit is
   P-invariant at print precision.
3. **Exonerated: cuSolverMp FFI (potrf/potrs/getrf/getrs) on the 4×4 grid** — unit tests
   at machine precision; production ζ reproduced by an independent solver to 1e-10.
4. **Exonerated: W-solve staging/backend, q/k pad shards, phdf5 valid_shape, accum-mode,
   solver-kind dispatch** — no P-dependent branch differs between 4 and 16 GPU.
5. **Side-findings (not the bug, worth tickets):**
   - `common/cusolvermp_batched_test.py` crashes (process_allgather of the donated A);
     standalone replacement in this report dir.
   - Legacy-LU fallback at 16 GPU: planner sizes r_chunk for the cuSolverMp path → immediate
     21.3 GiB OOM in `z_q_phase` at `memory_per_device_gb=30`; with budget 12 the run passes
     ζ-fit but crashed twice in bispinor V_q one-tile (`v_q_g_flat.py:429`) with NCCL
     `invalid argument` (once on a reused node set, once fresh). The ζ files it wrote first
     were sufficient for the discriminator; the fallback path itself deserves a health check.
   - Transverse LU ridge is weakly P-dependent (1e-12·|tr(C_pad)|/n_rmu_padded: trace +4 from
     pad identity, n 668→672) — irrelevant next to finding 1 but should become P-invariant
     (use logical trace / logical n).

## Artifacts
- `B_bispinor/head_16g_luoff/` — solver-swap control run (ζ files in `tmp/`, `cusolvermp_lu=off`,
  mem 12 GB; V_q stage crashed post-ζ, see side-finding).
- `reports/device_invariance_2026-07-08/zeta_compare.py` — ζ cross-run comparator (single rank).
- `reports/device_invariance_2026-07-08/potrs_4x4_check.py` — donation-safe potrf/potrs test.
- SLURM: 4-node alloc **55674933** (lx-alloc-jackm) left RUNNING (~1 h) for follow-up runs.
