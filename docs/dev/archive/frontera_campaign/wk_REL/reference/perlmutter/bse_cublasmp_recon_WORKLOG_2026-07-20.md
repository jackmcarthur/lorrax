# 2-D-distributed cuBLASMp V_Q reconstruction (large-n_μ generalization)

Branch `agent/bse-cublasmp-recon` off `agent/bse-multinode` @ c6de94c, worktree
`lorrax_A_bse_integration`. Allocation JID 56206654 (4 nodes / 16× A100, 4×4).

## Goal

Generalize the V_Q interpolation coarse-prep reconstruction to n_μ so large that
a single n_μ×n_μ tile CANNOT be replicated on one proc. The cusolverMp→recon
seam in the multinode path (`reports/bse_multinode_2026-07-20/HLO_AUDIT.md`)
GATHERED R from `P('x','y')` (160×160/dev) to `qb3`-replicated (640×640/dev,
6.2 MiB/tile) and ran the reconstruction R g Rᴴ, Sc Vδ Sc replicated per q. This
work KILLS that gather: R stays 2-D sharded and the whole reconstruction runs
through the cuBLASMp batched distributed GEMM, so every n_μ² intermediate is
`P(None,'x','y')` (per-proc = n_μ/Px × n_μ/Py) and NO full tile lands on one proc.

## What changed (single-sourced)

`src/bse/vq_interp.py`:
- `_recon_body(lam, evec, ZG, v_ref, v_lr, *, gram, gram_outer, gemm, conj,
  constr, eps_tik)` — the Tikhonov-clean → SR/LR-split reconstruction expressed
  ONCE, dispatched to a matmul BACKEND. The arithmetic (S = R g_ε Rᴴ,
  Vδ = conj(A)Aᵀ over the sphere, V_SRc = conj(S) Vδ conj(S), zt = S ZG) is
  backend-agnostic; only the layout/BLAS differ.
- `_replicated_prims` (default) — the local-dot einsums, bit-identical to the
  original `_clean_split`. `_clean_split` now calls `_recon_body` with these.
- `_distributed_prims` — cuBLASMp primitives on 2-D-sharded tiles:
  * `gram` = S = R g Rᴴ from the RAW cusolverMp buffer Qraw via `transa='C'`
    (allowed on multi-rank) — R = conj(Qraw)ᵀ is NEVER materialised, NO transpose
    reshard. Per-element: S_μν = Σ_r conj(Qraw_rμ) g_r Qraw_rν.
  * `gram_outer` / `gemm` need op(B)≠N which cuBLASMp forbids on multi-rank, so
    the transpose operand (Aᵀ / R side) is PRE-materialised (`swapaxes` +
    sharding constraint → still full-mesh 2-D sharded, one all-to-all, never
    replicated).
- `_recon_distributed_chunk` — per-q distributed eigh (cusolverMp, R stays
  `P('x','y')`), stack → `P(None,'x','y')`, G-pad the sphere factors to a
  multiple of lcm(Px,Py) (zeros → exact), then `_recon_body` with the cuBLASMp
  prims.
- `prepare_coarse(..., distributed_recon=False, mem_per_device_gb=0.0)` — gates
  the path. `False` (default) = replicated-batched (small-n_μ efficient, 640
  case does not regress). `True`/`"auto"` = 2-D-distributed (requires
  cusolverMp). The per-q phase-factoring of zt → Fch is a SHARED host step
  (`_phase_fch`), identical both backends.
- `_dist_recon_q_chunk` — the q-chunk memory knob that REPLACES "the tile must
  fit replicated per proc": per proc holds ~8 sharded n_μ² tiles ×
  (n_μ/Px)(n_μ/Py) per q; cap the chunk from `mem_per_device_gb`.

`src/bse/exciton_bands.py`: `--distributed-recon off|on|auto`, wired to
`prepare_coarse` with the auto-detected device budget.

`tests/test_bse_vq_recon_distributed.py`: 1-GPU gate (1×1 mesh, cuBLASMp).

## Per-element math (the index contraction for every GEMM)

    g_ε(λ)_br = λ_br² / (λ_br² + (ε_tik·max_r λ_br)²)                  [real, O(n_μ)]
    S_bμν     = Σ_r conj(Qraw_brμ) g_br Qraw_brν      (= R g_ε Rᴴ)     gram, transa='C'
    A_ref_bμG = ZG_bμG √v_ref_bG ,  A_lr_bμG = ZG_bμG √v_lr_bG        [elementwise]
    Vδ_bμν    = Σ_G conj(A_ref_bμG) A_ref_bνG − (lr)                  gram_outer (op(B)=Aᵀ)
    T1_bμσ    = Σ_ρ conj(S)_bμρ Vδ_bρσ ;  V_SRc_bμν = Σ_σ T1_bμσ conj(S)_bσν   gemm
    zt_bμG    = Σ_ρ S_bμρ ZG_bρG                                       gemm

## Results (real logs, JID 56206654, 4×4 mesh)

### 1-GPU math gate — PASS
`tests/test_bse_vq_recon_distributed.py` (cuBLASMp on 1×1): distributed prims vs
replicated prims agree to ≤1e-9 on S / V_SRc / zt; the analytic filter identity
S(C²+c²I)=C² holds. `1 passed in 7.37s`.

### THE CAPABILITY PROOF — per-proc memory replicated vs distributed (synthetic
random Hermitian C_q, `scripts/recon_capability.py`, `logs/capability.log`)

| n_μ | replicated tile / proc | distributed shard / proc | ‖RΛRᴴ−C‖/‖C‖ | filter id ‖S(C²+c²I)−C²‖/‖C²‖ | eigh (s) | recon GEMMs (s) |
|----:|-----------------------:|-------------------------:|-------------:|------------------------------:|---------:|----------------:|
| 640   | 0.01 GiB | (160,160)   = 0.4 MiB  | 3.18e-15 | 3.01e-15 | 5.9  | 2.5* |
| 4096  | 0.25 GiB | (1024,1024) = 16 MiB   | 5.88e-15 | 6.02e-15 | 4.5  | 1.7  |
| 16384 | 4.00 GiB | (4096,4096) = 256 MiB  | 1.15e-14 | 1.82e-14 | 22.2 | 20.3 |
| 32768 | 16.00 GiB| (8192,8192) = 1024 MiB | 1.69e-14 | 5.72e-14 | 109.7| 85.5 |

`*` 640's recon-GEMM wall includes one-time cuBLASMp descriptor/context warmup.
Per-proc shard is exactly n_μ/4 × n_μ/4 (empirical `addressable_shards` shape),
NEVER n_μ×n_μ — the tile is genuinely 2-D distributed.

**Replicated-batched OOM boundary (same script):**
- n_μ=16384: replicated recon attempt = **ok** (4 GiB tile fits 40 GB).
- n_μ=32768: replicated recon attempt = **OOM** —
  `XlaRuntimeError: RESOURCE_EXHAUSTED: Failed to allocate request for 48.00GiB
  (51543802528B) on device ordinal 0`. The 2-D-distributed path completes the
  SAME n_μ=32768 at **1024 MiB/proc** and closes the filter identity to 5.7e-14.

So: a single n_μ×n_μ tile need never fit on one proc. The replicated path dies
between n_μ=16384 and 32768 (one 16 GiB tile + eigh workspace > 40 GB); the
distributed path scales the tile down by Px·Py = 16 and keeps going.

### CORRECTNESS — full exciton bands via distributed-recon vs dir10 — PASS (0.0000 meV)
`--distributed-recon on --eigh-backend cusolvermp` over the 40-Q 8v8c path
(MoS2 12×12, n_μ=640, nq=144), compared to `ref_dir10_40interp_8v8c.dat`
(`logs/exciton_drecon.log`, `logs/compare_drecon_vs_dir10.log`):
- **GLOBAL max|ΔE| = 0.0000e+00 meV**, mean 0.0000e+00; E_1(Γ)=1.141460 eV both.
- The distributed-recon nulls held on the real fixture: `run_nulls`
  F_own_rebuild_vs_cleaned_LR_tile = **1.87e-9 OK** (built from the distributed
  S/Fch) — the reconstruction is correct on production data, not just synthetic.
So routing the reconstruction through cuBLASMp on 2-D-sharded tiles reproduces
the exciton physics to the .dat precision.

### 2 — n_μ=640 bit-match (distributed vs replicated recon, MoS2 12×12, nq=144) — PASS
`scripts/recon_bitmatch_640.py` (`logs/recon_bitmatch_640.npz`), same cusolverMp
eigh both paths, only the recon backend differs:

| output | relF(distributed vs replicated) |
|--------|--------------------------------:|
| S       | 1.70e-16 |
| V_SRc (host)   | 2.34e-13 |
| V_SRc (device stack, eval consumes) | 2.34e-13 |
| Fch     | 1.92e-15 |

**PASS at 1e-9.** The cuBLASMp reconstruction reproduces the replicated
`_clean_split` to ~1e-13 (the V_SRc's two extra GEMM reassociations) / ~1e-16
(S). This is the n_μ=640 bit-match that gates the full exciton run above.

### GOLDEN GATES — 26 passed (1 GPU)
`tests/test_gw_jax_regression.py test_symmetry_unfold.py test_bse_vq_interp.py
test_minibz_average.py test_bse_vq_recon_distributed.py` → **26 passed in 101.6s**
(`logs/golden_gates_1gpu.log`). No regression; the new distributed-recon test is
green.

### TIMING + crossover
| stage (wall s)     | dir10 4-GPU native | 16-GPU replicated-recon | 16-GPU **distributed-recon** |
|--------------------|-------------------:|------------------------:|-----------------------------:|
| htransform_psi_cQ  | 155                | 15.5                    | 15.5                         |
| **vq_prepare**     | 59                 | 398                     | **418**                      |
| solve_scan cold    | 202                | 165                     | 161                          |
| TOTAL              | 636                | 758                     | 774                          |

At n_μ=640 the distributed recon costs **+5%** on vq_prepare (418 vs 398 s): both
16-GPU paths are dominated by the 144 per-q cusolverMp eighs (cross-node NCCL
latency, the multinode "small-case" regime); the extra cuBLASMp reconstruction
GEMMs add only ~20 s. So the distributed recon does NOT win on SPEED at n_μ=640 —
it wins on **CAPABILITY**: it is the only path that runs at all once one
replicated n_μ×n_μ tile no longer fits. **Crossover (capability): n_μ ≈ 24k** —
replicated recon completes at n_μ=16384 (4 GiB tile) but OOMs at 32768 (16 GiB
tile + eigh workspace = 48 GiB > 40 GB); the distributed path is comfortable at
both (256 MiB, 1024 MiB/proc) and would keep scaling with Px·Py. The distributed
recon-GEMM wall itself scales ~n³/(Px·Py): 20.3 s (16384) → 85.5 s (32768).

### N_μ²-DISTRIBUTION AUDIT — PASS (9/9 full-mesh)
See `NMU2_DISTRIBUTION.md` (owner directive). Every ≥O(n_μ²) intermediate
(Qraw, S, Sc, A_ref, A_lr, V_delta, T1=Sc@Vδ, V_SRc, zt) verified on 16 GPUs
(`scripts/recon_shard_audit.py`, `logs/shard_audit.log`) at
`spec=P(None,'x','y')`, per-proc shard **(2, 160, 160)** = (n_μ/Px, n_μ/Py) —
NEVER (640,640) replicated nor (640,160)/(160,640) half-distributed. Only λ and
g are replicated `P()` (O(n_μ), allowed). **VERDICT: ALL n_μ² tensors FULL-MESH
2-D sharded (9/9).** Compiled S-gram GEMM sharding lines: `logs/recon_gram_hlo.txt`.

## Notes
- `nvshmem detect topo failed status 27` warnings print during FFI init but are
  BENIGN: NCCL is the comm path (bit-accurate 1e-15 results confirm), nvshmem is
  an unused fallback probe. Not a blocker.
