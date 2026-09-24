# ζ fit by μ-batches: Z(G) first, C⁺ once

Status: design, branch `feat/zeta-mubatch-2026-09-23` (not on main).
Evidence: `runs/runtime/zeta_mubatch_20260923/` in the sandbox.
Owner design 2026-09-23, refined the same day (direction-agnostic
transpose; ψ(r) cache when it fits, plane-regenerated ψ only when it does
not).

## The identity

ISDF least squares per selected q:

```text
Z_q(μ, r) = Σ_k Σ_{ab} D^L_{k,ab}(μ, r) · conj(D^R_{k+q,ab}(μ, r))      (k-convolution)
D^X_{k,ab}(μ, r) = Σ_n w^X_n ψ_{nka}(r_μ) ψ*_{nkb}(r)                  (X = L, R windows)
C_q(μ, ν) = Z_q(μ, r_ν)                    ζ_q = C_q⁺ Z_q               (C⁺ acts on μ only)
```

`C_q⁺` is the operator `factor_c_q` builds today (rank-truncated
pseudo-inverse or Cholesky, the same rcond, the same pad-diagonal and
LR+RL completion). Because it acts on μ only, it commutes with the
r→G transform:

```text
ζ_q(μ, G) = C_q⁺ Z_q(μ, G),     Z_q(μ, G) = FFT_r[e^{-iq·r} Z_q(μ, r)] on the ζ sphere
V_q = conj(ζ) diag(v_q) ζᵀ = conj(C_q⁺) M_q conj(C_q⁺),   M_q = conj(Z_q) diag(v_q) Z_qᵀ
```

`C_q⁺` is Hermitian, so `(C⁺)ᵀ = conj(C⁺)`. Both forms of `V_q` are the
same operator. Truncating to the sphere before or after the solve is
identical, so ζ(G) must match the r-chunk fit to rounding. The LR+RL
completion `Z_q ← Z_q + conj(Z_{−q})` is applied in r space, before the
q selection and before the transform, exactly where the r-chunk loop
applied it.

## The loop

The r-chunk loop (orbit-closed tiles; each tile built for all μ, solved
and accumulated into an all-G ζ(G) buffer) is replaced by:

```text
setup:  C_q, factor (unchanged);  Z store (Q, μ, N_G), zeroed
        r blocks R_p (one per rank);  ψ(r) source (cache or regeneration)
for each μ batch B (b centroids, serial):
    X_B = ψ_{nks}(r_μ ∈ B), all full-BZ k, all fit bands     (replicated, small)
    every rank p, for each r sub-block r_s ⊂ R_p:
        D^L, D^R(k, a, μ_B, b, r_s) = X_B · ψ*(r_s)          (band-chunked GEMM)
        Z_q(μ_B, r_s) for all q = k-conv tail                (the existing tail math)
        LR+RL completion, then select the Q stored q rows
    ONE all-to-all per batch: rows (q, μ_B) × r-split  →  row-owned × full r
    each rank: e^{-iq·r}, local full-box FFT per row, gather the ζ sphere
    write the rows into the Z store (write-once; no accumulator)
after all batches:  ζ = C⁺ Z (G-chunked, the existing solve) when a consumer
                    needs ζ;  V_q = conj(C⁺) M conj(C⁺) otherwise
```

The transpose is direction-agnostic: `R_p` may be any equal split of the
flat grid. No distributed FFT and no second redistribution exist: the
all-to-all lands whole rows on their owners, and each row's full-box FFT
is local.

### Row ownership (the Z-store layout)

The store is `(Q, μ_pad, N_G)` at `P(None, ('x','y'), None)`, the layout
of the old accumulator, so the existing solve, writer and V_q contraction
read it unchanged. Batches are strided over the μ shards: batch β takes
`b/P` consecutive μ from every rank's μ block, so the all-to-all splits
the batch's μ axis and every rank writes its own rows at local offset
`β·b/P`. This is the "(q, μ_B) pairs" assignment and works for any Q,
including Q = 1. The q-local variant (`P(('x','y'),None,None)`, split the
q axis instead) is the same kernel with the other split axis; it is
chosen when the q-local solve tier is (see R4) and is what lets V_q run
with no communication.

### ψ(r_local) source

- **Cache (preferred).** When `nk·nb·ns·R·16` plus the batch workspace
  fits, regenerate ψ once: each rank full-box IFFTs its own band shard
  (the existing transform), one all-to-all band→r, cache across all
  batches. `R_p` is a flat-index block; no box axis enters.
- **Regeneration (small P only).** Otherwise `R_p` is a block of whole
  planes along the largest box axis (padding when uneven). Per batch and
  per band chunk each rank evaluates its own bands on every plane
  through the ψ cylinder (the plane-pruned partial DFT already on this
  branch), one all-to-all plane→owner, then 2D IFFTs of its own planes.
  This is the only place a box axis enters; P larger than that axis
  (pencils) refuses by GATE until needed.

## Per-rank memory model

Symbols: `Q` stored q rows, `nk` full-BZ k, `ns` spinor, `μ` packed centroid
carrier, `nb` fit bands (transport-padded), `N_r` grid, `N_G` ζ sphere
(`ngkmax`), `R = ⌈N_r/P⌉`, `b` μ batch, `r_s` r sub-block, `bc` band chunk,
`16` bytes per c128.

| object | sharding | bytes/rank |
|---|---|---|
| C factor | `P(None,'x','y')` or q-local batch | `Q·μ²·16/P` |
| Z store | `P(None,('x','y'),None)` | `Q·μ·N_G·16/P` (device, else host, else slab_io disk) |
| ψ(r) cache (cache route) | r-block `R_p` | `nk·nb·ns·R·16` |
| ψ(G) resident (regen route) | bands over `('x','y')` | `nk·nb·ns·N_Gψ·16/P` |
| centroid face, full BZ | `μ_X × band_Y` | `nk·ns·μ·nb·16/P` (the parent faces serve C) |
| X_B | replicated | `nk·ns·b·nb·16` |
| D^L, D^R | local | `2·nk·ns²·b·r_s·16` |
| k-conv transients | local | `≈3·nk·b·r_s·16` |
| Z batch (pre-transpose) | r-block | `Q·b·R·16` |
| rows (post-transpose) | row-owned | `Q·b·N_r·16/P` |
| row FFT box | local, row-chunked | `c·N_r·16·f_FFT` |
| regen transient (regen route) | local | `nk·bc_p·ns·n_col·n_a·16` + `nk·bc·ns·r_s·16` |

The planner (`gw.gflat_memory_model`, the one budget) picks, in order:
the ψ route (cache if it fits with the smallest batch), the store tier
(device, host, disk), then the largest `b` (a multiple of P dividing the
μ carrier's per-rank share when possible), then `r_s` and `bc`. No new
deck key or environment knob exists.

### Communication and flops (whole fit, per rank)

| term | volume / flops | VI3 12×12 P16 | P100 |
|---|---|---|---|
| Z transpose | `Q·μ·N_r·16/P` | 637 GB | 102 GB |
| ψ regeneration (regen route) | `n_batch · nk·nb·ns·n_col·n_a·16/P` | 28 GB per batch | 0 (cache fits) |
| pair GEMM | `16·nk·ns²·μ·nb·R` | 9.2e14 | 1.5e14 |
| ζ = C⁺Z on the sphere (only if ζ is written) | `8·Q·μ²·N_G/P` | 5.4e13 | 8.6e12 |
| M = conj(Z) v Zᵀ | `8·Q·μ²·N_G/P` | 5.4e13 | 8.6e12 |
| C⁺ M C⁺ | `16·Q·μ³/P` | 4.7e12 | 7.6e11 |
| retired r-space solve | `8·Q·μ²·N_r/P` (2× for rank-truncate) | 1.0e15 | 1.6e14 |

(VI3: Q = nk = 144, ns = 2, nb = 360, μ = 3200, N_r = 1 382 400,
N_G = 72 541.) Honest accounting: the pair GEMM and the k-convolution are
unchanged in total; the M contraction is the same `N_G·μ²` GEMM V_q runs
today. The saving is the solve (two `μ³` products per q instead of `N_r`
columns), the accumulator's per-tile transforms, the ψ(tile) sources, and
the ζ write and re-read.

## ζ consumers

| consumer | needs ζ on disk? | handled by |
|---|---|---|
| scalar V_q (`compute_all_V_q`) | no | `conj(C⁺) M conj(C⁺)` from the store |
| g0 one-leg and the head channel | a few G columns | `C⁺ Z[:, :, G_sel]` |
| ζ reuse on a later run, restart head channel | yes | write ζ (G-chunked solve) |
| bispinor V_q, BSE `vq_interp`, downfold, exciton bands | yes | write ζ |

## Regime guard

A single q's `(μ, N_G)` row set plus workspace must fit a rank for the
q-local tier; the μ-strided layout needs only `μ·N_G·16/P` per q. The
planner refuses with `GATE zeta-mubatch-capacity` naming the shortfall
when even `b = P` does not fit. The G-chunked 2D factor application for
μ ~ 1e5 supercells is future work.

## Scope

Charge channel, ns = 1 and ns = 2. The current (transverse) channels
keep their incumbent loop until their vertex tails are ported; the
coupled μ = 1,2,3 coordinator is unchanged.
