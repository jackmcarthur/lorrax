# ζ fit by μ-batches: Z(G) first, C⁺ once

Status: implemented for the charge channel on branch
`feat/zeta-mubatch-2026-09-23` (not on main): `isdf.zeta_mubatch`,
`gw.isdf_fitting._fit_mubatch`, planner `gw.gflat_memory_model.plan_zeta_mubatch`.
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
after all batches:  stream the store by G tiles:  ζ_t = C⁺ Z_t,
                    V_q += conj(ζ_t) diag(v_q) ζ_tᵀ,  keep the G≈0 shell;
                    write ζ_t to zeta_q.h5 only when a consumer needs the file
```

V_q is formed ζ-first, not as `conj(C⁺) M conj(C⁺)`: the two are the same
operator in exact arithmetic, but the second loses `≈ ε·κ(C)²` because it
applies the rank-truncated `C⁺` twice to an `M` that has already mixed
scales.  Measured: 3.3e-14 on core fixture A, 1.7e-3 on CrI3 8×8
(κ(C) = 1.0e8).  ζ-first costs the same `N_G·μ²` GEMM per tile plus one
`μ²·G_tile` solve application, and only `V_q` (plus the shell) is resident.

The transpose is direction-agnostic: `R_p` may be any equal split of the
flat grid. No distributed FFT and no second redistribution exist: the
all-to-all lands whole rows on their owners, and each row's full-box FFT
is local.

### Row ownership (the Z-store layout)

`ZStore` is one write-once resource; the planner places it on the device,
in host memory (one numpy tile per addressable device) or in a slab_io
scratch dataset.  SlabIO datasets are contiguous, so the owner's
`(q, G_tile, μ_batch)` chunking lives in the dataset shape:

- **q-owned** (the q-local solve tier, `Q ≥ P` and a q's factor plus a G
  tile fit): `(Q_pad, n_Gt, n_batch·b, G_tile)` at `P(('x','y'), …)`.  The
  all-to-all splits the q axis; rank p owns whole q rows, a batch write is
  whole `(q, G_tile, μ_B)` chunks per writer, and a G-tile read is local
  on host and device — V_q runs with no communication.
- **μ-owned** (otherwise, any Q including Q = 1): `(n_Gt, Q, n_batch·c,
  G_tile)` per rank, `c = b/P`; the all-to-all splits the batch's μ axis;
  a G-tile read reaches the solve's layout with one all-to-all.

The μ axis is stored in batch-slot order (`β·b + j`); `read_tile` gathers
the packed carrier through `MuOrbitBatches.packed_to_slot` (−1: a layout
pad centroid, an exact zero row).  The device holds only the batch being
written unless the planner places the whole store there.

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
| Z store | q-owned or μ-owned tiles | `Q·μ·N_G·16/P` (device only when it fits at the same `b`, else host (0.6 of the node's MemTotal per task), else slab_io disk) |
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

## Symmetry: parent k

The loop above is written on the full BZ.  The r-chunk loop's symmetry
saving carries over unchanged: the pair GEMM runs on the raw parent k̄
only and the pair projectors are unfolded to full k before the
k-convolution (TRS is used in the unfold, never inside the k FFT; TASTE 7).
The unfold gathers on both endpoints, so each must be closed under the rows
`plan.sym_idx` selects — the batch on μ, the rank block on r.  Branch
`feat/zeta-mubatch-sym-2026-09-23`; the builders live in
`gw.centroid_k_unfold`, the tail in `isdf.core`.

| name | returns | layout |
|---|---|---|
| `orbit_mu_batches(plan, mu_pad, P, *, b_target)` | `mu (n_batch, b)` packed centroid per slot (−1 pad); `left_perm (n_batch, 2n_sym, b)` batch-local source slot; `left_L (…, 3)` wraps; `.rank_mu`, `.packed_to_slot(mu_pad)` | host; one batch's tables replicated `P()` |
| `orbit_r_blocks(plan, fft_grid, P, *, r_s_target, route)` | `points (P, n_sub, r_s)` flat grid index (−1 pad); `local_perm (P, n_sub, 2n_sym, r_s)`; `wraps (…, 3)`; `planes (P, n_sub, n_pl_max)`, `plane_axis` | host; device copy `P(('x','y'))` on the rank axis |
| `mu_batch_tables`, `r_block_tables` | the tables of a given partition; refuse a split orbit by name | host |
| `parent_projector_kconv(D_l, D_r, *, plan, left_perm, left_L, right_perm, right_L, kgrid, vertex_l, vertex_r, pair_kernel)` | `Z (nk, b, r_s)`, every full-k q | manual mode, rank-local |

Batches are unions of whole orbits (not per rank: `X_B` is replicated and
the unfold precedes the transpose); `b` is a multiple of P and at least the
largest orbit, so the planner prices the returned `b`, not `b_target`.  The
layout's pad centroids are in no batch — their Z rows are exactly zero — so
the store starts zeroed and rows land through `packed_to_slot`.  The r
blocks come from `build_real_grid_orbit_tiles(fill='owner_contiguous')`:
rank p takes the p-th contiguous run of the orbit plane order and `r_s` is
the smallest cap that holds the grid in `P·n_sub` blocks.  `route` changes
no point; on a group that mixes all three axes (cubic; TaAs in its
primitive bct cell) a block touches most planes, which `planes` prices.

Per batch β, rank p, sub-block s:

```text
D^X(k̄, a, μ_B, b, r_s) = Σ_n w^X_n ψ_{n k̄ a}(r_μ) ψ*_{n k̄ b}(r)      k̄ = raw parents
Z(q, μ_B, r_s) = parent_projector_kconv(D^L, D^R, left = batch β tables,
                                        right = rank p block s tables)
```

`parent_projector_kconv` is the r-chunk `_z_q_face_parent` tail factored
out (bit-identical there): typed transport by
`symmetry_maps.unfold_operator_local` with `left_mesh_axis=right_mesh_axis=None`
(the tables are already local), spin by `open_spin_block_coefficient`, or
the native `conv_kparent` arm built by `make_fused_conv_kparent(mesh, kgrid,
ns, (b, r_s), …)`.  It replaces `_kconv_tail`, the identity-plan arm.

What scales with k: the pair GEMM, ψ(r) cache or regeneration, `X_B` and
`D^L, D^R` fall from `nk` to `n_parent` rows; the k-convolution and its
transients stay on the full zone; the unfold adds `ns²` gathers of
`nk·b·r_s` per sub-block.  The r tables cost `2n_sym·R·16` bytes per rank.

| deck | n_parent / nk | GEMM and ψ-cache factor |
|---|---|---|
| core fixture A (P1, TRS) | 5 / 9 | 1.8× |
| A-cubic (48 ops) | 3 / 8 | 2.7× |
| TaAs 4×4×4 (I4₁md, TRS) | 13 / 64 | 4.9× |
| Si 4×4×4 SOC | 8 / 64 | 8.0× |
| Bi 8×8×8 | 65 / 512 | 7.9× |
| TaAs 8×8×8 | 59 / 512 | 8.7× |
| Fe bcc FM (ntran = 1) | 64 / 64 | 1× |

Verification: `tests/test_zeta_mubatch_orbit_tables.py` (tables, red
twins), `tests/test_zeta_mubatch_sym_parity.py` (CPU 2×2) and
`tests/multi_device/zeta_mubatch_sym_p4.py` (P4, native arm): the batch
layout against the incumbent r-chunk kernel, the full-BZ children and
direct sums, then ζ and V_q through one C⁺.

## ζ consumers

The fit returns `ZetaG`, a lazy ζ over the store (`zeta_layout =
'G_flat'`), in place of the file path when no consumer needs the file.

| consumer | needs ζ on disk? | handled by |
|---|---|---|
| scalar V_q (`v_q_g_flat._compute_V_q_g_flat_one_tile`) | no | `ZetaG.contract_v`: streamed ζ-first V_q |
| g0 one-leg unfold (non-IBZ) | the G≈0 shell | `ZetaG.shell` (`|q+G| ≤ max|q_full|`, pads at the FFT-box sentinel) |
| head channel (`compute_head_channel_zeta`) | a few G columns | `ZetaG.head_columns(sel)` from the shell |
| `write_restart_tensors = true`, restart | yes | the same streamed tiles are written (`contract_v(…, zeta_io=…)`) |
| bispinor / transverse V_q, BSE `vq_interp`, downfold, exciton bands | yes | the file is written; consumers open `ZetaG.path` |

## Regime guard

One planner rule picks the solve: q-local (q-owned store, R4) when
`Q ≥ P` and one q's factor plus a G tile fit a rank, else the 2D
distributed factor applied per G tile (μ-owned store; kept for huge
`N_μ`).  `check_mubatch_solve` refuses (`GATE zeta-mubatch-solve-tier`)
a store/tier pair that disagree.  The planner refuses with `GATE
zeta-mubatch-capacity` naming the shortfall when even the smallest
whole-orbit batch does not fit, and `GATE zeta-mubatch-pencils` when the
plane route would need P above the largest box axis.  Nq = Nk = 1 runs
the μ-owned layout (core fixture B).

## Scope

Charge channel, ns = 1 and ns = 2. The current (transverse) channels
keep their incumbent loop until their vertex tails are ported; the
coupled μ = 1,2,3 coordinator is unchanged.
