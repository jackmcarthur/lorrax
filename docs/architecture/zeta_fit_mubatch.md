# ζ fit by μ batches (route G)

`gw.isdf_fitting._fit_mubatch` runs every fresh charge-channel ζ fit, at
ns = 1, 2 and 4 (the bispinor charge lift). It forms the right-hand side
Z_q(μ, G) on the ζ sphere in batches of centroids, then applies C_q⁺ and
contracts V_q one G tile at a time. The three current channels use the
[real-grid tile loop](zeta_fit_face_psi_cct.md#current-channels). The theory
is on the [ISDF page](../theory/isdf-zeta-vq.md). Code: `isdf.zeta_mubatch`
(the kernel, `ZStore`, `ZetaG`), `isdf.pair_kernels`, `isdf.cplus`, and the
planner `gw.gflat_memory_model.plan_zeta_route_g`.

## The identity

```text
D^X_{k,ab}(μ, r) = Σ_n w^X_n ψ_{nka}(r_μ) ψ*_{nkb}(r)          X = L, R band windows (0/1 weights)
Z_q(μ, r)        = Σ_k Σ_ab D^L_{k,ab}(μ, r) conj D^R_{k+q,ab}(μ, r)
C_q(μ, ν)        = Z_q(μ, r_ν),      ζ_q = C_q⁺ Z_q,      C_q⁺ = B Bᴴ  (isdf.cplus)

Z_q(μ, G) = FFT_r[e^{-iq·r} Z_q(μ, r)]    on the ζ sphere |q+G|² ≤ zeta_cutoff
ζ_q(μ, G) = C_q⁺ Z_q(μ, G),               V_q = conj(ζ_q) diag(v_q) ζ_qᵀ
```

C_q⁺ acts on the centroid index only, so it commutes with the r → G
transform of the other index. The fit therefore never forms ζ(r), a full FFT
box of ζ, or an accumulator over r. It writes Z_q(μ, G) once and applies C⁺
afterwards. When the L and R windows differ, the LR+RL completion
Z_q ← Z_q + conj Z_{−q} is applied in r space before the q selection, as it
is for C_q ([normal equations](zeta_fit_face_psi_cct.md#normal-equations)).

## One μ batch

conj ψ(G) of the n_p raw parents k̄ is resident, with G slots sharded
`P(None, None, None, ('x','y'))`: N'_G = N_Gψ/P slots per rank. It comes from
the single ψ(G) read (`common.psi_G_store.load_parent_psi_G`), which also
samples the centroid faces that C_q is built from. A batch has b = P·c
slots. Rank p owns slots p·c + [0, c), and each owner's bin holds whole
centroid orbits.

```text
1  X_B = ψ_{n k̄ s}(r_μ)      partial DFT over the rank's G slots, one psum (replicated)
2  D̃^X_k̄(a, μ, b, G)         Σ_n w^X_n X_{n k̄ a}(r_μ) conj c_{n k̄ b}(G): one batched GEMM
                               per side on the rank's G slice (isdf.pair_kernels)
3  all-to-all                 G split → μ owner, [L | R] rows owner-major
   on the owner (c centroids, every G):
4  typed unfold k̄ → k        D̃_k = (U_k⊗Ū_k) T_k[D̃_k̄(perm μ, pslot G) e^{2πi L·k̄} conj(phase)];
                               the ψ spheres' occupied (b,c) columns, DFT along the longest grid
                               axis onto every plane: the D cylinder (k, plane, s, 2c, s, column)
5  per group of n_pg planes   columns → planes, 2D FFT: D(k, μ, r_plane) up to its Bloch phase;
                               k-convolution → Z_q(μ, r_plane) for every q of the full zone;
                               LR+RL, stored-q selection, e^{-iq·r}, forward 2D FFT, one matmul
                               onto the ζ-sphere cylinder (columns × axis values)
6  ζ-sphere slots             rows Z_q(μ_B, G), μ-owned → ZStore while batch β+1 runs
```

Each batch runs two collectives, the X_B psum and the pair-projector
all-to-all; `LORRAX_DEBUG_PRINT=1` counts them from the compiled HLO. Every
FFT, the typed unfold and the k-convolution run locally on the owner of whole
μ rows, so no FFT is distributed.

Step 4 is the exact Fourier image of the r-space transport that C_q is built
with (`typed_child_G_tables`). It applies the rotation as a permutation of
sphere slots and the phase e^{-2πi(k̄+G)·t} with t = round(N·Sτ)/N. It then
applies the spinor rotation U_k, conjugation on antiunitary rows, and the
centroid permutation and lattice wrap from the owner's orbit tables. Because
each owner holds whole orbits, the centroid gather is local.

The k-convolution in step 5 is the pair convolution on the identity plan,
where every k is its own parent: `ffi.fft.make_fused_conv_kplane(D, F)`. It
reads the 2D-FFT output `D (N_k, n_pg, ns, 2c, ns, p)` where the FFT left it
and applies the Bloch phase `F (N_k, n_pg, p)` and the L | R split of the 2c
slots on its load (mathdx mode 6 on CUDA; the XLA composition and the host
plans on CPU), so no phased, split or transposed copy of D is written. It
correlates over k by FFTs on the k grid, at O(N_k log N_k) per (μ, r) point.
Kernel contracts are on [the FFI layer](ffi_layout.md#k-convolution-router-and-the-mathdx-family).

**Cost per batch and rank** (n_b fit bands, n_⊥ = N_r/n_a points per plane,
n_col × n_s the ψ cylinder):

| step | arithmetic | traffic |
|---|---|---|
| X_B | n_p·n_b·ns·N'_G·b MACs | psum of n_p·n_b·ns·b·16 B |
| pair GEMM | 2·n_p·ns²·b·n_b·N'_G MACs | none |
| all-to-all | none | 2·n_p·ns²·b·N'_G·16 B |
| axis DFT (owner) | N_k·ns²·2c·n_col·n_s·n_a MACs | none |
| plane FFTs (owner) | N_k·ns²·2c·n_a transforms of n_⊥ points, then Q·c·n_a forward | none |
| k-convolution (owner) | O(c·N_r·ns²·N_k log N_k) | none |

Summed over the μ/b batches, each rank does 2·n_p·ns²·μ·n_b·N_Gψ/P GEMM MACs,
sends 2·n_p·ns²·μ·N_Gψ·16/P bytes, and does O(μ·N_k·ns²·N_r·(log N_r +
log N_k)/P) of owner transforms.

## Z store

`ZStore` holds Z_q(μ, G) as μ-owned rows. Each row is written once, and the
store never lives on the device. G is cut into n_Gt tiles of G_tile slots in
tile-major order, so a batch write and a G-tile read are each a few
contiguous blocks. The store has two placements:

- **host.** One pageable numpy block `(n_Gt, Q, n_batch, c, G_tile)` per
  local device, exactly Q·n_batch·c·n_Gt·G_tile·16 bytes per rank.
- **disk.** A slab_io scratch dataset `(n_Gt, Q, n_batch·b, G_tile)` in
  `zeta_Z_store.scratch.h5` next to the ζ file, deleted on close.

The planner uses the host when the store fits the host share, and disk
otherwise. The host share is 0.8·MemAvailable at plan time, divided among the
processes on the node, taking the minimum over processes. The ψ read releases
its host staging first (`WfnLoader.release_read_staging`). A tile read
reaches the finalize layout with one all-to-all, then gathers slot order into
packed centroid order (`OwnerOrbitBatches.slot_of_packed`). Tile t+1 is read
while tile t is contracted.

## Finalize: ζ, V_q and the head columns

`ZetaG` holds ζ as the pair (Z store, factor B). `ZetaG.contract_v` streams
the G tiles once:

```text
for each G tile t:   ζ_t    = B (Bᴴ Z_t)              at the logical μ extent (solve_at_logical)
                     V     += conj(ζ_t) diag(v_q) ζ_tᵀ  v and ζ masked to each q's sphere
                     shell ← ζ_t at the kept slots
                     ζ_t → zeta_q_G                  only when a file consumer needs it
```

The back-solve tier ([factor and back-solve](zeta_fit_face_psi_cct.md#factor-and-back-solve))
fixes the layout. Under `local`, Z tiles are q-local,
`P(('x','y'), None, None)`, and V accumulates on the q owner. Under
`replicated`, Z tiles are G-split, and each rank accumulates partial sums over
its own G columns. Each finished object (V, the shell, or a ζ tile for the file) leaves its
accumulator in one explicit collective (`_to_mu_owner`: an all-to-all or a
reduce-scatter), so V lands at `P(None,'x','y')` without being replicated.
Per rank, C⁺ and V each cost O(Q·μ²·N_G/P). The finalize holds three
things per rank:

- V: ceil(Q/P)·μ²·16 under `local`, or Q·μ²·16 of partial sums under
  `replicated`;
- the factor;
- about 6·ceil(Q/P)·μ·G_tile·16 bytes of Z and ζ tiles, which the choice of
  G_tile keeps below target/4.

V is formed ζ-first. The equal conj(C⁺) M_q conj(C⁺), with
M_q = conj(Z_q) diag(v) Z_qᵀ, applies C⁺ twice to a product formed before the
solve, so its rounding error grows as ε·κ(C)² instead of ε·κ(C).
`LORRAX_DEBUG_PRINT` also forms that product and prints the difference.

The V_q consumer names the columns to keep (`v_q_g_flat._head_shell`): slot 0
for the full-zone g0, the IBZ one-leg unfold sources, and the head channel's
slots (`vcoul.head_slot_table(...).sel`). `ZetaG.head_columns(sel)` reads them
from the shell.

## ζ consumers

The fit returns a `ZetaG` (`zeta_layout = 'G_flat'`) in place of a ζ loader.

| consumer | reads | source |
|---|---|---|
| scalar charge V_q (`v_q_g_flat`) | all of ζ, once | `ZetaG.contract_v` |
| g0, IBZ one-leg unfold, head channel (`compute_head_channel_zeta`) | a few G columns per q | the shell |
| restart and reuse, the BSE bundle, the four-current V_q | `zeta_q.h5` | written when `write_restart_tensors = true` (the default) or `bispinor = true`, by one extra pass over the store |

## Memory per rank and the planner

`plan_zeta_route_g` sizes everything from the one device budget,
`memory_per_device_gb` times the fragmentation target. No deck key or
environment variable sizes the fit.

| object | bytes per rank | live |
|---|---|---|
| conj ψ(G) slice † | n_p·n_b·ns·N'_G·16 | whole loop |
| centroid faces, C factor | the parent faces; ceil(Q/P)·μ²·16 (`local`) or Q·μ²·16 (`replicated`) | whole fit |
| sphere tables | 12·N_k·N'_G + 4·N_k·n_col·n_s + 8·Q·N_G | whole fit |
| Z rows, current batch and one lookahead | 2·Q·c·N_G·16 | every stage |
| X_B and its phase matrix † | 2·n_p·n_b·ns·b·16 + n_p·N'_G·b·16 | stage 1 |
| pair projectors, GEMM output and all-to-all output † | 4·n_p·ns²·b·N'_G·16 | stage 1 |
| pair projectors on the owner † | 2·n_p·ns²·b·N'_G·16 | stage 2 |
| D cylinder, all planes | N_k·n_a'·ns²·2c·n_col·16 | stages 2–3 |
| one plane group | 2·N_k·n_pg·ns²·2c·n_⊥·16 | stage 3 |
| k-convolution and Z workspace | (9·N_k + 3·Q)·c·n_pg·n_⊥·16 | stage 3 |
| ζ-cylinder accumulator | Q·c·n_zc·n_za·16 | stage 3 |

Here n_a' is n_a rounded up to a multiple of n_pg, and n_zc, n_za are the ζ
sphere's columns and axis values. A batch's working set is the maximum over
its three stages, not their sum. † The planner prices these rows at N_k (the
full zone), but the kernel holds only the n_p raw parents, so the plan is an
upper bound whenever n_p < N_k.

The planner decides in this order:

1. **Resident ψ(G).** Let M_f be the device target minus the fixed terms.
   ψ(G) stays resident when M_f − Ψ holds the smallest batch (b = P,
   n_pg = 1); otherwise the plan refuses.
2. **Batch width.** For each n_pg = 1, 2, 4, …, n_a, b is the largest
   multiple of P whose working set fits M_f − Ψ, capped at ceil(μ/P)·P and
   balanced across batches.
3. **Plane-group width.** Each candidate's modelled time per batch is the two
   collectives (`gw.comm_model.comm_time`), plus 3 ms per plane group, plus
   0.65 s·(c+1)/2 of owner work, times the batch count. The cheapest wins,
   and the receipt names the runner-up.
4. **G tile.** G_tile is the largest multiple of P with
   G_tile·(6·ceil(Q/P)·μ·16, plus Q·μ²·16/N_G when G-split) ≤ target/4,
   capped at ceil(N_G/P)·P.

The fit then packs whole orbits into bins of width c ≤ b/P with the least
padded work, n_batch·(c+1) (`best_owner_orbit_batches`). The planner does not
see orbit sizes, so the executed c can be smaller than the planned one.

The receipt states each term as a multiple of the Green's-function tile
N_k·ns²·μ²·16/P, the GW run's unit of memory. It also reports, without
enforcing, whether the minimum configuration lies within 4·G_tile, about the
GW run's own bottleneck. For scale, see CLAIMS 2676 (VI3 12×12, P16): ψ(G)
read 58 s, C_q 9 s, factor 4 s, μ-batch loop 105 s, V_q 23 s, and a 35 GB
per rank host Z store.

## Refusals and pads

| refusal | condition | way out |
|---|---|---|
| `GATE zeta-mubatch-capacity` | ψ(G) plus the smallest batch exceed the device target (ψ(G) streaming is not implemented) | more ranks or more memory per device |
| `GATE zeta-mubatch-shell` | a head consumer reads a ζ column that the V_q pass did not keep | name the slot in `_head_shell`'s lists |
| `_fit_mubatch` | the loader's full-BZ rows are not the C-order k grid | none |
| `typed_child_G_tables` | a child k is not an image of its parent, or needs a G outside the parent's sphere | none |

`runtime.padding` owns the pads ([mesh-padded axes](padding.md)). The stored
q rows (divisor P) and the ζ sphere in whole G tiles (divisor G_tile) are
`PaddedAxis` receipts made once in `_fit_mubatch` and shared by the kernel,
the store and the finalize. The ψ G slots, the plane groups and the fit bands
are padded by name. Pad q rows and pad G slots carry v = 0 and ngk = 0, so
they contribute nothing to V. Pad batch slots are −1 in
`OwnerOrbitBatches.mu`, and their Z rows are zero.

## Verification

- **`tests/multi_device/zeta_mubatch_p4.py`** (P = 4) checks the kernel, both
  store placements and both read layouts against the dense full-BZ sum at
  1e-12. It covers a glide group with spin mixing and an antiunitary row
  (ns = 2), the 48-operation A-cubic fixture (ns = 1), and a ragged deck where
  no axis divides the mesh. Its red twin shifts the ζ-sphere axis index by one
  and must miss by more than 1e-3.
- **`tests/test_zeta_mubatch_orbit_tables.py`** checks the whole-orbit batch
  tables against a direct Seitz evaluation, and checks that a split orbit
  refuses.
- **`tests/multi_device/pair_kernels_p4.py`** checks the pair GEMM;
  `tests/multi_device/kconv_router_p4.py` checks the router.
