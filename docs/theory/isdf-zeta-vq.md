# 11. ζ-fit and V_q on the G-flat pipeline (algorithms, sharding, communication)

This section is the single source of truth for the **current** ζ + V_q
implementation: rank-5 open-spin pair density, scan-INSIDE-shard_map
r-chunk loop, G-flat on-disk ζ̃, per-q G-chunked V_q kernel, and the
IBZ cascade (ζ written for IBZ q's, V_q computed at IBZ q's, full BZ
recovered post-loop by `unfold_v_q`).

It replaces and extends the previous §11 of
`physics.md`. Companion to §3–5 (historical r-space
ζ-on-disk) and to §7 (sharding summary). The math in §3–5 is still
valid pointwise; the algorithm / sharding / I/O layer documented below
has wholly replaced its r-space sibling for the GW driver path.

**Sharding notation.** Mesh is `Mesh(devices.reshape(p_x, p_y),
('x','y'))`, `P = p_x · p_y`. Inline axis subscripts:

* `μ_X` — sharded on `'x'`.
* `μ_Y` — sharded on `'y'`.
* `μ_XY` — flat-sharded on the tuple axis `('x','y')`.
* unsubscripted — replicated.

Reshards are written `A[μ_X, …] → A[μ_XY, …]` so the mesh axis that
moves is visible. Source-of-truth citations are inlined as
`file.py:line` and refer to `sources/lorrax_B`.

**Sizes (typical → extreme).**

| Symbol | Range | Notes |
|---|---|---|
| `n_rtot = nx·ny·nz` | 5k–2M+ | FFT-box bottleneck; everything that materialises one must chunk over it |
| `n_band` | 50–10k | per-channel L/R windows |
| `n_rmu_padded` | 500–100k (≈10·n_band) | centroid count, rounded up to `P = ∏ p_a` so single-axis sharding is divisible |
| `n_G_sph ≈ 0.05·n_rtot` | 1k–100k | per-q `(q+G)` sphere, padded to `ngkmax = max_q ngk[q]` |
| `n_k = n_q` | 1–10k | flat-k / flat-q axis |
| `n_q_disk` | 1–n_q | IBZ subset (factor `ntran` reduction) |
| `n_bc` | 1–~30 | band-chunk count = ⌈n_band / band_chunk_size⌉ |

---

## 11.0 Where the centroids come from

Everything below takes `{r_μ}` as given. It is chosen by `centroid.kmeans_cli`
before any of this runs, and by one of two selectors
(`--centroid-selector`, [drivers → centroids](../drivers.md#centroids--centroidkmeans_cli)):

* **`kmeans`** (default) — a density-weighted Lloyd iteration produces an
  over-sampled candidate pool, which greedy pivoted Cholesky on the q=0
  pair-density Gram prunes to `n_rmu`. Seeded, and the seed is not a
  formality: on Si 4×4×4 the seed alone moves Σ_x by tens of meV, which is
  why the standing recipe is best-of-five draws ranked by the generator's
  own independent-direction count.
* **`pivoted_full_grid`** — the same greedy, with the candidate pool taken to
  be the *whole* real-space grid in C-order and orbits consumed member by
  member, so the delivered set is orbit-closed and every point in it is a
  certified independent direction. No RNG anywhere.

What the ζ-fit cares about is the second column of that choice: the number
of **independent directions** the point set spans, because the ζ back-solve
truncates every mode below `zeta_rcond` and a point that adds no direction
adds a truncated mode instead of a basis function. Measured on Si 4×4×4 at
24³ with `nband = 100` (§7.7.12): a 1692-point seeded set spans 1246
directions and the back-solve keeps 1246 of 1692 per q; the deterministic
1394-point set spans 1282 and keeps **1394 of 1394**.

The reason `pivoted_full_grid` is a backup and not the default belongs to
§11.2's memory model rather than to the physics: its candidate Gram is
`O(n_rtot²)`, which is 3.06 GB at `n_rtot = 13824` and roughly 200 GB at
`n_rtot = 110592`. It is affordable on small, high-symmetry cells at modest
cutoff, and on nothing else. The Gram build is column-blocked on every mesh
(`build_gram_q0_via_loadwfns`), so the binding object is the assembled
`(M, M)` Gram itself and not the pair-density transients.

The hard ceiling on this axis is the Gram's own numerical rank — 1474 on
that deck at that band window. No point set on that grid, chosen by any
rule, spans more directions than that, so `n_rmu` beyond it buys redundancy
and the truncation that goes with it.

---

## 11.1 Pipeline overview

The pipeline is **r-chunked** (r is the dominant memory axis) and
streams ψ(G) from a host cache via `io_callback`. ψ(G) is **never** a
jit argument; q, μ, n, G are either small or sharded.

```
                  WFN.h5
                     │
              (build_psi_G_store)               host RAM
                     ▼                       ┌──────────────┐
              ψ(G) host slab cache  ◄────────│  per-process │
              n flat-sharded over ('x','y')   │  numpy tiles │
                     │                        └──────────────┘
                     │ io_callback (one bc per scan iter)
                     ▼
       ┌──────────────────────────────────────────────────┐
       │  fit_zeta_to_h5  (isdf_fitting.py:1751)          │
       │                                                  │
       │  Stage A  load centroids                         │
       │     ψ[k, μ_XY, n, ns]  ← load_centroids_band_chunked │
       │                                                  │
       │  Stage B  CCT once, factor once  (per channel)   │
       │     c_q_from_psi_sm → C_q[q, μ_X, ν_Y]           │
       │     factor_c_q     → L_q[q, μ_X, ν_Y]            │
       │                                                  │
       │  Stage C  r-chunk loop                           │
       │     for each r-chunk r ∈ [r0, r0 + r_chunk):     │
       │        zeta_chunk ← fit_one_rchunk(…)            │
       │                       (ZCT + IBZ slice + solve)  │
       │            ζ[q_ibz, μ_XY, r]                     │
       │        gflat_acc ← accumulate_rchunk_to_gflat(   │
       │                       zeta_chunk, gflat_acc, …)  │
       │            ζ̃[q_ibz, μ_XY, G_sph]   (donated)     │
       │                                                  │
       │  Stage D  post-loop                              │
       │     mask ngk[q] pad slots → SlabIO.write_slab    │
       └──────────────────────────────────────────────────┘
                     │
                     ▼
              zeta_q_G.h5
              dataset 'zeta_q_G':  (n_q_disk, n_rmu, ngkmax) c128
              mf_header (copied verbatim from WFN.h5)
              isdf_header  (centroids, sphere, vertex_μ_L, …)
                     │
                     │ read one q at a time (batched IBZ pre-read)
                     ▼
       ┌──────────────────────────────────────────────────┐
       │  V_q g-flat                                      │
       │  compute_all_V_q_g_flat  (v_q_g_flat.py:499)     │
       │                                                  │
       │  resolve IBZ q list                              │
       │  pre-read ALL IBZ ζ̃ slabs in one batched call    │
       │  for each q ∈ IBZ:                               │
       │     V_q ← per-q G-chunked GEMM kernel            │
       │  unfold IBZ → full BZ  (symmetry_maps.unfold_v_q)│
       │                                                  │
       │  V_qmunu[q, μ_X, ν_Y]   IN MEMORY (no on-disk V) │
       └──────────────────────────────────────────────────┘
```

Lifetime-ordered objects: (1) **ψ(G) host slab cache** —
`PsiGStore` numpy tiles, persistent across the whole
`fit_zeta_to_h5`; (2) **ψ at centroids** (`psi_rmu_Y`,
`psi_rmuT_X`) — live across the r-chunk loop; (3) **L_q factor** —
`(n_q, n_rmu_padded, n_rmu_padded)` Cholesky / pivoted-LU, live
across the r-chunk loop; (4) **`gflat_acc`** —
`(n_q_disk, n_rmu_padded, ngkmax)` G-flat ζ̃ accumulator, donated
in place every iter; (5) **ζ_chunk** —
`(n_q_disk, n_rmu_padded, r_chunk)`, per-iter only; (6) **V_q,μν**
— `(n_q_full, n_rmu, n_rmu)` in-memory result, no on-disk V_q for
the charge channel (bispinor off-diag tiles write via
`BispinorVqWriter`).

Three invariants: (a) the full ζ never sits in memory — it lives
only as `gflat_acc` (G-flat, IBZ-only, μ-sharded) and is read back
one q at a time; (b) no replicated transient is acceptable — every
device-side buffer is sharded or scan-aliased
(`PATH_D_PICKUP.md §0`); (c) the IBZ cascade gates on centroid orbit
closure (`compute_centroid_sym_perm(extend_trs=True)` must succeed,
else fall back to full-BZ).

---

## 11.2 The four workspaces (memory model)

`reports/zeta_v_q_g_flat_reference_2026-05-12/report.md §5` is the
living quantitative reference. Recap of the four workspaces that
contest the per-rank HBM budget. See [`MEMORY_MODEL.md`](../architecture/memory-model.md)
for the full chunk-size selection rules.

### 11.2.1 ψ G-space slab cache (host RAM, replicated per process)

Source: `common/psi_G_store.PsiGStore` (see `psi_G_store.py`).

* Layout: per-process numpy buffer of shape
  `(nk, n_band_per_process, ns, ngkmax)`, c128, **on host**. The band
  axis is flat-sharded over `('x','y')` — process `(x_idx, y_idx)`
  owns bands `[id · bpd_max, (id+1) · bpd_max)` where
  `id = x_idx · p_y + y_idx`.
* Population: built once at the top of `fit_zeta_to_h5`
  (`isdf_fitting.py:2305-2311`) by `build_psi_G_store`. Two modes:
  * `gspace_mode="host_cache"` (default) — all band-chunks held in
    host RAM for the lifetime of the fit.
  * `gspace_mode="file_reread"` — host tiles rebuilt at each r-chunk
    via phdf5 collective I/O, freed at `psi_G_store.end_rchunk()`.
* **Access pattern**: per-rank, per-bc slices pulled into device via
  `io_callback` inside the scan body
  (`isdf_fitting.py:639-643`). The slicer
  `_slice_local_tile_bc(x_idx, y_idx, bc_idx)` returns the rank-local
  bc tile padded to a static `(nk, _bpd_max, ns, ngkmax)`.
* **Critical contract**: ψ(G) is **NOT** a jit argument. Passing it as
  such would force the entire `n_band` × `n_k` × `ngkmax` tensor onto
  device. The io_callback + host cache pattern keeps only one bc on
  device at a time (the rank-5 carry's lifetime).

The replicated-per-process residency (~5 GB / rank on CrI3 6×6 80 Ry
charge) is the **only** legitimate replicated buffer in this pipeline.
Every device-side intermediate is sharded.

### 11.2.2 Pair-density workspace (rank-5, inside `fit_one_rchunk`)

Source: `isdf_fitting.py:628-637` (carry init), `:699-705` (einsum into
carry).

Inside `z_q_from_psi_sm._local` the carry is

$$P_{l,\,k}^{(\mathrm{open})}(\alpha, r, \mu, \beta), \;\; P_{r,\,k}^{(\mathrm{open})}(\alpha, r, \mu, \beta) \in \mathbb{C}^{n_k \times n_s \times r_\mathrm{loc} \times \mu_\mathrm{loc} \times n_s}$$

with `mu_loc = n_rmu_padded / p_x` and `r_loc = r_chunk_size / p_y`.
The carry lives **inside the shard_map body**: it is rank-local, never
a global jax.Array, and aliased across `lax.scan` iters by XLA's
scan-internal slot allocator (single physical slot, lifetime = 1
iter).

The pre-IFFT band-FFT box that feeds the rank-5 carry is

```
psi_Y_bc_local_full_r : c128[nk, _bpd_max, ns, n_zchunk]   per rank
```

(from `to_rchunk_inner` at `isdf_fitting.py:650-653`). This is the
**transient FFT box**; same single-slot lifetime as the carry.

**Why r_loc gates after gather, not before.** Round 6 lesson encoded
at `isdf_fitting.py:615-626`: slicing the r-axis per-rank BEFORE the
all_gather over bands mixes r-slabs from different y-ranks at the same
gathered band position. The correct order is

1. Compute full-r-chunk slab per rank (`r_loc = n_zchunk`, not
   `n_zchunk / p_y`).
2. all_gather along the band axis.
3. dynamic_slice on the r axis to this y-rank's `r_loc` slab.

XLA's scan-internal allocator aliases the full-r slab across iters →
single slot of cost `nk · bpd_max · ns · n_zchunk · 16 B` regardless
of p_y.

### 11.2.3 ζ̃ G-flat on-disk format (per-q sphere, padded)

Source: `isdf_fitting.py:2256-2273` (dataset create), `wfn.h5.spec`-style
header (`file_io/mf_header.py`, `file_io/isdf_header.py`).

```
zeta_q_G.h5
├── mf_header          (copied verbatim from WFN.h5)
└── isdf_header
│       r_mu_fft_idx           (n_rmu_logical, 3)      int32
│       fft_grid                (3,)                   int32
│       density                'scalar' | 'current'
│       vertex_mu_L            int
│       zeta_layout            'G_flat'
│       gvec_components        (n_q_disk, 3, ngkmax)   int32
│       ngk_per_q              (n_q_disk,)             int32
│       zeta_cutoff_ry         f64
│       zeta_is_done           bool   (flipped at the very end)
└── zeta_q_G            (n_q_disk, n_rmu, ngkmax)      c128
    chunks: (1, n_rmu, ngkmax)   ← per-q reads are one HDF5 chunk
```

Conventions:

* **G-flat means cell-periodic FFT on the per-q sphere.** ζ̃ on disk
  satisfies

  $$\boxed{\tilde\zeta_{q,\mu}(K) \;=\; \frac{1}{N_r}\,\mathrm{FFT}_{r\to G}\big[e^{-2\pi i\,q\cdot r}\,\zeta_{q,\mu}(r)\big](K), \qquad K = q + G.}$$

  The 1/N_r factor is absorbed into the V_q kernel's `v(q+G)` overlay
  (the writer uses `norm='backward'`, the V_q reader applies the
  scaling factor).

* **`ngkmax = max_q ngk[q]`**. Per-q sphere lengths `ngk[q]` vary; the
  on-disk array is padded to a uniform `ngkmax`. Pad slots
  `[ngk[q], ngkmax)` are zeroed at the very end of the fit
  (`isdf_fitting.py:2494-2502`) so V_q reads see zero contribution
  from non-physical G.

* **`sphere_c[q, 0] == 0`**: G = (0, 0, 0) is always slot 0 on every
  q's sphere (`coulomb_sphere.py:120-247` writer convention). This is
  why `g0_q = zeta_L[:, 0]` reads the head term in
  `v_q_g_flat.py:135`.

* **q-axis is IBZ when `write_ibz_only=True` (default).** Disk shrink
  factor is `n_q_full / n_q_irr` (e.g. 4.5× on CrI3 6×6×1 with
  `ntran=12`, combined with the r → G-sphere saving for 21× total).

* **q-axis fractional vectors** follow the BGW wrap convention
  `q > kgrid/2 → q − kgrid` (`isdf_fitting.py:2070-2073`) then `/
  kgrid`. The unfold and V_q kernels must use the same convention.

* **μ-padding**: μ axis padded to `n_rmu_padded ≡ ∏ p_a` for mesh
  divisibility; writer clips back to logical `n_rmu` via
  `valid_shape=` (`isdf_fitting.py:2510-2514`).

### 11.2.4 V_q,μν output (sharded, in memory only)

Source: `v_q_g_flat.py:394-409` (allocator), `:91-143` (per-q kernel).

```
V_acc : c128[n_q_ibz, n_rmu_L_padded, n_rmu_R_padded]   P(None, 'x', 'y')
g0_acc: c128[n_q_ibz, n_rmu_L_padded]                   P(None, 'x')     (CC tile only)
```

Persistent across the q-loop, donated to the per-q kernel each iter.
After the loop, `unfold_v_q` rotates and replicates IBZ rows into the
full-BZ shape `(n_q_full, n_rmu, n_rmu) P(None, 'x', 'y')`. There is
no on-disk V_q for the charge channel — downstream consumers (Σ^X,
χ⁰) read the in-memory buffer directly. For bispinor `(μ_L, ν_L)`
off-diagonal tiles V_acc is written to disk via the bispinor
orchestrator's `BispinorVqWriter`; see §11.8 of the original chapter
for the seven-tile layout.

---

## 11.3 `fit_one_rchunk` algorithm — kernel inside `z_q_from_psi_sm`

Source: `isdf_fitting.py:378-744` (helper), `:1467-1624`
(`_make_fit_one_rchunk_kernel`), `:1627-1717` (`fit_one_rchunk` entry).

Inputs and outputs of the cached `@jax.jit` kernel:

```
in  : psi_l_rmuT_X_fit   (nk, n_rmu_padded, nb_l, ns)   P(None, 'x', None, None)
      psi_r_rmuT_X_fit   (nk, n_rmu_padded, nb_r, ns)   P(None, 'x', None, None)
      L_q                (n_q_full, n_rmu, n_rmu)       P(None, 'x', 'y')
      norms_l, norms_r   (nb_l,), (nb_r,)               replicated
      r_start_dyn         scalar int32                   traced
      gamma_perm, gamma_phase   (ns,), (ns,)             replicated; identity when charge
      cct_trace_per_q    (n_q_full,)                    replicated; only used for LU
out : zeta_chunk          (n_q_disk, n_rmu_padded, r_chunk)
                          P(None, ('x','y'), None)
```

The kernel runs **one fused jit** per r-chunk. Inside that jit, one
`shard_map` body carries out:

1. **IFFT-first band gather**: per-bc ψ(G) → ψ(r-slab), then
   all_gather along the band axis.
2. **L/R per-bc band mask** with bc-validity gate (handles short final
   bc).
3. **Two einsums into rank-5 carries** `P_l_acc`, `P_r_acc`.
4. **Post-pair tail**: IFFT_k → γ̃·γ̃ contract → FFT_k → transpose.
5. **IBZ slice** of `Z_q` (and `L_q`) — only IBZ rows feed the solve.
6. **`solve_zeta`** triangular / LU solve into `ζ[q_ibz, μ_XY, r]`.

### 11.3.1 Per-iter scan body (inside the shard_map)

`isdf_fitting.py:633-705`. The scan iterates `bc_idx ∈ [0, n_bc)` —
one band-chunk per iter:

```python
def body(carry, bc_idx):
    P_l_acc, P_r_acc = carry
    # (1) PULL — io_callback into rank-local host tile.
    psi_G_bc_local = io_callback(_slice_local_tile_bc, ...,
                                 x_idx, y_idx, bc_idx, ordered=False)
    # (2) FFT → r-slab; per-rank cuFFT, no resharding.
    psi_Y_bc_local_full_r = to_rchunk_inner(psi_G_bc_local, ...)
    # (3) ALL-GATHER across both mesh axes along the band axis.
    psi_Y_bc_full_r = jax.lax.all_gather(
        psi_Y_bc_local_full_r, axis_name=('x','y'), axis=1, tiled=True)
    # (3b) Slice r-axis to this y-rank's r_loc slab (AFTER gather).
    psi_Y_bc = jax.lax.dynamic_slice_in_dim(
        psi_Y_bc_full_r, r0_y_offset, r_loc, axis=3)
    # (4) L/R masks per global band index + bc-valid gate.
    g_axis  = b_lo_global[bc_idx] + jnp.arange(bpd_max_global)
    bc_valid = g_axis < b_hi_global[bc_idx]
    l_mask = (g_axis >= L_lo_g) & (g_axis < L_hi_g) & bc_valid
    r_mask = (g_axis >= R_lo_g) & (g_axis < R_hi_g) & bc_valid
    psi_l_Y_bc = jnp.where(l_mask[None,:,None,None], psi_Y_bc, 0)
    psi_r_Y_bc = jnp.where(r_mask[None,:,None,None], psi_Y_bc, 0)
    # (5) per-bc X-side slices + masks (band axis replicated → local).
    psi_l_X_bc = jnp.where(l_mask[None,None,:,None],
                           dynamic_slice_in_dim(psi_l_X_padded, ...), 0)
    psi_r_X_bc = jnp.where(r_mask[None,None,:,None],
                           dynamic_slice_in_dim(psi_r_X_padded, ...), 0)
    # (6) Two einsums into the rank-5 carries.
    delta_P_l = jnp.einsum('kmna,knbr->karmb', psi_l_X_bc, psi_l_Y_bc)
    delta_P_r = jnp.einsum('kmna,knbr->karmb', psi_r_X_bc, psi_r_Y_bc)
    return (P_l_acc + delta_P_l, P_r_acc + delta_P_r), None
```

`karmb = (k, α, r, μ, β)`; `ψ_X` (centroid leg) and `ψ_Y` (sample leg)
carry distinct spinor labels α, β, so the rank-5 carry holds the
open-spin pair density. Charge channel has `gamma_L=gamma_R=None`;
the γ̃ contract is deferred to the post-scan tail (§11.3.3).

**IFFT-first / gather-second.** Gather-first would inflate the FFT
box to ~80 GB/rank at CrI3 6×6 80 Ry. IFFT-first keeps the FFT box
at `bpd_max × n_rtot × 16 B` per rank.

### 11.3.2 CCT pre-loop (once per channel)

Outside the r-chunk loop, `c_q_from_psi_sm` is called once per channel
at `isdf_fitting.py:1953-1963`. Structurally **identical** to
`z_q_from_psi_sm` except:

* `sample_kind = 'centroids'` instead of `'flat_r_slab'`.
* `n_col = n_rmu` instead of `r_chunk`.
* No post-gather r-slice (the col axis is `n_rmu`, sharded on `'y'`
  inside the gather → the einsum produces the per-rank `n_rmu / p_y`
  slab natively).

Output `C_q[q, μ_X, ν_Y]` is then factored once by `factor_c_q`
(`isdf_fitting.py:914-1080`) into `L_q[q, μ_X, ν_Y]` — Cholesky for
charge (μ_L=0, PSD CCT), pivoted LU + ridge for transverse (μ_L ∈
{1,2,3}, indefinite CCT). The factored `L_q` lives across all
r-chunks.

### 11.3.3 Post-pair tail (inside the same shard_map)

`isdf_fitting.py:715-737`. After the scan completes, each rank holds
`P_l, P_r : c128[nk, ns, r_loc, mu_loc, ns]`. The tail reshapes
`nk → (nkx, nky, nkz)` and runs IFFT_k → γ̃·γ̃ contract → FFT_k →
transpose:

```python
P_l_R   = jnp.fft.ifftn(P_l_3d, axes=(0,1,2), norm='forward')
P_r_R   = jnp.fft.ifftn(P_r_3d, axes=(0,1,2), norm='forward')
Z_R     = gamma_double_contract(jnp.conj(P_l_R), P_r_R,
                                perm_L, phase_L, perm_R, phase_R,
                                spin_axes=(3, 6))
Z_q_3d  = jnp.fft.fftn(Z_R, axes=(0,1,2), norm='forward')
return jnp.transpose(Z_q_3d.reshape(nk, r_loc, mu_loc), (0, 2, 1))
# -> Z_q[q, mu_loc, r_loc]  at P(None, 'x', 'y')
```

The IFFT_k → FFT_k pair runs on the k-axis, which is **replicated**
in this layout, so each rank runs a local cuFFT — no resharding.

### 11.3.4 IBZ slice + per-channel solver

`isdf_fitting.py:1601-1621`. The FFT-built `Z_q` is naturally full-BZ.
The triangular solve has no inter-q coupling, so the kernel slices
IBZ rows of `L_q` and `Z_q` inside the jit via `L_q[q_irr_idx_j]` /
`Z_q[q_irr_idx_j]` with the closure-static `q_irr_idx_j` array
(`None` → fall back to full-BZ solve). The gather emits a
`dynamic-slice`-style indexed gather on the q-axis.

`solve_zeta` (`isdf_fitting.py:1125-1230`) dispatches by solver-kind:

| `solver_kind` | When | Backend | Native output spec |
|---|---|---|---|
| `cusolvermp_cholesky` | true 2D mesh, μ_L=0 | `ffi.cusolvermp.batched_distributed_cholesky_solve` | `P(None, 'x', 'y')` |
| `sharded_cholesky` | 1×P or P×1 mesh, μ_L=0 | in-tree shard_map triangular | `P(None, None, ('x','y'))` |
| `cusolvermp_lu` | true 2D mesh, μ_L≠0 | `ffi.cusolvermp.batched_distributed_lu_solve` | `P(None, 'x', 'y')` |
| `lu` | 1×P or P×1 mesh, μ_L≠0 | per-q `jnp.linalg.solve` with ridge | `P(None, 'x', 'y')` |

The solver landing reshard discipline
(`_reshard_zeta_mu_X_r_Y_to_mu_XY` / `_reshard_zeta_r_XY_to_mu_XY`,
`isdf_fitting.py:1080-1122`) stages through `P(None, 'x', 'y')` so
each step is a single-mesh-axis all-to-all:

```
cusolvermp branch:    P(None, 'x', 'y') → P(None, ('x','y'), None)
                      single all-to-all on 'y' (r → μ)

shard_map fallback:   P(None, None, ('x','y'))
                      → stage 1: P(None, 'x', 'y')          all-to-all on 'x'
                      → stage 2: P(None, ('x','y'), None)   all-to-all on 'y'
```

Two-mesh-axis reshards as one op trigger XLA's "Involuntary Full
Rematerialisation"; the staged form avoids it.

### 11.3.5 Sharding spec table (scan body)

| Object | Shape | Sharding | Where |
|---|---|---|---|
| `psi_l_X`, `psi_r_X` (jit args) | `(nk, n_rmu_padded, nb, ns)` | `P(None, 'x', None, None)` | shard_map in_specs at line 554 |
| `psi_G_bc_local` (io_callback out) | `(nk, bpd_max, ns, ngkmax)` | rank-local | `:639-642` |
| `psi_Y_bc_local_full_r` (post-IFFT) | `(nk, bpd_max, ns, n_zchunk)` | rank-local | `:650-653` |
| `psi_Y_bc_full_r` (post-gather) | `(nk, P·bpd_max, ns, n_zchunk)` | rank-local | `:658-660` |
| `psi_Y_bc` (post r-slice) | `(nk, P·bpd_max, ns, r_loc)` | rank-local | `:666-667` |
| Carry `P_l_acc`, `P_r_acc` | `(nk, ns, r_loc, mu_loc, ns)` | rank-local | `:628-631` |
| `Z_q` (return) | `(nq, n_rmu_padded, r_chunk)` | `P(None, 'x', 'y')` | `:556, :735-737` |
| `L_q[q_irr_idx]` | `(n_q_ibz, n_rmu_padded, n_rmu_padded)` | `P(None, 'x', 'y')` | `:1602` |
| `ζ` (solve output) | `(n_q_ibz, n_rmu_padded, r_chunk)` | `P(None, ('x','y'), None)` | `solve_zeta`'s output reshard |

`mu_loc = n_rmu_padded / p_x`. `r_loc = n_zchunk / p_y`. Pre-flight
check at `:454-457` raises if `n_zchunk % p_y != 0`.

---

## 11.4 `accumulate_rchunk_to_gflat` — r-chunk additive G-flat update

Source: `wfn_transforms.py:439-626`. For each r-chunk `R_c`,

$$\boxed{\tilde\zeta_{q,\mu}(K) = \sum_c \mathrm{FFT}_{r\to G}\!\big[\mathrm{pad}_{R_c\to[0, n_\mathrm{rtot})}(e^{-2\pi i\,q\cdot r}\,\zeta_{q,\mu}(r))\big](K),\quad K=q+G.}$$

One `shard_map` over `('x','y')`. Inputs and output:

```
in  : rchunk      (n_q_disk, n_rmu_padded, r_len)    P(None, ('x','y'), None)
      gflat_acc   (n_q_disk, n_rmu_padded, ngkmax)   P(None, ('x','y'), None)   ← donated
out : (n_q_disk, n_rmu_padded, ngkmax)               P(None, ('x','y'), None)
```

Per-rank flat-axis chunking on `(n_q · n_mu_local)` rows
(`:584-620`). Inside `lax.scan`:

1. Slice `cs` rows of the rank-local `rchunk` flat view.
2. Apply separable phase `phx[q_row] · phy[q_row] · phz[q_row]`
   (q-row clipped to `n_q-1` for pad rows; pad rows are zero so the
   clip is harmless).
3. Scatter into a zero FFT box `(cs, nx, ny, nz)`.
4. Local cuFFT `jnp.fft.fftn(box, axes=(-3,-2,-1), norm='backward')`.
5. `take_along_axis` at `sphere_c[q_row]` produces the per-q-G slice
   `(cs, ngkmax)`.
6. `dynamic_update_slice` into the donated `acc_flat`.

**No collective fires inside the body.** All cross-rank coordination
has already happened upstream — the rchunk's μ-axis is already
sharded, the r-axis (which is now contiguous in `r_flat = rx·ny·nz +
ry·nz + rz`) is replicated, and the FFT axes don't move across
ranks.

The chunker has a single tunable, `chunk_size` (env override
`LORRAX_GFLAT_CHUNK_SIZE`, cohsex.in `gflat_chunk_size`). Memory
bound per rank: `cs · n_rtot · 16 B` for the per-iter FFT box.
Suggested set: `cs ≈ 8e7 / n_rtot` for ~1 GB/rank.

---

## 11.5 V_q g-flat per-q kernel

Source: `v_q_g_flat.py:56-146` (`_make_per_q_kernel`),
`:294-492` (`_compute_V_q_g_flat_one_tile`).

### 11.5.1 The math

For one IBZ q and one (μ_L, ν_L) tile:

$$\boxed{V_q^{\mu_L \nu_L}[\mu,\,\nu] \;=\; \sum_{G \in \mathrm{sphere}(q)}\,\overline{\tilde\zeta_L^{\mu_L}(q, \mu, G)}\,v(q+G)\,t^{\mu_L \nu_L}(q+G)\,\tilde\zeta_R^{\nu_L}(q, \nu, G)}$$

`v(q+G)` is the dimension-aware bare Coulomb scalar
(`compute_vcoul.compute_v_q_per_G`); `t^{μ_L ν_L}(K)` is the bispinor
weight (`v_q_bispinor.py:127-174`):

| Tile | `t^{μ_L ν_L}(K)` |
|---|---|
| CC (0, 0) | 1 |
| TT diag (i, i) | `1 − K̂_i²` |
| TT off (i, j), i<j | `−K̂_i K̂_j` |
| (0, i), (i, 0) | 0 (Coulomb-gauge) |

Per-q the kernel computes one `(n_rmu_L, n_rmu_R)` block; the
accumulator `V_acc[q_ibz, μ, ν]` is updated via
`dynamic_update_slice`.

The CC tile additionally extracts the head term `g0_μ(q) = ζ̃_L(q, μ,
G=0) = ζ̃_L[μ, 0]` (by the sphere convention `sphere[q, 0] == 0`).

### 11.5.2 G-chunked accumulation via `lax.scan`

`v_q_g_flat.py:108-126`. The inner kernel chunks the G-axis with
`g_chunk = _pick_g_chunk(ngkmax, target=4096)` (largest divisor of
`ngkmax` ≤ 4096; `:226-231`). `n_chunks = ngkmax / g_chunk` (1 for
MoS2 3×3 `ngkmax=1963`; ~14 for CrI3 6×6 80 Ry `ngkmax ≈ 55k`).

```python
def _g_chunk_body(V_carry, i):
    start   = i * g_chunk
    L_chunk = dynamic_slice_in_dim(zeta_L, start, g_chunk, axis=-1)  # (n_rmu_L/p_x, g_chunk)
    R_chunk = dynamic_slice_in_dim(zeta_R, start, g_chunk, axis=-1)  # (n_rmu_R/p_y, g_chunk)
    v_chunk = dynamic_slice_in_dim(v_q,    start, g_chunk, axis=0)   # (g_chunk,)
    L_w     = jnp.conj(L_chunk) * v_chunk[None, :]
    return V_carry + L_w @ R_chunk.T, None

V_q, _ = jax.lax.scan(
    _g_chunk_body, V_q, jnp.arange(n_chunks, dtype=jnp.int32))
```

The scan keeps the HLO bounded (one body, `n_chunks` iters at
runtime) in contrast to the historical static Python loop, which
unrolled the HLO `n_chunks ×` and grew compile time linearly with
system size (CrI3 6×6 80 Ry hit ~14× unroll).

Each iter's GEMM is `(n_rmu_L/p_x, g_chunk) @ (g_chunk, n_rmu_R/p_y)
= (n_rmu_L/p_x, n_rmu_R/p_y)`. Both operands have G replicated (see
§11.5.3); the contraction is a **local cuBLAS GEMM with no
collective inside the scan body**.

### 11.5.3 Sharding of ζ_L and ζ_R

The two ζ slabs arrive from disk at read sharding
`P(None, ('x','y'), None)` (flat-μ across the mesh). The kernel
reshards them to single-axis before the GEMM
(`v_q_g_flat.py:96-102`): `zeta_L_q` (flat-XY on μ) →
`P('x', None)` for the left leg and `P('y', None)` for the right
leg. This is the **mesh-axis split** that lets the einsum

$$V[\mu_X, \nu_Y] = \sum_G \overline{L[\mu_X, G]} \cdot v[G] \cdot R[\nu_Y, G]$$

reduce over the (replicated) G axis with M sharded on `'x'` and N
sharded on `'y'`. The reshard from flat μ_XY → single-axis μ_X (and
μ_Y) is a combination of `all-to-all` and `all-reduce` (see §11.7.4
for the HLO-verified inventory).

| Array | Shape | Sharding | Notes |
|---|---|---|---|
| `zeta_L_q` (post-read) | `(1, n_rmu_L, ngkmax)` | `P(None, ('x','y'), None)` | disk → device, flat-μ |
| `zeta_L_3d` | `(1, n_rmu_L, ngkmax)` | `P(('x','y'), None, None)` | leading reshard |
| `zeta_L` | `(n_rmu_L, ngkmax)` | `P('x', None)` | left leg of einsum |
| `zeta_R` | `(n_rmu_R, ngkmax)` | `P('y', None)` | right leg |
| `v_q` per-q row | `(ngkmax,)` | `P(None)` | replicated |
| `V_q` per-q tile | `(n_rmu_L, n_rmu_R)` | `P('x', 'y')` | μ × ν tile |
| `V_acc` | `(n_q_ibz, n_rmu_L, n_rmu_R)` | `P(None, 'x', 'y')` | persistent, donated |
| `g0_acc` (CC only) | `(n_q_ibz, n_rmu_L)` | `P(None, 'x')` | persistent, donated |

`same_zeta=True` for CC `(0,0)` and TT-diag `(i, i)` tiles — caller
aliases one buffer for both operands. `same_zeta=False` for TT-off
`(i<j)` tiles where ζ_L and ζ_R come from distinct files (charge vs
current centroids, possibly different `n_rmu`).

### 11.5.4 The batched IBZ pre-read

`v_q_g_flat.py:435-447`. Before the q-loop, all IBZ ζ̃ slabs are
read in one batched HDF5 call via `read_L.read_all_ibz(n_q_ibz)`
(and similarly for `read_R` when `same_zeta=False`). The historical
pattern (`concatenate([read_L(q) for q in range])`) issued one
`read_slab` call per q, each producing a distinct `_per_rank`
closure id that triggered a JAX trace-cache miss in the FFI
shard_map dispatch. The 2026-05-12 switch to one batched read drops
trace retries by 9× (MoS2 3×3 bispinor: 63 → 7 retraces).

Per-rank cost: `n_q_ibz · n_rmu / p_x · ngkmax · 16 B` —
~10 MB / rank on MoS2 3×3; ~0.8 GB / rank on CrI3 6×6 80 Ry.

### 11.5.5 Async prefetch (not shipped)

`v_q_g_flat.py:33` documents but does not ship async prefetch — the
historical worker thread deadlocked against the kernel's NCCL
collective under heavy mesh contention. The sync per-q loop with
pre-read-all is already ~6× faster than the legacy μ × ν tile
driver, so prefetch is shelved.

### 11.5.6 Per-q kernel orchestrator

`_compute_V_q_g_flat_one_tile` (`v_q_g_flat.py:294-492`) end-to-end
flow: (1) resolve IBZ list (§11.6.1); (2) build `v_q_table` via per-tile
`v_per_G_builder` (bare Coulomb for CC, `1 − K̂_i²` for TT-diag,
`−K̂_iK̂_j` for TT-off); (3) pad `n_rmu_L`/`n_rmu_R` up to `∏ p_a`;
(4) zero-init `V_acc`, `g0_acc` at production shardings;
(5) compile `kernel = _make_per_q_kernel(...)` (compile-once per
`(mesh_xy, n_rmu_L, n_rmu_R, ngkmax, g_chunk, write_g0, same_zeta)`);
(6) batched IBZ pre-read (§11.5.4); (7) per-q loop calls
`kernel(V_acc, g0_acc, zeta_L_q, ..., v_q_dev[q], jnp.int32(q))` with
`jax.block_until_ready(V_acc)` after each q to bound the working set
(~1 GB on MoS2 3×3, ~6 GB on CrI3 6×6 80 Ry); (8) post-loop IBZ →
full-BZ unfold (§11.6.3).

---

## 11.6 IBZ cascade integration

The IBZ cascade activates **only if** the centroid set is closed
under the WFN sym group `{S | τ}`. The closure check is
`symmetry_maps.compute_centroid_sym_perm` at three sites:

1. **In `fit_zeta_to_h5`** (`isdf_fitting.py:2040-2063`): pre-flight
   check. Fail → set `write_ibz_only = False`, fall back to full-BZ
   on-disk q-axis.
2. **Inside `fit_one_rchunk`** (`isdf_fitting.py:1601-1610`):
   closure pre-resolved at `fit_zeta_to_h5` level, threaded via
   `q_irr_full_idx`. Kernel slices IBZ rows of L_q and Z_q before
   the solve.
3. **In `_compute_V_q_g_flat_one_tile`** (`v_q_g_flat.py:177-206`
   via `_resolve_ibz_q_list`): independent re-check. Fail → set
   `use_ibz = False` and iterate the full BZ in the q-loop
   (post-loop unfold becomes a no-op).

When the cascade fires:

* `n_q_disk = n_q_irr` (e.g. 8 for CrI3 6×6×1 with `ntran=6`
  spatial / 12 TRS-augmented; 2 for MoS2 3×3 with `ntran=8`; 8 for
  Si 4×4×4 with `ntran=48`).
* `zeta_q_G.h5` contains only the IBZ q's.
* V_q is computed at the IBZ q's only.
* `unfold_v_q` (post-loop) rotates each IBZ V_q row into all full-BZ
  q's that share its star, with centroid double-permute and umklapp
  L-phase.

### 11.6.1 `_resolve_ibz_q_list`

`v_q_g_flat.py:154-223`. Returns `(q_irr_kgrid_int, q_irr_frac,
full_to_irr_idx, full_to_irr_sym, sym_perm, L_table, use_ibz)`.

1. If `LORRAX_FORCE_FULL_BZ=1`, skip the IBZ cascade entirely
   (`:177`). Bypass switch for the V_q unfold during debugging.
2. Else call `compute_centroid_sym_perm(..., extend_trs=True)` to
   verify that every sym op permutes the centroid set. **Raises
   `RuntimeError` if not orbit-closed** — typical for `kmeans_cli
   --no-orbit` outputs.
3. On success: `q_irr_kgrid_int = sym.q_irr_kgrid_int`,
   `q_full_to_irr_idx = sym.irr_idx_q`, `q_full_to_irr_sym =
   sym.sym_idx_q`, `use_ibz = True`.
4. On failure: `q_irr = full BZ`, `use_ibz = False`. The unfold then
   becomes a no-op.

The `extend_trs=True` flag is **load-bearing**: `sym_perm` is then
shape `(2·n_tran, n_rmu)` with the second half duplicating the
spatial rows (centroid permutation is unchanged under TRS, only the
unfold's `conj` differs).

### 11.6.2 `write_ibz_only` — ζ writer side

`isdf_fitting.py:2075-2092`. When active, sets `n_q_disk = n_qpt_irr`
and `q_irr_frac = _bgw_wrap_q(sym.q_irr_kgrid_int) / kgrid`. The
`q_irr_full_idx` is threaded as a closure-static int array into
`fit_one_rchunk`'s jit; the IBZ gather of L_q / Z_q happens
**inside** the fused kernel rather than post-solve.

The `_compute_V_q_g_flat_one_tile` reader side
(`v_q_g_flat.py:344-349`) double-checks: `gvec_components.shape[0]`
must equal the resolved `n_q_ibz`, else the file was written with a
different `write_ibz_only` setting (incompatible).

Bispinor charge tiles write IBZ-only by default; bispinor μ_L>0
tiles currently write full-BZ until the bispinor V_q orchestrator
gains IBZ support (`isdf_fitting.py:1775`).

### 11.6.3 `unfold_v_q` (post-loop, full-BZ recovery)

Source: `symmetry_maps.unfold_v_q` (`services/symmetry_maps/src/symmetry_maps/maps.py`).

The bilinearity argument in ζ kills the τ-phase: V_q is bilinear in
ζ̃, and ζ̃ transforms as
$\tilde\zeta_{Sq, \pi_s(\mu)}(SG) = e^{-i(Sq+SG)\cdot\tau}\tilde\zeta_{q,\mu}(G)$,
so the two τ-phase factors `(+i)(−i)` cancel exactly. What survives
is a **centroid double-permute** plus a **per-centroid umklapp
phase** from the integer real-space lattice wrap
`L_μ = L_table[s, μ]`:

$$\boxed{V_{\mathrm{full}}[q, \mu', \nu'] \;=\; e^{2\pi i\, q_{\mathrm{irr}} \cdot (L_{s,\mu'} - L_{s,\nu'})}\; V_{\mathrm{ibz}}\big[i(q),\,\pi_s(\mu'),\,\pi_s(\nu')\big]}$$

where `i(q) = sym.irr_idx_q[q]`, `s(q) = sym.sym_idx_q[q]`,
`q_irr = q_irr_frac[i(q)]`. For TRS-augmented rows
(`s ≥ n_sym_spatial`) the rule is augmented with `conj`:

$$V_{\mathrm{full}}[\mathrm{TRS}\text{-}q, \mu, \nu] \;=\; \mathrm{conj}\big(V_{\mathrm{full}}[q, \mu, \nu]\big).$$

The L-phase factor is **essential** whenever `S r_μ + τ` exits the
unit cell (i.e. `L_μ ≠ 0`). On CrI3 (P-3 hex), every non-trivial
full-BZ q has nonzero L → the phase is the difference between an
ISDF-noise-floor V_q dump and a ~unity-relative-error wrong V_q.
See `reports/trs_sym_audit_2026-05-14/SYMMETRY_CONVENTIONS.md` for
the empirical validation.

**Implementation** (`symmetry_maps/maps.py`): a `@shard_map` body inside a
content-hashed `@jax.jit` with `in_specs = out_specs = P(None,'x','y')`.
The naive `take_along_axis` on a sharded μ (or ν) axis silently
forces XLA to all-gather that axis (Px× or Py× single-tile peak per
rank), which is unacceptable at large $P_x \cdot P_y$. The shard_map
body replaces the all-gather with two volume-preserving
`lax.all_to_all(tiled=True)` redistributions per spatial axis:

1. `all_to_all('x', split_axis=ν, concat_axis=μ)` — input
   `(n_q, μ/P_x, ν/P_y)` → output `(n_q, μ, ν/(P_x · P_y))`. Each
   rank now holds the full μ axis at the cost of an extra split on ν.
   **Per-rank byte count is unchanged** (`n_q · μ · ν / (P_x · P_y)`):
   it is a redistribution, not a broadcast.
2. `take_along_axis(perm_q[:, :, None], axis=μ)` on the now-local μ
   axis (no inter-rank traffic).
3. `all_to_all('x', split_axis=μ, concat_axis=ν)` — reverse step (1)
   back to the canonical sharding.
4. Same triple on `'y'` for the ν permutation.

Per-rank peak memory stays at exactly **1× single-tile**
(`n_q · n_rmu² / (P_x · P_y)`) for the entire unfold, regardless of
`P_x · P_y` (including the case `P_x · P_y > n_q^\text{full}`, which
breaks any q-axis-splitting design). Wire traffic per `all_to_all`
is the standard $(P-1)/P$ × tile per rank — NCCL `ncclAllToAll` on
GPU.

The umklapp phase `exp(2π i q_irr · L_μ)` is computed replicated
inside the body and applied via two `dynamic_slice_in_dim` calls
indexed by `lax.axis_index('x')` and `lax.axis_index('y')` to pick
this rank's μ-tile and ν-tile slices.

`mode='promise_in_bounds'` skips XLA's OOB-fill branch (`perm_q` is
permutation-by-construction so the check is gratuitous) and dodges
an HLO verifier `s32/s64` mismatch under shard_map+x64.

**Cache by signature.** `_get_unfold_v_q_jit` memoises the compiled
HLO by `(V_q shape, sym-table bytes, mesh id)`. V_q's call and W_q's
call (and any future caller at the same shape) share one compile.
The sym tables are closure-captured (constant-folded by XLA); a
runtime-args form is ~2× slower per call because JAX must marshal
the tables every invocation.

**Trivial-IBZ short-circuit**: when `ntran=1` (e.g. nosym runs), the
IBZ is already the full BZ — `irr_idx` is identity, `sym_idx` is all
zeros. The helper returns `V_q_ibz` directly without entering the
shard_map. Nosym runs incur zero collective cost.

**Callers.** Three call sites share this single helper:

* `gw/v_q_g_flat.py` — V_q g-flat path (the canonical V_q computation).
* `gw/v_q.py` — V_q legacy path (deprecation track).
* `gw/gw_jax.py` — W_q. The chi0/W block slices `V_q_full` and
  `chi0_q_full` to the IBZ via `slice_q_full_to_ibz` (same module),
  factors $(1 - V_q \chi_q)$ and solves at IBZ q's only, then
  unfolds the resulting $W_q$ with `unfold_v_q`. Wired 2026-05-16
  with the same Cholesky/LU helpers used by V_q. Reduces the dominant
  $W_q$ linalg cost by $\sim n_\text{tran}$×.

`slice_q_full_to_ibz(arr_full, q_irr_full_idx, *, out_sharding=...)`
is the IBZ-projection counterpart: a jit-cached gather along the q
axis preserving the `P(None,'x','y')` sharding. Together
`slice_q_full_to_ibz` + `unfold_v_q` form the IBZ↔full-BZ pair used
end-to-end for both V_q and W_q.

---

## 11.7 Sharding + communication

### 11.7.1 Mesh and axis assignment

```
Mesh(devices.reshape(p_x, p_y), ('x', 'y'))
```

Typical production meshes:

| System | Mesh | Notes |
|---|---|---|
| MoS2 3×3 (1 node) | 2×2 = 4 GPUs | bispinor + charge in one job |
| Si 4×4×4 60 Ry | 1×2 or 2×2 | full Oh symmorphic, IBZ cascade active |
| CrI3 6×6 30 Ry | 4×4 = 16 GPUs | charge-only, IBZ cascade active |
| CrI3 6×6 80 Ry | 4×4 = 16 GPUs (80 GB HBM) | charge-only; `hbm80g` required |

`p_x` and `p_y` are independent. `1×P` and `P×1` are degenerate
cases (some collectives become no-ops; cuSolverMp falls back to
in-tree sharded routines per `_resolve_solver_kind_charge/transverse`
at `isdf_fitting.py:831-911`).

### 11.7.2 Axis-sharding conventions

| Pipeline stage | Axis sharded | Reason |
|---|---|---|
| ψ at centroids (`psi_rmu_Y`) | μ on `'y'` | output of centroid loader; r-axis convention |
| ψ at centroids (`psi_rmuT_X`) | μ on `'x'` | conjugate-transposed; pair-density einsum's left side |
| ψ(G) host slab | n on `('x','y')` | flat band-sharded; each process owns 1/P bands |
| Pair-density carry | rank-local | inside shard_map body; `mu_loc` × `r_loc` per rank |
| `C_q` | μ_X, ν_Y | output of `c_q_from_psi_sm` |
| `L_q` | μ_X, ν_Y | factor preserves CCT's sharding |
| `Z_q` | μ_X, r_Y | output of `z_q_from_psi_sm` |
| ζ (solve output) | μ_XY, r | `solve_zeta` reshard discipline |
| `gflat_acc` (ζ̃ on disk's in-memory form) | μ_XY, G | μ flat-sharded; G replicated |
| `zeta_L` (V_q kernel left leg) | μ on `'x'` | post-reshard |
| `zeta_R` (V_q kernel right leg) | μ on `'y'` | post-reshard |
| V_q,μν output | μ on `'x'`, ν on `'y'` | square tile spec for downstream Σ_X |
| `g0` head (CC only) | μ on `'x'` | one-axis tile |

The single principle: **whatever axis the next consumer needs
sharded on `'x'` is sharded on `'x'` going in.** Reshards are
minimised by fixing one canonical sharding per object type.

### 11.7.3 Collective inventory (where each one fires)

The audit below is grounded in the actual HLO of the production
kernels — in particular
`runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/run_sym_hlo_dump_2026-05-15/
xla_dump/module_0490.jit_fn.sm_8.0_gpu_after_optimizations.txt`
for the V_q per-q kernel, and the round-6 audit notes in
`reports/zeta_rchunk_memory_model_2026-05-13/round6_discussion.md`
for the ζ-fit kernel.

| Site | File:line | Collective | Per-call size (CrI3 6×6 80 Ry, 4×4 mesh) |
|---|---|---|---|
| ζ-fit band gather | `isdf_fitting.py:658` | `all_gather` over `('x','y')` along band axis | ~27 MB / rank per bc iter |
| IBZ-q gather of L_q / Z_q | `isdf_fitting.py:1602` | `dynamic-slice`-style gather (q-axis replicated → indexed) | ~24 MB per r-chunk for L_q, ~70 MB for Z_q |
| `solve_zeta` solver landing reshard | `isdf_fitting.py:1080-1122` | one-axis `all_to_all` per stage (`'y'` → μ; optionally `'x'` → μ on shard_map path) | byte volume = ζ itself; ~3 ms / call on Perlmutter |
| V_q ζ_L reshard (XY → 'x') | `v_q_g_flat.py:101` | `all_to_all` and an `all_reduce` (see §11.7.5) | ~50 MB / q (MoS2) → ~840 MB / q (CrI3 80 Ry) |
| V_q ζ_R reshard (XY → 'y') | `v_q_g_flat.py:102` | `all_to_all` (skipped when `same_zeta=True`) | same |
| V_q G-chunk inner GEMM | `v_q_g_flat.py:123` | **none** — local cuBLAS GEMM | n/a |
| `unfold_v_q` μ-permute on 'x' | `symmetry_maps/maps.py` (shard_map body) | 2× `all_to_all(tiled=True)` on `'x'` | (P_x−1)/P_x × tile per rank, peak 1× tile |
| `unfold_v_q` ν-permute on 'y' | `symmetry_maps/maps.py` (shard_map body) | 2× `all_to_all(tiled=True)` on `'y'` | (P_y−1)/P_y × tile per rank, peak 1× tile |
| `slice_q_full_to_ibz` | `symmetry_maps/maps.py` | local gather on q axis; no collective | n/a (q axis is replicated) |
| W_q IBZ slice + unfold (gw_jax) | `gw/gw_jax.py` | same as `unfold_v_q` + `slice_q_full_to_ibz` above | identical to V_q's unfold |

The ζ-fit band gather is the single biggest collective per-call: it
fires once per bc-iter (10–25 iters per r-chunk × ~16 r-chunks at
CrI3 scale). The Round 6 HLO confirms it is **the only**
inter-shard_map collective inside the body — no nested psum, no
cross-replica-sum, no whole-band gather elsewhere.

### 11.7.4 V_q kernel collectives — HLO-verified

The G-axis is replicated on both `zeta_L` (`P('x', None)`) and
`zeta_R` (`P('y', None)`), so the einsum inside `_g_chunk_body` is a
purely local cuBLAS GEMM — **no collective fires inside the `lax.scan`
body**.

Verified against module 0490 of the Si 4×4×4 HLO dump (1×2 mesh,
single-tile CC kernel):

```
collective    shape              source_line   purpose
all-to-all    c128[2,1,216,588]  v_q_g_flat:96    leading reshard P(None,('x','y'),None) → P(('x','y'),None,None)
all-reduce    c128[1,432,588]    v_q_g_flat:101   squeeze: drop 'y' sharding → P('x',None) (select-zero + sum)
all-to-all    c128[1,2,216,588]  v_q_g_flat:102   right-leg reshard P(('x','y'),None,None) → P('y',None)
```

Three collectives total. All three are **μ-side reshards**, not
G-axis reductions. The `all-reduce` at line 101 is the XLA
implementation of the flat-`('x','y')` → single-`'x'` step: it
zeroes out non-owned μ rows (`loop_select_fusion`) and sums across
'y' to replicate. The `all-to-all` at 102 is skipped at the HLO
level when `same_zeta=True`.

The scan body itself emits a cuBLAS GEMM (`__cublas$gemm`) with no
collective. The G-axis combine across scan iters is a per-rank
accumulation into `V_carry`; the final `V_q` at `P('x', 'y')` has M
(μ) and N (ν) sharded on distinct mesh axes with G already
replicated, so no reduction across ranks is needed at the GEMM
contracting axis.

**Correction to earlier round notes.** Round-5 and Round-6
narratives that listed "one `all-reduce` per q on the G-axis" were
mis-attributing the reshard collective at line 101 to the
contraction. The all-reduce exists, but it is bound to the μ-axis
flat-to-single reshard, not to the G contraction. P2's HLO audit
listing "3 × all-reduce" double-counted the start/done/fusion
triple; there is only **one** logical all-reduce per q.

### 11.7.5 What is NOT in the audit (negative findings)

The following collectives DO NOT appear in the HLO of the current
production kernels:

* No `psum` anywhere in `fit_one_rchunk` (the rank-5 carry is
  rank-local and consumed locally in the post-pair tail).
* No `cross_replica_sum` — we use `all_gather` for the band axis,
  not `psum`.
* No whole-band gather (band axis is gathered per bc, not all at
  once).
* No collective inside `accumulate_rchunk_to_gflat`
  (`wfn_transforms.py:439-626` body has one shard_map and zero
  in-body collectives — the scan body is purely local FFTs +
  scatters).
* No collective inside the V_q `_g_chunk_body` scan
  (`v_q_g_flat.py:114-123`). The G-axis contraction is a pure
  local GEMM.
* No `all_gather` or sum-reduce inside `unfold_v_q`. The original
  implementation issued a full `all_gather` along μ and ν (Px× /
  Py× single-tile peak per rank). The current design uses paired
  `all_to_all(tiled=True)` redistributions that keep per-rank peak
  at 1× tile and scale to arbitrary `Px·Py`, including the regime
  `Px·Py > n_q^\text{full}`. See §11.6.3 for the design and
  `symmetry_maps/maps.py` for the body.

The clean count is the **direct payoff** of the scan-INSIDE-shard_map
discipline: every collective is a deliberate site (band gather, IBZ
gather, single-axis reshards, take_along_axis); there are no
incidental psums from JAX folding a multi-axis reshard into a
nested-collective form. The post-rewrite slot count is **1**
FFT-box slot per shard_map body (vs. 58 pre-rewrite for the
Python-unrolled bc loop).

---

## 11.8 Memory model — brief

See [`MEMORY_MODEL.md`](../architecture/memory-model.md) for the full per-stage
formulas and chunk-size derivation.

**Persistent pool** (fixed at problem-setup time):

```
B_persist = 2·nk · ns · n_rmu · n_band · 16 / P      ψ at centroids (L+R)
          + n_q · n_rmu² · 16 / P                    L_q factor
          + n_q_disk · n_rmu · ngkmax · 16 / P       gflat_acc
```

MoS2 3×3: ~0.5 GB / rank. CrI3 6×6 80 Ry: ~3 GB / rank
(`gflat_acc` dominates at ~0.66 GB).

**Workspace pool** `W_pool = B − B_persist`. Three transients alias
sequentially inside each r-chunk iter:

```
  Step         Block     Size                                  Knob
  ψ(G)→ψ(r)    W_wfn     k_chunk · band_chunk · ns · n_rtot    band_chunk_size,
               (1 FFT     · 16 · fft_factor / P                 psig_k_chunk_size
                slot)
  C_q / Z_q    W_zeta    3 · n_q · ns² · n_rmu · r_chunk · 16  r_chunk_size
               (3 pair-   / P                                   (dominant lever)
                density)
  ζ → G-flat   W_accum   gflat_chunk_size · n_rtot · 16 ·       gflat_chunk_size
               (1 FFT     fft_factor / P
                slot)
```

The three blocks **do not co-exist in time** within one iter; XLA
aliases them so the binding peak is `max(W_wfn, W_zeta, W_accum)`
rather than the sum. `W_zeta` is the binding peak at any reasonable
`r_chunk`.

**Chunk-size rule**: pick each chunk size as **large as memory
allows**. `r_chunk` first (biggest win), then `band_chunk`, then
`gflat_chunk_size`. Each draws from the same pool; doubling
`r_chunk` halves the per-iter overhead.

**Scan-INSIDE-shard_map invariant** (Round-5/6 fix; `MEMORY.md` entry
`feedback_path_d_scaffolding_pattern`): Python-unrolled inner loop
in jit ⇒ N× unsharded slots; `fori_loop` SPMD-replicates the
sharded carry; `scan(unroll>1)` preallocates an N× temp.
`scan(unroll=1)` inside `shard_map(check_rep=False)` is the only
pattern that preserves both per-iter slot aliasing **and** the
rank-5 carry's rank-local shape. Used in `z_q_from_psi_sm._local`
(`isdf_fitting.py:563-737`) and
`accumulate_rchunk_to_gflat._local` (`wfn_transforms.py:584-620`).

**V_q kernel memory**: peak bounded by the two pre-read ζ slabs
(`n_q_ibz · n_rmu/p_x · ngkmax · 16 B` each; ~0.8 GB/rank on CrI3
80 Ry, ~1.6 GB/rank on off-diag tiles). No W_zeta-class
scale-with-r_chunk term — V_q runs **after** the r-chunk loop.

**Known open memory cases**:
* Unsharded `W_wfn` at CrI3 6×6 80 Ry — loader's FFT box in
  `psi_G_store.fetch_psi_rchunk` is replicated; manual
  `psig_k_chunk_size = 6` cap is the workaround. Real fix:
  `with_sharding_constraint` at `load_wfns.py:657`. Open issue per
  `reports/zeta_v_q_g_flat_reference §5.8 fix priority #1`.
* cuSolverMp internal scratch (~`n_rmu²`-class) not modelled in the
  chunker; small at MoS2 3×3, ~few-GB at CrI3 6×6 80 Ry.
* No `W_vq` planner term; V_q's `vq_g_chunk_size` knob defaults to
  `_pick_g_chunk(ngkmax)` capped at 4096.

---

## 11.9 Performance

### 11.9.1 Si 4×4×4 60 Ry — overhead-dominated

`reports/gflat_perf_before_after_mos2_2026-05-12 §3`. Si is small
enough that the V_q hot loop is overhead-dominated: launch latency,
IBZ → full-BZ unfold, and FFI handshakes account for >40% of wall.
The IBZ cascade gives an **8× disk shrink** on `zeta_q_G.h5`:

| Stage | r-space ζ (legacy) | G-flat ζ (current) |
|---|---|---|
| ζ-fit wall | 14 s | 11 s |
| ζ-fit disk write | 21 s | 1.2 s |
| `zeta_q.h5` size, sym IBZ | 252 MB | 35 MB |
| `zeta_q.h5` size, nosym | 252 MB | 252 MB |

V_q wall-clock is overhead-dominated; the per-q kernel takes
microseconds per call.

### 11.9.2 MoS2 3×3 80 Ry, 2×2 mesh

`reports/gflat_e2e_bispinor_mos2_3x3_2026-05-11 §3-4`. WFN load +
centroids 2.1 s; CCT 1.4 s; Cholesky 3.6 s; r-chunk loop (4 chunks)
7.5 s; ζ write 1.2 s; V_q charge loop 1.8 s (2 IBZ × ~0.9 s);
unfold 0.4 s. Per-rank peak ~12.8 GB on 80 GB A100.

### 11.9.3 CrI3 6×6 30 Ry — q-loop dominates

`reports/gflat_e2e_bispinor_mos2_3x3_2026-05-11 §5`. At larger scale
the V_q q-loop dominates. The IBZ cascade gives a `4.5×` reduction
in q-loop work; the per-q GEMM dominates so the total V_q time
scales linearly with `n_q_ibz`. Measured **~6× speedup** in V_q
against the legacy μ × ν tile driver.

| Stage | Legacy μ×ν driver | G-flat (current) |
|---|---|---|
| ζ-fit | 188 s | 71 s |
| V_q (charge, 8 IBZ) | 73 s | 12 s |
| V_q unfold | n/a (no IBZ) | 0.7 s |
| `zeta_q.h5` size | 18 GB | 1.7 GB |

### 11.9.4 CrI3 6×6 80 Ry — production target

`reports/zeta_v_q_g_flat_reference_2026-05-12 §10`. Required cohsex
knobs:

```ini
memory_per_device_gb  = 60.0
band_chunk_size       = 16
r_chunk_size          = 0     # planner picks ~12 500
gflat_chunk_size      = 64    # bound accumulate FFT box ≤ ~1 GB/rank
psig_k_chunk_size     = 6     # bound the unsharded band-load FFT box
```

* ζ-fit per r-chunk: ~12 s (16 r-chunks × ~12 s ≈ 3 min for full
  ζ-fit).
* CCT + Cholesky (once): ~25 s (cuSolverMp `potrf` on 4×4 mesh).
* V_q per IBZ q: ~3 s (8 q's × 3 s ≈ 24 s for V_q full IBZ pass).
* `unfold_v_q`: ~0.5 s.

End-to-end ζ + V_q: ~4–5 min wall-clock on 16 GPUs (4×4 mesh,
80 GB HBM). This is the regime where the G-flat refactor + IBZ
cascade together deliver the 21× disk-size win documented in
`reports/gflat_e2e_bispinor_mos2_3x3_2026-05-11`.

The fused kernel's irreducible XLA floor is ~28 GiB / rank — only
80 GB A100s fit. The `accumulate_rchunk_to_gflat` peak is 1.15 GB
per rank at `chunk_size=64`.

---

## 11.10 File pointers

### 11.10.1 Source

| File:Line | Symbol | Role |
|---|---|---|
| `isdf_fitting.py:82-148` | `pair_density`, `accum_pair_density` | rank-5 open-spin P_k,αβ(μ, col) |
| `isdf_fitting.py:256-376` | `c_q_from_psi_sm` | CCT pipeline (centroid-sample variant) |
| `isdf_fitting.py:378-744` | `z_q_from_psi_sm` | ZCT (r-chunk-sample) scan-INSIDE-shard_map |
| `isdf_fitting.py:914-1080` | `factor_c_q` | Cholesky (μ_L=0) / pivoted-LU (μ_L≠0) |
| `isdf_fitting.py:1080-1124` | `_reshard_zeta_*` | post-solve μ_XY landing reshards |
| `isdf_fitting.py:1125-1465` | `solve_zeta` | per-q triangular / LU + cuSolverMp dispatcher |
| `isdf_fitting.py:1467-1717` | `_make_fit_one_rchunk_kernel`, `fit_one_rchunk` | fused jit factory + entry point |
| `isdf_fitting.py:1751-2567` | `fit_zeta_to_h5` | top-level driver, r-chunk loop, IBZ resolver |
| `wfn_transforms.py:439-626` | `accumulate_rchunk_to_gflat` | r-chunk → G-sphere FFT-and-accumulate |
| `psi_G_store.py` | `PsiGStore`, `build_psi_G_store` | host-resident ψ(G) cache + slicer |
| `symmetry_maps` (service door) | `compute_centroid_sym_perm` | centroid sym closure + L-table |
| `gw/v_q_g_flat.py:56-146` | `_make_per_q_kernel` | compile-once per-q G-chunked GEMM kernel |
| `gw/v_q_g_flat.py:154-223` | `_resolve_ibz_q_list` | IBZ cascade gate + centroid orbit closure |
| `gw/v_q_g_flat.py:226-231` | `_pick_g_chunk` | largest divisor of `ngkmax` ≤ `target=4096` |
| `gw/v_q_g_flat.py:234-287` | `_make_read_q` | ZetaReader / ZetaLoader, batched `read_all_ibz` |
| `gw/v_q_g_flat.py:294-492` | `_compute_V_q_g_flat_one_tile` | per-tile end-to-end (read + q-loop + unfold) |
| `gw/v_q_g_flat.py:499-569` | `compute_all_V_q_g_flat` | charge-channel public entry point |
| `gw/v_q_bispinor.py:57-174` | `UNIQUE_TILES`, `_make_v_per_G_for_tile` | 7-tile enumeration + per-tile weight |
| `gw/v_q_bispinor.py:482-…` | `compute_V_q_bispinor_g_flat_to_h5` | bispinor orchestrator |
| `symmetry_maps` (service door) | `find_irreducible_bz_points` | IBZ k/q resolver |
| `symmetry_maps` (service door) | `slice_q_full_to_ibz` | full-BZ → IBZ q-axis gather (sharding-preserving, jit-cached) |
| `symmetry_maps` (service door) | `unfold_v_q` | IBZ → full BZ centroid double-permute + L-phase + TRS conj; shard_map + paired `all_to_all` for 1×-tile peak per rank |
| `symmetry_maps` (service door) | `SymMaps` | sym tables, IBZ index helpers |
| `gw/gw_jax.py` | chi0/W block | W_q = (1-V_qχ_q)⁻¹V_q solved at IBZ via `slice_q_full_to_ibz` + Cholesky/LU, unfolded with `unfold_v_q` |
| `file_io/zeta_reader.py` | `ZetaReader` | G-flat per-q HDF5 slab reader |
| `file_io/slab_io.py` | `SlabIO` | phdf5 writer with `valid_shape=` μ-pad clip |

### 11.10.2 Cross-references

| Doc | Focus |
|---|---|
| §3–5, §7 of `physics.md` | Historical r-space ζ-on-disk path; sharding map |
| `architecture/memory-model.md` | Per-stage memory formulas + chunk-size selection rules |
| `reports/zeta_v_q_g_flat_reference_2026-05-12/report.md` | Living engineering reference: donations, shardings, chunker envs, CrI3 validation log |
| `reports/zeta_rchunk_memory_model_2026-05-13/PATH_D_PICKUP.md` | Zero-replicated-intermediates principle; scan-INSIDE-shard_map context |
| `reports/zeta_rchunk_memory_model_2026-05-13/round{5,6,8}_*.md` | Round 5 plan, Round 6 HLO audit, Round 8 unified-FFT design |
| `reports/trs_sym_audit_2026-05-14/{SYMMETRY_CONVENTIONS,hlo_findings}.md` | BGW sym + TRS-row rule; HLO audit of `unfold_v_q` |
| `reports/zeta_ibz_2026-05-11/report.md` | IBZ-only ζ-on-disk schema design + symmetry derivations |
| `reports/v_q_bispinor_plan_2026-05-08/report.md` | Lorentz tile sectorization + V_q_bispinor container |
| `reports/gflat_e2e_bispinor_mos2_3x3_2026-05-11/report.md` | First G-flat end-to-end bispinor; 21× disk-size win |
| `reports/gflat_perf_before_after_mos2_2026-05-12/report.md` | Before/after performance on MoS2 3×3 + Si 4×4×4 |

---

## 11.11 Quick reference

| Quantity | Formula |
|---|---|
| Pair density (open-spin) | `P_{k,αβ}(μ, col) = Σ_n ψ*_{n,k,α}(r_μ) ψ_{n,k,β}(r_col)` |
| CCT (lattice convolution) | `C_q = FFT_{R→q}[γ̃·γ̃ · conj(IFFT(P_l)) · IFFT(P_r)]` |
| ZCT | same with `col = r_chunk` instead of `col = ν` |
| ISDF normal equations | `C_q ζ_q = Z_q` per q, per μ_L |
| Cell-periodic ζ | `z_{q,μ}(r) = e^{-2πi q·r} ζ_{q,μ}(r)` |
| G-flat ζ (on disk) | `ζ̃_{q,μ}(K) = (1/N_r) FFT_{r→G}[z_{q,μ}(r)](K=q+G)` |
| r-chunk additivity | `ζ̃(q,μ,K) = Σ_c FFT[pad(phase · ζ_c)](K)` |
| ζ sym transform | `ζ̃_{Sq, π_s(μ)}(SG) = e^{-i(Sq+SG)·τ} ζ̃_{q,μ}(G)` |
| V_q (G-chunked) | `V_q^{μ_L ν_L}[μ,ν] = Σ_G conj(ζ̃_L) · v(q+G) · t^{μ_L ν_L}(q+G) · ζ̃_R` |
| V_q transverse weight | `t^{ij}(K) = δ^{ij} − K̂_i K̂_j` |
| V_q scalar unfold | `V_full[q,μ',ν'] = e^{2πi q_irr·(L_μ − L_ν)} · V_irr[i(q), π_s(μ'), π_s(ν')]` |
| V_q TRS unfold | `V_full[TRS-q, μ, ν] = conj(V_full[q, μ, ν])` |

Algorithmic spine (one-liners):

* **Pair-density scan body**:
  `for bc in n_bc:  psi_Y = all_gather(IFFT(io_callback(bc)));  P_l/r += einsum(psi_X[bc], mask(psi_Y))`.
* **CCT/ZCT tail** (post-scan):
  `Z = FFT_k(γ̃·γ̃(conj(IFFT_k(P_l)), IFFT_k(P_r)))`.
* **r-chunk-additive G-flat update**:
  `gflat_acc[q,μ,K] += FFT_3d(pad_R_c(phase·ζ_chunk[q,μ,r]))[sphere[q,K]]`.
* **V_q per-q G-chunked GEMM**:
  `V_q[μ,ν] = Σ_chunks (conj(L_chunk) ⊙ v_chunk) @ R_chunk.T` — local cuBLAS, no in-body collective.
* **IBZ → full unfold**:
  `V_full[q,μ',ν'] = e^{2πi q_irr·(L_μ − L_ν)} · V_ibz[i(q), π_s(μ'), π_s(ν')]`,
  `conj`-wrap when `s(q) ≥ n_sym_spatial` (TRS row).

---
