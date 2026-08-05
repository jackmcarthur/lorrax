# Padding-cleanliness audit — Scope 2: V_q / W / symmetry unfold / head rank-1

Repo: `sources/lorrax_D`, branch `agent/memplanner-cleanup` @ `62b0365` (+1 uncommitted edit).
Question: is μ-padding handled with the fewest lines and ONE convention around the key GW
operations, judged against the ideal — pad born once at ingest, `(n_logical, n_padded)`
carried in ONE place (meta), consumers either structurally pad-neutral or using ONE
canonical helper?

Uncommitted at audit time: `src/gw/gw_init.py` only (+6/−1: clip `g0_mu` to logical
before the zeta-h5 disk write). The `mu_logical_mask` in `ppm_sigma._prepare_sigma_state`
is COMMITTED (62b0365), as are the logical-extent solves (b0b0626) and the
`LORRAX_EXTRA_MU_PAD` knob (083d209). New untracked test files:
`tests/test_mu_pad_invariance.py`, `tests/multi_device/`.

## 1. The convention as it exists

There IS one convention, stated in `runtime/padding.py:1-29` and SHARDING_RULES §2:
in-memory μ extent = `padded_mu_extent(n_rmu, ∏p_a)` with exact-zero pad rows; disk stores
the logical extent (SlabIO `valid_shape=`); `Meta` carries both `n_rmu` and
`n_rmu_padded` (`meta.py:24,40-45,136`). Post-fix, all three scope-2 pad-extent producers
route through the single `padded_mu_extent` function (meta, `v_q_g_flat.py:359-366`,
`v_q_bispinor.py:513-516`, `gw_init.py:209-211`, `wfn_transforms.py:1762`) — commit
083d209 explicitly unified two V_q-side local round-ups that had drifted ("the knob
exposed that as a shape crash").

But the carrier is NOT one place: `Meta` holds only the charge-channel pair. The bispinor
per-tile L/R extents are re-derived at 3 sites (v_q_g_flat tile `_pad`, BispinorVqReader
`_padded_shape_LR`, gw_init transverse `meta_curr` refresh), each recomputing
`p_x·p_y` locally. And **three divisor conventions are alive** in the wider dataflow:
mesh-product (`padded_mu_extent`), host-count (`meta.n_rmu_jax`, `gw_init.py:201`,
"legacy field"), and `lcm(gx,gy)` (`bse_io.py:439-440`, the head-injection consumer).

## 2. Site inventory (scope 2)

Classes: **SN** = structurally neutral (zero pad-specific code; zero rows kill the
contribution by construction) · **CH** = canonical helper (`padded_mu_extent` /
SlabIO `valid_shape` / meta) · **AH** = ad-hoc per-site logic · **LATENT** = neutrality
assumed, not structural.

| # | Site | File:lines | Class | Pad LOC |
|---|------|-----------|-------|---------|
| 1 | Per-q V kernel (`conj(ζ_L)·v·ζ_Rᵀ` G-chunked GEMM) | v_q_g_flat.py:88-141 | **SN** — ζ pad rows zero ⇒ V pad rows/cols and g0 pad entries exactly zero | 0 |
| 2 | ζ read into padded buffer | v_q_g_flat.py:232-264 (`valid_mu=` → zeta_reader.py:267-271 `valid_shape=`) | **CH** | ~4 |
| 3 | Tile pad extents L/R | v_q_g_flat.py:355-373 (`_pad`) | **CH**, but re-derivation (not meta-carried) | ~12 |
| 4 | V/g0 accumulators born at padded extent, `P(None,'x','y')` | v_q_g_flat.py:376-385 | **CH** (uses #3) | ~3 |
| 5 | `unfold_v_q` perm-table pad: identity block appended to `fwd_perm`, zero block to `L_table`, pad-invariant guard | symmetry_maps.py:282-310 | **AH** | ~23 |
| 6 | `unfold_v_q` divisibility gate `n_rmu_padded % (Px·Py)` | symmetry_maps.py:381-389 | **CH** (guard on meta contract) | ~9 |
| 7 | `unfold_v_q` gather/phase/TRS on padded rows | symmetry_maps.py:392-468 | **SN given #5** — see §3 | 0 |
| 8 | `_unfold_g0_ibz_to_full` perm pad | v_q_g_flat.py:605-635 | **AH — duplicate of #5** (argsort variant) | ~14 |
| 9 | χ₀ (G bilinear in ψ; minimax τ-scan) | w_isdf.py:48-197 | **SN** — ψ pad rows zero ⇒ G, χ₀ pad rows/cols zero | 0 |
| 10 | W Dyson solve, JAX_NATIVE: per-q LU μ-sliced to `n_log`, W pad zero-filled via zeros-init `W_acc` | w_isdf.py:206-291 (Fix 1b, commit b0b0626) | **CH intent, AH mechanics** — hand-rolled `[:n_log,:n_log]` slice per-site, no shared "solve-at-logical" helper | ~30 |
| 11 | W-solve nq round-up/pad/slice-back (q axis) | w_isdf.py:255-260,289-290 | **AH** (inline round-up, not the helper) but SN in effect (pad q: V=0 ⇒ A=I, W=0) | ~8 |
| 12 | W Dyson solve, CUBLASMP_FFI | w_isdf.py:297-329,372-378 | **LATENT (documented)** — runs at PADDED extent; ≤1e-8-rel pad-extent sensitivity retained (":374-376") | ~3 (comment) |
| 13 | `n_rmu_logical` plumbing `getattr(meta,'n_rmu',n_rmu)` | w_isdf.py:380-388 | **CH** (meta-carried) with a soft silent fallback | ~8 |
| 14 | Bispinor tile writes: padded memory → logical disk | v_q_bispinor.py:330-349,386-395 | **CH** (SlabIO `global_shape=valid_shape=` logical) | ~14 |
| 15 | `BispinorVqReader._padded_shape_LR` + `get_tile`/`get_g0_CC`/`_zero_tile` | v_q_bispinor.py:482-563 | **CH** (routes through `padded_mu_extent`; docstring pins it to the ψ-side extent) — re-derivation, not meta | ~30 |
| 16 | TT Lorentz mixing at padded extent | v_q_bispinor.py:362-397 (`unfold_v_q_bispinor_lorentz`) | **SN** — per-q linear mix of tiles that all have zero pad rows | 0 |
| 17 | gw_init bispinor re-pad after logical-disk read | gw_init.py:465-475 | **AH** (inline `jnp.pad` to `meta.n_rmu_padded`; correct, commented) | ~10 |
| 18 | gw_init `g0_mu` zeta-h5 write clipped `[..., :meta.n_rmu]` | gw_init.py:511-519 | **AH — UNCOMMITTED** host-side slice | ~6 |
| 19 | gw_init transverse meta refresh (`n_rmu/_jax/_padded` on `meta_curr`) | gw_init.py:200-222 | **CH** (meta as carrier — the right pattern) but carries the legacy `n_rmu_jax` host-count divisor | ~14 |
| 20 | Head rank-1 `conj(g0)⊗g0` injection | head_correction.py:731-815 | **SN** — g0 pad entries zero ⇒ pad rows/cols of the update zero, at any μ extent | 0 |
| 21 | Restart-state writes: `V_qmunu`, `G0_mu_nu` (gw_init.py:688-693), `W0_qmunu` (gw_jax.py:310, tagged_arrays.py:103-114) | file_io/tagged_arrays.py:60-88 | **MISSING** — written at the PADDED in-memory extent, no `valid_shape` clip. See §4.1 | 0 (that's the problem) |
| 22 | BSE head consumer: `lcm(gx,gy)` pad + G0 re-pad-if-smaller (no trim-if-larger) | bse_io.py:438-475 | **AH + LATENT** (third divisor convention; inherits #21's extent) | ~12 |

Canonical module itself: `runtime/padding.py` = **402 lines**. Live in production:
`padded_mu_extent` + `extra_mu_pad` (~65 lines incl. docs). The advertised array-level
canon — `PadAxis`, `pad_shape_to_mesh`, `pad_array_to_mesh`, `unpad_array_from_mesh`,
`valid_shape_from_pad_meta`, `logical_shape_from_padded`, `round_up_to_mesh_product` —
has **zero consumers in `src/`** (only `tests/test_padding.py`). ~230 lines of dead
canonical API; the codebase de facto chose SlabIO `valid_shape` as its array-level
mechanism instead.

**Counts:** 22 pad-aware sites in scope · SN 5 · CH 8 · AH 6 · LATENT/MISSING 3.
Pad-specific LOC in consumers ≈ **190** (incl. comments/docstrings, which are roughly
half — the fixes are heavily annotated); plus the 402-line helper module (~230 dead).

## 3. Does the sym permutation ever index pad rows?

No — but by per-site construction, not structurally, and a mistake would be silent.
`fwd_perm` logical entries are `< n_rmu_logical` by construction
(`centroid/orbit_syms.py:compute_centroid_sym_perm` permutes the logical centroid set);
pad rows are appended as an identity block (`symmetry_maps.py:287-290`) so pad→pad,
gathering zeros; `L_table` pad rows are zeros ⇒ phase `exp(0)=1` on rows that are zero
anyway; TRS `conj(0)=0`. So the unfold IS pad-neutral. Two caveats:

1. Every gather uses `mode='promise_in_bounds'` (symmetry_maps.py:413-439,
   v_q_g_flat.py:656-657). If the pad-extension block were ever wrong/missing, the OOB
   would clip/wrap **silently** — the exact failure shape of the TRS-blind sym bug.
   The `elif n_rmu_padded != n_rmu_logical: raise` guard (:291-295) covers only the
   too-small direction.
2. The pad-extension logic is duplicated (site #5 vs #8) with a real divergence hiding
   in the duplicate: `unfold_v_q` uses `fwd_perm` directly (the order-3-correct
   convention per its own comment :273-281) while `_unfold_g0_ibz_to_full` uses
   `inv_perm = argsort(sym_perm)` (:625). Documented as unobservable (only the Γ slot
   of g0 is consumed), but it is exactly the "two parallel sym-action helpers" pattern
   the unified-sym-action rule forbids.

## 4. Latent-bug candidates (pad neutrality assumed, not structural)

1. **Restart files store P-dependent padded extents** (site #21) — `V_qmunu`,
   `G0_mu_nu`, `W0_qmunu` are written with `global_shape = arr.shape` = the padded
   in-memory extent and NO `valid_shape` clip (`tagged_arrays.py:63-65,110-114`).
   This violates the §2 disk contract ("disk stores logical so any process count can
   re-read") on the *output* side of the exact stage this audit covers. Consequences:
   a restart/BSE file written at P=16 (μ=1216) differs in shape from one written at
   P=4 (μ=1204); `bse_io` recovers "n_rmu" from the dataset shape (:411) so pad rows
   masquerade as physical centroids downstream; `bse_io.py:468-471` re-pads G0 only
   if *smaller* — a file with a *larger* pad than the reader's `lcm`-pad has no trim
   path (shape mismatch or silent extent leak). Zero rows keep the *math* neutral in
   the bilinear ring contraction, but the extent-leak-to-disk is the same defect class
   as ROOT_CAUSE, one hop later. The uncommitted gw_init edit fixes this for the
   zeta-h5 `g0_mu` copy but NOT for the restart `G0_mu_nu` written 170 lines later in
   the same function (gw_init.py:690) — the same vector, two conventions, one function.
2. **CUBLASMP_FFI W-solve still factorizes at the padded extent**
   (w_isdf.py:372-378) — documented ≤1e-8-rel pad-extent sensitivity (block-cyclic
   layout needs mesh divisibility). Accepted residual, but it means "solve at logical"
   is a per-backend property, not an invariant; the Tier-1 bit-identity gate holds
   only for the JAX_NATIVE path.
3. **`_resolve_w_solve_fn`'s `getattr(meta, 'n_rmu', n_rmu)` fallback**
   (w_isdf.py:387) — a caller passing a meta-like object without `n_rmu` silently gets
   a padded-extent LU back (the pre-fix behavior). Fine for the one synthetic-meta test
   caller, but it makes the invariance fix opt-out-by-omission rather than fail-loud.
4. (adjacent, flagged by ROOT_CAUSE) distributed ζ-Cholesky keeps the padded extent
   at multi-GPU — outside this scope (isdf/core.py) but the same non-structural class.

## 5. Verdict: clean-ish convention, not minimal — concrete reductions

**One convention exists and the recent fix series measurably converged on it** (single
`padded_mu_extent`, SlabIO valid_shape, meta as carrier, zero-row invariant + Tier-1
gate). The heavy operations themselves (V bilinear, χ₀, TT mixing, head rank-1) are
structurally neutral with zero pad code — that part is as clean as the ideal. The
residue is at the edges, and fewer sites/lines ARE achievable:

1. **Bake the pad into the sym tables at construction** — have
   `compute_centroid_sym_perm` / the SymMaps accessor emit `fwd_perm` and `L_table`
   already at `n_rmu_padded` (identity/zero tail). Deletes sites #5 and #8 (~37 lines,
   one of them a duplicate with a divergent inv/fwd convention) and closes the
   silent-OOB exposure in one place. This is the "unified sym action" direction already
   on record.
2. **Clip restart writes to logical** — `valid_shape=(nq, meta.n_rmu, meta.n_rmu)` in
   `write_restart_state_to_h5` / `write_w0_qmunu_to_h5` (~4 changed lines), and re-pad
   on read via `padded_mu_extent` like gw_init.py:465-475 already does for bispinor
   tiles. Kills latent #1 and lets `bse_io` drop its own `lcm` convention (site #22)
   for the standard one. Highest value-per-line change in this scope.
3. **Fold the uncommitted g0 clip into (2)** — with logical-on-disk done centrally in
   the writer, the hand-rolled `[..., :int(meta.n_rmu)]` slice (site #18) becomes the
   same one-argument `valid_shape` idiom instead of a fourth mechanism.
4. **Decide the PadAxis layer's fate** — adopt it (sites #11, #17 become 2-line calls)
   or delete it (−~230 lines). Today it is the worst state: a documented canonical API
   nothing uses, which future contributors will read as the contract.
5. **One `solve_at_logical` idiom** — the μ-slice+zero-fill pattern now exists in three
   hand-rolled variants (w_isdf per-q LU; isdf/core per-q ζ solve and cuSolverMp path,
   per ROOT_CAUSE Fix 1). A single shared helper (slice both operands to `n_log`, solve,
   embed into zeros) would make "solves run on logical" a grep-able invariant instead of
   a per-site discipline — the defect class ROOT_CAUSE proved this codebase repeatedly
   falls into.
6. **Retire `n_rmu_jax` (host-count divisor)** or document why it must differ; two
   Meta fields with different round-ups of the same quantity is the seed of the next
   extent bug.

Net: with (1), (2), (4-delete) the scope-2 pad-specific consumer code drops from
~190 lines / 22 sites to roughly ~110 lines / 15 sites, the disk contract becomes
universal, and the only remaining ASSUMED site is the documented FFI solve.
