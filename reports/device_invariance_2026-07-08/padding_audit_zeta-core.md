# Padding cleanliness audit — SCOPE 1: birth sites + ζ-fit core

Tree: `sources/lorrax_D` @ `agent/memplanner-cleanup` (HEAD 62b0365).
Uncommitted at audit time: `src/gw/gw_init.py` (+6/-1: `g0_mu` disk write clipped to
logical μ — see C11) plus new untracked test files (`tests/test_mu_pad_invariance.py`,
`tests/multi_device/`, `.padprobe/`, `.tier2_cross_p/`). NOTE: contrary to the tasking
note, `mu_logical_mask` in `ppm_sigma._prepare_sigma_state` is already COMMITTED
(62b0365); the only in-flight source edit is the g0 clip.

Judged against the ideal: pad born ONCE at ingest; (n_logical, n_padded) carried in ONE
place (meta); consumers either structurally neutral or using ONE canonical helper.

---

## 1. The birth-site design (verdict first)

**The scheme itself is minimal and sound.** `n_rmu_padded = round_up(n_rmu, ∏p_a)`
with exact-zero pad rows is the weakest padding that makes every PartitionSpec
(single-axis or product-axis) divide, and zero rows make every *bilinear* consumer
neutral by construction (pair density, CCT, V_q, Σ projections — zero code at those
sites). Because the pad is always a trailing contiguous block, the **integer pair
(meta.n_rmu, meta.n_rmu_padded) is the minimal carrier** — a threaded boolean mask
would be strictly more API for no more information. The one consumer that genuinely
needs a mask (PPM mode census) builds it in one line from `meta.n_rmu`
(`ppm_sigma.py:600`). So: no, the pad mask should NOT be constructed once and
threaded; the n_log integer should be, and post-fix it mostly is.

**But the implementation around the scheme is not minimal.** Three concrete failures:

### 1a. A dead second convention lives in `runtime/padding.py`

`padding.py` is 402 lines. Production uses exactly two functions from it:
`padded_mu_extent` + `extra_mu_pad` (~65 lines, the single source of truth for the
μ extent — good, landed in 083d209). The other **~337 lines** — `PadAxis`,
`pad_array_to_mesh`, `unpad_array_from_mesh`, `pad_shape_to_mesh`,
`valid_shape_from_pad_meta`, `logical_shape_from_padded`, `round_up_to_mesh_product`,
`_spec_axes`, `_spec_divisor` — have **zero callers in `src/`**. Their only consumer is
`tests/test_padding.py` (427 lines testing dead API). `slab_io.py:190-191` still
points readers at "``runtime.padding`` in the agent/padding-refactor branch" — an API
that was designed, tested, and never adopted. This is exactly the two-parallel-paths
antipattern: a future contributor can pick the PadAxis convention while every live
site uses `meta.n_rmu_padded` + ad-hoc `jnp.pad`. **Delete it (or adopt it everywhere;
given the no-new-API-layers rule, delete): −337 src lines, −427 test lines.**

### 1b. round-up arithmetic exists in ≥4 spellings

`padded_mu_extent` (canonical), `meta._round_up` (bands, meta.py:8), an inline
`((n + p - 1)//p)*p` in `gw_init.py:201` (legacy n_rmu_jax refresh), and **three**
inline `((n_zchunk + Py - 1)//Py)*Py` copies inside `solve_zeta`
(core.py:1266, 1311, 1346). Not wrong, but five sites where one `round_up(n, d)`
would do.

### 1c. Dead legacy padded fields in Meta

`n_rmu_jax`, `nbnd_jax`, `n_rtot_jax` (meta.py:127-129, `round_up(·, n_proc)` —
documented as the WRONG divisor, host count not device count) have **zero consumers**
outside meta.py itself and the gw_init.py:201 refresh that dutifully keeps
`n_rmu_jax` up to date for nobody. Delete all three fields + the refresh (~10 lines,
removes a booby trap: any future reader of `n_rmu_jax` gets a wrong-divisor shape).

### Where padding is actually born (5 sites, all correct)

| # | Site | What | Class |
|---|------|------|-------|
| B1 | `meta.py:136` | `n_rmu_padded = padded_mu_extent(n_rmu, world)` | CANONICAL (the one place) |
| B2 | `meta.py:109` | `b_id_4 = _round_up(b_id_4_user, world)` band pad | canonical-ish (own helper) |
| B3 | `wfn_transforms.py:1762,1844-48` | ψ centroid μ-pad: `padded_mu_extent` + `jnp.pad` inside `_reshard_all` — the array-level birth | AD-HOC pad (canonical `pad_array_to_mesh` unused), extent canonical |
| B4 | `wfn_transforms.py:1803-1877` | band pads: past-file zero-concat, pad slice-off, user-band zeroing | ad-hoc, structurally neutral (zeros) |
| B5 | `gw_init.py:198-217` | bispinor `meta_curr` refresh (transverse channel has its own n_rmu) | canonical (routes through `padded_mu_extent`; rebuilding Meta is right) |

**Doc bug at B3:** the `load_centroids_band_chunked` docstring
(wfn_transforms.py:1689-1696) still says *"no padding needed at this layer"* —
directly contradicted by the `n_rmu_padded` pad eight lines below (added by 083d209).
Stale guarantee, same failure mode as the false `_identity_pad_block_diagonal`
docstring that ROOT_CAUSE.md had to correct. Fix the docstring.

**μ-extent recomputation:** besides Meta, three sites recompute padded extents from
logical counts (`wfn_transforms:1762`, `v_q_g_flat.py:359-366` local `_pad`,
`v_q_bispinor.py:510-516` `_padded_shape_LR`). Post-083d209 all route through
`padded_mu_extent`, so they can no longer diverge (the knob exposed the previous
divergence as a `dot_general 1204 vs 1216` crash). The V_q sites cannot simply read
`meta.n_rmu_padded` because bispinor carries per-channel extents (charge ≠
transverse); acceptable as-is, though carrying the (logical, padded) pair on the
channel/tile metadata would remove the recompute pattern entirely.

---

## 2. Consumer inventory (ζ-fit core)

Classes: **N** = structurally neutral (zero pad rows, no code) · **H** = canonical
helper/seam · **A** = ad-hoc per-site logic · **M** = missing (latent bug).

| # | Site | What | Class | ~lines |
|---|------|------|-------|-------|
| C1 | `c_q_from_psi_sm` / `z_q_from_psi_sm` bilinears; `gflat` FFT accumulate; V_q GEMM | pad rows = exact zeros ⇒ zero contribution | **N** | 0 |
| C2 | `core.py:765-830` `_identity_pad_block_diagonal` + call in `factor_c_q:1007` | pad-block identity so the padded buffer is non-singular; docstring now honestly scoped (Cholesky ≤1e-7, LU fails) | **H** | 66 |
| C3 | `core.py:1049-1063` `factor_c_q` P=1 path: slice to logical, Cholesky, re-embed | logical-extent factorisation (Fix 1b) — but lines 1056-1059 **re-implement** the helper's pad+diag re-embed instead of calling `_identity_pad_block_diagonal` | **A** (duplication) | 15 |
| C4 | `core.py:1239-1253, 1294-1341` cuSolverMp-LU branch: μ-slice L/Z to `n_log`, logical ridge, zero-fill ζ pad rows, mesh-divisibility fallback | Fix 1 (the eV bug). Correct, but per-branch inline | **A** | ~35 |
| C5 | `core.py:1379-1405` `_ridge_indef_solve`: slice → LU → re-pad | Fix 1, per-q path | **A** | 12 |
| C6 | `core.py:1407-1423` `_tri_solve_logical`: slice → tri-solve ×2 → re-pad | same 4-line slice/re-pad skeleton as C4/C5, third copy | **A** | 10 |
| C7 | `core.py:1264-1270, 1309-1315, 1344-47, 1473-76, 1529, 1540, 1553` NRHS/n_zchunk column pad + trim, three branches | same round-up+`jnp.pad`+`[:, :, :n_zchunk]` triplet ×3 | **A** | ~18 |
| C8 | `isdf_fitting.py:448-458` `cct_trace_per_q` over `L_q[:, :n_rmu, :n_rmu]` | logical-block trace for the LU ridge (perf-hoisted out of solve_zeta) | **A** | 9 |
| C9 | `isdf_fitting.py:790-809, 971-981` `gflat_acc` at padded extent; `write_slab(..., valid_shape=(…, n_rmu, …))` | SlabIO `valid_shape` IS the canonical disk seam (memory padded / disk logical) | **H** | 12 |
| C10 | `isdf_fitting.py:986-992` allgather-backend fallback: `_f['zeta_q_G'][...] = _g` | `_g` is gathered at **padded** μ; dataset was created at **logical** n_rmu (line 691) ⇒ h5py shape-mismatch crash whenever mu_pad > 0 on the H5PY_ALLGATHER backend. Loud, not silent — but it means the pad-invariance knob + allgather path (CPU CI, local dev) cannot run. Fix: `_g[:, :n_rmu, :]` | **M** | 0 (missing 1) |
| C11 | `gw_init.py:511-517` **UNCOMMITTED**: `g0_mu` write clipped `[..., :meta.n_rmu]` | before this edit G0 was written to disk at the padded extent — a real violation of the disk-stores-logical contract (any cross-P re-read of g0_mu got extent mismatch). In-flight fix is correct; classify the pre-fix state as the 2nd MISSING site | **A** (was M) | 6 |
| C12 | `isdf_fitting.py:954-968` G-axis pad-slot mask before write (`ngk[q]` sphere pad) | different axis (G, per-q sphere), masks physical garbage the gather put in pad slots | **A** (justified) | 14 |
| C13 | `core.py:485-537, 584-607, 662-688` bc band front/back pads + L/R masks; `psi_G_store` `_bpd_max` uniform pad, `np.zeros` pad rows, `_zero_user_band_pad_in_shard` | band-axis alignment pads; masks make them neutral; `np.zeros`-not-`np.empty` is load-bearing and documented | **A** but neutral-by-mask, well-commented | ~75 |
| C14 | `zeta_reader.py:167,238`, SlabIO `read_slab(shape=padded, valid_shape=logical)` | read-side: logical prefix into padded buffer, pad slots zero-filled | **H** | ~8 |
| C15 | `core.py:1620-1650, 1689-97` fit-kernel `n_rmu = meta.n_rmu_padded`; `solve_phase(..., n_rmu_logical=int(meta.n_rmu))` | the ONE place the kernel picks its extents; threads n_log into solve_zeta | **H** | ~8 + comments |

**Known accepted residual (not a site to fix here):** the multi-device charge
factorisations (2D-blocked Cholesky, cuSolverMp potrf) still run at the PADDED extent
— block-cyclic layouts require mesh divisibility (`core.py:1043-1048`). Measured
≤1e-7 rel pad sensitivity; this is the documented floor the Tier-2 gate tolerances
absorb, and removing it means never padding μ in kernels (architecture change,
per ROOT_CAUSE §AS-FIXED).

### Tally

- **Sites in scope: 20** (5 birth + 15 consumer).
- **Structurally neutral: 1 family** (all bilinears — the majority of the physics, 0 lines; this is the scheme working as designed).
- **Canonical-helper sites: 6** (B1, B5, C2, C9, C14, C15 + the `padded_mu_extent` routing at B3/V_q).
- **Ad-hoc sites: 10** (C3-C8, C11-C13, B3's raw `jnp.pad`), of which the slice/re-pad skeleton is **triplicated** (C4/C5/C6) and the NRHS pad **triplicated** (C7).
- **Latent-bug candidates: 2 confirmed** — C10 (allgather padded write, crash) and the
  wfn_transforms stale docstring (misleads the next agent); plus 1 booby trap
  (dead `n_rmu_jax` wrong-divisor field).
- **Pad-specific lines in scope: ~300 executable** (≈470 counting the contract
  comments/docstrings, which are mostly earning their keep post-ROOT_CAUSE) **+ ~337
  lines of dead helper API + 427 lines of tests for the dead API**. Dead code
  outnumbers live pad logic.

---

## 3. Is fewer achievable? Concretely, yes: ~5 sites and ~800 lines

1. **Delete the dead half of `runtime/padding.py`** (PadAxis, pad/unpad_array_to_mesh,
   pad_shape_to_mesh, valid_shape_from_pad_meta, logical_shape_from_padded,
   round_up_to_mesh_product) + `tests/test_padding.py` + the `slab_io.py:190` stale
   reference. Keep `padded_mu_extent`/`extra_mu_pad` and add one plain
   `round_up(n, d)`. **−~760 lines, kills the second convention.**
2. **One `_solve_at_logical` wrapper in `solve_zeta`**: `slice L/Z → fn → zero-fill ζ
   pad rows` used by all three solver branches (C4/C5/C6), and one
   `_pad_nrhs/_trim_nrhs` pair for C7. Collapses 6 ad-hoc blocks into 2 helpers,
   ~65 → ~30 lines, and makes it impossible for a 4th solver branch to forget the
   slice — the exact defect class ROOT_CAUSE identified.
3. **C3: call `_identity_pad_block_diagonal` for the P=1 re-embed** instead of
   re-implementing its pad+diag (−5 lines, one source of truth for the pad block).
4. **Delete `n_rmu_jax`/`nbnd_jax`/`n_rtot_jax`** and the gw_init:201 refresh
   (−~12 lines, removes the wrong-divisor trap).
5. **Fix C10** (one-line slice) and the B3 docstring.
6. Optional, lower value: have `factor_c_q` return `(L_q, cct_trace_per_q)` so the
   logical-trace slice (C8) lives next to the logical-slice logic it mirrors, instead
   of in the orchestrator.

Not recommended: threading a pad mask or a PadInfo object through the pipeline. The
trailing-block invariant + the (n_rmu, n_rmu_padded) pair in Meta is already the
minimal representation, and post-fix the two functions that do dense math on μ²
(`factor_c_q`, `solve_zeta`) both take `n_rmu_logical=` explicitly — callers state the
logical extent once and the pad handling is internal. That is the right shape; the
remaining dirt is duplication *inside* those functions and the dead parallel API
*outside* them.

## 4. Cleanliness verdict for this scope

**Design: clean. Implementation: cluttered — grade B-.** One extent authority
(`padded_mu_extent` → `Meta.n_rmu_padded`), zero-row neutrality doing the heavy
lifting for free, and the post-fix `n_rmu_logical=` threading is the correct minimal
convention. But: a 337-line dead alternative API (plus 427 test lines) shadows the
live convention, the logical-slice/re-pad pattern is hand-copied 3×, the NRHS pad 3×,
round-up arithmetic 5×, one padded-write latent crash (C10) survives in the allgather
fallback, one disk-contract violation was live until the uncommitted g0 fix (C11),
and a birth-site docstring asserts padding doesn't happen where it does. All of the
above is a ~1-day deletion/consolidation pass with no behavior change outside C10/C11.
