# ζ-fit CCT on the two-face ψ carrier (`low_mem_bands=true`)

Landed: `feat/zeta-fit-face-psi-2026-08-22`, on top of
`integ/low-mem-bands-2026-08-22`. Continues the audit
(`reports/gwjax_low_mem_bands_audit_2026-08-22/report.md`) and its revision
round, which scoped this exact gap: "The FULL contract — product-sharding
the surviving X-form's band axis — collides with the r-chunk kernel's
existing `all_to_all('y')` and needs a distributed-algorithm redesign."

## What changed, and what did not

Two independent halves of the ζ fit consume ψ:

| stage | before this change | after this change |
|---|---|---|
| CCT (Gram build, `fit_zeta_to_h5` STEP 2) | 4 single-axis copies (`psi_l_rmu_Y`, `psi_r_rmu_Y`, `psi_l_rmuT_X`, `psi_r_rmuT_X`), never all-P sharded | **all-P face carrier** (`psi_nmu`/`psi_mun`, `2·S/(Px·Py)`), never single-axis |
| r-chunk loop (STEP 6, `fit_one_rchunk`/`z_q_from_psi_sm`) | 2 single-axis X-form copies (`2·S/Px`) | **unchanged** — still single-axis X-form, refused-by-name for this session |

`low_mem_bands=false` (the default) is bit-identical to the pre-existing
path: every new branch is a static, structurally-dead arm under that flag
(`layout='legacy'` in `isdf.core.c_q_from_psi_sm`, `low_mem_bands=False`
in `fit_zeta_to_h5`).

## Why the CCT half factors as a band GEMM

CCT's per-k pair density is

```
P_{ab}(μ, ν; k) = Σ_n conj(ψ_{n,k,a})(μ) · ψ_{n,k,b}(ν)
```

— a contraction over the band index `n`, with `(μ, ν)` as the two
uncontracted (output) indices. That is *exactly* the shape
`gw.greens_function_kernel.build_G` already computes for the Green's
function (`G_μν(k) = Σ_n ψ_n(μ) · w_n · ψ*_n(ν)`), which the low-mem-bands
face G-build already ships as one planned `distrib_la.gemm_plan` N,N GEMM
over the two-face carrier. `isdf.core._c_q_face` reuses that exact
mechanism — same GEMM-seam merge (`common.contract_bands.
merge_spin_centroid`/`split_spin_centroid`, same axis positions `(1,2)`
and `(2,3)`), same "conjugate the μ/row operand, leave the ν/col operand
un-conjugated" convention — rather than re-deriving it.

```
psi_mun : (nk, s, μ, n)  P(None, None, 'x', 'y')   -- μ on X, n(band) on Y
psi_nmu : (nk, n, s, μ)  P(None, 'x', None, 'y')   -- n(band) on X, μ on Y
```

The band axis `n` is sharded *oppositely* on the two operands — `y` on
`psi_mun`, `x` on `psi_nmu` — which is exactly what a SUMMA-style
distributed GEMM needs: `A[m_x, k_y] @ B[k_x, n_y] -> D[m_x, n_y]`, with
the provider (cuBLASMp, via `distrib_la.gemm_plan`) doing the ring
communication that reduces over the two halves of `k` living on different
mesh axes. This is the mechanism, not a metaphor for it — it is the
*same* compiled kernel shape the face G-build already certifies, and its
communication cost is the one the revision round measured (~0.18 s/call
at P16/4-node for the G-build shape; CCT's own shape is smaller per call
but pays the identical mechanism).

After the GEMM, `split_spin_centroid` twice restores the open-spin
`(nk, s, μ, s', ν)` rank-5 pair density — no transpose. `isdf.core.
_c_q_face` then runs the SAME three primitives `_c_q_legacy`'s own inline
body uses (`local_ifftn3`, `gamma_double_contract`, `local_fftn3`) over
this NATURAL axis order rather than legacy's `'karmb'` convention: there
is no FFI `pair_kernel` on the face path to dictate a specific
trailing-axis order (the face CCT always takes the XLA IFFT/FFT arm), so
adopting `'karmb'` anyway would buy nothing and cost a real transpose.

**Two OOMs were found and fixed this way, both the same shape: an extra
top-level buffer this design did not need.**

1. A `jnp.transpose` from the GEMM's natural `(k, a, μ, b, ν)` order into
   `'karmb'` (`(k, a, ν, μ, b)`) is not free at production `μ` — it is a
   genuine `μ²`-scale data movement, not a bitcast. Dropping it (keeping
   the GEMM's own natural order all the way through, and choosing
   `gamma_double_contract`'s `spin_axes` to match instead of legacy's)
   removed one live buffer, but did not by itself fix the OOM below.
2. The real culprit: the band GEMM (`gemm(A, B)`, called once for `P_l`
   and once for `P_r`) and the IFFT/γ̃/FFT tail were three SEPARATE
   top-level `jax.jit` dispatches. Each is its own compiled executable
   with its own committed output buffers — XLA's buffer assignment
   cannot alias or reuse memory ACROSS that boundary the way it can
   inside one fused program, which is exactly what lets
   `_c_q_legacy`'s single monolithic `shard_map` hold its own two
   pair densities in a small working set. Both symptoms measured
   together, real 16-rank CUDA, MoS2 6×6×1 / 626 bands / μ≈5282 / P=16
   (4×4): `RESOURCE_EXHAUSTED: Out of memory while trying to allocate
   11.28GiB` at `C_q.block_until_ready()`, byte-identical before and
   after removing the transpose alone — confirming the transpose was
   not the dominant term. Folding BOTH `gemm(...)` calls and the
   `shard_map` tail into ONE outer `@jax.jit` (mirroring
   `gw.w_isdf._get_chi_minimax_kernel_face`'s own `_build_Gv_Gc`, which
   already wraps two calls into the same `g_plan` in one outer jit for
   the identical reason) fixed it outright, with no other change.

Only the band contraction differs between the two layouts now;
everything downstream of the pair density uses the same three physics
primitives, addressed by two different (and each individually free)
axis conventions, inside one compiled program per layout.

## Sidestepping the L/R window's own divisibility problem

`distrib_la.gemm_plan` requires its contraction extent `k` to divide
*both* mesh axes. The ζ fit's L/R band windows
(`band_range_left = (b0, b3)`, `band_range_right = (b0, b4)`) are **not**
mesh-divisible in general — `b3` is an arbitrary sigma-window edge (report
obstacle #3: "Band windows are not automatically legal face matrices").

The fix is the same one `gw.greens_function_kernel.build_G_tau`'s
`phases`/`mask` argument already uses for exactly this problem
(`Wavefunctions.band_mask`'s own docstring names it "the bring-up
substitute"): **weight, don't window.** `_c_q_face` runs the GEMM over the
carrier's *full* loaded band extent — `BandSlices.b4` is already padded to
the world size, an existing invariant, not a new one — and multiplies one
GEMM operand by a `{0.0, 1.0}` per-band weight vector before the call.
Zero-weighted bands contribute exactly zero to the band sum (the
contraction is bilinear in ψ — `TASTE.md`'s "a zero pad is inert for
operators LINEAR or BILINEAR in the padded axis" applies directly), so
this reproduces the L/R-sliced legacy sum bit-for-bit up to
summation-order roundoff, with **no new pad**, no new divisibility
requirement, and one shared `gemm_plan` for both `P_l` and `P_r` (only the
weight differs).

## Why the r-chunk half is NOT ported this session

`z_q_from_psi_sm`'s per-band-chunk body already uses BOTH mesh axes for
two *different* purposes in the same scan iteration:

* `'y'` carries the **r-scatter**: `lax.all_to_all('y', split_axis=r,
  concat_axis=band)` turns this rank's own bands (held over the *full*
  r-chunk) into every rank's bands restricted to *this* rank's `r_loc`
  r-slab — the mechanism that buys the `p_y×` memory win the r-chunk
  redesign already shipped (129 GB/rank -> 12.9 GB/rank at MoS2 12×12,
  per that kernel's own comment).
* The X-form operand it then contracts against
  (`psi_l_X_bc`/`psi_r_X_bc`) needs its band axis **replicated**, because
  the per-bc slice + `(m,n,a),(n,b,r)->(a,r,m,b)` einsum is a rank-local
  contraction, not a GEMM — the band axis must be fully present on every
  rank for that local contraction to be correct.

Giving that X-form operand a genuinely face-sharded band axis (bands also
on `'y'`) would put the SAME mesh axis in two structurally different
roles inside one scan body — carrying the r-scatter's post-`all_to_all`
r-slab identity on one operand, and a face-sharded contraction axis
needing its own SUMMA-style reduction on the other — and the two
communication patterns do not commute for free. Making this correct is a
distributed-algorithm redesign of the streaming kernel itself (a
band-sharded contraction *inside* the same scan iteration that already
does the r-scatter), not a mechanical port of the CCT recipe above. It is
named, not silently worked around:

* `fit_zeta_to_h5(low_mem_bands=True)` builds `psi_l_rmuT_X_fit`/
  `psi_r_rmuT_X_fit` exactly as before (single-axis X-form,
  `2·S/Px` — `GFlatChunkPlan.stage_cd_psi_bytes`) and passes them to the
  unmodified `fit_one_rchunk`/`z_q_from_psi_sm`.
* `KNOWN_LORRAX_ISSUES.md`, "zeta-fit r-chunk all-P psi" row, names the
  collision above as the reason and the redesign it needs.
* The planner (`gflat_memory_model._persistent_bytes`) and its docstring
  say explicitly that Stage A/B pricing is now accurate under
  `low_mem_bands=true` while Stage C/D remains an under-estimate for the
  same reason.

## Call-site sequencing (`gw.gw_init.prepare_isdf_and_wavefunctions`)

The face carrier is built **once**, immediately after the fresh load, and
reused for both the fit and the post-fit `Wavefunctions` bundle — not
rebuilt twice:

```
psi_rmu_Y, psi_rmuT_X = load_centroids_band_chunked(...)     # single-axis
if low_mem_bands:
    psi_nmu_fresh = with_sharding_constraint(psi_rmu_Y, PSI_NMU_SPEC)   # free
    psi_mun_fresh = with_sharding_constraint(conj(psi_rmuT_X).T, PSI_MUN_SPEC)  # free
    del psi_rmu_Y                      # its ONLY consumer, CCT, now reads the face
fit_zeta(..., psi_rmu_Y, psi_rmuT_X, psi_nmu_fresh=psi_nmu_fresh, psi_mun_fresh=psi_mun_fresh)
    -> fit_zeta_to_h5(low_mem_bands=True, psi_nmu_fresh=..., psi_mun_fresh=...)
       STEP 1: X-forms built as before (r-chunk loop needs them);
               Y-forms SKIPPED ENTIRELY (never materialized, not even transiently)
       STEP 2: C_q via isdf.core.c_q_from_psi_sm(layout='face', ...)
       STEP 6: fit_one_rchunk(...) unchanged, reads the X-forms
wfns = wavefunctions_face_from_restart(psi_nmu_fresh, psi_mun_fresh, ...)  # reused, not rebuilt
del psi_rmuT_X                          # r-chunk loop is done; only the face survives
```

Both `with_sharding_constraint` calls are FREE resharding constraints (no
transpose collective, no gather) — see
`gw.wavefunction_bundle.build_wavefunctions_face`'s own docstring, whose
derivation this mirrors exactly, just executed earlier (before the fit
instead of after it) and reused for two purposes instead of rebuilt for
each.

Net effect: under `low_mem_bands=true`, the single-axis Y-form of ψ is
**never resident**, not even transiently — the "100% of the time,
including through the fit" contract holds for that half of the psi
inventory. The single-axis X-form remains resident throughout the whole
fit (unchanged from before this session) — the explicitly scoped,
named remainder.

## Refusals added

`fit_zeta_to_h5(low_mem_bands=True)` refuses, by name, before any
compute:

* `vertex_mu_L != 0` (bispinor/transverse) — already refused upstream by
  `gw_config.refuse_unsupported_low_mem_bands`; re-checked here as a
  second, cheap line of defense, since `isdf.core._c_q_face` has no
  non-identity-γ̃ arm.
* `band_norms is not None` (pseudobands) — the face CCT path has no
  weighted-norms arm this session; a real feature gap if ever needed
  together, not a silent approximation.
* `psi_rmu_Y is not None` — catches a caller that forgot to drop the
  single-axis copy before calling (the whole point of the redesign).

`isdf.core.c_q_from_psi_sm(layout='face')` separately refuses
`gamma_L`/`gamma_R` not both `None`, for the same reason as the first
bullet — defense in depth at the lower-level primitive too.

## Verification

This section was rewritten by the landing session (`feat/zeta-fit-face-psi-
2026-08-22`) after independently re-checking every claim below on real
hardware; the draft it replaced described what was PLANNED, not what had
run. See that branch's commits for the exact diff each check landed with.

* **Algebra parity, isolated, real 4-rank CUDA**
  (`tests/test_isdf_cq_face_parity.py`, `lx run -N 1 -G 4 -n 4 ... --mesh
  2x2`, JID 57438326, step `lx-Xg4-205215-1914834-7751`, exit 0): builds
  the SAME synthetic ψ, feeds it to `c_q_from_psi_sm(layout='legacy')`
  and `layout='face'`, diffs `C_q`. 3/3 cases PASS — `ns=1` with an
  asymmetric upper edge (5296→... the L/R-window-not-mesh-divisible
  case), `ns=2` (the GEMM-seam spin-merge order), and `ns=1` with an
  asymmetric LOWER edge (`band_range_left[0] != band_range_full[0]`,
  the case a window-offset bug would show up in first). Max relative
  diff 3.8e-16 to 2.1e-15 across the three — float64 noise from a
  different summation order (a SUMMA-distributed cuBLASMp GEMM vs. a
  rank-local `jnp.einsum`), not a discrepancy; this is the same
  "relative, never bit-exact" engine-parity bar (`RTOL = 1e-12`) the
  package's own `test_distrib_la_multiproc.py` uses for the identical
  reason. This test is gated at a 1e-10 relative bar, two orders of
  margin over what was measured.
* **End-to-end, real 16-rank CUDA, production scale**: MoS2 6×6×1, 626
  bands, μ≈5282, P=16 (4×4), `compute_mode=cohsex`, fresh fit,
  `low_mem_bands=true`, `head=off` — the same `86_bgw_lorrax_scaling`
  k6_c600 deck the audit's revision round used. JID 57438326, step
  `lx-Xg4-202445-2208353-6591`, **rc=0, 253.0 s wall**. Artifacts:
  `runs/MoS2/86_bgw_lorrax_scaling_20260819/points/
  k6_c600_lowmem_ab_20260822/face_headoff_zetafit_facecct_2026-08-22/`.
  `zeta_fit.CCT` (the CCT stage, band GEMM + IFFT/γ̃/FFT tail) timed at
  **6.28 s** (of which the GEMM plan build+warm itself is 2.60 s), up
  from the incumbent single-axis fit's ≈0.8 s at this shape — the
  accepted memory-for-communication trade, smaller in absolute terms
  than the ~tens-of-seconds ballpark floated before landing, and well
  short of dominating the 253 s total (screening + Cholesky dominate).
  `eqp0.dat` (1224 QP rows) vs. the pre-existing face reference
  (`.../face_headoff/`, the incumbent-fit face leg from the audit's
  revision round): **max|ΔE_QP| = 7.0e-9 eV**, mean 2.2e-9 eV,
  max|ΔE_DFT| = 0 (the mean-field input is untouched, as it must be) —
  the same ~1e-9 eV floor the revision round measured between its own
  face mem56-vs-mem20 variants (different reduction order, same
  physics), not a new discrepancy this redesign introduces.
* **Memory, real 16-rank CUDA, `LORRAX_MEM_DEBUG=1`**: same deck, JID
  57438326, step `lx-Xg4-203000-2222395-5541`, rc=0, 261 s. Artifacts:
  `.../face_headoff_zetafit_facecct_memdebug_2026-08-22/gw_memdebug.out`.
  At `zeta_fit_start`, the run's own **shard-level** instrumentation
  (`.addressable_shards`, not the sharding-blind global-shape `mem_probe`
  table — see `KNOWN_LORRAX_ISSUES.md`'s `mem_probe` row) reports
  `psi_nmu_fresh=0.2440 GB psi_mun_fresh=0.2440 GB` per rank, matching
  `2·S/(Px·Py)` at this shape to 4 significant figures — the face
  carrier is genuinely 2-D-sharded, not a same-global-shape single-axis
  array (the ambiguity a naive global-shape read cannot resolve). The
  X-form (`psi_rmuT_X`, single-axis, unported this session) is still
  resident throughout — expected and disclosed, not a defect: total
  `in_use=1.53 GB` at `zeta_fit_start` is consistent with `0.244 + 0.244
  GB` (face) `+ ≈0.98 GB` (X-form at `3.90 GB / p_x=4`) plus small
  metadata, not with a leftover single-axis Y-form (which would add
  another ≈0.98 GB at this shape).
* **Regression, real 4-rank CUDA** (`lx run -G 4 -n 1 ...`, JID
  57438326): `tests/test_zeta_mesh_invariance.py` **7/7 PASS**
  (unaffected — the small-system fast path never sets `low_mem_bands`);
  `tests/test_wavefunction_bundle_face_carrier.py`,
  `tests/test_low_mem_bands_envelope.py`, `tests/test_contract_bands.py`
  together: 44 passed, 13 skipped (mesh-size/feature skips, not
  failures), 0 failed.
* **`low_mem_bands=false` bit-identical**: `_c_q_legacy` is untouched
  code (the pre-`layout=` body, moved verbatim into its own function —
  diffed against the pre-session source to confirm no line inside it
  changed); every new branch in `fit_zeta_to_h5`/`c_q_from_psi_sm`/
  `prepare_isdf_and_wavefunctions` is reached only when
  `cfg.memory.low_mem_bands` is true.

See the commit messages on `feat/zeta-fit-face-psi-2026-08-22` for the
exact job IDs and artifact paths.
