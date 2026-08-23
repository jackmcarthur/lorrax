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

## The r-chunk half: the collision, and how it's avoided (landed
`feat/zeta-fit-rchunk-face-psi-2026-08-22`)

`z_q_from_psi_sm`'s per-band-chunk body already used BOTH mesh axes for
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
communication patterns do not commute for free. This was the collision
this design note originally named and deferred on. It is real, and the
landed fix does not dissolve it by making the X operand a genuine SUMMA
operand — it sidesteps it by never materializing a *resident* X-form at
all, replacing it with a *bounded, per-band-chunk* reconstruction from
the already-resident face carrier.

### Why not a second SUMMA GEMM (the CCT recipe, mechanically reused)

The natural first idea, symmetric with the CCT half: replace the
rank-local einsum with `distrib_la.gemm_plan`, exactly as
`isdf.core._c_q_face` does for the CCT Gram. This does not fit, for a
reason specific to `gemm_plan`'s contract rather than to the physics:

`GemmPlan.__call__` (`services/distrib_la/src/distrib_la/matmul_plan.py`)
is itself a top-level `jax.jit`-wrapped `shard_map` — it expects its `A`/
`B` operands as GLOBALLY-sharded `jax.Array`s with a declared
`NamedSharding` (`P(None,'x','y')`), and does its OWN internal
shard_map/collective dispatch. `z_q_from_psi_sm`'s scan body, in BOTH
layouts, already runs *inside* a manual-mode `shard_map` (the one that
owns the `all_to_all`/`all_gather` r-scatter) — its operands there are
bare per-rank LOCAL arrays with no sharding annotation, which is exactly
what manual mode means. Calling `gemm_plan` from inside that body is not
expressible: it needs annotated global arrays, and the body has none by
construction. Making room for it would mean pulling the band contraction
*out* of the manual shard_map entirely — restructuring the r-scatter
itself into a mix of Auto-mode ops (for the GEMM) and a nested manual
region (for the `all_to_all`) — which is precisely the "distributed-
algorithm redesign of the streaming kernel itself" this note originally
flagged as out of scope for a mechanical port. It was not attempted.

### The landed design: masked gather + `psum('y')`, not a GEMM

`isdf.core._z_q_face` (dispatched from `z_q_from_psi_sm(layout='face')`,
mirroring `_c_q_face`'s own `layout=` split from `_c_q_legacy`) keeps the
r-scatter's `shard_map`/`lax.scan` **completely unchanged** — same
`all_to_all('y')`, same `all_gather('x')`, same Y-compaction table, same
IFFT → γ̃·γ̃ → FFT tail. The ONLY change is how the scan body obtains its
X operand, per band-chunk `bc`, per scan iteration:

1. The persistent face carrier's `psi_mun` (`gw.wavefunction_bundle.
   PSI_MUN_SPEC`, `P(None,None,'x','y')` — μ on `'x'`, bands on `'y'`,
   resident at `2·S/(Px·Py)` for the WHOLE fit, since it is the SAME
   array STEP 2's CCT Gram already reads) is a `shard_map` input, same as
   `psi_r_cache`.
2. For each output position `p` in `[0, bpd_max_global)` (this bc's
   padded band width — the SAME `bpd_max_global` the Y-side already
   uses), the GLOBAL band index is `b_lo_rel[bc] + p`, and psi_mun's OWN
   `'y'`-sharding assigns it to EXACTLY one owning rank:
   `owner_y = global_band // shard_w` (`shard_w = nb_face / p_y`, a
   STATIC per-rank constant).
3. Every rank computes a locally-SAFE (index-clamped, always in-bounds)
   `jnp.take` of its OWN local `psi_mun` shard at that position, zeroes
   it via `jnp.where` unless it IS the owner
   (`owner_y == jax.lax.axis_index('y')`), and `jax.lax.psum(..., 'y')`
   sums the (at-most-one-nonzero-per-position) contributions — a
   **selective broadcast-from-owner**, not a full `all_gather('y')` of
   the shard and not a resident copy.
4. The result, `x_full_bc` — shape `(nk, ns, mu_loc, bpd_max_global)`,
   bounded to ONE band-chunk's width — is masked by `weight_l`/
   `weight_r` (the SAME "weight, don't window" vectors `_c_q_face` builds
   for STEP 2, reused unchanged rather than rebuilt: same L/R window,
   same `band_range_full`-relative indexing) to produce
   `psi_l_X_bc`/`psi_r_X_bc`, which feed the UNCHANGED einsum
   (`'kamn,knbr->karmb'` — same contraction as legacy's `'kmna,knbr->
   karmb'`, only the free-index label order differs for the X operand's
   different natural axis layout; no transpose).

This is deliberately NOT "insert breakpoints so every band-chunk lies
within one `'y'`-shard" (a design considered and rejected before
implementation): psi_mun's `'y'`-shard width (`nb_full/p_y`, e.g. 18 at
this report's k6_c600/4×4 shape) is generally NOT a multiple of
`band_chunk_size` (e.g. 16), so most band-chunks straddle a shard
boundary — and forcing alignment would mean fragmenting the SHARED
`band_chunk_ranges` grid the Y-side/`psi_G_store` also uses, risking that
machinery's own `p_xy`-divisibility invariant for no necessary reason.
The per-position gather-then-`psum` handles an ARBITRARY straddle for
free (each position independently finds its one true owner), so no
alignment is required at all.

**Correctness pitfall found and fixed during this port, worth recording**:
the LAST band-chunk is typically SHORT (`bpd_max_global` is the padded
MAXIMUM across all chunks, sized for the largest one), so `global_band`
overruns both this chunk's own true end AND — for the FINAL chunk
specifically — `weight_l`/`weight_r`'s own array extent (`nb_face`). An
un-clamped `jnp.take` on a real (non-`x_full_bc`) array at an
out-of-range index silently NaN-fills in this JAX version; because the
scan accumulates via `P_l_acc + delta_P_l` (`+=`), ONE NaN contribution
from the trailing pad positions poisoned the ENTIRE output — measured:
100% NaN across every element, all three parity cases, before the fix.
The X-side `psi_mun` gather is immune BY CONSTRUCTION (`owner_y` for an
out-of-range `global_band` computes to `>= p_y`, which no real rank ever
equals, so `owns` is always False there — a safe zero, not garbage); the
weight lookup needed its own explicit `bc_valid = global_band <
b_hi_rel[bc]` guard, clamping the index before the take and masking the
result to zero outside — mirroring `_z_q_legacy`'s own identically-named
`bc_valid`/`g_axis` pattern, which this design otherwise does not need
(see `isdf.core._z_q_face`'s comment beside `_b_hi_rel_np`).

### What stayed exactly as it was

* The r-scatter `shard_map`/`lax.scan` (io_callback or `psi_r_cache`
  read, `all_to_all('y')`, `all_gather('x')`, Y-compaction) — untouched,
  reused verbatim by copying the SAME code into `_z_q_face`'s body
  (`_z_q_legacy` itself is UNTOUCHED, per this note's own precedent for
  the CCT half).
* `fit_one_rchunk`/`_make_fit_one_rchunk_kernel`'s public shape — extended
  with a `layout=`/`psi_mun=`/`weight_l=`/`weight_r=` arm, exactly
  mirroring how `fit_zeta_to_h5` already dispatches STEP 2.
* `low_mem_bands=False` — `_z_q_legacy` is the frozen pre-`layout=` body;
  every new branch in `z_q_from_psi_sm`/`fit_one_rchunk`/
  `_make_fit_one_rchunk_kernel`/`fit_zeta_to_h5`/`gw_init.
  prepare_isdf_and_wavefunctions` is reached only when
  `cfg.memory.low_mem_bands` is true.

### What is now resident, and what changed at the call site

`fit_zeta_to_h5(low_mem_bands=True)` no longer builds `psi_l_rmuT_X_fit`/
`psi_r_rmuT_X_fit` AT ALL (STEP 1 sets them to `None` under this layout);
STEP 6 reads `psi_mun_fresh` directly instead. Because of this,
`gw.gw_init.prepare_isdf_and_wavefunctions` now drops BOTH single-axis
copies (`psi_rmu_Y` AND `psi_rmuT_X`) immediately after building the face
carrier, before `fit_zeta` is even called — not `psi_rmu_Y` alone,
followed by `psi_rmuT_X` only after the whole fit returns, as the
CCT-only landing left it. See "Call-site sequencing" below for the
updated sequence.

`KNOWN_LORRAX_ISSUES.md`'s "zeta-fit r-chunk all-P psi" row is CLOSED by
this session (narrative moved to `claims/legacy_KNOWN_LORRAX_ISSUES.md`).
`GFlatChunkPlan.stage_cd_psi_bytes` (`gflat_memory_model.py`) is
layout-aware now: under `low_mem_bands=True` it reports the new bounded
per-call/per-bc transient terms instead of the retired `2·psi_one/p_x`
X-form figure (still reported, unchanged, under `low_mem_bands=False`).

## Call-site sequencing (`gw.gw_init.prepare_isdf_and_wavefunctions`)

The face carrier is built **once**, immediately after the fresh load, and
reused for both the fit and the post-fit `Wavefunctions` bundle — not
rebuilt twice:

```
psi_rmu_Y, psi_rmuT_X = load_centroids_band_chunked(...)     # single-axis
if low_mem_bands:
    psi_nmu_fresh = with_sharding_constraint(psi_rmu_Y, PSI_NMU_SPEC)   # free
    psi_mun_fresh = with_sharding_constraint(conj(psi_rmuT_X).T, PSI_MUN_SPEC)  # free
    del psi_rmu_Y, psi_rmuT_X          # BOTH single-axis copies drop HERE now —
                                        # neither has a consumer left anywhere in
                                        # the fit (STEP 2 AND STEP 6 both read the
                                        # face carrier)
fit_zeta(..., psi_rmu_Y, psi_rmuT_X, psi_nmu_fresh=psi_nmu_fresh, psi_mun_fresh=psi_mun_fresh)
    -> fit_zeta_to_h5(low_mem_bands=True, psi_nmu_fresh=..., psi_mun_fresh=...)
       STEP 1: BOTH X-forms and Y-forms SKIPPED ENTIRELY (neither ever
               materialized, not even transiently)
       STEP 2: C_q via isdf.core.c_q_from_psi_sm(layout='face', ...)
       STEP 6: fit_one_rchunk(..., layout='face', psi_mun=psi_mun_fresh,
               weight_l=weight_l_face, weight_r=weight_r_face) —
               isdf.core._z_q_face reads psi_mun_fresh per band-chunk
wfns = wavefunctions_face_from_restart(psi_nmu_fresh, psi_mun_fresh, ...)  # reused, not rebuilt
# no further deletion needed: psi_rmu_Y/psi_rmuT_X are already gone
```

Both `with_sharding_constraint` calls are FREE resharding constraints (no
transpose collective, no gather) — see
`gw.wavefunction_bundle.build_wavefunctions_face`'s own docstring, whose
derivation this mirrors exactly, just executed earlier (before the fit
instead of after it) and reused for two purposes instead of rebuilt for
each. `weight_l_face`/`weight_r_face` (STEP 2's own "weight, don't
window" vectors) are built once, alongside the CCT GEMM plan, and kept
alive (not deleted with the plan) specifically so STEP 6 can reuse them
without rebuilding — two `(nb_face,)` float64 vectors, not worth pricing.

Net effect: under `low_mem_bands=true`, NEITHER single-axis form of ψ is
**ever resident**, not even transiently, anywhere in the fit — the
"100% of the time, including through the fit" contract now holds for the
WHOLE psi inventory, not just the Y-form half. The only ψ-derived objects
alive during STEP 6 are the persistent face carrier (already counted,
`2·S/(Px·Py)`, shared with the post-fit bundle) and the bounded per-bc
transients `_z_q_face` builds and frees each scan iteration.

## Refusals added

`fit_zeta_to_h5(low_mem_bands=True)` refuses, by name, before any
compute:

* `vertex_mu_L != 0` (bispinor/transverse) — already refused upstream by
  `gw_config.refuse_unsupported_low_mem_bands`; re-checked here as a
  second, cheap line of defense, since neither `isdf.core._c_q_face` nor
  `_z_q_face` has a non-identity-γ̃ arm.
* `band_norms is not None` (pseudobands) — the face CCT/r-chunk paths
  have no weighted-norms arm; a real feature gap if ever needed together,
  not a silent approximation.
* `psi_rmu_Y is not None` — catches a caller that forgot to drop the
  single-axis Y-form copy before calling.
* `psi_rmuT_X is not None` — the r-chunk half's OWN addition this
  session: catches a caller that forgot to drop the single-axis X-form
  copy (the whole point of this session's redesign) — mirrors the
  `psi_rmu_Y` check exactly, added beside it.

`isdf.core.c_q_from_psi_sm(layout='face')` and `z_q_from_psi_sm
(layout='face')` separately each refuse `gamma_L`/`gamma_R` not both
`None`, for the same reason as the first bullet — defense in depth at
the lower-level primitives too.

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
  asymmetric, non-mesh-divisible upper edge (the L/R-band-window "weight,
  don't window" case), `ns=2` (the GEMM-seam spin-merge order), and `ns=1` with an
  asymmetric LOWER edge (`band_range_left[0] != band_range_full[0]`,
  the case a window-offset bug would show up in first). Max relative
  diff 3.8e-16 to 2.1e-15 across the three — float64 noise from a
  different summation order (a SUMMA-distributed cuBLASMp GEMM vs. a
  rank-local `jnp.einsum`), not a discrepancy; this is the same
  "relative, never bit-exact" engine-parity bar (`RTOL = 1e-12`) the
  package's own `test_distrib_la_multiproc.py` uses for the identical
  reason. This test is gated at a 1e-10 relative bar, two orders of
  margin over what was measured.
* **`zeta_q.h5` on-disk parity, production scale — value-level, NOT
  bit-exact, and the gap is explained, not merely bounded.** Same k6_c600
  deck, incumbent single-axis fit (`.../face_headoff/tmp/zeta_q.h5`) vs.
  this session's face-CCT fit (`.../face_headoff_zetafit_facecct_2026-
  08-22/tmp/zeta_q.h5`): identical dataset names/shapes/dtypes
  (`g0_mu` (36,5282) c128, `zeta_q_G` (36,5282,3012) c128). Numerically,
  `g0_mu` max\|diff\|=1.85e-5 (max rel 2.9e-5); `zeta_q_G`, sampled at
  q∈{0,12,24,35}, max\|diff\|=1.73e-5, **max rel diff=6.4e-4** — three to
  four orders of magnitude ABOVE the CCT Gram's own 1e-15-relative parity
  measured above. This is the SAME mechanism `TASTE.md`'s "Arbitrary
  choice under degeneracy" entry documents for this exact fit family ("a
  finite ISDF basis is... fitted, truncated or ill-conditioned... a solve
  can be conditioned badly enough that the difference decides the
  answer," with its own worked example of one ULP flipping a pivoted-
  Cholesky pivot order): `zeta_q_G` is not `C_q` itself but the OUTPUT of
  STEP 3's rank-truncated Cholesky solve against `C_q`, and that solve is
  the amplifier — a ~1e-15-relative perturbation in the Gram entering an
  ill-conditioned factorization is expected to surface as a much larger
  relative change in the solved coefficients, all while the underlying
  physics is unchanged. The number that actually matters for whether this
  redesign is safe to gate is the END-TO-END observable it feeds
  (below): **eqp0 agrees to 7.0e-9 eV**, not the ~1e-9 eV a summation-
  order change alone would predict on a WELL-conditioned quantity, but
  still five orders of magnitude below any physically meaningful QP
  energy scale. Per `TASTE.md` rule 15 (declare the parity class before
  gating), this redesign's `zeta_q.h5` parity is **value-level / gauge
  κ·ε**, not bit-exact — the CCT Gram itself (the actual output this
  session's code changes) IS bit-exact-class (1e-15 relative); the
  amplification happens downstream, in unmodified STEP 3 code, and would
  reproduce on ANY two Gram matrices differing at float64 noise level,
  face-vs-legacy or otherwise.
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

## Verification — the r-chunk half (landed
`feat/zeta-fit-rchunk-face-psi-2026-08-22`)

Every check below was re-run and read from disk this session; none is
carried over from a plan.

* **`_z_q_legacy` is byte-identical to the pre-session
  `z_q_from_psi_sm`.** AST-extracted both function bodies (from the
  branch point `d699f697` and from this branch) and diffed the source
  text from the first executable statement onward: **0 bytes differ**
  (20,178 characters, exact match) — not a visual diff, a programmatic
  one. `fit_one_rchunk`/`_make_fit_one_rchunk_kernel`'s `layout='legacy'`
  arm reaches this function with the exact same call signature as
  before.
* **Algebra parity, isolated, CPU-emulated 2×2 mesh (no GPU needed —
  `_z_q_face` uses no `gemm_plan`/cuBLASMp, only `all_to_all`/
  `all_gather`/`psum`, all portable)** — `tests/
  test_isdf_zq_face_parity.py`, subprocess worker,
  `--xla_force_host_platform_device_count=4`: builds ONE random ψ
  source, feeds it to `z_q_from_psi_sm(layout='legacy')` and
  `layout='face'` against the SAME synthetic `psi_G`/`psi_r_cache` (both
  the `psi_r_cache`-supplied production path AND the `psi_r_cache=None`
  io_callback compatibility path independently verified), diffs `Z_q`.
  3/3 cases PASS — `ns1_asym` (L window `[0,21)` straddles psi_mun's own
  `'y'`-shard boundary at 18, the case the masked-`psum` gather exists
  for), `ns2_spinor` (ns=2, the free-index label reordering in the
  `'kamn,knbr->karmb'` einsum), `ns1_lower_asym` (L window's LOWER edge
  `[5,30)` also off the array origin). **Max\|diff\| = 0.0 in all three
  cases — bit-exact**, not merely within a relative tolerance (the gate
  gave 1e-9 headroom expecting float64 noise from the mask-then-`psum`
  reduction; none was measured, because `psum` over an at-most-one-
  nonzero-term set is a select, not a re-ordered sum).
* **Algebra parity, real 4-rank CUDA** (`lx run -N 1 -G 4 -n 4 ...
  --mesh 2x2`, JID 57448156, step `lx-Xg4-221309-962979-2026`, exit 0):
  the SAME three cases, same mechanism, on real multi-process CUDA with
  `jax.distributed`/`initialize_communicator_stack` bootstrapped
  (mirrors `test_isdf_cq_face_parity.py`'s own `__main__` pattern — the
  first attempt without it printed `processes=1, devices=1` per rank and
  refused every case; fixed by moving the bootstrap before any `jax`
  import in the CLI path only). 3/3 PASS, **max\|diff\| = 0.0** again.
* **This session's own correctness pitfall, caught by the parity test
  before it ever reached production**: the very first run of the CPU
  parity check gave **100% NaN** in `Z_face` (both `psi_r_cache` and
  io_callback paths, all three cases). Root cause: the LAST band-chunk's
  padded width (`bpd_max_global`) overruns both that chunk's true end
  and — for the final chunk specifically — `weight_l`/`weight_r`'s own
  `(nb_face,)` array extent; an un-clamped `jnp.take` on those arrays at
  an out-of-range index silently NaN-fills (this JAX version), and the
  scan's `+=` accumulator turns ONE NaN contribution into a fully NaN
  `Z_q`. Fixed by clamping the index and masking the result with a
  `bc_valid` guard mirroring `_z_q_legacy`'s own identically-named
  pattern (see `isdf.core._z_q_face`'s comment beside `_b_hi_rel_np`);
  the X-side `psi_mun` gather needed no equivalent fix — it is immune by
  construction, since an out-of-range `global_band` computes an
  `owner_y >= p_y` that no real rank ever equals, giving a safe zero
  rather than garbage.
* **End-to-end, real 16-rank CUDA, production scale, apples-to-apples
  against the landed CCT-only reference (SAME deck, SAME
  `memory_per_device_gb=20`, SAME `head_correction=off`)**: MoS2 6×6×1,
  626 bands (`zeta_nband=626`, narrowed below the padded `b4=640` —
  exercises the `_bfe_transport <= nb_face` bound this design's weight-
  lookup safety relies on), μ≈5282, P=16 (4×4), `compute_mode=cohsex`,
  fresh fit. JID 57438326, step `lx-Xg4-220303-828572-4807`, **rc=0,
  257.495 s wall** (reference `.../face_headoff_zetafit_facecct_2026-
  08-22/`: rc=0, 253.131 s). Artifacts: `runs/MoS2/
  86_bgw_lorrax_scaling_20260819/points/k6_c600_lowmem_ab_20260822/
  face_headoff_zetafit_rchunk_2026-08-22/`.
    * **`eqp0.dat`, `eqp1.dat`, `sigma_diag.dat`, `sigma_freq_debug.dat`
      — BYTE-IDENTICAL to the reference** (`diff` on every file, header
      timestamp line excluded: 0 differing lines). Not value-level, not
      κ·ε — file-precision bit-exact, end to end through screening and
      Sigma.
    * **`zeta_q.h5` (`g0_mu`, and `zeta_q_G` sampled at q∈{0,9,18,35}) —
      also bit-exact**, max\|diff\|=0.0, max\|rel diff\|=0.0. This is a
      STRONGER result than the CCT half's own `zeta_q.h5` parity
      (6.4e-4 relative, amplified through STEP 3's ill-conditioned
      Cholesky from a genuine ~1e-15-relative summation-order change):
      here there is no summation-order change to amplify, because the
      masked-`psum` mechanism reduces to a select at every position, not
      a re-ordered reduction — the SAME mechanism the isolated parity
      test's own 0.0 already predicted.
    * **Timing, same deck/breakdown labels, reference vs this session**
      (`zeta_fit.*` section timings, `timing.section`): `zeta_fit.CCT`
      6.884 s vs 6.282 s (± noise, unrelated to this session — CCT is
      unmodified); `zeta_fit.chunk_loop` **48.304 s vs 46.910 s** (23
      r-chunks, +1.394 s / +3.0%); `zeta_fit.chunk.z_q_build`
      (`fit_one_rchunk`'s `z_q_phase`, the substage THIS session's code
      change touches directly) **17.601 s vs 16.393 s** (+1.208 s /
      +7.4%); `zeta_fit.chunk.solve` (unmodified back-solve code)
      30.218 s vs 30.045 s (± noise). TOTAL wall +4.364 s / +1.7%. This
      is far below the "tens of seconds" ballpark floated when this half
      was deferred — the per-bc masked-gather/`psum` mechanism's real
      communication cost is a single-digit-percent addition to an
      already-communication-bound stage, not a new bottleneck.
* **Memory — shard-level instrumentation (`.addressable_shards`, real
  16-rank CUDA, `LORRAX_MEM_DEBUG=1`, `LORRAX_MAX_RCHUNKS=3`,
  `LORRAX_EXIT_AFTER_ZETA=1` for a fast leg), same deck**: JID 57438326,
  rc=0. Artifacts: `.../face_headoff_zetafit_rchunk_memdebug_2026-08-22/
  gw_memdebug.out`.
    * The face carrier's own shard-level figure is unchanged by this
      session (it was already all-P, from the CCT half):
      `psi_nmu_fresh=0.2440 GB psi_mun_fresh=0.2440 GB` per rank.
    * **The planner's own disclosure**
      (`GFlatChunkPlan.stage_cd_psi_bytes`, `gflat_memory_model.py`) —
      layout-aware as of this session — prints
      **`Stage C/D ψ floor (post-CCT, face r-chunk incremental) =
      0.299 GB/dev`**, replacing the pre-session reading at the
      IDENTICAL deck/point, `Stage C/D ψ floor (post-CCT, X-forms only,
      both layouts) = 2.001 GB/dev` (`.../
      face_headoff_zetafit_facecct_memdebug_2026-08-22/gw_memdebug.out`)
      — a **6.7× reduction** in the modeled incremental term, and
      squarely inside the task's own "~0.5 GB/dev + chunk slab" target.
    * **Measured (not modeled), genuine per-rank HBM
      (`bytes_in_use` from `jax.local_devices()[0].memory_stats()` —
      real allocator state, not the sharding-blind global-shape
      `live_arrays` table `KNOWN_LORRAX_ISSUES.md`'s `mem_probe` row
      warns about)**, at the IDENTICAL `rchunk_start chunk=0`/
      `after_fit_one_rchunk chunk=0` probe points, same deck:
      reference `in_use` 6.86 GB / 7.36 GB; this session 4.91 GB /
      5.41 GB — **a consistent −1.95 GB/rank drop at both points**,
      the genuine, measured signature of the retired single-axis
      X-form's `2·S/Px` residency (in the right range of, and
      corroborating, the planner's independently-modeled 2.001 →
      0.299 GB/dev delta). No large single-axis-shaped array appears in
      either probe's `live_arrays` table that was not already present
      before this session's change; the one NEW entry at
      `after_fit_one_rchunk chunk=0` (`complex128 (36,5296,2624) x1 =
      8.00 GB` global / 0.50 GB/rank) is `Z_q`'s own ordinary per-chunk
      output, present in the reference run too.
* **Regression, CPU-emulated (real jax collectives, no GPU needed),
  `lx run -N 1 -G 0 ...`**: `tests/test_zeta_mesh_invariance.py`
  **7/7 PASS** (unaffected — confirmed via an ISOLATED run after an
  earlier combined invocation's 2 unrelated failures turned out to be
  this session's OWN launch-wrapper bug — `JAX_PLATFORMS=cuda,cpu`
  forced on a `-G 0`/no-GPU step, fixed by a separate CPU-only wrapper,
  not a `_z_q_face` defect); `tests/test_isdf_zq_face_parity.py`,
  `tests/test_conv_kpair_plan.py`, `tests/
  test_zq_from_psi_sm_bit_identity.py`, `tests/
  test_band_chunk_size_floor.py`, `tests/
  test_wavefunction_bundle_face_carrier.py`, `tests/
  test_low_mem_bands_envelope.py`, `tests/test_contract_bands.py`,
  `tests/test_gflat_planner_zq_seam.py`, `tests/
  test_zeta_nband_decoupling.py`, `tests/test_transverse_rank_truncate.py`
  together: **134 passed, 13 skipped (mesh-size/feature skips, not
  failures), 0 failed.**
    * `test_conv_kpair_plan.py::test_cq_and_zq_both_enter_the_shared_
      conv_plan` needed a fix, PRE-EXISTING and not introduced by this
      session: it asserted `"_conv_kpair_setup(" in inspect.getsource(
      c_q_from_psi_sm)` — a check that was ALREADY false once the CCT
      session made `c_q_from_psi_sm` a thin dispatcher (the string lives
      in `_c_q_legacy`'s body, not the wrapper's). This session's own
      `z_q_from_psi_sm` dispatcher hit the identical shape of failure,
      so both assertions were corrected together to check `_c_q_legacy`/
      `_z_q_legacy` directly — the functions that actually call
      `_conv_kpair_setup`.
* **`low_mem_bands=false` bit-identical**: `_z_q_legacy` is untouched
  code (see the AST diff above); every new branch in `z_q_from_psi_sm`/
  `fit_one_rchunk`/`_make_fit_one_rchunk_kernel`/`fit_zeta_to_h5`/
  `prepare_isdf_and_wavefunctions` is reached only when
  `cfg.memory.low_mem_bands` is true.

See the commit messages on `feat/zeta-fit-rchunk-face-psi-2026-08-22`
for the exact job IDs and artifact paths this section summarizes.
