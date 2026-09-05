# ζ-fit CCT on the two-face ψ carrier (`low_mem_bands=true`)

> **Current status (2026-08-29, `origin/main` at `2aee7e60`).** This page
> begins with the 2026-08-22 face-carrier landings, then records later
> extensions. The production transverse schedule is now the coupled μ=1,2,3
> route described in [Current coupled transverse schedule](#current-coupled-transverse-schedule-2026-08-29).
> Its code owners are `gw.gw_init`, `gw.isdf_fitting`, and `isdf.core`.
> Memory arithmetic belongs to the [memory model](memory-model.md), not here.

Original landing: `feat/zeta-fit-face-psi-2026-08-22`, on top of
`integ/low-mem-bands-2026-08-22`. Continues the audit
(`reports/gwjax_low_mem_bands_audit_2026-08-22/report.md`) and its revision
round, which scoped this exact gap: "The FULL contract — product-sharding
the surviving X-form's band axis — collides with the r-chunk kernel's
existing `all_to_all('y')` and needs a distributed-algorithm redesign."

## What changed, and what did not

Two independent halves of the ζ fit consume ψ:

| stage | before this change | after this change |
|---|---|---|
| CCT (system-matrix build, `fit_zeta_to_h5` STEP 2) | 4 single-axis copies (`psi_l_rmu_Y`, `psi_r_rmu_Y`, `psi_l_rmuT_X`, `psi_r_rmuT_X`), never all-P sharded | **all-P face carrier** (`psi_nmu`/`psi_mun`), never single-axis |
| r-chunk loop (STEP 6, `fit_one_rchunk`/`z_q_from_psi_sm`) | 2 resident single-axis X-form copies | The face kernel reconstructs one bounded X band block from `psi_mun`; its planner-selected Y cache reuses the canonical transform/scatter. Fresh transverse μ=1,2,3 fits may share both transactions. |

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

## Raw-parent contraction on orbit-closed real-grid tiles (2026-09-05)

Both fit projectors are band sums that need every full-zone k only AFTER
the sum: at a child `k = g·kbar` the pair projector is the typed symmetry
image of the raw parent's,

    D_k = U_g · [ exp(2πi kbar·(L_mu − L_r)) D_kbar(alpha_g(mu), alpha_g(r)) ] · U_g†,

conjugated once more on antiunitary rows, with `D_k[a,mu,b,r] = sum_n w_n
psi_{nka}(mu) psi*_{nkb}(r) = conj(P_k)`.  So when the deck runs the face
carrier on a symmetry-reduced WFN with an orbit-closed centroid set, the
fit contracts on the WFN's raw rows only (`gw.centroid_k_unfold.
CentroidKUnfoldPlan`, the same plan the screening Green function uses) and
transports the completed operator with the service's collective-free local
gathers.  The full-k faces are not read by the fit at all; the parent faces
are `n_parent/nk` of their size and the parent `PsiGStore`/ψ(r) cache the
same fraction.

The image permutes BOTH endpoints.  Centroids are orbit-packed so the
`alpha_g` gather stays on one X owner.  A contiguous r slab is not closed
under the group, so `_z_q_face_parent` streams orbit-closed **tiles**
(`gw.centroid_k_unfold.RealGridOrbitTiles`: complete orbits, each whole on
one Y owner, owner-local zero pads, one static width ≤ the planner's
`chunk_r`, tables as runtime operands) and the r gather stays on one Y
owner.  The kernel accumulates the open-spin parent projectors over the
band chunks exactly as `_z_q_face` does (Y block through
`to_rpoints_inner` at the tile slots, X block by the masked `psum('y')`),
then per OUTPUT spin block `(a,b)` accumulates
`sum_cd U[a,c]U*[b,d] · unfold_operator_local(D[c,:,d,:])`, conjugates,
and runs the unchanged IFFT/product/FFT tail. The output spin blocks run
through a `lax.scan` inside the local `shard_map`: one full-k IFFT pair
is live at a time. A Python-unrolled loop lets the compiler overlap these
buffers, increasing the live set and cold compile time. The scan leaves
the parent band contraction and service-owned spin/phase/TRS action intact.  The Z_q tile
leaves the kernel with centroids in the run's packed order (the one
in-memory order, see below) and r in slot order; the q-selected RHS goes
straight to the unchanged solve, and
`accumulate_rchunk_to_gflat(r_indices=...)` scatters the slots into the
G-flat box.  `_c_q_face_parent` is the square case: one planned GEMM per
side on `n_parent` rows, `plan.unfold_operator`, conj, the `_c_q_face`
tail.

Admission (`gw.gw_init.prepare_isdf_and_wavefunctions`): `low_mem_bands`,
`bispinor = false`, one- or two-component spinors, `wfn.nkpts < nk_tot`,
an orbit-closed centroid set, and a fresh charge fit.  No deck key; a run announces
`C_q on raw parents` / `Z_q on raw parents` with the tile census.  The
charge vertex only: a current vertex needs the Cartesian action the plan
does not own.  Parity gates: `tests/test_isdf_zq_parent_parity.py`
(children generated from parents by the typed action, glide + k
reduction + TR row + SU(2) mixing; parent Z_q on every tile and parent C_q
equal the full-k face kernels at <1e-10) and
`tests/test_parent_projector_unfold_oracle.py` (the transport against
`WfnLoader`'s own ψ unfold on the in-tree Si WFN, 1.7e-15; the wrong
phase sign, dropped antiunitary conjugation and identity spinor arms are
O(1)).

### One in-memory centroid order; files stay canonical

Every centroid axis a gwjax run computes on is in the ORBIT-PACKED order of
`common.grouped_layout` (`common.centroid_basis.PackedCentroidBasis`,
built once in `gw_jax` from the centroid table and the point group and
carried as `meta.mu_basis`; `meta.n_rmu_padded` is its packed extent).
Whole symmetry orbits sit on one X (or Y) shard, so every symmetry action
is a rank-local gather; each shard ends in exact-zero pad slots, so the
pads are interleaved per shard and are NOT a global suffix.  The loader
samples ψ at the packed centroid table and zeroes the pad slots; from
there ψ faces, Z_q, C_q, ζ(μ,G), V, χ0, W, pole residues, G and Σ all
inherit the order and no kernel converts.  Two things change because the
pads are interleaved: dense factors/solves run at the whole packed extent
with 1 on the pad diagonal of C_q (its pad rows/columns are exact zeros,
Z's pad rows are zero, so ζ's are; `meta.mu_solve_extent`), and the
GN-PPM "pad modes born dead" selector is the active-slot mask
(`meta.mu_active_mask`) rather than an index prefix.  The Dyson matrix
`1 - Vχ` already carries 1 on its pads.  A bispinor deck, a trivial group
or a non-closed centroid set uses the identity layout (canonical order,
every conversion a no-op).

Files keep the CANONICAL centroid-file order, suffix padded to the mesh
multiple, so ζ h5, restart tensors and the MPA sample/pole store stay
processor-grid agnostic and BSE/htransform/downfold read them unchanged.
Conversion happens ONLY at the I/O seam, through the basis: a writer
unpacks (`unpack_axis`/`unpack_operator`: ζ(μ,G) before its slab write,
V/W0/ψ faces before the restart writer, χ/W samples and GN-PPM poles
before the store), a reader packs (`pack_axis`/`pack_operator`: V after
its assembly from the ζ file, restart tensors and faces after the reader,
pole batches after the store read, the head-channel columns).  Each
conversion is one volume-preserving all-to-all round trip per axis with a
rank-local prefix pad/crop inside the shard (never an all-gather; the
extent change 836↔840 on Si is a per-shard slice).  In-memory q-wedge
unfold tables come from the canonical resolution conjugated into the packed
order (`_resolve_ibz_q_list(mu_basis=)` → `basis.pack_tables`); the W0
pre-unfold capture converts itself back (`PreUnfoldCapture.canonical`) so
the writer's table cross-check still compares the producer's own tables.

Validation on the Si 4×4×4 80-band deck at P4 (sandbox
`runs/Si/99_psi_irr_zeta_2026-09-05/`, legs 20 and 21, JID 57941637): the
fresh packed-order run agrees with the canonical-order run of the same
quadrature schedule to 5.3 µeV in E_QP over 224 rows (ζ file 9e-8
relative; the eigensolver regroups the permuted C_q, TASTE 77), and a
restart from a copy of its canonical tensors reproduces it to the printed
digit.  One pitfall recorded there: the pad diagonal of C_q must be C's own
scale (tr C/n), not 1 — under `charge_zeta_solve = rank_truncate` a unit pad
becomes λ_max when C is small and the cut drops real modes.

### Σ on the parents and parents-only storage

The same plan and carrier serve Σ (`gw.wavefunction_bundle.ParentSigmaRoute`,
`sigma_face_kernel_kwargs`, `parent_sigma_operands`): G is contracted on the
parent faces (`build_G(..., k_unfold_plan=plan)`) and unfolded to full k in
the run's order, the full-k Σ operator after the FFT convolution is selected
on the parents' own full-k rows (`plan.parent_full_rows` =
`SymMaps.kirr_fullids`), projected with the same parent faces, and the band
matrix is broadcast to full k by
`symmetry_maps.unfold_file_wedge_band_operator(trs_rule="transpose")`.  The
rule is the transpose because Σ transforms like G: with ψ_Θk = Θψ_k,
G_Θk(r,r') = G_k(r',r) at the same complex frequency, so Σ_Θk,mn = Σ_k,nm
(the conj rule would flip Im Σ on the diagonal).  Gate:
`tests/test_sigma_parent_projection.py`.  There is no per-τ-node basis
move anywhere (the former canonical restore cost four all-to-all exchanges
per node on Si).

When every wavefunction consumer of a run is parent-capable, `gw_init`
never forms the full-k faces at all: the loader samples the raw parents
(`load_centroids_band_chunked(k_domain="ibz")`), the face bundle carries
`psi_nmu = psi_mun = None` with the parent carrier as its only ψ, and
`face_extents` names the full-k shapes to the kernel factories from the
carrier.  The run announces `ψ storage: parents only`.  ζ reuse is
allowed (the plan does not depend on the fit).  Restart tensors carry the
raw-parent faces in canonical order (`psi_parent_y`, `psi_parent_y_mun`,
`psi_parent_k_rows`) and no `psi_full_y`; the restart branch packs them,
rebuilds the plan and the carrier and announces `ψ storage: parents only
(restart)`.  Fractional occupations are parent-capable: the contour χ0
rides the Green transport, and the static-Γ / direct-q pair scans unfold
each band tile from the packed parents inside the scan
(`symmetry_maps.unfold_wavefunction_local`, the one-endpoint typed action).
The self-consistent map's rotation acts on the carrier's faces at the
parents' rows (`wavefunction_bundle.rotate_wavefunctions`).

Consumers that still read full-k ψ keep the run on full k and are named
in the log: `head_correction = full` (dynamic head wings: the static wing
is a |ψ|² permutation, the dynamic wings need the plan's missing Cartesian
action), `qp_solver = self_consistent` (the density rebuild loads ψ(G) on
the full BZ; the IBZ path is `rho_from_wfns(kweights, sym_perm)`), and
non-RPA diagrams.  BSE and downfold read `psi_full_y` and refuse a
parents-only restart file by name (`file_io.tagged_arrays.
require_full_k_psi`) until they take the parent unfold.

## Current face-route boundaries

`fit_zeta_to_h5(low_mem_bands=True)` refuses, by name, before any
compute:

* `band_norms is not None` (pseudobands) — the face CCT/r-chunk paths
  have no weighted-norms arm; a real feature gap if ever needed together,
  not a silent approximation.
* `psi_rmu_Y is not None` — catches a caller that forgot to drop the
  single-axis Y-form copy before calling.
* `psi_rmuT_X is not None` — the r-chunk half's OWN addition this
  session: catches a caller that forgot to drop the single-axis X-form
  copy (the whole point of this session's redesign) — mirrors the
  `psi_rmu_Y` check exactly, added beside it.

Charge and monomial transverse vertices are supported. The coupled
transverse coordinator is narrower: it requires a four-spinor face carrier,
all three channels fresh, and the planner-selected bounded face-Y cache.
Otherwise the same face kernels run on the sequential schedule.

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
  that historical source's STEP 3 conditioning/back-solve against `C_q`, and
  that solve is
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
  short of dominating the 253 s total (screening + the ζ solve dominate).
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
* **Regression:** `tests/test_zeta_mesh_invariance.py` **7/7 PASS**, JID
  57438326, historical one-rank/four-visible-GPU launch (`-G 4 -n 1`):
  single-rank numerical evidence, not a P=4 gate. The small-system fast path
  never sets `low_mem_bands` and was unaffected;
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
      solve from a genuine ~1e-15-relative summation-order change):
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

## `GemmPlan.local_call`: the manual-mode composition, resolved — and why
the r-chunk half still keeps the masked `psum` (appended 2026-08-23,
`feat/gemm-in-manual-shardmap-2026-08-22`)

The section above ("Why not a second SUMMA GEMM") named the blocker as
`GemmPlan.__call__`'s own contract — a top-level `jax.jit`+`shard_map`
that needs globally-annotated operands — not the underlying cuBLASMp FFI.
The owner's directive this session was to test that distinction for
real rather than accept it as a permanent wall: *"gemm_plan's jit doesn't
compose inside a shard_map — this is a significant structural issue that
should be able to be overcome."*

**It was.** `distrib_la.matmul_plan.GemmPlan.local_call` (new this
session) exposes the SAME `D = alpha*A@B (+ beta*C)` cuBLASMp GEMM as a
bare function on LOCAL per-rank tiles — no `shard_map`, no `jax.jit` of
its own — safe to call directly from inside somebody else's manual-mode
`shard_map`/`lax.scan`. The refactor that enabled it is small: the
transpose/`ffi_call`/transpose body that `_build_kernel` already wrapped
in its own `shard_map` (`_local_gemm_call` now, factored out as the one
shared implementation) needed nothing from that wrapper except being
handed a correctly-shaped local tile — which a caller's own manual body
already has. Descriptor/workspace lifetime (the owner's stated worry) is
a non-issue: `batched_gemm_ffi.cc`'s `cublasMpMatrixDescriptor`/
`cublasMpMatmulDescriptor` are built and destroyed FRESH on every FFI
call; only the NCCL communicator (`ctx_handle`, resolved once by
`gemm_plan()`) persists, and `local_call` reuses the identical handle
`__call__` does.

**Proven on real 4-rank CUDA, inside a manual `shard_map` + `lax.scan`**:
`check_gemm_plan_manual_shard_map`,
`services/distrib_la/tests/test_distrib_la_multiproc.py`, JID 57448156,
step `lx-Xg4-230200-1474729-7510`, exit 0. Four sub-cases, all against a
numpy reference: a bare single call, a call streamed through `lax.scan`
(the r-chunk's own shape), a `beta!=0` donated-`C` accumulate, and
donated `out=` — all from inside one manual `shard_map`. Max relative
residual `2.5e-16` (complex128) / `9.6e-17` (float64), both dtypes, well
under the package's `RTOL=1e-12` engine-parity bar. `matmul_plan.py`'s
module docstring carries the full writeup under "Composition inside a
MANUAL-mode shard_map."

### Why `_z_q_face`'s X-operand reconstruction does not switch to it

Two independent, structural reasons — checked by reading the mechanism,
not by running the (unbuilt) alternative, because both are decisive
before any code is written:

1. **`GemmPlan`'s output is never replicated.** Every `D` it returns is
   `P(None,'x','y')` — genuinely 2-D-sharded on BOTH mesh axes, because
   that is what a SUMMA-distributed provider computes. `_z_q_face` needs
   the opposite for its X operand: the SAME band-chunk window PRESENT on
   every rank (feeding a rank-LOCAL einsum against the Y-side's own
   band-replicated operand — see the section above). Turning a GEMM's
   `'y'`-sharded output into a `'y'`-replicated one needs an explicit
   `all_gather('y')` *after* the GEMM's own SUMMA communication — strictly
   MORE data movement than today's single `psum('y')`, never less. This
   is a shape mismatch a benchmark cannot fix, because no amount of tuning
   changes what `D`'s sharding *is*.
2. **The band-chunk window is not psi_mun's own block-cyclic layout.**
   Even setting aside (1) — say the consumer were rewritten to accept a
   `'y'`-sharded X, as CCT's own consumer does — a valid `k`-extent SUMMA
   operand needs `k` to divide both mesh axes AND to be laid out exactly
   as cuBLASMp's block-cyclic descriptor expects: mb=`k/px` contiguous
   rows per rank, nb=`k/py` contiguous columns. `psi_mun`'s OWN resident
   layout already satisfies this for `k = nb_face` (the CCT half's own
   `A` operand, unmodified, no gather). But the r-chunk's per-scan-
   iteration contraction extent is `bpd_max_global`
   (`band_chunk_size`-driven, chosen for the fit's OWN memory/streaming
   reasons) — a DIFFERENT, independently-chosen grid that generally does
   not align with `psi_mun`'s fixed `'y'`-shard width (`shard_w =
   nb_face/p_y`; this doc's own measured example, 16 vs. 18, is exactly a
   non-aligned pair). A window that straddles `psi_mun`'s shard boundary
   is not psi_mun's own block-cyclic tile for ANY `k`, so a genuine SUMMA
   call over it needs its OWN preceding reshard to materialize a properly
   block-cyclic-aligned local tile first — communication that is at least
   as large as the masked-gather-then-`psum` it would be replacing, since
   both are moving the same "assemble this rank's slice of an
   arbitrarily-windowed band range from a differently-sharded source"
   data. Forcing the two grids to align (making `band_chunk_size` a
   multiple of psi_mun's `'y'`-shard width, or vice versa) is exactly the
   alignment this design deliberately rejected before landing (see "The
   landed design," above): it would fragment the SHARED
   `band_chunk_ranges` grid the Y-side/`psi_G_store` machinery also uses,
   for a benefit neither reason above actually delivers.

Both reasons are about what data has to move, not about whether the call
can be EXPRESSED — `local_call` genuinely removes the expression
obstacle the owner named, and the two reasons above are what remains once
it is removed. **The masked-gather+`psum('y')` route in `_z_q_face` is
therefore kept, unchanged** — this session made no functional change to
it, re-confirmed by the unchanged parity/mesh-invariance gates below.

### What this session DID change, and its own regression check

Only `services/distrib_la/src/distrib_la/matmul_plan.py` (the new
`local_call` surface, plus a pure refactor of `_build_kernel` into a
shared `_local_gemm_call`+`_gemm_attrs` helper — no behavior change to
the existing `__call__` path) and its test file. Because `_c_q_face`'s
own CCT-half GEMM calls (`gemm(A, B)`, this doc's own "Why the CCT half
factors as a band GEMM" section) run through the SAME refactored
`_build_kernel`, that shared code is the one place this session's change
could have silently regressed production ζ-fit numerics — so it was
re-verified, not merely argued:

* `tests/test_zeta_mesh_invariance.py`: **7/7 PASS**, real jax collectives
  (CPU-emulated, `lx run -N 1 -G 0 ...`), unaffected — the small-system
  fast path never touches `matmul_plan.py`. Confirmed against THIS
  worktree specifically, not merely assumed: a bare `lx run` without an
  explicit `PYTHONPATH` wrapper silently resolves `isdf.core`/
  `common.shard_map`/`gw.gw_config` from the OTHER agent checkout
  `lorrax_A` even with `cwd` inside this worktree (module `__file__`
  printed to confirm, matching `KNOWN_SANDBOX_ERRORS.md`'s
  `retarget_pythonpath` row) — so this result was taken through a wrapper
  script that exports `PYTHONPATH`/`JAX_PLATFORMS`/`XLA_FLAGS` before
  `exec python3`, with `common/shard_map.py`'s own deprecation-warning
  file path in the pytest output confirming the worktree copy actually
  ran.
* `tests/test_isdf_cq_face_parity.py` / `tests/test_isdf_zq_face_parity.py`:
  unaffected by inspection — these gate `isdf.core.c_q_from_psi_sm`/
  `z_q_from_psi_sm`, neither of which this session edited — but NOT
  re-run in isolation this session; see the honest gap below.
* **Real 4-rank CUDA, the shared code path itself, not the deck around
  it**: `check_gemm_plan_cublasmp` (unchanged behavior) and
  `check_gemm_plan_manual_shard_map` (new), both in
  `services/distrib_la/tests/test_distrib_la_multiproc.py`, JID 57448156,
  step `lx-Xg4-230200-1474729-7510`, exit 0. `check_gemm_plan_cublasmp`
  drives `_build_kernel` through `GemmPlan.__call__` exactly as
  `_c_q_face`'s `gemm(A, B)` calls do (eager, nested-`jit`, `lax.scan`,
  `beta`-accumulate, donated `out=`) and matched a numpy reference to
  `1.3e-16`–`3.3e-16` relative — this IS the refactored code CCT's own
  GEMM calls run through, so this is direct (not by-analogy) evidence
  the refactor is behavior-preserving for the auto-mode path.
* **Honest gap: no k6_c600 production end-to-end regression run this
  session.** The intended check — rerun the exact
  `face_headoff_zetafit_rchunk_2026-08-22` reference deck
  (`memory_per_device_gb=20`, `head_correction=off`, fresh fit, P=16
  4×4) on this branch and diff `eqp0.dat`/`sigma_diag.dat`/`zeta_fit.CCT`
  timing against that reference — was attempted
  (`runs/MoS2/86_bgw_lorrax_scaling_20260819/points/
  k6_c600_lowmem_ab_20260822/face_headoff_zetafit_rchunk_gemmplan_localcall_2026-08-22/`,
  symlinked to the reference run's `WFN.h5`/`kin_ion.h5`/`dipole.h5`/
  `centroids.txt`/`cohsex.in`) and did NOT complete: the deck-directory
  `-m gw.gw_jax` launch path mis-resolved source/environment across three
  successive attempts and a fourth (after working around the first
  three) hung at 4-node NCCL bootstrap with zero measured progress on
  every rank, killed after ~2 minutes rather than left occupying the
  shared pool — see `KNOWN_SANDBOX_ERRORS.md`, "2026-08-23 — a
  deck-directory `lx run -N n -G 4 -n P python3 -u -m gw.gw_jax` also
  mis-resolves source..." for the full sequence and the specific
  failures (wrong checkout, wrong `.so`, missing `libfabric`, then the
  cross-node hang). This is a sandbox launcher/environment limitation,
  not a code finding — it neither confirms nor refutes anything about
  `_z_q_face` or the CCT half's own numerics, which are unmodified code.
  The regression claim this section actually supports is the narrower
  one above: the shared `_build_kernel`/`_local_gemm_call` refactor
  matches its pre-refactor numerics on real hardware, at the operation
  level: a production-scale confirmation remains open, and should be the
  first thing a follow-up session with a working multi-node launch path
  does before this capability is used in another manual-mode kernel.

## γ̃ VERTEX: the transverse (bispinor) channel lands (appended
2026-08-23, `feat/transverse-zeta-face-2026-08-23`)

Both `_c_q_face` and `_z_q_face` accept `gamma_L`/`gamma_R` — the same
`(perm, phase)` tuple calling convention `_c_q_legacy`/`_z_q_legacy`
already had — closing the LAST gap in the bispinor+`low_mem_bands`
census row. `gw.isdf_fitting.fit_zeta_to_h5`'s `vertex_mu_L != 0`
refusal under `low_mem_bands` is dropped; the former
`low_mem_bands_bispinor_unported` row was deleted. Full narrative and every
verification number: `claims/0442.md`.

### The mechanism: endpoint application, not post-IFFT contraction

`_c_q_legacy`'s vertex insertion happens AFTER the IFFT, inside
`gamma_double_contract`, on the ALREADY-formed real-space open-spin
pair densities `P_l`/`P_r` — `gamma_apply` transforms `P_r` alone
(both its spin axes, via two sequential calls), `P_l` untouched. The
face path reproduces the exact same physics via a structurally
DIFFERENT mechanism: **endpoint application**, folding γ̃ into the psi
FIELDS (`psi_mun`/`psi_nmu`) themselves, BEFORE the band GEMM
(`_c_q_face`) or the per-band-chunk masked-gather (`_z_q_face`) —
mirroring `gw.wavefunction_bundle.with_lorentz_vertices`'s own
field/axis table (`_G_VERTEX_FIELDS`) rather than `_c_q_legacy`'s own
mechanism. Both are valid because γ̃ acts ONLY on the spin axis, which
is REPLICATED in the face carrier and untouched by every band-
contraction/collective mechanism either kernel uses — a linear
transform on an axis a contraction doesn't touch commutes freely
across that contraction, regardless of which SIDE of it the transform
sits on. `P_l`'s construction stays untransformed either way, matching
`_c_q_legacy`'s own asymmetry (`gamma_apply` never touches `P_l`).

**The conjugation-convention trap, found and avoided.** `psi_mun`
plays OPPOSITE conjugation roles in the two kernels this vertex touches
(same field, different sign): `_c_q_face`'s `A = conj(merge_spin_
centroid(psi_mun, 1, 2))` conjugates `psi_mun` (`psi_nmu` stays
unconjugated); `greens_function_kernel._build_G_face`'s `B =
merge_spin_centroid(jnp.conj(psi_nmu), 2, 3)` conjugates `psi_nmu`
instead (`psi_mun` is the DIRECT/unconjugated operand there). This was
found by READING `_build_G_face` directly — not assumed from
`with_lorentz_vertices`'s own G-build precedent — because an
identity-vertex check (γ̃⁰ = I, both mu_L=nu_L=0) cannot discriminate a
swapped conjugation convention: `gamma_apply` with an identity
perm/phase is a no-op regardless of where the conjugate sits, so the
existing `test_isdf_cq_face_parity.py`/`test_isdf_zq_face_parity.py`
identity cases would pass either way. Consequence: for `_c_q_face`,
`gamma_L` (`mu_L`, the "left"/direct-role vertex per `with_lorentz_
vertices`'s own naming) is applied to `psi_mun` — but AFTER conjugating
it (`gamma_apply(jnp.conj(psi_mun_), perm_L, phase_L, axis=1)`), using
the ORIGINAL phase — not before, which would need a `conj(phase_L)`
compensation (`conj(gamma_apply(X, perm, phase)) ==
gamma_apply(conj(X), perm, conj(phase))`; conjugating first sidesteps
the correction entirely and is what the landed code does). `gamma_R`
applies to `psi_nmu` directly (never conjugated on the CCT path), with
no compensation needed either way. `_z_q_face` inherits the identical
convention on its own per-bc operands (`x_full_bc`, already conjugated
via `psi_mun_conj` upstream; `psi_Y_bc`, never conjugated).

### Why `_z_q_face` applies γ̃ AFTER its collectives, not before

The task's own framing ("apply... BEFORE the band GEMM") is followed
literally in `_c_q_face` (gamma applied to `psi_mun_`/`psi_nmu_`
before `merge_spin_centroid`+`gemm`, since the R-window's own GEMM
call is already separate from `P_l`'s — no extra dispatch either way).
`_z_q_face` instead applies γ̃ to `x_full_bc`/`psi_Y_bc` — the SHARED,
already-collected outputs of the masked-`psum('y')` gather and the
`all_to_all`/`all_gather` r-scatter respectively — rather than to their
pre-collective sources. This is not a deviation from the same
principle, it is the SAME principle applied where it actually saves
work: γ̃ commutes with the collective regardless of order (proven
above), so doing it after means ONE gather/scatter serves BOTH the
untransformed (`P_l`-role) and gamma-transformed (`P_r`-role) uses,
instead of a second masked-gather-then-`psum` purely to serve the
R-role. Zero extra communication, confirmed by the isolated parity
gate's own bit-exact result (identical to the identity-channel case's
own bit-exact result — no summation-order change was introduced
either way).

### The other half: the transverse centroid set's own face carrier

Porting the two kernels alone does not produce a runnable bispinor
deck — the ζ_T fit and Σ^B both need a face `Wavefunctions` bundle
sampled at the TRANSVERSE centroid set (`gw.gw_init.
_transverse_wfn_data`), a SEPARATE array from the charge channel's own
carrier. `_transverse_wfn_data` now builds it internally (the SAME
`PSI_MUN_SPEC`/`PSI_NMU_SPEC` `with_sharding_constraint` path the
charge channel's own caller uses, not re-derived), because it is the
ONE function BOTH the fresh-fit bispinor loop and the ζ-reuse early
return call — a real `KeyError: 'psi_nmu_fresh'` surfaced at exactly
this seam during the end-to-end gate, when a second run against an
already-fit ζ took the reuse branch instead of the fresh-fit branch a
first (duplicated, not-yet-refactored) attempt had covered. See
`claims/0442.md` for the full account and every job/step id.

### End-to-end result

MoS2 3×3 bispinor GN-PPM fixture (`tests/regression/bispinor_debug/
bispinor_test.in`), real 4-rank CUDA, `low_mem_bands=false` vs `=true`,
both `rc=0`: `sigma_diag_bispinor_test.dat` 0 differing non-comment
lines; `eqp0.dat`/`eqp1.dat` max|ΔE_QP| = 1.7e-8 eV, max|ΔE_DFT| = 0.0
eV — the same order of magnitude as this project's other lifted rows
(QSGW rotation: 3.3e-8 eV). Both bundles' own printed layout disclosure
(`"...face layout: psi_nmu/psi_mun..."` for charge, `"Σ^B transverse ψ
inventory (layout='face')..."` for the transverse bundle) confirms
neither ever falls back to a legacy single-axis copy under this deck.
Artifacts: `runs/MoS2/90_bispinor_lowmem_smoke_2026-08-23/`.

## Current coupled transverse schedule (2026-08-29)

The charge fit remains independent. For the three current vertices, the
driver first checks each `zeta_q_mu{1,2,3}.h5` separately. Coupling is eligible
only when all three files need a fresh fit, `low_mem_bands=true`, and the
transverse planner selected the bounded face-Y cache. Partial reuse fits only
the missing channels, sequentially. A capacity miss also uses the sequential
schedule; it is a safe fallback, not a different equation.

The important global layouts are:

```text
psi_mun[k,s,mu_X,n_Y]
C_q[mu_L,q,mu_X,nu_Y]       # three separately prepared systems
Z_mu123[mu_L,q,mu_X,r_Y]    # one shared r-chunk build
zeta_q[q,mu_XY,r]           # one channel after its solve
zeta_G[q,mu_XY,G]           # persistent G-flat output for that channel
```

`mu_L` is a three-element replicated selector, not a processor axis.
`mu_XY` means one centroid axis flattened over the product mesh. The Y and X
caches live inside the manual `shard_map`, so they are local working buffers,
not additional global arrays with public `NamedSharding` contracts. Their
conceptual layouts are `psi_Y[bc,k,n,s,r_Y]` with the active band block
replicated, and `psi_X[bc,k,s,mu_X,n]` with that block broadcast from its Y
owner.

For each outer r chunk, all three host threads meet at the coordinator. μ=1
builds `Z_mu123`; μ=2 and μ=3 wait for it. `_z_q_face_coupled_mu123` performs
the canonical ψ(G)→ψ(r) transform, the Y `all_to_all('y')` plus
`all_gather('x')`, and the X-owner broadcast once per band chunk. It also
reuses the channel-independent left
pair density. The right density and vertex phase are evaluated for μ=1, then
2, then 3, with only one channel's right carry live. This preserves the
accepted scalar-spin-pair and band-scan order.

The leading three-channel RHS is not solved as one 108-system batch. The
coordinator hands out one `Z_q[q,mu_X,r_Y]` slice at a time, and the existing
per-channel solver runs in μ=1→2→3 order. This retains the accepted three
36-q arithmetic boundaries; the former stacked solve changed physical CrI3
results at about the 1e-9 relative level and is not selected by `gw_init`.

There are two solve transports:

- `batch_reshard`: keep the raw CCT for each channel; the trace-scaled ridge
  is added inside each per-r-chunk solve. On every r chunk,
  `distrib_la.Plan.batched` moves independent q systems from
  `C_q[q,mu_X,nu_Y]` / `Z_q[q,mu_X,r_Y]` to whole matrices distributed over
  the batch axis, runs local JAX LU solve, and returns the face-sharded result.
  It deliberately refactors each r chunk. The coupled policy selects this
  route automatically only on its audited A100 square P4/P16 envelope and
  only when the full live-set and operand-floor checks pass.
- distributed factor-token route: factor each channel once during ordered
  preparation. The LU factors and pivots remain inside an opaque
  `distrib_la.FactorToken`; each r chunk sends its 2-D-sharded RHS through the
  matching provider back-solve. No factor is exposed, concatenated, gathered,
  or moved into the local batch route.

An explicit `batch_reshard` request is never silently converted to the
distributed provider. If the coupled local live set does not fit, the driver
keeps that requested per-channel solve route but drops coupling. With
`distrib_la_batched_route=auto`, it may keep coupling on the distributed-token
route when that complete live set fits.

Each channel's G-flat accumulator is moved to process-local host memory before
the coordinator releases preparation. For every channel slice, the code
restores only that accumulator, calls the canonical
`accumulate_rchunk_to_gflat`, and spills it again immediately. Thus all three
completed outputs exist concurrently on the host, but only the active one is
on device. After the last r chunk, a final-ready barrier prevents μ=1 from
starting its G-flat write while μ=2 or μ=3 still solves. Final restore, write,
close, and provenance remain ordered μ=1→2→3.

The capacity equations, host-spill pricing, solve operand floor, and
fragmentation target are owned by [the memory model](memory-model.md) and
`gw.gflat_memory_model`. The large-centroid alternatives and their sharding
costs are summarized in [`docs/dev/large_nmu_operation.md`](../dev/large_nmu_operation.md).
