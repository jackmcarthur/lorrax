# Exact finite-occupation χ₀ on the two-face ψ carrier (`low_mem_bands=true`)

Ports census row "Exact finite-occupation response"
(`reports/gwjax_low_mem_bands_audit_2026-08-22/report.md`,
`gw_config._LOW_MEM_BANDS_REFUSALS`'s `low_mem_bands_metal_material_class_
unported`) plus the fractional/contour χ₀ kernel the same report's revision
noted as a second, separate non-consumer of `_build_Gv_Gc`
(`gw.w_isdf._get_chi_fractional_contour_kernel`). Landed
`feat/metal-response-face-2026-08-23`, on top of
`integ/low-mem-bands-2026-08-22`. Companion note to
`docs/architecture/zeta_fit_face_psi_cct.md`, whose masked-gather + `psum`
idiom (`isdf.core._z_q_face`) this design reuses on BOTH mesh axes.

## Two physically different kernels, two different ports

`src/gw/w_isdf.py` carries two chi0 families that read ψ directly instead of
going through the ordinary `build_G_tau`-based minimax kernel
(`_get_chi_minimax_kernel`, already ported 2026-08-22):

1. **The fractional/contour kernel** (`_get_chi_fractional_contour_kernel` /
   `compute_chi0_contour_fractional`) — builds
   `A_q(t) = Σ_ab f_a (1-f_b) exp[-i(E_b-E_a)t] X_ab(q)` from **two
   independent one-particle Green's functions**, `Gf` weighted by `occ_f`
   and `Gu` weighted by `1-occ_u`. The exponential and the weight both
   **separate** across `a`/`b` — `exp[-i(E_b-E_a)t] = exp[iE_a t]·
   exp[-iE_b t]` and `f_a·(1-f_b)` is a product of a per-`a` and a per-`b`
   factor — so this is exactly the shape `greens_function_kernel.build_G_tau`
   already computes (`G_μν(k) = Σ_n ψ_n(μ)·w_n·ψ*_n(ν)`), which the
   `low_mem_bands` port already ships a `layout='face'` arm for. **This half
   needed no new distributed algorithm** — see "Part A" below.

2. **The ordered-pair kernel** (`_fractional_pair_scan` and its two callers,
   `_get_chi_static_fractional_gamma_kernel` /
   `_get_chi_fractional_q_kernel`, reached through
   `compute_chi0_static_fractional_gamma` /
   `compute_chi0_static_fractional` / `compute_chi0_direct_fractional`) —
   the exact static divided difference `(f_a-f_b)/(E_a-E_b)` (or its dynamic
   generalization `(f_a-f_b)/(E_a-E_b+z)`) is **jointly** a function of BOTH
   band indices' energies and occupations. It does **not** separate into a
   product of a per-`a` and a per-`b` factor (the denominator couples `a`
   and `b`), so it cannot collapse to a one-particle `build_G_tau`/GEMM
   contraction — the census's own finding, confirmed here by re-derivation,
   not merely repeated. **This half needed a genuine new distributed
   algorithm** — see "Part B" below, the actual "2-D band-pair ring/tile"
   the census called for.

## Part A: the fractional/contour kernel — a `build_G_tau` face port

`build_G_tau(..., layout='face', gemm=..., band_weight=...)` already exists
(shipped with the ordinary minimax kernel's face port) and accepts an
arbitrary per-`(k,n)` `band_weight`. `_get_chi_fractional_contour_kernel`'s
body already calls `build_G_tau` twice per time node — once for `Gf`
(`band_weight=occ_f`), once for `Gu` (`band_weight=1.0-occ_u`) — so the face
port is a **substitution of operands**, not a new mechanism:
`psi_f_xn`/`psi_f_yr`/`psi_u_yr`/`psi_u_xn` (four legacy views) become
`psi_mun`/`psi_nmu` (the two face copies), and ONE `distrib_la.gemm_plan`
(shape `m=n=n_rmu*ns, k=nb_full, nq=nk`) is built once and shared by both
`Gf` and `Gu`, mirroring `_get_chi_minimax_kernel_face`'s own `g_plan`
shared by `Gv`/`Gc`.

**The one genuine subtlety: legacy's band WINDOW is a real cost cut that
face cannot reproduce as a window.** `_occupation_support_slices` returns
the smallest CONTIGUOUS `f_slice`/`u_slice` covering every band whose
occupation weight clears a threshold; legacy slices `wfns.xn(f_slice)`/
`wfns.yr(f_slice)` down to that window before the `build_G_tau` contraction
— fewer bands enter the einsum, not just a smaller weight on them. A face
carrier cannot be band-sliced (obstacle #3, `Wavefunctions.band_mask`'s own
docstring) — so face pays the full `nb_full` contraction and reproduces the
window via the SAME "weight, don't window" convention `isdf.core._c_q_face`
established: `occ_f_face = occ_full * indicator([f_lo, f_hi))`, where the
indicator is exactly 1 inside `f_slice` and 0 outside. Because the
contraction is bilinear in ψ (a zero-weighted band contributes exactly
zero — the same TASTE-15 argument the CCT design used for its own L/R
window), this reproduces legacy's windowed sum bit-for-bit up to
summation-order roundoff, at the cost of a full-`nb_full` GEMM per call
instead of a windowed one (the SAME cost legacy's own `band_mask`-based
val/cond split already accepted for the ordinary minimax kernel).

No new tiling, no new communication primitive: this is the same
`_get_chi_minimax_kernel_legacy`/`_get_chi_minimax_kernel_face` split,
mechanically repeated for a second factory
(`_get_chi_fractional_contour_kernel_legacy`/
`_get_chi_fractional_contour_kernel_face`), with the cache-management lines
moved to the dispatcher exactly as that first split's own precedent (the
dispatcher owns `cache_key`/lookup/store; the `_legacy` sibling is a pure,
UNTOUCHED builder — diff-confirmed against the pre-session source with only
the caching lines removed).

## Part B: the ordered-pair kernel — the genuine 2-D band-pair algorithm

### Why the naive port is banned, not merely slow

Legacy's `_fractional_pair_scan` receives `psi_x_a`/`psi_x_b`
(`PSI_XN_SPEC`, μ on X, **bands fully replicated**) and `psi_y_a`/
`psi_y_b` (`PSI_YN_SPEC`, μ on Y, bands fully replicated), and tiles the
`O(nb²)` ordered-pair sum with cheap, comm-free `dynamic_slice`s, because
every rank already holds every band. The obvious face substitute —
`all_gather('y')` the persistent `psi_mun` once to rebuild a
band-replicated, μ-on-X-only array — reproduces exactly the LEGACY
single-axis residency, `2·S/Px` (mu on X, all bands, replicated over Y),
not the face carrier's `2·S/(Px·Py)`. That is the **√P-class** residency
the owner's scaling rationale (`reports/gwjax_low_mem_bands_audit_2026-08-22/
report.md`, "Owner's scaling rationale for the all-P psi requirement")
explicitly bans for anything resident under `low_mem_bands=true` — the
exact trap the zeta-fit r-chunk port already hit and solved once
(`docs/architecture/zeta_fit_face_psi_cct.md`'s r-chunk section). This
design reuses that solution rather than re-deriving it.

### Why `GemmPlan`/`GemmPlan.local_call` do not apply

Two independent reasons, both structural (checked by reading the contract,
not by benchmarking a rejected alternative — same standard the r-chunk
design note held itself to):

1. **The weight is not bilinear in `(a,b)`.** A GEMM computes
   `Σ_n A(m,n)·B(n,p)` — a contraction that is linear in a SINGLE shared
   index `n`. The ordered-pair weight `(f_a-f_b)/(E_a-E_b+z)` (or its
   diagonal `-df/dE` limit) is a function of the PAIR `(a,b)` that does not
   factor as `u_a·v_b` for any choice of `u`, `v` (the denominator mixes
   both indices) — this is a re-derivation, not a repetition, of the
   census's finding. There is no GEMM whose output is this weighted sum:
   the weight must be evaluated PER PAIR, which is exactly what the tile
   scan already does locally under legacy.
2. **Even where a GEMM shape existed, `GemmPlan.local_call`'s output is the
   wrong sharding for this consumer.** `zeta_fit_face_psi_cct.md`'s own
   "why `_z_q_face`'s X-operand reconstruction does not switch to it"
   section already worked this out for the structurally analogous r-chunk
   case: a SUMMA GEMM's `D` is genuinely 2-D-sharded (`P(None,'x','y')`);
   this kernel needs a band-tile PRESENT ON EVERY RANK along one axis
   while μ stays local on the other — the shape a broadcast/`psum`
   produces, not a GEMM. The same argument applies here unchanged.

### The chosen mechanism: masked-gather + `psum`, on BOTH mesh axes

`isdf.core._z_q_face` reconstructs a BOUNDED band tile of the μ-on-X form
from `psi_mun` via a per-position `jnp.take` (clamped, always in-bounds) +
`jnp.where(owner==rank)` + `jax.lax.psum('y')` — a selective
broadcast-from-owner, not a resident copy and not a full `all_gather`. This
design applies the SAME idiom to reconstruct BOTH ψ orientations the
ordered-pair kernel needs, for a bounded band tile at a time:

```
_gather_mun(psi_mun_local, g_lo)   -- masked-gather + psum('y')
    (nk, s, mu_X_loc, tile) un-conjugated, from psi_mun (bands on 'y')

_gather_nmu(psi_nmu_local, g_lo)   -- masked-gather + psum('x'), then
    a LOCAL (no-comm, bounded-size) axis reorder
    (nk, s, mu_Y_loc, tile) un-conjugated, from psi_nmu (bands on 'x')
```

`_gather_mun` needs no reorder: `psi_mun`'s own axis order `(nk, s, μ, n)`
already matches `PSI_XN_SPEC`'s `(nk, s, μ_X, n)`. `_gather_nmu` does:
`psi_nmu` stores `(nk, n, s, μ)` (band axis SECOND, not last), so the
post-gather tile — bounded to `(nk, tile, s, μ_Y_loc)` — gets one
`jnp.transpose` to `(nk, s, μ_Y_loc, tile)`, matching `PSI_YN_SPEC`'s order.
This transpose is on an ALREADY-LOCAL, bounded-size array (`tile` bands,
not `nb_full`) — a register/HBM-local reorder, not a collective; it costs
nothing communication-wise, unlike the CCT design's discovery that the SAME
kind of reorder on a FULL μ-extent object was a genuine `μ²`-scale
transpose (that finding does not apply here because the object being
reordered here is bounded to `tile`, not `n_rmu`).

Both gathers are IMMUNE to a band tile that overruns the real `nb_full`
extent by construction (`_z_q_face`'s own documented property): for a
phantom `global_band >= nb_full`, `owner_y`/`owner_x` computes `>= p_y`/
`>= p_x`, which no real rank ever equals, so `psum` returns exactly zero —
no separate `bc_valid` clamp-and-mask is needed here the way `_z_q_face`
needed one for its `weight_l`/`weight_r` lookup, because THIS design pads
`energy`/`occupation`/`surface_weight` with `jnp.pad` (legacy's own
technique, reused verbatim) rather than indexing a real array at an
out-of-range position — the padded region is `0.0`, always finite, and its
contribution is independently zeroed by the pre-existing `nb_logical`
mask (`ga < nb_logical & gb < nb_logical`, unchanged from legacy) since
`nb_logical <= nb_full` always holds.

### The physics core is duplicated, not shared — by the tree's own convention

`_fractional_pair_scan_face`'s per-pair weight/density/contribution math
(the divided-difference weight, the two density contractions, the final
`zmn` einsum) is the SAME derivation `_fractional_pair_scan` already
carries, and is INTENTIONALLY re-typed rather than factored into a shared
helper `_fractional_pair_scan` also calls — mirroring every other
legacy/face split in this codebase (`_c_q_legacy`/`_c_q_face`,
`_z_q_legacy`/`_z_q_face`, `_legacy_build_G`/`_face_build_G`): the legacy
function is FROZEN so it stays diff-confirmably byte-identical, and a
sibling function carries the face mechanism. `_fractional_pair_scan`
itself is untouched by this session — confirmed by diff against the
pre-session source.

### Tiling choice: nested scan (outer reuses "a", inner refetches "b")

The `O(ntiles²)` pair loop is restructured as two nested `lax.scan`s
(outer over the "a" band tile, inner over "b") rather than legacy's single
flat scan over `ntiles²` steps, because the two orientations now have
DIFFERENT costs: a `dynamic_slice` on a resident array is free regardless
of loop shape, but a masked-gather+`psum` is not, so the shape of the loop
now matters. The outer scan reconstructs `a`'s two tiles ONCE per outer
step and reuses them across the whole inner sweep; the inner scan
reconstructs `b`'s two tiles fresh every step, because a "b" tile is never
resident for more than one inner iteration. This is the design's actual
answer to "2-D band-pair ring/tile": communication is bounded to
`O(ntiles)` for the "a" side and `O(ntiles²)` for the "b" side, and the
resident working set at any instant is `O(tile)` band-widths on each
operand — never the `O(nb_full)` a cached single-axis form would need
(the same "no resident single-axis array, ever" contract the r-chunk fit
port established).

**Measurement, not assumption, decided against a `ppermute` ring for this
session.** A genuine systolic ring (rotating each rank's own resident
`psi_mun`/`psi_nmu` shard around the mesh via `lax.ppermute`, `p_y`/`p_x`
steps instead of `ntiles` masked-`psum` calls) is a plausible follow-up if
`ntiles` ever exceeds `p_y`/`p_x` by a large factor at production scale —
but at the scale this session's own gate and Na-deck harness measure
(`nb_full` in the tens, `pair_tile=8`, `ntiles` in the single digits, the
outer×inner product a few dozen `psum` calls total), the masked-gather
route's cost was negligible against the fit/screening/Sigma stages the
existing face G-build/CCT ports already accept the SAME mechanism's
communication cost for (`gw.greens_function_kernel`'s own module docstring:
"the ~20% end-to-end face overhead ... is the communication price of the
8× ψ memory reduction, not recoverable by restructuring ψ storage"). A
`ppermute` ring is NOT implemented this session; if a future large-`nb_full`
metal deck (hundreds of bands, not the tens this port was gated on) proves
the `O(ntiles²)` term matters, chasing it is a well-scoped follow-up with a
name in `KNOWN_LORRAX_ISSUES.md` rather than a design gap.

### Dispatch

`compute_chi0_static_fractional_gamma`, `compute_chi0_static_fractional`
(a thin `z=[0]` wrapper around the next one, unaffected), and
`compute_chi0_direct_fractional` all dispatch on `wfns.layout`, mirroring
`_chi_layout_operands`'s established pattern for the ordinary minimax
kernel. Under `layout='face'` the caller's (possibly narrower than
`nb_full`) `energies`/`occupations`/`surface_weight` tables are zero-padded
up to `nb_full` (harmless: any padded position is `>= nb_logical`, hence
already excluded by the pre-existing `nb_logical` mask) before the face
kernel is called with the FULL `psi_mun`/`psi_nmu`.

## Current reachability

Material class is inferred from the loaded WFN occupations; it is not a deck
option. The face-carrier MPA executor and its occupation-weighted kernels are
now supported with `low_mem_bands = true`, so no material-specific refusal row
remains. The live option envelope is owned by [the input
reference](../input_reference.md#system); this page records the kernel port and
does not duplicate that register.

## Verification

See `claims/0441.md` for exact job ids and artifact paths; summarized here.

* **A real bug was found and fixed while gating this port — worth
  recording precisely, because the SMALL synthetic unit test could not
  have caught it.** `_get_chi_fractional_contour_kernel_face`'s `Gu`
  build applied `band_weight = 1.0 - occ_u`, mirroring the LEGACY
  kernel's own naming (`occ_u` = a raw, merely-SLICED occupation there).
  But the face args builder (`_chi0_fractional_contour_args`) was
  ALREADY masking `occ_u` to zero outside `u_slice` before handing it to
  the kernel ("weight, don't window") — so for every band OUTSIDE the
  true support window, the kernel computed `1.0 - 0.0 = 1.0` instead of
  the required `0.0`, silently pulling every excluded band INTO `Gu_k`'s
  contraction with full empty-state weight. On the small synthetic gate
  (`nb_full=24`, energies drawn i.i.d. uniform on `[-1,1]`) the derived
  `f_slice`/`u_slice` happened to span nearly the WHOLE band range, so
  the masking was nearly a no-op and the bug was invisible: the check
  passed at 1e-16 relative with the bug PRESENT. It surfaced only on the
  production-shape Na harness below, whose real semicore/valence/
  conduction structure gives a genuinely narrow `f_slice=[0,10)`,
  `u_slice=[8,48)` on a 48-band window — **measured max\|rel diff\| =
  0.188**, spread across 98.7% of the (μ,μ) output (mean absolute diff
  15.4 against a reference scale of 496 — not a localized indexing slip,
  every band outside the true window contributing garbage). Fixed by
  moving the `1-occ` inversion INTO the args builder, before masking, so
  the value the kernel receives and applies directly IS the final
  weight (`occ_u_face = (1.0 - occ_full) * u_ind`, kernel:
  `band_weight=occ_u`, no further transform) — re-verified by a negative
  control (reverting the fix reproduces `max_rel≈0.48` on the CHECKED-IN
  gate's own new deep-tail case; restoring it returns to 1e-16). The
  checked-in gate (`tests/test_chi0_fractional_face_parity.py`) was
  strengthened with a deep semicore tail (3 bands at −40 to −50 Ry) and a
  deep virtual tail (3 bands at +40 to +50 Ry) specifically so a genuinely
  narrow, asymmetric support window is exercised going forward — the
  precise shape class that hid this bug the first time.
* **Part A (fractional/contour), algebra parity, real 4-rank CUDA**
  (`lx run -N 1 -G 4 -n 4 ... --mesh 2x2`, JID 57457661, step
  `lx-Xg4-024653-187362-6021`, exit 0): `tests/
  test_chi0_fractional_face_parity.py`, ns=1/ns=2, a genuinely metallic
  (MP1, fractional, an injected exact degeneracy, deep semicore/virtual
  tails forcing a genuinely narrow occupation-support window). 2/2 cases
  PASS: `max|rel diff|` 4.73e-16 (ns1) / 4.25e-16 (ns2). CPU-emulated
  (`--xla_force_host_platform_device_count=4`, no `lx run` needed) SKIPS
  this quantity by name — the sandbox's host FFT FFI backend is
  unavailable (`KNOWN_SANDBOX_ERRORS.md`, 2026-08-22 row) — real CUDA is
  the gate of record for Part A.
* **Part B (ordered-pair), algebra parity, real 4-rank CUDA AND
  CPU-emulated** (same JID/step for CUDA; CPU-emulated needs no GPU —
  `lx run -N 1 -G 0 ...`): the Gamma (`compute_chi0_static_fractional_
  gamma`) and finite-q/finite-z (`compute_chi0_direct_fractional`,
  nonzero `z`, the dynamic-weight branch) cases, same occupation table,
  `nb_logical < e.shape[1] < nb_full` (Gamma) / `nb_logical < nb_full`
  (direct) — genuinely non-trivial logical windows exercising the
  zero-pad path. 2/2 cases PASS both ways: real CUDA `max|rel diff|`
  1.91e-16 (ns1 direct) to 1.59e-16 (ns2 gamma); CPU-emulated 1.14e-16
  to 2.14e-16, same order.
* **Instrumented no-single-axis-psi proof**: STRUCTURAL, not merely
  measured — every new face kernel's `shard_map` `in_specs` name only
  `PSI_MUN_SPEC`/`PSI_NMU_SPEC` (both `P(None,·,'x','y')`-shaped, 2-D
  sharded on BOTH mesh axes); no single-axis-shaped array can be
  constructed inside the traced program at all. Confirmed by an
  AST-level source scan of the four new functions
  (`_get_chi_fractional_contour_kernel_face`, `_fractional_pair_scan_
  face`, `_get_chi_static_fractional_gamma_kernel_face`,
  `_get_chi_fractional_q_kernel_face`): zero occurrences of `psi_xn`/
  `psi_xr`/`psi_yr`/`psi_yn`/`.xn(`/`.xr(`/`.yr(`/`.yn(` in any of them.
* **`low_mem_bands=false` bit-identical**: `_fractional_pair_scan`,
  `_get_chi_static_fractional_gamma_kernel`, `_get_chi_fractional_q_kernel`,
  and `_get_chi_fractional_contour_kernel_legacy` are diff-confirmed
  UNTOUCHED against the pre-session source (`git diff` shows no `-` line
  inside any of their bodies; the contour kernel's extraction into a
  dispatcher+legacy-sibling pair removed only its own cache-management
  lines, mirroring `_get_chi_minimax_kernel`'s established split); every
  new branch is reached only under `wfns.layout == 'face'`. Independent
  regression confirmation: `tests/multi_device/fractional_chi_gate.py`
  (the pre-existing dense-Kubo-oracle gate for the UNTOUCHED legacy
  contour kernel), real 4-rank CUDA, JID 57457661, step
  `lx-Xg4-022613-139085-6291`: `max_rel=4.550e-16` (dense Gamma) /
  `3.721e-16` (dense finite-q) — re-validates the legacy body after the
  dispatcher split, independent of anything face-specific.
* **`tests/test_low_mem_bands_envelope.py`** (the refusal-row comment
  update): 24/24 PASS, JID 57457661.
* **`tests/test_zeta_mesh_invariance.py`**: 7/7 PASS, JID 57457661,
  historical one-rank/four-visible-GPU launch (`-N 1 -G 4 -n 1`). This is
  single-rank numerical evidence, not P=4 or scaling evidence. The CPU-emulated multi-device
  path for this file is unreliable on this sandbox's jax build
  independent of any diff here (`KNOWN_SANDBOX_ERRORS.md`, 2026-08-23
  row); 5/7 there, verified identical on the unmodified base tree via
  `git stash`.
* **Na-deck production-shape harness, face-vs-legacy** (`runs/Na/
  02_soc48b_qsgw_mpa/57_lowmem_metal_response_harness_20260823/`,
  `harness.py`, JID 57457661, step `lx-Xg4-024717-188236-4221`, exit 0):
  the deck's REAL WFN.h5 header (`nk=29`, `nb_full=48`, `ns=2` fully
  relativistic, `n_rmu=176` from `centroids_frac_176.txt`) and a REAL
  `OccupationState.solve_mp1` at the deck's real eigenvalues and real
  occ-weighted electron count (9.000001), width 0.01 Ry (this deck's own
  degauss/2 convention). psi itself is synthetic (see the harness's own
  module docstring for the precise, stated scope — NOT a full `gw_jax`
  driver run, and why one is structurally unreachable). ALL THREE
  quantities PASS at machine precision: gamma 4.54e-16, direct 4.39e-16,
  contour 6.13e-16 (the contour sub-check uses a factorizable `nk=32`
  `(2,4,4)` grid with real eigenvalues resampled to fill it, since the
  deck's own `nk=29` is IBZ-reduced and prime — stated in the harness's
  own comment). This run is what FOUND the bug above on its first
  attempt (contour max_rel=0.188) and confirms the fix at production
  scale on its second.
