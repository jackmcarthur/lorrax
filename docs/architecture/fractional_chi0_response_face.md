# Exact finite-occupation χ₀ on the two-face ψ carrier (`low_mem_bands=true`)

> 2026-09-06 parent-route update: production minimax and fractional-contour
> response factories now accept only canonical faces or typed raw parents;
> their legacy four-copy factories and operand branches are deleted.
> The contour response validation uses an independent band-pair sum. The
> ordered-pair band scan this note also ported (static Γ, finite-q and
> finite-z) was deleted on 2026-09-17: static χ₀ and the MPA metal first
> sample come from `compute_chi0_matsubara`.
> Historical port descriptions below are retained as derivation context,
> not as current route-selection instructions; see [decisions](decisions.md).

Ports census row "Exact finite-occupation response"
(`reports/gwjax_low_mem_bands_audit_2026-08-22/report.md`, former gate
`low_mem_bands_metal_material_class_unported`) plus the fractional/contour χ₀
kernel the same report's revision
noted as a second, separate non-consumer of `_build_Gv_Gc`
(`gw.w_isdf._get_chi_fractional_contour_kernel`). Landed
`feat/metal-response-face-2026-08-23`, on top of
`integ/low-mem-bands-2026-08-22`. Companion note to
`docs/architecture/zeta_fit_face_psi_cct.md`, whose masked-gather + `psum`
idiom (`isdf.core._z_q_face`) this design reuses on BOTH mesh axes.

## The kernel

`src/gw/w_isdf.py` carries one fractional-occupation chi0 family that reads ψ
through its own time nodes instead of the ordinary `build_G_tau`-based minimax
kernel (`_get_chi_minimax_kernel`, already ported 2026-08-22):

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

## What this does NOT unblock, and why (read before touching the refusal row)

`gw_config._validate_metal_compute_mode` REQUIRES `compute_mode = mpa`
whenever `mpa_material_class = metal` (`gw_config.py:3827-3853`) — there is
no deck combination that reaches this session's ported kernels without also
setting `compute_mode = mpa`. `compute_mode = mpa` is UNCONDITIONALLY
refused under `low_mem_bands = true` by the separate, pre-existing
`low_mem_bands_dynamic_ppm_unported` row (`gw.mpa.sigma`'s own executor —
a completely different subsystem: frequency-domain Σ_c(ω) integration, not
this note's χ₀/response subject — was mechanically ported in an earlier
session but never end-to-end gated). So **lifting
`low_mem_bands_metal_material_class_unported` alone cannot unblock a live
`low_mem_bands=true` metal deck**: a deck that cleared this row would
immediately refuse at the OTHER row instead, with a different rule id. This
is not a gap in this session's port; it is a genuinely separate, unrelated
census item this session was not scoped to touch. `docs/input_reference.md`
and the historical lift record in `gw_config.py` say this explicitly
so the next reader does not have to re-derive it — the row STAYS REFUSING,
narrowed to name the real remaining blocker, per this session's own
verification section below.

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
* **Instrumented no-single-axis-psi proof**: STRUCTURAL, not merely
  measured — every new face kernel's `shard_map` `in_specs` name only
  `PSI_MUN_SPEC`/`PSI_NMU_SPEC` (both `P(None,·,'x','y')`-shaped, 2-D
  sharded on BOTH mesh axes); no single-axis-shaped array can be
  constructed inside the traced program at all. Confirmed by an
  AST-level source scan of the new functions (the contour kernel face, and
  the three pair-kernel functions since deleted): zero occurrences of `psi_xn`/
  `psi_xr`/`psi_yr`/`psi_yn`/`.xn(`/`.xr(`/`.yr(`/`.yn(` in any of them.
* **`low_mem_bands=false` bit-identical** (at the time; legacy bodies since
  deleted): `_get_chi_fractional_contour_kernel_legacy` and the legacy pair
  kernels were diff-confirmed
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
  driver run, and why one is structurally unreachable). The contour
  quantity PASSES at machine precision, 6.13e-16 (the since-deleted pair
  kernel's gamma and direct rows gave 4.54e-16 and 4.39e-16; the contour sub-check uses a factorizable `nk=32`
  `(2,4,4)` grid with real eigenvalues resampled to fill it, since the
  deck's own `nk=29` is IBZ-reduced and prime — stated in the harness's
  own comment). This run is what FOUND the bug above on its first
  attempt (contour max_rel=0.188) and confirms the fix at production
  scale on its second.
