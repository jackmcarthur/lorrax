# Bispinor TT (transverse-transverse) q=Γ head/wing correction — design note

Date: 2026-08-23
Branch: `feat/bispinor-static-headwings-2026-08-23`, base
`integ/low-mem-bands-2026-08-22@b70d3bcd`.
Scope: the **present static UNSCREENED** bispinor method (`sigma_x_bispinor.py`'s
`compute_sigma_x_bispinor`, the bare-Breit Σ^B). Screened transverse/CT/TC
heads are explicitly out of scope — see "What this is not" below.

## Binding guides and where this note sits in them

Two 2026-08-22 audits are the binding design references for the eventual
fully-screened bispinor pathway:

* `reports/full_bispinor_gw_audit_2026-08-22/report.md` — the full
  screened-photon Dyson/Sigma architecture.
* `reports/bispinor_screened_wings_q0_audit_2026-08-22/report.md` — the
  q→0 head/wing physics for that architecture.

Both name the SAME first step, independently of each other and of this
note: the second guide's "Recommended landing order" item 1 is *"Land the
already documented bare TT mini-BZ correction using the existing `vcoul`
sampler. It is independent, known missing, and leading order for a
slab."* The first guide's own bottom line calls the bare TT head "the
already known TT mini-BZ correction" that a later landing order item
(6) assumes is already in place before screened heads are attempted. This
work IS that first step — not a shortcut around the two guides, but their
own named entry point, scoped to the exchange-only Σ^B that already exists
in production.

## The physics: what is missing, and why it is the current-current analogue of the charge head

### The charge channel's q→0 problem, and its existing fix

The bare exchange sum Σ_X = Σ_q V(q) M(q) (schematically) needs V(q) at
every q on the finite k-grid, including q=Γ. In 3D bulk V(q) ~ 8π/q²
diverges; in LORRAX's slab-truncated 2D kernel it is a milder 1/q cusp
(`docs/BISPINOR_DHFB_DESIGN.md` §11). Either way, a single point evaluation
at q=Γ is undefined, so the code zeros that grid slot
(`compute_v_q_per_G`'s `denom_zero` guard) and the CORRECT discrete-BZ-sum
replacement is the mini-BZ Voronoi CELL AVERAGE of the (regular except at
the origin) integrand — `vc0 = ⟨v(q)⟩_mBZ` — computed by
`vcoul.{slab_2d,bulk_3d}.*.q0_average`'s Sobol sampler and consumed by the
charge exchange head machinery (`gw.head_correction`, `StaticHeadTerms`).

This works for CC because the CHARGE structure factor has an EXACT,
direction-independent limit: `M_mn(k, q→0, G=0) → δ_mn` (wavefunction
orthonormality). So the missing slot's replacement is just the scalar
`⟨v⟩` times the band-diagonal identity — no angular structure needed.

### The current channel's q→0 problem is qualitatively different

The bare transverse (current-current) exchange sum has the same divergent
grid slot, but TWO things differ from the charge case:

1. **The kernel itself has no q→0 limit.** The Coulomb-gauge transverse
   propagator carries the spatial block
   `−P^T_ij(K̂) = −(δ_ij − K̂_iK̂_j)`
   (`gw.v_q_bispinor._make_per_q_v_builder_for_tile`), which depends only
   on DIRECTION and is therefore discontinuous at K=0 — there is no single
   number `t_ij(0)` to fall back on, unlike the charge kernel's isotropic
   `1/q²`/`f_2D(q)` which at least has a well-defined (divergent, but
   direction-independent) radial profile.
2. **The current structure factor does NOT collapse to δ_mn.** The
   relevant q→0 vertex is `j^i_mn(k) = ⟨mk|α^i|nk⟩` — diagonal elements are
   band velocities (finite, generically nonzero), and with spin-orbit
   coupling the OFF-diagonal (spin-mixing) elements are generically
   nonzero too. So even the "which observable does the missing slot
   couple to" question has a different, richer answer than the charge
   case.

Point (1) is why a naive "just reuse `vc0`" fix is wrong: `v(q)` alone is
not the object that needs averaging; `−v(q)·P^T_ij(q̂)` is, and that product
DOES have a well-defined (anisotropic) mini-BZ cell average even though
neither factor has a limit on its own. Point (2) is why the fix must land
in the (μ,ν) CENTROID-BASIS V-tile that Σ^B's existing `sigma_sx_k`
convolution consumes, rather than as a scalar band-diagonal shift bolted
onto Σ^B afterward the way the charge channel's `StaticHeadTerms` does it
(`cohsex_sigma.compute_cohsex_sigma`'s `x_head =
static_head_terms_to_kij(...)` call, added to `sig_x` AFTER the μν
convolution already ran) — that shortcut is only valid because the charge
structure factor IS exactly diagonal at q=0; a current-channel version
would silently discard the off-diagonal band structure the design doc
below already measured as nonzero.

### What is actually missing, named precisely

The Gamma-cell singular term for the current-current interaction — the
transverse-channel analogue of the charge channel's v(q)→∞ head — is the
**q=Γ, G=0 slot of every TT (μ_L,ν_L ∈ {1,2,3}) V-tile**, currently left
at exactly zero by `gw.v_q_bispinor._make_per_q_v_builder_for_tile`
(the bare `v` factor is already zero there, so `v·t` is zero regardless of
`t`'s direction-dependence). The correct replacement is the mini-BZ
Voronoi cell average of the FULL transverse kernel,

```
D^TT_ij = −⟨ v(q) P^T_ij(q̂) ⟩_mBZ
```

which — unlike `P^T_ij` alone — is perfectly well-defined: it is the same
kind of Monte-Carlo cell average `vc0` already is, just weighted by the
projector instead of by 1.

This is not a new physics claim. It is the SAME correction
`docs/BISPINOR_DHFB_DESIGN.md` §11 ("q=Γ treatment of the CC, TT and CT
tiles") already derived and measured on the bi4 (MoS2 4×4, 402 charge +
143 transverse centroids) deck, job 7885325, and registered as
`KNOWN_LORRAX_ISSUES.md`'s bispinor row (claim 41): the missing rank-1
head is comparable in Frobenius norm to the WHOLE stored q=Γ TT slab
(ratio 0.97/1.04/6.0 for the 11/22/33 tiles) and shifts eqp by ≈0.2 meV at
4×4 — sub-meV at that grid, but decaying only as ~1/√N_k, "the same slow
decay that makes the CC head correction mandatory in 2D." That row also
names the smallest fix: replace the q=Γ, G=0 slot of the TT builders with
`−⟨v P_T^{ij}⟩`.

For an isotropic 3D cell the closed form is `⟨t_ij⟩_angle = (2/3)δ_ij`
(the projector's trace is 2 in every direction). For the in-plane mini-BZ
of a slab it is `diag(1/2, 1/2, 1)` — exactly the historical bi4
measurement (0.4993, 0.5007, 1.0000).

### What is NOT missing (and stays out of scope here)

* **CT/TC tiles.** In Coulomb gauge the bare propagator has `t^{0,i} ≡ 0`
  identically — a property of the interaction, independent of the
  vertices — so no correction is missing at the bare level, at any q.
  (This changes once transverse SCREENING is added: the screened wings
  audit's whole point is that `chi_0i` generates nonzero screened CT/TC
  blocks. Out of scope for the static unscreened method.)
* **CC tile.** Already correctly handled by the existing scalar
  `compute_cohsex_sigma`/`StaticHeadTerms` machinery — see "why a
  band-diagonal shortcut works there" above. Untouched by this work.
* **Screened TT/CT/TC heads, magnetic/contact response, Ward-identity
  completion, Hall/Chern terms.** These belong to the fully-screened
  pathway the two binding guides describe (their own "Physics decisions
  required before certification" / "Required refusals" sections) and are
  explicitly NOT addressed by a correction to the BARE exchange V-tile.
  This work changes nothing about chi, W, or any dynamic/COHSEX channel.

## Where the guides leave a choice, and what this note picks

Both guides discuss the fully-screened photon head as a small
field-basis/Schur-complement service (`fold_cartesian_head_wings_sharded`
generalized to E/B fields, §3-4 of the q0 audit). That machinery is
designed for the SCREENED head (it needs `W_body`, a Dyson-coupled
mini-BZ solve, and a 6×6 `R_FF` response tensor) and is deliberately NOT
what this note implements: the static UNSCREENED TT exchange head has no
screening to fold, no body W, and no field-basis coupling to CC — it is a
single (3,3) tensor, computed once per run, injected as a rank-1 update to
a bare V-tile. Reusing the screened-head Schur machinery here would be
over-engineering for the object actually being corrected; the smallest
correct fix, and the one the design doc and the KNOWN_LORRAX_ISSUES row
already name, is the direct V-tile injection.

**Choice: inject at the `v_per_G` builder level, not as a separate
`g0_acc`/`StaticHeadTerms`-style band-diagonal term.** Two reasons:

1. It reuses the EXISTING (μ,ν) V-tile → `sigma_sx_k` convolution
   pipeline verbatim — the corrected tile flows through the same
   `compute_sigma_x_bispinor` code every other TT contribution already
   uses, so the resulting Σ^B correctly comes out as a FULL band matrix
   (diagonal AND off-diagonal), which is required because the current
   structure factor is not diagonal (point 2 above). A band-diagonal
   shortcut would silently drop that structure.
2. It is a **local, single-file, deterministic v-table change** — no new
   Dyson step, no new Sigma contraction, no forked kernel — matching
   rule 5 (microservice: one owner per mechanism) and the task's explicit
   "no forked kernels" instruction. The mini-BZ averaging itself is
   delegated entirely to `services/vcoul`, the existing canonical owner
   of Coulomb-kernel mini-BZ sampling (`q0_average`,
   `v_head_minibz_avg`, `minibz_moment_tensor` already live there for the
   charge/BSE cases); this work adds one sibling function
   (`minibz_transverse_head_avg`) that shares 100% of the draw/estimator
   machinery (`minibz_voronoi_batches`, `_minibz_kernel_bare`, the same
   two BGW estimator branches) rather than a second sampler.

**Choice: default OFF, one deck key, parse-time refusal.** Per rule 13
("unsupported paths refuse loudly") and the task's own instruction, this
changes physics defaults and baselines exactly as the design doc's
"landing it is an owner re-pin decision" line already flagged in 2026-08-01
— so it ships behind `bispinor_tt_head_correction = false` (default),
refusing loudly (`GATE bispinor_tt_head_unsupported`) under `bispinor =
false` or `sys_dim` not in `{2, 3}` (box truncation's q=Γ, G=0 slot is
already finite — `vcoul.box_0d.Box0D._v_bare_per_q` never zeros it — so
there is no missing slot there to fill). Every existing deck's TT tiles
stay byte-identical.

## Implementation summary (Stage 2, landed this session)

* `services/vcoul/src/vcoul/minibz.py`: new `minibz_transverse_head_avg`
  — the tensor sibling of `minibz_average`/`minibz_moment_tensor`, same
  two-branch BGW estimator, weighted by `t_ab(q̂) = δ_ab − q̂_aq̂_b` instead
  of `1` or `q_aq_b`. Bare units (no `1/celvol`), matching the module's
  existing convention.
* `services/vcoul/src/vcoul/{slab_2d,bulk_3d}.py`: new sibling method
  `q0_average_transverse_tensor`, same draw as `q0_average` (same `nmax`
  1↔3 on the same `analytic_sphere` flag) so the TT head samples from the
  identical mini-BZ cell the charge head's `vc0` does.
* `src/gw/v_q_bispinor.py`: new `_tt_head_tensor` (one `(3,3)` call per
  run, not per tile/q) and a `tt_head_correction` gate on
  `_make_per_q_v_builder_for_tile`/`compute_V_q_bispinor_g_flat_to_h5`
  that replaces the `K² ≤ eps_K2` slot (the unique q=Γ, G=0 point) of the
  returned `v·t` table with `T[i,j] / cell_volume`. Nothing else in the
  (μ,ν) contraction, the Σ^B kernel, or any other tile changes.
* `src/gw/gw_config.py`: `head.bispinor_tt_head_correction: bool`
  (default `false`), `refuse_unsupported_bispinor_tt_head_correction`
  (parse-time + driver-entry, mirroring `refuse_unsupported_low_mem_bands`'s
  two call sites).
* `docs/input_reference.md`: new row under `## Screening`, beside
  `head_minibz_average`.

## Verification obtained this session

CPU-only, `lx run -G 0` on a real compute node (`tests/
test_bispinor_tt_head_correction.py`, 14/14 PASS):

1. **Parity** — `tt_head_correction=False` (default): the q=Γ, G=0 slot of
   a TT tile is exactly `0`, and the CC tile is untouched even when the
   flag is set (the `is_CC` branch short-circuits before any `vcoul`
   call).
2. **Wiring** — `tt_head_correction=True`: the q=Γ, G=0 slot equals
   `_tt_head_tensor(...)[i,j] / cell_volume` exactly, and every OTHER
   `(q, G)` slot is bit-identical to the off case.
3. **Red twin — trace identity.** `tr(t_ij(q̂)) = 2` at every point on the
   unit sphere (`δ_ii=3` minus `|q̂|²=1`), so `tr(T) = tr(⟨v·t⟩_mBZ) =
   2⟨v⟩_mBZ = 2·vc0` EXACTLY, for ANY cell shape — a wrong index order, a
   wrong sign in the projector, or an off-by-factor in the volume
   convention would break this. Measured to `rtol=1e-10` on a synthetic
   slab cell (bulk_3d and slab kinds both checked in an ad hoc script
   before the pytest was written; both matched to 13 significant figures).
4. **Red twin — measured slab ratio.** On an in-plane-isotropic synthetic
   slab cell (unrelated geometry to the bi4 deck: a=12 bohr, c=30 bohr,
   4×4×1 grid), `T / vc0` reproduces `diag(1/2, 1/2, 1)` to Monte-Carlo
   tolerance (atol 5e-3, off-diagonal ≈0) — an INDEPENDENT numerical
   confirmation of the exact ratio `docs/BISPINOR_DHFB_DESIGN.md` §11
   measured on a completely different (real MoS2) cell and a different
   code path, which is strong evidence the physics, not a
   geometry-specific coincidence, is what is being reproduced.
5. Refusal envelope: rule id + all five message parts (`got/want/fix/
   why/doc`) for both named refusal conditions; default-off no-op
   confirmed; the `gw_init.py` driver-entry mirror call confirmed present
   by source inspection; the docs row confirmed present.

**Scope of this verification, stated honestly.** All of the above is
host-side numpy/jax algebra with NO SlabIO/HDF5 write path — this
worktree started this session with no built `liblorrax_ffi.so` (see
`KNOWN_SANDBOX_ERRORS.md`'s 2026-08-23 row), which blocked re-running the
EXISTING `tests/test_compute_V_q_bispinor_g_flat.py` end-to-end HDF5
round-trip regression (unmodified by this session) to directly confirm
the default-off path is still byte-identical all the way through the
on-disk tile write. The parity check above (item 1) gives strong indirect
evidence — it is the SAME function `_ref_tile_V` in that test calls, with
the SAME default arguments — but it is not a substitute for actually
running that file.

A follow-up attempt (after the commit below) built a genuine
`liblorrax_ffi_host.so` from THIS exact commit
(`src/ffi/cpp/build_host.sh` under `lx run --cpu`, the bare Milan
partition with working Cray PE modules) and got 4/6 tests in that file to
run; the 2 SlabIO-writing tests still refuse, on an MPICH SONAME mismatch
between the `--cpu`-partition build and the `-G 0` (jax-having) container's
own bundled MPICH — a different, more precisely diagnosed instance of the
same class of environment gap, not resolved. See the KNOWN_SANDBOX_ERRORS
row for the exact libraries and the reusable `.so` path. This also blocks
a real Σ^B GPU leg (Stage 3) until a CUDA FFI is available, which the host
build does not provide regardless.

## Landing order this note follows (subset of the two guides' own §)

1. Land the bare TT mini-BZ correction (this work). ✅ implemented,
   gated off, CPU-verified.
2. (Deferred — screened pathway) exact selected-current matrix elements
   + Ward-residual diagnostic.
3. (Deferred) `ward_subtracted_no_pair` experimental finite-q reference.
4. (Deferred) gauge-covariant static magnetic/contact response,
   `LongWaveResponse` artifact.
5. (Deferred) generalize the distributed charge fold to E/B fields; per-
   sample `(charge, T1, T2)` mini-BZ solve.
6. (Deferred) bordered head/body Sigma contractions;
   `cc_dynamic_noncc_static` / `full_static_cohsex`.

Items 2-6 are the fully-screened pathway's own remaining steps and are
untouched by this session.
