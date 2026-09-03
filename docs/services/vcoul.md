# vcoul — the bare and truncated Coulomb interaction

Measured at branch `svc/vcoul-2026-08-07`, HEAD `603519db`. The service was
extracted at `66f1bd6` on top of the flagged physics fix `358bb0b`; the A/B
probe that certifies the extraction (142 dumps through the old import
paths, byte-compared) lives in the phase scratchpad and its results are
quoted in the extraction and consolidation commit messages.

## Purpose

Everything LORRAX knows about `v(q+G)` lives here: the three truncation
schemes (3D bulk `8π/|q+G|²`, the Ismail-Beigi 2D slab, the Wigner-Seitz
0-D box), the per-q evaluation driver that serves them through one code
path, and the mini-BZ machinery — Voronoi-cell sampling, the BGW-ported
cell average with its Baldereschi-Tosatti analytic-sphere branch, and the
3D body head that the G-flat V_q path injects at the `argmin |q+G|`
slots. The BGW `vcoul` text-table reader rides along as `vcoul.bgw_parity`
because it is this family's external comparison surface, even though that
surface is currently dark (see the owner ledger).

The package is standalone by construction and by test: it imports numpy
and jax, optionally scipy, and nothing of LORRAX. Deck keys, wavefunction
loaders and `Meta` stay on the LORRAX side of the door; the service speaks
`CoulombGeometry`, explicit `kgrid`, and explicit `sys_dim`.

## API

The door is the top-level package; LORRAX imports `vcoul` public names and
never a submodule. The surface is 44 names; the ones that carry the
architecture are these.

`gauss_legendre_interval(order, left, right)` is the standalone host owner
for immutable float64 Gauss--Legendre rules on finite intervals.  The exact
Wigner--Seitz photon cubature and LORRAX's MPA/VNL consumers share this one
arithmetic implementation through the service door; `common.gauss_legendre`
is retained only as a compatibility re-export.

`CoulombGeometry` is a frozen dataclass of the four things every kernel
needs: the Cartesian reciprocal rows with `blat` already folded in, the
cell volume, and (for the 0-D box only) `bdot` and `fft_grid`.
`CoulombGeometry.from_wfn(wfn)` is the only place *in this service* where
`wfn.blat * wfn.bvec` is written, and it is also the only place left in
`src/gw/coulomb/`, where the product used to be hand-multiplied at five
call sites — the kind of repetition that eventually ships a transpose.
Read that as the scoped claim it is: the product still appears elsewhere
in LORRAX, at
`src/gw/gw_init.py:1106` and `src/gw/isdf_fitting.py:693-694` on the live
GW path (both of which should move to `CoulombGeometry.from_wfn(wfn).bvec`
and are registered for it), and in archived material under `misc/`. Nothing
gates the exclusivity, so it is a design intent to uphold rather than a fact
you can rely on when reading unfamiliar code.

`get_kernel(sys_dim)` returns the `Bulk3D`, `Slab2D` or `Box0D` kernel;
`v_qG_table(kernel, q_frac, gvec_components, *, geometry, ...)` is the one
production driver for `v(q+G)` on the writer's per-q G-sphere. Each kernel
contributes exactly one arithmetic method; the driver owns the per-q loop,
the `vcoul_cutoff_ry` mask, the head-slot injection, and their order (head
before cutoff, deliberately, so a head slot outside the bare-Coulomb
cutoff is zeroed like any other G).

`build_v_head_miniBZ_fn_3d(kgrid, bvec, cell_volume, *, nmc, seed)` is the
3D body head. It returns a *function* of the Cartesian `K = q+G` rather
than a per-q table, because the head is injected at every slot attaining
`argmin |q+G|²` and each of those slots has to be valued from its own `K`;
a per-q scalar is attached to `q_frac @ bvec`, which is not the argmin on
12 of Si's 64 q, so the old `(nkx, nky, nkz)` shape could not express the
rule. Its defaults (`nmc=2**18`, `seed=42`) are pins the frozen BSE
reference depends on, which is why it takes raw arrays rather than a
geometry: the signature itself is part of the frozen contract.
`build_miniBZ_dq_cart` is the draw underneath it, and it is on the door
because its centrosymmetry is a correctness requirement rather than an
implementation detail — see the Contract section.

`minibz_frac_to_cart(U, bvec)` and `minibz_cell_affine(bvec, kgrid)` own
the two pieces of mini-BZ geometry that used to exist as copies. The draw
map is the single place the row convention (`U @ bvec`, never
`U @ bvec.T`) is decidable; the 2026-05..08 head bias lived precisely in a
private copy of it. `minibz_voronoi_batches`, `minibz_average` and
`minibz_inscribed_sphere_r2` are the BGW `minibzaverage.f90` port;
`minibz_moment_tensor` (`⟨v q_a q_b⟩`) and `minibz_transverse_head_avg`
(`⟨v (δ_ab − q̂_a q̂_b)⟩`, the bare TT head) are the same draw and the same
two estimator branches with a tensor weight. `minibz_photon_cubature`
dispatches on the kernel type to the exact Wigner–Seitz rule: the
two-dimensional polygon issuer `slab_minibz_photon_cubature` or the
three-dimensional polyhedron issuer `bulk_minibz_photon_cubature`. Both issue
one authenticated `MinibzPhotonReceipt`, and the packed photon head consumes
either through `head_correction.complete_static_photon_q0`. The slab rule also
owns the scalar charge head `Slab2D.q0_average` (see below).
The kernel-facing spelling of the TT estimator is
`Bulk3D`/`Slab2D`.`q0_average_transverse_tensor`, which is what
`gw/v_q_bispinor.py` calls to value the bare TT Γ-head slot; it is still
on the Sobol draw and is **not** covered by that owner (registered).
Physics owner
for all of them: `docs/theory/four-current-head-corrections.md`; how they
are wired into the four-current pipeline:
[`architecture/four_current_wiring.md`](../architecture/four_current_wiring.md);
`wrap_points_to_voronoi` is the jitted Voronoi fold everything shares.

`bare_coulomb_sphere_indices` / `bare_coulomb_sphere_mask` answer "which G
are inside `|q+G|² ≤ cutoff`" and nothing else: the sentinel padding that
the ζ file format wants is the caller's business, so the old
`common.coulomb_sphere` wrapper applies it on the LORRAX side.

`read_bgw_vcoul` / `fill_v_grid_for_q` / `BGWVcoulTable` parse and scatter
BGW `vcoul` dumps. They moved verbatim, including the known missing
shifted-q₀ fallback in `find_q_index` — repairing that surface is an owner
decision, not something an extraction may do in passing.

## Contract

The volume convention is BerkeleyGW's: every `v` comes out multiplied by
`1/Ω_cell`, so the downstream `ζ v ζ†` contraction is in Rydberg with no
further factor. The one exception is `minibz_average`, which returns bare
kernel units because its two callers apply different volume conventions;
its docstring says so and the body-head table (which divides) is the
example of why.

`q+G = 0` is always zeroed by the kernels — the q→0 head is somebody
else's rank-1 term — except in the 0-D box, where the truncated `v(G=0)`
is finite and *is* the head. `Box0D` refuses `v(q+G)` at `q ≠ 0` rather
than serving the q=0 answer, because a box deck is Γ-only and a finite-q
request means the k-grid is wrong, not the kernel.

### One q→0 cell-average rule per dimension, named

`Slab2D.q0_average` takes a `rule` selection with no silent alternative:

| `rule` | what it is | default |
|---|---|---|
| `Q0_RULE_EXACT` = `"wigner_seitz_polygon"` | the exact mini-lattice Wigner–Seitz polygon, Γ-to-edge Duffy triangulation, fixed 16/24/32 Gauss–Legendre ladder — `slab_minibz_photon_cubature`'s own receipt | **yes** |
| `Q0_RULE_SOBOL_DEBUG` = `"sobol_debug"` | the superseded scrambled-Sobol Voronoi draw, kept verbatim so its numbers stay reproducible | no; DEBUG only, announces itself |

The scalar charge head and the packed bispinor Γ completion therefore
reduce the **same** nodes, weights and kernel values: measured on the
MoS2 3×3 deck geometry, `⟨v⟩` from `Slab2D.q0_average` and `⟨v⟩` from
`complete_static_photon_q0`'s own `static_photon_head_moment_chunk`
reduction agree to `2.274e-13`, and the screened companion
`⟨v/(1 − v qᵀSq)⟩` to `0.000e+00`
(`runs/MoS2/08_bisp_k_head_quadrature_20260901/head_owner_check.json`).
Before this, the two disagreed by 0.16 % of `⟨v⟩` — the Sobol estimator's
deterministic-per-seed sampling error on the `|q|` cusp — which was
+5.72 meV on every occupied state's SX+COH
(`reports/bisp_j_architecture_review_2026-09-01/report.md` §4,
claim 0586) and about half the scalar route's disagreement with
BerkeleyGW (claim 582).

For `sys_dim=3`, `bulk_minibz_photon_cubature` constructs the true Voronoi
cell of the mini-lattice rows `b_i/N_i`. A centered fundamental
parallelepiped bounds the WS covering radius by
`R = 1/2 sum_i |b_i/N_i|`; lattice half-spaces longer than `2R` are therefore
redundant. The smallest singular value converts that length bound to one
finite integer-neighbour shell, which is clipped once—there is no open-ended
search. Each face is triangulated about its center and joined to Gamma. The
tetrahedral Duffy map has Jacobian `r^2 u |det(v0,v1,v2)|`, so its radial
`r^2` cancels the bulk `8 pi/q^2` kernel exactly before summation. The fixed
10/16/22 Gauss–Legendre ladder uses interior nodes only, normalized weights,
and authenticates the polytope, cell volume, weight-sum/centroid identities,
physical counts, padded counts, and payload digest in the same receipt family
as the slab rule.

Consequences a caller sees:

* `nsamples`, `method` and `qmc_reps` are the `sobol_debug` rule's dials
  and select nothing under the production rule, which is exact and has no
  order knob. The deck-facing wrappers (`gw.vcoul.compute_q0_averages`,
  `gw.head_densify.gamma_cell_head_scalar`) still thread their historical
  values through; on a slab they are inert, and both docstrings say so.
* `analytic_sphere` — the deck key `head_minibz_average` — is **refused**
  on the slab rather than ignored (`GATE
  slab_q0_analytic_sphere_unavailable`): it only ever widened the Sobol
  draw's Voronoi fold, and the exact rule integrates the Wigner–Seitz
  cell itself. There is no Baldereschi–Tosatti sphere in 2D; the slab
  head is a `|q|` cusp, not a `1/q²` pole.
* The production rule prints its ladder certificate once per process —
  polygon edges, orders, node counts, `⟨v⟩`, and the 24→32 mixed
  absolute+relative error ratio — and REFUSES above 1
  (`GATE slab_q0_polygon_not_converged`, budget `atol 1e-12`,
  `rtol 1e-8`, the same one the packed completion applies to its own
  ladder). The scalar head had no convergence bound before.
  It also emits the typed `SlabQ0Certificate` through an optional
  `certificate_fn`. LORRAX's `HeadResolver` sends that receipt to the
  top-level production reporter, so `gwjax.out` contains `Slab WS cert`
  with the actual order ladder, physical node counts, polygon edge count,
  evaluation count and worst final error ratio; the run record no longer
  depends on component stdout escaping the production sink.

**`Bulk3D` is unchanged and takes no `rule`.** The polygon construction is
two-dimensional; there is no 3D owner for it, so the bulk head keeps the
scrambled-Sobol draw plus the Baldereschi–Tosatti analytic sphere, with
every dial live. The paragraph below is therefore a statement about the
3D head and about `sobol_debug`.

Explicit requests refuse; only `auto` demotes, and it announces once.
`method="sobol"` without scipy is a `RuntimeError` naming the fix;
`method="auto"` falls back to the uniform draw with a single
`warnings.warn` that also names the silent `nsamples` bump the old code
performed without telling anyone. With scipy present — every production
machine — `"sobol"` is bit-identical to the pre-extraction code.

The head slot is `argmin |q+G|²`, all of it, and every tied slot gets the
tied set's mean. This was settled on 2026-08-08 and it is the one place
in this service where a physics choice is made rather than preserved.

The rule that shipped before then picked the slot whose Miller index was
literally `(0,0,0)`. That label is not equivariant under `q → −q`: the BGW
wrap sends a boundary component to `+1/2` and never to `−1/2`, so at 12 of
Si's 64 q the Miller-(0,0,0) slot at `+q` pairs with a slot at `−q` that
carries the bare value instead of the head. Measured on the Si fixture's
own G-lists, `max |v(+q, i) − v(−q, pair(i))|` was 1.293e-2 under the
label rule and is exactly 0.0 under this one; in production `V_qmunu` the
per-q reciprocity went from 63 of 64 q failing at a median 4.286e-3 to 0
of 64, worst 1.356e-15.

One slot per q is provably impossible on an even k-grid, which is why the
rule injects at all of the argmin rather than tie-breaking. The argmin is
degenerate on 13 of Si's 64 q (multiplicities 1:51, 2:7, 4:6, identical
across tie tolerances from 1e-14 to 1e-6 — these are exact symmetry
degeneracies, and the smallest relative gap among the non-degenerate q is
0.73), and 7 of those q are self-paired (`−q ≡ q` as a grid index) with
the pairing *swapping* their tied slots. At such a q, injecting at one
tied slot and not its partner makes `V_q ≠ conj(V_q)` by construction, no
matter which one you pick. Every even grid tested (fcc, sc, bcc,
hexagonal, triclinic) has exactly those 7; every odd grid has none.

The tied slots share their mean rather than keeping their own values
because two symmetries have to hold at once. `q → −q` is satisfied by
per-slot values already; `K → RK` for R in the little group is not, since
the mini-BZ is a parallelepiped whose symmetry is lower than the crystal's
and `⟨v⟩` therefore distinguishes slots that the point group exchanges.
The IBZ-plus-unfold arm needs both: measured there, the label rule left 9
q bad (worst 7.459e-3), per-slot argmin values left 6 (worst 7.709e-5, and
they newly broke 4 q that had been fine), and the tied-set mean left none
(worst 2.647e-7, which is the arithmetic floor).

`head_tie_rtol` (default 1e-9) is the relative window on `|q+G|²` that
defines "attaining the argmin". It sits about five decades above float64
noise on the dot product and six below the smallest real gap, and the tie
count is flat across that whole range, so its value is a non-decision.

The caller owes one thing in return: `v_head_fn` must satisfy
`f(K) = f(−K)`. For a Monte-Carlo mini-BZ average that means a
centrosymmetric δq sample set, which `build_miniBZ_dq_cart` guarantees by
unioning its draw with its own negation — an ordinary one-sided draw is
even only in the `nmc → ∞` limit and leaves an MC-sized residual in
reciprocity even once the slot selection is right. A caller that still
holds the retired `(nkx, nky, nkz)` head table gets a `TypeError` naming
the replacement, never a silent fallback to the label rule.

## Backends

There are none to speak of, and that is a statement, not an omission: the
service is host-side numpy with one jitted jax helper
(`wrap_points_to_voronoi`, CPU or GPU alike) and no `.so`, no mesh, no
process count. The only capability axis is scipy for the Sobol generator,
handled by the announce-or-refuse gate above. This is why the service's
conftest does not force `XLA_FLAGS` device counts the way `distrib_la`'s
does — there is nothing here for emulated devices to cover.

## Tests

`services/vcoul/tests` holds the service's own suite (markers `services`
and `vcoul`, staged into the main run by the shared bootstrap, deselectable
with `--no-services`). Import isolation runs the whole public surface in a
`python -S` subprocess with only the service on the path — including the
no-scipy refusal arm — with a red twin, so "standalone" is a measured
property. The door smoke tests exercise every kernel, both refusal
classes, the head-slot rule (argmin selection at a q where it disagrees
with the label, the tied-set mean, the Γ skip, and the refusal of a per-q
head table), and the head-before-cutoff order on synthetic arrays, cubic
and hexagonal both. The consolidation pins hold golden body-head values
plus a composition test that rebuilds the head from the exported pieces
and a red twin that shows the composition pin can fail, and they pin the
δq set's closure under negation with a red twin on the one-sided draw.

`test_vcoul_head_slot_reciprocity.py` is the gate for the head-slot rule
itself, and it is service-local because everything it needs is service
arithmetic: it rebuilds the Si per-q G-lists from
`bare_coulomb_sphere_mask` (the same predicate the ζ writer uses), shows
the +q/−q slot pairing is exactly `K ↦ −K`, reproduces the multiplicity
histogram and the seven self-paired-with-swap q, and then measures
`max |v(+q, i) − v(−q, pair(i))| == 0.0`. Two red twins keep it honest:
the Miller-(0,0,0) label rule reinstated locally, which must fail by
~1e-2, and a one-sided δq draw with the right slots, which must also
fail. There is a third, quieter arm — the bare kernel with no head at all
is already reciprocal — so nothing the twins measure can be attributed to
the Coulomb formula rather than the head.

The physics guard for the 358bb0b draw fix is
`tests/test_vcoul_minibz_head_draw.py` in the main suite: the
Si-immunity algebra, two-sided mirror discrimination on a hexagonal cell,
a paired seed-band bias statistic (z_hex = 80, z_si = 2.4 at test scale),
and a rejection-sampled ground truth. Silicon cannot see that bug class;
that file can, and it is the only thing in the tree that can.

The Perlmutter evidence chain for the fix — Si COHSEX byte-identity, BGW
Σ stats identical to full float precision, the expected BSE frozen-arm
red at 7.07e-5 eV with the BGW band undisturbed — is recorded in
`tests/KNOWN_FAILURES.md` and the `si_bse_debug` fixture README, with
artifacts under `/pscratch/sd/j/jackm/svc_vcoul/_gates_{before,after}/`.

## Performance

The service is a one-shot setup cost, not a hot loop: `v_qG_table` runs
once per consumer setup and the head draw once per run. Baselines
(claims-style, machine/op/shape/seconds) live in
`services/vcoul/bench/baselines/` and were recorded on the WSL dev box —
the numbers that matter operationally are the head at production
`nmc=2**18` (seconds-scale) and `v_qG_table` at Si-fixture shapes
(milliseconds-scale). The 2026-08-08 rule doubled the δq sample count
(centrosymmetrisation) and moved the head from 64 table entries to the
~76 argmin slots the 64 q select, so the head's cost roughly doubled from
a seconds-scale base; it is still a one-shot setup cost and still off the
GPU steady state.

## Antipatterns

Do not hand-roll a fractional-to-Cartesian draw. `randvals @ bvec.T`
passed every volume and normalisation check for three months while biasing
the hexagonal head by half the size of the correction it computed, because
the transposed parallelepiped has the right volume and the wrong shape.
If you need points in a reciprocal cell, call `minibz_frac_to_cart`; if
you need mini-BZ offsets, take `minibz_voronoi_batches` whole.

Do not multiply `wfn.blat * wfn.bvec` at a call site. That product is
`CoulombGeometry.from_wfn`'s job, and the door will not take the pair.

Do not test a Coulomb formula against a reference built by calling the
formula. The pre-extraction suite pinned `v(q+G)` only self-referentially,
which is how a constant `42.0` slab kernel would have passed; value tests
against the metric-form closed expression are the pattern to copy.

Do not add a cubic-only test for anything in the mini-BZ family. Cubic
cells satisfy `bvec.T = P·bvec`, which makes them structurally blind to
the entire draw-convention bug class; every new test gets a hexagonal or
lower-symmetry row or it is not evidence.

Do not evaluate a q→0 cell average with a Monte-Carlo draw when an exact
rule for that cell exists in this service. The slab head's Sobol estimator
was correct, converged-looking and wrong by 0.16 %: the error is
deterministic per seed, so it does not present as noise, and two
production-style draws straddled the exact value by +0.16 % and −0.02 %.
A second draw is not a convergence study. If you add a q→0 average, say
which rule it uses and print a certificate that can refuse.

Do not loosen `ATOL_FROZEN_EV` or any bit-reproducibility pin to absorb a
physics change. The pin's value is that it cannot absorb one; refreeze
through the owner instead, with before/after measured, as `358bb0b` did.
