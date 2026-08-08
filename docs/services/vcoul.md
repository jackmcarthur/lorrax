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
3D body-head table that the G-flat V_q path injects at the Miller-(0,0,0)
slot. The BGW `vcoul` text-table reader rides along as `vcoul.bgw_parity`
because it is this family's external comparison surface, even though that
surface is currently dark (see the owner ledger).

The package is standalone by construction and by test: it imports numpy
and jax, optionally scipy, and nothing of LORRAX. Deck keys, wavefunction
loaders and `Meta` stay on the LORRAX side of the door; the service speaks
`CoulombGeometry`, explicit `kgrid`, and explicit `sys_dim`.

## API

The door is the top-level package; LORRAX imports `vcoul` public names and
never a submodule. The surface is 29 names; the ones that carry the
architecture are these.

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

`build_v_head_miniBZ_avg_3d(kgrid, bvec, cell_volume, *, nmc, seed)` is
the 3D body-head table. Its defaults (`nmc=2**18`, `seed=42`) are pins the
frozen BSE reference depends on, which is why it takes raw arrays rather
than a geometry: the signature itself is part of the frozen contract.

`minibz_frac_to_cart(U, bvec)` and `minibz_cell_affine(bvec, kgrid)` own
the two pieces of mini-BZ geometry that used to exist as copies. The draw
map is the single place the row convention (`U @ bvec`, never
`U @ bvec.T`) is decidable; the 2026-05..08 head bias lived precisely in a
private copy of it. `minibz_voronoi_batches`, `minibz_average` and
`minibz_inscribed_sphere_r2` are the BGW `minibzaverage.f90` port;
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

Explicit requests refuse; only `auto` demotes, and it announces once.
`method="sobol"` without scipy is a `RuntimeError` naming the fix;
`method="auto"` falls back to the uniform draw with a single
`warnings.warn` that also names the silent `nsamples` bump the old code
performed without telling anyone. With scipy present — every production
machine — `"sobol"` is bit-identical to the pre-extraction code.

The head slot is Miller-(0,0,0), not `argmin |q+G|²`. The two disagree on
12 of 64 Si q-points, BGW's own criterion is the argmin-like `ekinx <
avgcut`, and the choice moves numbers — so the shipped rule is preserved
exactly and the question is on the owner ledger, pinned by a test either
way.

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
classes, the head-slot rule, and the head-before-cutoff order on synthetic
arrays, cubic and hexagonal both. The consolidation pins hold golden
body-head values (bit-equality, captured pre-consolidation) plus a
composition test that rebuilds the head from the exported pieces and a red
twin that shows the composition pin can fail.

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
once per consumer setup and the head table once per run. Baselines
(claims-style, machine/op/shape/seconds) live in
`services/vcoul/bench/baselines/` and were recorded on the WSL dev box —
the numbers that matter operationally are the head table at production
`nmc=2**18` (seconds-scale) and `v_qG_table` at Si-fixture shapes
(milliseconds-scale). Nothing here touches the GPU steady state.

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

Do not loosen `ATOL_FROZEN_EV` or any bit-reproducibility pin to absorb a
physics change. The pin's value is that it cannot absorb one; refreeze
through the owner instead, with before/after measured, as `358bb0b` did.
