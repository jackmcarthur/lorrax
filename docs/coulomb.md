# The Coulomb interaction: why v(q+G) is never just 8π/|q+G|²

Everything downstream of the wavefunctions — screening, W, Σ, the BSE kernel —
contracts against the bare Coulomb interaction `v(q+G)`. In an infinite crystal
that is the textbook `8π/|q+G|²` in Rydberg, with the code's `1/Ω_cell`
convention folded in. On a finite k-grid it cannot be used as written, for two
distinct reasons, and the `vcoul` service exists so that each consumer does not
invent its own answer to either.

## Why the head needs averaging at all

A k-grid turns the Brillouin-zone integral of `v` into a sum over a finite set of
q-points, each standing in for a small cell of the zone — the "mini-BZ", the
Voronoi cell of the q-grid. For most (q, G) that replacement is harmless: `v` is
smooth across a mini-BZ, and the value at the centre represents the cell well
enough. It fails exactly where `v` is large and curved, which is at and near
`q + G = 0` — the head. At q = 0 the point value is infinite while the cell
average is finite. At small nonzero q the point value can misrepresent the cell
average by several percent, and because `v` is convex there, always in the same
direction, so the error accumulates rather than cancelling.

The cure is to replace the point value with the average of `v` over the mini-BZ.
That is what BerkeleyGW's `minibzaverage.f90` does and what LORRAX ports: at
strictly q = 0, an analytic Baldereschi-Tosatti sphere term handles the `1/q²`
singularity and Monte Carlo handles the smooth remainder; at finite q, the
average is pure (quasi-)Monte Carlo over uniform points in the mini-BZ.

Getting "uniform points in the mini-BZ" right is subtler than it looks. The
fractional draw maps to Cartesian through the **rows** of `bvec` (`U @ bvec`),
because that parallelepiped is a fundamental domain of the reciprocal lattice and
the Voronoi wrap is then measure-preserving. The transposed spelling has the
right volume and the wrong shape; it passes every normalisation check you would
think to write, and it shipped for three months as a bias worth about half the
size of the correction it was computing on hexagonal cells. Silicon could not see
it — cubic symmetry makes the two spellings statistically identical — which is
why it survived a green suite for a quarter. That map now exists in exactly one
function, `vcoul.minibz_frac_to_cart`, and the story is pinned by
`tests/test_vcoul_minibz_head_draw.py`.

How much does all this matter? Two measured numbers are worth carrying. Turning
the body-head average off against BerkeleyGW's matching convention moves Σ_X on
the Si 4×4×4 anchor by **136.2 meV MAE** over 128 (k, band) pairs; at the level
of the external gate, the same knob is worth **141.65 meV MAE**, about 95× that
gate's 1.5 meV budget. Head handling is not bookkeeping. It is meV-scale physics
in every number the code prints.

## What the truncation schemes are for

`sys_dim` decides how much of space the Coulomb interaction is allowed to see.

In a genuinely periodic 3D crystal (`sys_dim = 3`, `Bulk3D`) nothing is
truncated. In a slab calculation (`sys_dim = 2`, `Slab2D`) the supercell repeats
a two-dimensional system with vacuum in between, and untruncated `v` would let
neighbouring images interact; the Ismail-Beigi factor
`1 − e^{−z_c|k_∥|} cos(k_z z_c)` cuts the interaction at half the cell height so
that each slab sees only itself. For molecules (`sys_dim = 0`, `Box0D`) the
interaction is confined to the Wigner-Seitz box by a real-space FFT construction;
there the q = 0, G = 0 element is finite and *is* the head, so none of the
mini-BZ machinery is needed — and the kernel refuses `v(q+G)` at q ≠ 0 outright,
because a box calculation is Γ-only by construction and a finite-q request means
the k-grid is wrong rather than the kernel.

There is no wire truncation (`sys_dim = 1`) in LORRAX today. That absence is a
registered owner item, not an oversight.

## Why the head-slot convention matters

The mini-BZ-averaged head has to be injected into the `v(q+G)` table at one G per
q — but which G? LORRAX injects at Miller index (0,0,0). BerkeleyGW's criterion
(`ekinx < avgcut`) effectively selects the smallest `|q+G|`, and for 12 of the 64
Si q-points those two disagree: a q near the zone boundary can have its smallest
`|q+G|` at an umklapp G. The two conventions move numbers. The shipped
Miller-(0,0,0) rule is therefore preserved exactly and pinned by tests, and the
choice is deliberately parked as an owner question — the honest state is
"unsettled and pinned", not "settled by whoever refactored last".

Related and equally deliberate: the head is injected **before** the bare-Coulomb
cutoff mask, so a head slot that falls outside the cutoff is zeroed like any
other G.

## What the deck keys do

`sys_dim` selects the truncation, as above.

`mc_average_vcoul_body` (default true) mini-BZ-averages the G = 0 slot of every
q ≠ 0 — the "body head". Its BerkeleyGW correspondence is exact and worth
memorising: LORRAX `false` matches BGW `cell_average_cutoff 1d-12`, which
averages only the literal q+G = 0 element and is what the Si anchor decks pin;
LORRAX `true` matches BGW's 3D semiconductor default in spirit, though BGW's
default averages every (q, G) rather than only G = 0.

`head_minibz_average` upgrades the q → 0 head from the historical pure-Sobol mean
to the Baldereschi analytic-sphere form — but note that on every deck which pins
`vhead` or `whead_0freq` (that is, all current gate decks) the override wins, and
this key is parsed, stored, and never read.

`bare_coulomb_cutoff` zeroes `v` beyond `|q+G|²` in Ry. `vhead`, `whead_0freq`
and `whead_imfreq` inject externally computed (typically BerkeleyGW) q → 0
scalars, bypassing the native ladder; `wcoul0_source` and `wcoul0_eta` steer that
ladder when it does run. `use_bgw_vcoul` with `bgw_vcoul_file` overlays BGW's
full `v(q, G)` table for bit-level comparison — currently a dark surface, with no
deck and no test exercising it and a known unrepaired gap in the reader, all
registered. `zeta_cutoff` bounds the ζ G-sphere.

The key-by-key reference stays in [Input reference](input_reference.md); the
service API and its contracts are in [`docs/services/vcoul.md`](services/vcoul.md).

## Where the code is

The physics lives in the standalone `vcoul` service
(`services/vcoul/src/vcoul/`): the geometry object, the three kernels, the single
`v_qG_table` driver that serves all of them through one code path, the mini-BZ
machinery, and the BerkeleyGW parity reader.

`CoulombGeometry.from_wfn(wfn)` is the only place in the tree where
`wfn.blat * wfn.bvec` is written. Before the extraction that product was
hand-multiplied at five call sites, which is the kind of repetition that
eventually ships a transpose.

On the LORRAX side, `gw.compute_vcoul`, `gw.vcoul`, `gw.coulomb.*` and
`common.coulomb_sphere` remain — not as forwarding shims, but as the deliberate
translation layer that turns deck- and wavefunction-facing signatures into door
calls, plus a CLI. The door itself speaks `CoulombGeometry`, an explicit `kgrid`
and an explicit `sys_dim`; it has never heard of a deck key or a `Meta` object,
and that is what keeps it standalone.

If you are about to hand-roll a fractional-to-Cartesian draw, a `blat*bvec`
product, or a Coulomb test whose reference is built by calling the formula under
test, the Antipatterns section of [`docs/services/vcoul.md`](services/vcoul.md)
names each of those graves. One is worth repeating here: **do not add a
cubic-only test for anything in the mini-BZ family.** Cubic cells satisfy
`bvec.T = P·bvec`, which makes them structurally blind to the entire
draw-convention bug class. Every new test in that family gets a hexagonal or
lower-symmetry row, or it is not evidence.
