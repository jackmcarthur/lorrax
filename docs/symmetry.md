# Symmetry: the IBZ, stars, and the two conventions that bite

A crystal's Brillouin zone is redundant. If the lattice has 48 point-group
operations, then a 4×4×4 k-grid's 64 points collapse to 8 inequivalent ones, and
computing anything on all 64 does eight times the necessary work. LORRAX
therefore computes on the **irreducible** wedge and reconstructs the rest by
symmetry. That reconstruction — *unfolding* — is what the `symmetry_maps` service
does, and it is worth understanding because its failure mode is unusually cruel:
a wrong unfold changes numbers by tens or hundreds of eV while leaving every
quantity a normal test would check completely unchanged.

This page is the story. The conventions and their derivations — the BerkeleyGW
`mtrx` and τ algebra, the ψ-unfold formula, the (α, L) decomposition — live in
[Symmetry conventions](theory/symmetry.md), and the service's contract lives in
[`docs/services/symmetry_maps.md`](services/symmetry_maps.md).

## What the maps actually are

`SymMaps(wfn)` reads a `WFN.h5` header and builds the tables everything else
uses. Two integer arrays carry the reduction: for every full-BZ k-point,
`irr_idx_k` says which irreducible point it descends from, and `sym_idx_k` says
which symmetry operation takes you there. A *star* is the set of full-BZ points
sharing an irreducible parent. Alongside those come the Cartesian rotation
tables, the spinor rotations, the fractional translations, and the umklapp
vectors that appear when a rotated G-vector falls outside the original sphere.

The operation table is **twice** as long as the crystal's spatial group. The
second half is the time-reversal-augmented half: Θ is antiunitary, so
`O(−k) = conj(O(k))`, and time reversal buys you another factor of up to two in
zone reduction at the cost of a complex conjugation appearing in the middle of
your reconstruction. Whether it is legal to use is not read off a flag — LORRAX
**measures** it (below).

`KStarMap` bundles the three arrays that always travel together — `irr_idx`,
`sym_idx`, and the spatial-operation count — so that no call site can supply two
of the three. It offers `select` (keep one row per star), `broadcast` (IBZ → full
BZ), and `spread` (a diagnostic: how far apart are values that ought to be
equal). `KStarMap.identity(nk)` is the no-reduction map, which lets a driver read
identically whether or not symmetry is in use.

## The single conjugation rule, and the 183.61 eV

Here is the sharp edge. When you reconstruct a full-BZ row from a stored one, do
you conjugate?

The naive answer is "conjugate if this row came from a time-reversal operation",
i.e. `sym_idx >= n_sym_spatial`. That answer is correct for exactly one flavour
of operand and silently wrong for the other, and the difference is a XOR.

The reason is that the thing you are reconstructing *from* is itself a row with a
symmetry label. `star_select` keeps the **first full-BZ member** of each star, in
full-BZ order — and that member may itself be a time-reversed image. Two rows
that are both time-reversed images of the same irreducible point are related to
each other *without* a conjugation; a spatial row related to a time-reversed
reference needs one. So the predicate is
`trs(member) XOR trs(reference_row)`, not `trs(member)`.

The two rules coincide only while every star's first full-BZ row happens to be
spatial — which is a property of the operation-selection policy, not of the
physics. On the gnppm deck, four of five stars have a time-reversal first row and
the two predicates disagree on eight of nine rows.

Getting it wrong cost **183.61 eV** in off-diagonal Σ. And here is why nothing
caught it: the norm, the hermiticity, the trace, the electron count and every
printed eqp column were *exactly* unchanged. Conjugating an entire star member
moves the diagonal min/max spread metric that the anchor test used by precisely
0.0 — measured live, 1.2130460739135742 both ways. Nothing in the suite looked
off the diagonal, so nothing in the suite could see it.

The structural fix is that the predicate now exists once, as
`_star_conj_flags`, read by four entry points inside the package and imported by
nothing outside it. The API-level fix is that the parameter naming the operand
flavour, `trs_reference`, has exactly two legal values and an unknown spelling
**raises** with both named rather than falling through:

* `"star_row"` (the default) — `A_irr`'s rows are values at the kept full-BZ
  rows, i.e. what `star_select` returns. The predicate is the XOR.
* `"ibz_slab"` — `A_irr` is the raw IBZ slab, read with no operation applied, so
  every row is TRS-false by construction and the predicate reduces to the
  member's own flag.

Because there *is* a default, a caller on the other flavour is one omitted
keyword away from the 183.61 eV. That is why the one such caller,
`gw/kin_ion_io.py`, passes the literal in the source, and why
`tests/test_kin_ion_star_broadcast.py` asserts *by AST* that it is a string
constant — not a variable, not a conditional.

## `R_cart` is the inverse rotation

The second trap is a transpose. `R_cart` is the **inverse** Cartesian rotation,
because BerkeleyGW's `mtrx` is the inverse real-space rotation while `mtrx.T` is
what acts on k and G. If you rotate a Cartesian index with `sym.R_cart[s]`
untransposed, you get the wrong answer — and, once again, norms, hermiticity and
traces all survive the mistake.

The rule is simply stated: **anything that rotates a rank ≥ 1 Cartesian index
uses `R_cart_forward`.** Two consumers are exempt and each says so in place:
`get_spinor_rotations`, whose transposed Shepperd form cancels the inversion, and
`unfold_v_q_bispinor_lorentz`, whose §A5 contraction compensates internally — and
whose docstring, rather than a comment eight hundred lines away, now says so.

## Folding and unfolding

Four operations do the reconstruction, at four different levels of the pipeline.

**`unfold_psi`** rebuilds a full-BZ wavefunction from its irreducible parent: the
spinor rotation, the τ phase, the G-list negation on the time-reversal half, and
the conjugation. It hard-raises unless the operation table is exactly twice the
length of the spinor-rotation table, because that mismatch is what a
half-augmented table looks like.

**`compute_centroid_sym_perm`** is the real-space half: it works out, for each
symmetry operation, where every ISDF centroid goes, and records the integer
lattice wrap in an `L_table`. It **refuses** a centroid set that is not
orbit-closed and names the regeneration fix — and you should not make that
refusal pass by regenerating a set, because the production sets' non-closure is a
measured, owner-scoped fact and regenerating means re-freezing the BerkeleyGW
anchor. Tests that need a non-closed set build one synthetically, by dropping a
centroid from a closed one.

**`unfold_v_q`** takes the sharded (q, μ, ν) Coulomb operator from the
irreducible q-set to the full grid: a double gather over centroids, the umklapp
L-phase, and the time-reversal conjugation, written as `shard_map` with paired
`all_to_all`s so that per-rank peak memory stays at one tile. Its four shape
refusals are load-bearing rather than tidy — the gathers it uses clip *silently*
on an out-of-bounds index, so an unrefused wrong shape is a wrong answer instead
of an error.

**The star helpers** (`star_select` / `star_broadcast` / `star_spread`) do the
same job for band-indexed quantities. `star_spread` is the diagnostic that can
see a gauge or conjugation mismatch that hermiticity, norms and electron counts
all survive.

## The one measurement in the stack

Everything above is bookkeeping derived from the header. `check_density_symmetries`
is different: it is a **measurement**. It takes the wavefunction coefficients,
builds two real-space densities, and compares them. It uses no `unfold_psi`, no
`SymMaps`, no spinor rotations and no τ — so a bug in the τ phase, in `U_spinor`,
or in the umklapp vector cannot move its verdict. That non-circularity is exactly
what makes it evidence rather than a restatement of an assumption, and it is why
`trs_holds` is a measured property of the file rather than an inference from
`noinv=.true.`.

It is tunable by environment — `LORRAX_TRS_CHECK` (`1` | `0` | `strict`),
`LORRAX_TRS_TOL`, `LORRAX_TRS_SPATIAL_TOL`, `LORRAX_TRS_MAX_K` — and note the
direction of that: the environment grants and tunes the *measurement*. It never
selects a symmetry convention.

## Traps

* **Do not test symmetry on silicon only.** Si has zero time-reversal rows at all
  64 k-points; every antiunitary branch is dead there. A suite that is green on
  Si alone proves nothing whatever about time reversal.
* **Do not derive star test tables from a generated grid.** Lex-minimal orbit
  representatives are always spatial, so a derived grid has no TRS first row,
  every discriminating test silently becomes a tautology, and the suite goes
  green while testing nothing. The in-tree tables are hand-typed and
  production-confirmed, and a meta-test pins the trap by asserting that a
  lex-min-derived grid has zero TRS-first stars.
* **Do not use a diagonal min/max spread as a symmetry gate.** It is the right
  check for anchor agreement and the wrong one for conjugation errors, which move
  it by exactly zero.
* **`nspinor = 2` means noncollinear, not spin-orbit.** The code branches on the
  spinor axis, never on SOC.
* **Do not re-derive the conjugation predicate.** It exists once. If you find
  yourself writing `sidx >= n_sym_spatial`, you are one operand flavour away from
  the 183.61 eV.
