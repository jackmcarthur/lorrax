# The symmetry register

**Read this before reading symmetry code.** It is the map: every distinct
symmetry operation LORRAX performs, which backend implements it, and where it is
called from. It is organised by OPERATION, not by file, because the question you
almost always have is "where does this kind of rotation live", not "what is in
this module".

The rule the tree is held to: **all symmetry goes through
`services/symmetry_maps`.** No bespoke star reconstruction, no coordinate
`argmin`, no energy fingerprints, no tolerance-based k-matching. Sites that still
violate it are listed in `tests/KNOWN_FAILURES.md` under "Bespoke IBZ→full-BZ
unfolding outside the symmetry service", with the reason each has not been fixed.

Surveyed 2026-08-15. Line numbers are a snapshot; the operation names are not.

---

## 0. The two conventions that cause the most damage

**k and G both transform with `sym_mats_k = mtrx.T`** (column form:
`G' = sym_mats_k @ G`). Real space uses `Rinv = inv(mtrx)`: `r' = Rinv @ r + τ`.
`mtrx` is BGW's raw stored matrix. A comment in `maps.py` asserted the opposite
until 2026-08-15, and the known-broken `tests/bench/charge_density.py:159-174`
adopted that wrong convention — it rotates G with `R_grid` (= `mtrx`).

**`SymMaps.translations` is raw BGW `tnp` = 2π·τ.** G-space consumes it
**undivided** (`tau_phase_row`). Real space **divides by 2π** (every
`orbit_syms` entry point). The same array means different things to the two
sides, and nothing checks it — passing one's argument to the other is a 2π error
no shape or dtype catches.

---

## 1. Spinor rotation — SU(2) from a 3×3

| | |
|---|---|
| **Backend** | `symmetry_maps/maps.py` `SymMaps.get_spinor_rotations` (Markley/Shepperd quaternion → Pauli) |
| **Input convention** | `syms_crystal_to_cartesian` — single source; improper ops flipped by `det<0 ⇒ R=-R` |
| **Called from** | exactly one place: `SymMaps.__init__` → `self.U_spinor` |
| **Duplicates** | **none in production.** Test fixtures build small U tables by hand; `misc/archived_tests/get_interp_vectors.py` uses the pre-2026-05 convention and is wrong (no `iσ_y·conj`, no τ phase, `Rinv_grid` for G) |

**Applying U to a symmetry row (the TRS augmentation)** is separate and also
single-backend: `maps.py` `spinor_rotation_for_sym_row` (alias
`trs_augment_U`), implementing `U_trs = iσ_y · conj(U)` for rows with
`sym_idx >= n_tran`. Two live callers: `unfold_psi` (host) and
`wfn_loader/loader.py` (device). Both call; neither restates.

**Four-component raw-parent action.** `SymMaps.spinor_action(rows,
nspinor=4)` returns `diag(U_2, det(R_cart) U_2)`, with `iσ_y K` on each
block for antiunitary rows. `U_spinor` stays 2×2; the named helper receives
`R_cart` explicitly because SU(2) alone has discarded spatial parity.
`SymMaps.lorentz_action` supplies `diag(1, polar time-odd R)` for operator
indices, and `mix_lorentz_blocks` applies its tensor product separately to
rectangular CC/CT/TC/TT sectors. Missing source blocks are zero. Complex
conjugation belongs to the scalar operator unfold, never the real mixer.
The old TT mixer and its private cache/export are deleted. Explicit numerical
fixtures retain the former TT rule as an independent oracle.
`CentroidKUnfoldPlan.unfold_face` owns the GW endpoint seam; static photon
and bare TT Sigma apply their vertices after that typed transport and sum
Lorentz sectors on parents before `unfold_file_wedge_band_operator`.

Gate: `services/symmetry_maps/tests/test_bispinor_actions.py` tests all
96 Si SOC rows, including inversion and antiunitary current covariance,
exact TT equality to the incumbent mixer, and rectangular CT transport.
This action does not change any centroid fixture.

## 2. Cartesian and Pauli-vector actions

`SymMaps.cartesian_action` owns forward orientation, polar/axial parity, and
time parity. `SymMaps.lorentz_action` supplies the polar, time-odd current
index used by `mix_lorentz_blocks`. TT carries two current indices; CT and
TC carry one. Scalar centroid transport owns antiunitary conjugation.
No caller derives a separate rotation or reuses a two-index TT rule for CT.

## 3. Wavefunction / orbital rotation

**ψ coefficient transform.** `SymMaps.unfold_wavefunction` owns the host
action; the device loader obtains its row, spinor, and translation factors
through `operation_rows`, `spinor_action`, and `reciprocal_phase`.

**G rotation + umklapp** (`G' = sym_mats_k @ G − kg0`). **Re-counted by
reading, 2026-08-15: the "four implementations" this page used to claim is
TOO LOW.** Split by what each thing actually is:

- **Rotation carrying an umklapp subtraction — 3 live, not 2.** Canonical
  `wfn_loader/loader.py:604`; `services/vcoul/src/vcoul/bgw_parity.py:199`
  (registered, blocked); and **`src/file_io/epsreader.py:136`
  (`unfold_eps_comps`), which was registered NOWHERE.** It has no in-tree
  caller — only `misc/archived_tests/` — and self-documents at `:126` that it
  has "NO SUPPORT FOR TAU (FRAC TRANS)", i.e. it is known-wrong on every
  non-symmorphic deck and nothing outside the file said so. Now in
  `tests/KNOWN_FAILURES.md`.
- **Umklapp-vector *solvers* — 4 independent.** `maps.py:1960`, a second
  differently-rounded branch in the *same function* (`maps.py:1953-1956`),
  `bgw_parity.py:71-75`, and `src/common/kq_mapping.py:71-73` (the `k−q`
  translation flavour).
- **Bare `S·G`, no umklapp — 3 more live spellings.** `maps.py:843`
  (`(S @ G.T).T`, inside `tau_phase_row`), `maps.py:102/122/143`
  (`einsum('sij,qj->sqi') % kg`, which *discards* the umklapp — see below),
  `tests/bench/charge_density.py:167` (`@ R.T`, and with the wrong matrix).
- **Archived: 7, not 2.** `misc/archived_tests/cohsex_noisdf.py:291,456`;
  `symtest.ipynb:232,407`; `get_interp_vectors.py:46,270,271`. All genuinely
  dead — the package has an empty `__init__` and those modules import a
  `wfnreader` that does not exist anywhere in the tree.

`loader.py:604` and `get_umklapp_vector` are **one composed path, not two
answers**: the loader calls the service for `Gkk` rather than restating it.
Counting them as two of four was a category slip.

**The vcoul blocker HOLDS, but its recorded reason is greppably wrong.**
`KNOWN_FAILURES.md` says "the service returns no `kg0`" — a reader who greps
finds `SymMaps.get_umklapp_vector` immediately and concludes the register is
out of date. The true blocker: that method is *index-keyed* to a
`(SymMaps, WFNReader)` pair, and `bgw_parity.fill_v_grid_for_q` receives raw
fractions and a bare matrix stack, never those objects. Separately,
`find_irreducible_bz_points` computes the rotation and throws the umklapp away
at `maps.py:102` (`% kg` **is** the discarded `kg0`), which is where a
`return_umklapp=True` would go in about two lines.

**Band-index unfold** (`star_broadcast`) is *not* a rotation — pure gather plus
conjugation on TRS rows. One adapter, `file_io/kin_ion.py
broadcast_ibz_to_full_bz`, with an AST gate asserting exactly one
`star_broadcast` call. Consumers: `file_io/sigma_output.py`,
`bandstructure/htransform.py`, `bse/absorption_common.py`.

## 4. Non-symmorphic phase

`maps.py tau_phase_row` — `exp(-i (S·G)·τ)`, one backend, two callers
(`unfold_psi`, the loader). Restated only in tests, which agree.

Not to be confused with the **structure factor** `exp(-2πi G·τ_atom)` in
`psp/` — that τ is an atomic position, a different operation needing no shared
backend.

## 5. Time reversal — consolidated 2026-08-15

Two genuinely different predicates, and the difference is priced at
**183.61 eV**:

- `trs_reference="star_row"` → XOR of two flags (`_star_conj_flags`) — for
  operands that came from `star_select`, i.e. rows of a full-BZ array, each
  carrying a `sym_idx` of its own.
- `trs_reference="ibz_slab"` → the member's own flag — for a wedge slab read
  verbatim off disk with no symmetry applied.

The other three spellings (`sym_idx >= n_sym_spatial` at ψ level) are one
predicate in three places and are consistent.

**`trs_reference` now has NO DEFAULT.** It used to default to `"star_row"`,
which is right for a `star_select` operand and wrong for a file slab. A default
is the wrong shape for a choice whose wrong branch is invisible, so a caller who
has not thought about it gets a `TypeError` rather than a plausible wrong
matrix. All 13 call sites in the tree — production, tests, tools and
multi-device gates — now state their flavour, and
`tests/test_unfold_through_the_service.py` sweeps the AST to keep it that way
(opt-out only via a greppable `trs-reference-exempt` comment; its one user is
the refusal cell itself).

`tests/multi_device/star_invariance_gate.py` used the **wrong** predicate —
comparing against the star's first FULL-BZ row while deciding conjugation with
the member's own flag. **FIXED and verified by execution at P=4**: on
`cohsex_debug` the old predicate classifies 4 spatial + 2 TRS pairs at a
residual of 1.400e-02, the fixed one 3 + 3 at 5.599e-10 — one pair in the wrong
bucket, seven orders of magnitude.

Two facts that execution surfaced and inference would not have: the gate
**passes on `si_cohsex_debug` with `TRS pairs=0`**, so that deck cannot exercise
the branch at all; and it **cannot pass on `cohsex_debug`**, whose noise floor
(5.6e-10) exceeds the gate's `RTOL = 1e-10` — a tolerance calibrated for Si's
1.2e-15. Both legs fail those two checks identically, before and after the fix.
Use a TRS-bearing deck for TRS coverage, and do not read that red as new.

### The q axis: TRS is checked, and the branch is gone (2026-08-22)

For a 2c DFT reference, `density_symmetry_check` compares occupied subspaces
using raw, spatial-only, or TRIM evidence and publishes the verdict as
`WfnLoader.trs_holds` → `SymMaps.trs_allowed`. **The q axis did not read it.**
`gw/v_q_g_flat.py`, `gw/screening.py` and `gw/screening_bse.py` each composed
q with −q through Θ and projected every self-negative q row onto its
Θ-invariant part, unconditionally, in three separate spellings.

On ferromagnetic CrI3 the q solve is the full 81-point BZ with q and −q as
**independent** irreducible parents, so there was no relation to compose and
the guard refused — after the 685.96-GB ζ fit had already closed (Perlmutter
JID 57271494, `GATE trs_pair_unfold_map` naming parents `(1→1, 8→8)`). Where
the parents had happened to coincide it would instead have silently replaced
one independently solved row with the conjugate of the other.

The fix is a policy object, not a guard: `symmetry_maps.QgridTrsPolicy`, built
through the one announcing door `gw.qgrid_symmetry.qgrid_trs_policy_for`, with
`trs_measured` keyword-only and **no default** (the house style the
`trs_reference` predicate already uses across 13 explicit call sites). On a
magnetic deck the policy contains no time-reversal operation at all — identity
row map, no projector — so there is no TRS code path left to guard. It also
refuses a table that names a Θ row beside a magnetic verdict, which can only
mean the tables and the verdict came from different objects.

**What is NOT gated on the verdict, and must not be.** `V_{−q} = conj(V_q)`
is not a statement about time reversal of the wavefunctions: the pair
densities fitted at −q are the conjugates of those at +q with bra and ket
relabelled, for any mean field, and `v(|q+G|)` is real and even. The
reciprocity gate stays armed on a ferromagnet — and there it is *more*
informative than on a nonmagnetic deck, because q and −q were solved
independently, so the gate is a measurement rather than an identity of the
unfold.

### `check_q_conjugate_reciprocity` is blind at every TRIM, not only at Γ

The docstring already said a `q=0`-only check is worthless because `−0 == 0`.
The same degeneracy holds at **every** self-negative q — all eight TRIM of an
even mesh — and there the statistic is "`A_q` is real" and nothing else. On Na
8×8×8 SOC c464 those eight are the only q the gate passes (3.915e-17 at Γ,
6.436e-17 at H) while the point-group covariance the unfold assumes of the
stored parent tile is violated there by **1.240e-02** and **2.411e-02**, its
two largest values. The printed residual is a lower bound on the defect.

Two comments justified that gate's tolerance with "the unfold builds `V_{−q}`
from `V_q` by symmetry so reciprocity holds BY CONSTRUCTION". It does not: the
unfold applies a **spatial** operation and reciprocity is a **conjugation**
statement, and the two coincide only if the finite ISDF ζ basis is point-group
covariant. Both comments are corrected in place, and the quoted `1.16e-7`
"empirical floor" is relabelled as what it is — one deck's ζ covariance, not
an arithmetic floor any deck inherits.

The discriminating statistic is `symmetry_maps.little_group_covariance_residual`
(reported by `common.sanity.report_parent_covariance`), which applies the
unfold's OWN formula with an op that maps the parent q to itself and asks
whether the tile comes back. It needs no second convention, costs one
permutation per op with no extra tile movement, and returns `nan` — not a
pass — when no non-identity little-group op exists.

### What can and cannot see a wrong predicate — MEASURED, and the surprise

| check | sees it? |
|---|---|
| electron count, hermiticity, spectrum, eqp.dat V_H | **no** |
| the diagonal star-spread metric (`compare_to_bgw`) | **no** |
| `symmetry_maps.star_spread` (full matrices, self-consistency) | **no** ← |
| comparison against independently computed full-BZ values | **yes** |

The third row is the one to know. The natural assumption is that the full-matrix
spread catches what the diagonal one misses. **It does not**: the wrong
predicate conjugates an ENTIRE STAR uniformly, and a uniformly conjugated star
is still perfectly self-consistent under the star relation. `star_spread`
measures self-consistency, so it reports ~0 on both.

What the mix-up changes is the star's relation to the data it came from, not its
internal consistency. So the only thing that can catch it is a comparison
against independently computed full-BZ values — which is how 27cc885 was caught
(183.61 eV "against an independently computed V_H") and what
`tests/test_star_offdiag_gate.py` does on the committed full matrices in
`cohsex_debug/sigma_mnk.h5`. Do not add `star_spread` as the guard for this
class and believe you are covered; a cell in
`test_unfold_through_the_service.py` asserts the blindness so that belief fails
loudly.

## THERE ARE TWO DIFFERENT "IBZ"s, AND THEY ARE NOT THE SAME SIZE

Measured 2026-08-15 over **every** committed deck, not three. **This is the
single most important thing on this page for anyone writing a wedge API.**

| deck | `nk_tot` | **file wedge** `nk_red` | **star wedge** (`star_select`) | coincide? | same ROW ORDER? |
|---|---|---|---|---|---|
| `si_cohsex_debug` | 64 | 8 | 8 | **yes** | yes |
| `si_bse_debug` | 64 | 8 | 8 | **yes** | yes |
| `hbn_cohsex_debug` | 18 | 18 | 18 | **yes** | yes |
| `cohsex_debug` | 9 | 4 | **3** | no | no |
| `gnppm_debug` | 9 | 9 | **5** | no | no |
| `bispinor_debug` | 9 | 9 | **5** | no | no |

`bispinor_debug` is a **fourth** diverging deck; earlier revisions of this
page listed three decks and missed it and `si_bse_debug`/`hbn_cohsex_debug`.

**The row ORDER is a second fact and it is the one that bites quietly.** Two
wedges of the same LENGTH can still be different row orders, so a length
match is not a k-set match. Measured `kirr_fullids` against `star_select`'s
row order: identical on the three that coincide; on `gnppm_debug` and
`bispinor_debug` `kirr_fullids` is the identity `[0..8]` while the star rows
are `[0, 1, 3, 4, 5]`; on `cohsex_debug`, `[0, 1, 2, 4]` against `[0, 1, 4]`.
Anything that picks a k-set by matching a count — as `dump_qp_wfn_artifacts`
did until 2026-08-15 — is asserting the order too, without checking it.

- **The file wedge** is `wfn.kpoints` — the k the WFN file stores, length
  `sym.nk_red`, addressed by `kirr_fullids`. This is what `eqp{0,1}.dat`,
  `sigma_diag.dat` and every text output are indexed by, and what BerkeleyGW
  means by the IBZ.
- **The star wedge** is what `star_select` keeps — one row per symmetry star,
  length = number of distinct `irr_idx_k` labels.

They differ whenever the WFN's k-set is finer than the symmetry strictly
requires. On `cohsex_debug`, WFN k #1 is the **time-reverse** of WFN k #2
(`sym_idx_k[kirr_fullids] = [0, 12, 0, 0]`, and 12 = ntran = pure time
reversal), so the file carries 4 k where symmetry has only 3 orbits.

**They coincide on `si_cohsex_debug`** — the deck most gates run on. So an API
that conflates them is *correct on the deck you test with* and wrong on two
others, which is the same trap shape as the TRS predicate. Where the lengths
differ a mistake raises; where they coincide it is silent.

`irr_idx_k` indexes the FILE wedge (`maps.py`: "`sym_idx_k[ik_full]` maps
`wfn.kpoints[irr_idx_k[ik_full]]` → `unfolded_kpts[ik_full]`"), with values in
`[0, nk_red)`. On `cohsex_debug` its distinct values are `{0, 2, 3}` — **file
wedge row 1 is never a parent**, and `broadcast_ibz_to_full_bz` therefore
reconstructs that k as `conj` of row 2 and discards row 1's own stored data.
That is self-consistent, but it is not what "one row per stored k" suggests.

### A documented contract that does not hold

`maps.py` states, as the justification for the `ibz_slab` predicate, that
"`sym_idx_k[kirr_fullids]` is the identity, so a row taken here is the STORED
wavefunction rather than a rotated one". **Measured, it is the identity only on
`si_cohsex_debug`**: `cohsex_debug` gives `[0, 12, 0, 0]` and `gnppm_debug`
gives `[0, 2, 0, 2, 2, 2, 0, 0, 0]`. The companion property
(`unfolded_kpts[kirr_fullids] == wfn.kpoints`) DOES hold on all three
(≤ 3.3e-10). Not repaired here — the code may well be right and only the
stated reason wrong — but nothing should be built on the identity claim until
someone establishes which.

### A test that pins the FIXTURE CORPUS, not the code

`test_at_least_one_fixture_has_the_two_wedges_coinciding_and_one_not` asserts
that the tree contains **both** a deck where the two wedges coincide
(`si_cohsex_debug`, 8 = 8) and one where they diverge (`cohsex_debug` 4 vs 3,
`gnppm_debug` 9 vs 5). It is unusual: it constrains the test corpus rather than
any behaviour.

It has to. If every deck diverged, conflating the two wedges would always raise
and nobody would need two names. If every deck coincided, it would never raise
and the bug would be permanently silent. The distinction is only testable
because the tree happens to contain both cases.

**Anyone pruning fixtures: removing `cohsex_debug` or `gnppm_debug` silently
disarms every check of the two-wedge distinction** — including the
reduce/unfold asymmetry cell, which needs a deck where a stored k is the
time-reverse of another. That is what this cell exists to say out loud.

## The SC loop crosses both wedges, and that is where it broke

`config.sc_on_ibz` runs H/E/U on the **star wedge** while Σ stays on the full
BZ. Every `.dat` writer wants the **file wedge**. Until 2026-08-15 the loop's
rows went straight to the writer:

    sc_iteration.py:1397 _write_sc_eqp_snapshot -> eqp_bgw.py:145
    ValueError: e_qp shape (5, 46) does not match e_dft (9, 46)

**There is no star-wedge → file-wedge operation and none should be added.**
The route that is right on every deck is *star wedge → full BZ → file wedge*:

```python
reduce_full_bz_to_file_wedge(sym, unfold_star_wedge_to_full_bz(sym, x))
```

Two boundaries take it — `_write_sc_eqp_snapshot` and
`dump_qp_wfn_artifacts` — and neither holds an index table. The second
replaced a `placements` dict that chose between `KStarMap.select` and the
full BZ **by matching lengths** against `wfn.nkpts`; per the order row above,
a length match is not a k-set match.

**`sc_on_ibz` now defaults True** (owner directive: `H^QP` is built and
`eigh`'d only on symmetry-reduced k). What paid for the flip, on
`gnppm_debug`:

| accelerator | on vs off, E_QP |
|---|---|
| `linear`, `mixing = 1` | **1e-6 meV** — the `%15.9f` print floor — at every iterate |
| `rcrop` (the default) | 24.45 meV by map call 5; 113.3 meV in the final `eqp0` |

The map is exactly k-set invariant; **rCROP is not**, because its
least-squares mixing minimises a residual summed over the loop's own k-set —
each orbit once on the star wedge, with its multiplicity on the full BZ. Same
fixed point, different path. So an *unconverged* rCROP run's iterates move
when this flag moves. NOT established: agreement at a converged fixed point
(both arms stop at 2–3 map calls, RMS ΔE ≈ 3 eV).

### The committed deck that exercises it, and the one that cannot

`tests/regression/gnppm_debug/gnppm_sc.in` is the **only** committed deck with
`qp_solver = self_consistent`. Before it, every deck was `one_shot_dft`, which
is why this rotted invisibly: flipping the default would have changed no suite
result. Dynamic self-consistency is now a supported configuration: the final
correlation cube is rotated from the last map's QP basis to the DFT output
basis one omega row at a time, and its at-DFT diagonal cache is rebuilt from
that rotated operator. `tests/test_qp_solver_config.py` keeps this deck as the
parse-level capability fixture; the full driver still has to satisfy the
ordinary numerical gates of the chosen ansatz and deck.

`cohsex_debug` (4 vs 3, the sharpest divergence) **cannot run `sc_on_ibz` at
all**: `centroids_frac_60.txt` is not orbit-closed, so the k-star spread of
Σ+V_H measures **2.763726e-01** against the 1e-6 refusal and the run stops.
That is the gate working on a genuinely non-star-invariant Σ — 2.762e-01 is
exactly that set's recorded closure residual, listed under the non-closed sets
below. **The centroid-closure question and the SC wedge question are coupled
through this deck**, and anyone who "fixes" that set changes which decks can
exercise the SC path.

## PRINCIPLE: a self-consistency check cannot detect a uniform error

**No check that asks "is this array consistent with itself under the symmetry
relation" can detect an error applied uniformly across a whole orbit.** Only a
comparison against independently computed values can.

This is not a remark about one function. It is a property of the *shape* of the
check, and it applies to any invariance residual — star spread, orbit closure
residuals, "does W(Sq) match the permutation of W(q)".

The worked instance, measured:

- The two TRS predicates differ by conjugating an **entire star** uniformly.
- A uniformly conjugated star still satisfies the star relation exactly.
- So `symmetry_maps.star_spread` — which compares each member to its star's
  reference by the correct XOR rule — reports **~0 on both the right and the
  wrong array**.
- The diagonal metric is blind for a second, independent reason: conjugating a
  Hermitian block leaves its real diagonal exactly intact.
- Historical consequence: 27cc885's wrong `trs_reference` measured **183.61 eV**
  and was caught only "against an independently computed V_H". The electron
  count, hermiticity, the spectrum and the eqp.dat V_H column were all
  unchanged, and nothing in the suite turned red for a month.

| check shape | detects a uniform per-orbit error? |
|---|---|
| self-consistency residual (`star_spread`, closure residuals) | **no** |
| any diagonal-only observable | no (and blind to conjugation twice over) |
| comparison against independently computed values | **yes** |

**Do not reach for `star_spread` as the guard for a conjugation or convention
error.** The guard is `tests/test_star_offdiag_gate.py`, which compares against
the committed full matrices in `cohsex_debug/sigma_mnk.h5` — independent data,
not self-consistency.

*How this was established, because the reasoning matters as much as the result:*
a cell was written asserting that `star_spread` **would** catch the mix-up —
the natural assumption being that the full-matrix check sees what the diagonal
one misses. The cell failed. The conclusion drawn was that the cell was wrong,
not the code, and the property above is what replaced it. The assumption had
been stated in a docstring for months without anyone running it.

## 6. Real-space / FFT-grid symmetry

`orbit_syms.py` owns this. The pair worth knowing:

- `centroid_source_map_and_wrap` — the **source** map `y = S·(r − τ)`.
- `fft_grid_pullback_perm` — the **forward** map `Rinv·r + τ`, inverted.

These are the same algebra computed two ways with two rounding strategies and
two failure modes. The package docstring warns they are *not* variants of each
other (direction), and cites a silent 4 eV gap on hex systems from confusing
them; what it does not say is that the wrap/snap logic must be **upgraded
twice**.

### Re-counted 2026-08-15: it is TWELVE live expressions, not "~6"

Every one is live production. This page previously said "~6" and named the
wrong byte-identical pair.

| where | direction | wrap |
|---|---|---|
| `orbit_syms.py:493` `centroid_source_map_and_wrap` | source | `rint`×grid → `floor_divide` → `rint` |
| `orbit_syms.py:956` `verify_centroid_orbit_closure` | **source** | `% 1.0`, min-image scoring |
| `orbit_syms.py:1321` `fft_grid_pullback_perm` | forward | `− np.floor` on UNSNAPPED floats |
| `orbit_syms.py:243` `orbit_images` | forward | `% 1.0` (jax, vmapped) |
| `orbit_syms.py:334`, `:342` | forward | `% 1.0` then `round(·×inv) % inv` |
| `orbit_syms.py:145` | forward, integer | `% N` |
| `centroid/charge_density.py:199` | forward, integer | `% N` |
| **`centroid/kmeans_isdf.py:270, 288, 378`** | forward | **no wrap at all** (min-image downstream) |
| **`centroid/kmeans_isdf.py:569`** | forward | `% 1.0` |

**The four in `src/centroid/kmeans_isdf.py` are the ones that matter**, and
this page did not name that module at all. It is the *generator* of the
centroid sets whose orbit closure the rest of this section is about, so an
upgrade to the r-space action convention would silently not reach the thing
that produces the data.

**The byte-identical pair was misidentified.** `charge_density.py:199` twins
**`orbit_syms.py:145`** — same expression *and* the same radix flatten on the
following line — not `fft_grid_pullback_perm`.

### Consolidation is 12 → ~5, not a tidy-up. RE-SCOPED, NOT DONE.

Three axes are irreducible and one of them is load-bearing:

1. **Direction.** Source (`S·(r − τ)`) vs forward (`Rinv·r + τ`). The package
   docstring already records that giving these one name shape is the mechanism
   of the 4 eV hex-system gap. A shared helper with a direction flag re-creates
   exactly that API.
2. **Rounding, and the two strategies are NOT interchangeable.**
   `centroid_source_map_and_wrap` snaps to FFT-grid integers *before* `floor`,
   and its own comment (`orbit_syms.py:494-503`) says why: naive `np.floor`
   flips an `L` component 0 → −1 on tiny negative noise, "which produces a
   spurious `exp(±iπ/2)` phase in `unfold_isdf_operator`" — measured on Si
   Fd-3m as 14 of 64 q at rel err ~0.8. `fft_grid_pullback_perm` does the
   forbidden thing (`− np.floor` on unsnapped floats) and is safe **only
   because it discards the integer part**. One keeps `L` and feeds it to an
   umklapp phase; one throws it away. Any shared helper must return `L` and
   pay the expensive path unconditionally.
3. **Backend.** `orbit_syms.py:243` and the four `kmeans_isdf.py` sites are
   inside `@jax.jit` / `lax.fori_loop`; three of those deliberately omit the
   wrap because the Lloyd metric handles periodicity through an explicit
   min-image table, so wrapping there would be **wrong**, not redundant.

### DONE 2026-08-16 — 12 → 5, seven restatements deleted, all bit-identical

The plan below was executed. What shipped, and the two things that changed
during it:

**The shared kernel is `snap_to_grid_and_split_wrap(images_frac, fft_grid)
-> (idx, L)`**, and it is what makes this a consolidation rather than a
tidy-up: the snap-before-floor rule used to be written inside the SOURCE map
and *not* inside the forward one, so it had to be upgraded twice and only one
copy ever implemented it. It now exists once and serves **both directions** —
it takes the images already computed, and the two maps differ in how those
were built, not in how they must be split. The 14/64 q evidence is written at
that function, and a test cell replays the exact negative-noise input and
asserts both that the snap gets it right and that a naive `floor` still gets
it wrong.

**S2's discards-`L` exemption is in the code**, with an AST test asserting the
pullback still throws `L` away — a pullback that started consuming it would
silently inherit S1's correctness requirement. **S1 and S2 are still two
functions** and a cell asserts that too.

**The seven deletions**, verified bit-identical on three decks
(si_cohsex_debug 48 ops/24³, gnppm_debug, cohsex_debug):
`orbit_syms.py:145` + `centroid/charge_density.py:199` → S4;
`orbit_syms.py:334, :342` → S3 stack form;
`centroid/kmeans_isdf.py:270, :288, :378, :569` → S3 single-op form.

**A numerical finding that shaped the design**: `einsum` and `@` differ by
**one ulp** (measured: 2 of 276 entries at 2.220e-16), because BLAS picks a
different summation order for a three-term dot. So **each call site kept the
contraction it already had** — the orbit sites were `einsum` and took the
stack form, the kmeans sites were `@` and took the single-op form. Crossing
them over would have been a one-ulp change to centroid selection dressed as a
refactor, and a pivoted-Cholesky pivot order is exactly what one ulp flips.

**`wrap` is required with no default** on both S3 forms, because three of the
four `kmeans_isdf` sites deliberately do NOT wrap — they feed a minimum-image
metric with its own offset table, so folding into `[0,1)` first would put the
image on the wrong replica. That is the most dangerous line in the diff.

`epsreader.py:136` was **deleted** in its own commit, with a tombstone.

### The plan as written, kept for the record

### THE PLAN — what the five survivors are, for the owner to scope

Written 2026-08-15 on request. **Not started.** Ordered by blast radius, and
the first two are worth doing whatever is decided about the rest.

**The five survivors.**

| # | survivor | absorbs | why it cannot merge further |
|---|---|---|---|
| S1 | `centroid_source_map_and_wrap` — SOURCE map, snap-then-floor, RETURNS `L` | `orbit_syms.py:493` | the only one whose integer wrap is consumed (umklapp phase); must keep the expensive rounding |
| S2 | `fft_grid_pullback_perm` — FORWARD map, DISCARDS `L` | `orbit_syms.py:1321` | inverts a permutation; safe with cheap rounding *because* it throws `L` away |
| S3 | `orbit_images` — FORWARD, jax, `% 1.0`, no `L` | `:243`, `:334`, `:342` | inside `jit`/`vmap`; the two numpy twins are the same expression at a different backend |
| S4 | `grid_point_image_perm` — FORWARD, INTEGER grid, `% N` | `orbit_syms.py:145`, `centroid/charge_density.py:199` | integer arithmetic end to end: no rounding question exists, so it must NOT be routed through S1/S2 |
| S5 | `verify_centroid_orbit_closure`'s scorer — SOURCE, min-image | `:956` | its residual is a *measurement*, not a map; merging it into S1 would make the check share code with the thing it checks |

`kmeans_isdf.py`'s four (`:270, :288, :378, :569`) fold into **S3** — but three
of them **deliberately omit the wrap**, because the Lloyd metric handles
periodicity through an explicit min-image table. S3 must therefore take
`wrap=False`, and a shared helper that always wraps would be *wrong* there, not
merely redundant. That is the single most dangerous line item in this plan.

**Which rounding strategy wins: S1's, and it is not a preference.**
`orbit_syms.py:494-503` records the mechanism — naive `np.floor` flips an `L`
component 0 → −1 on tiny negative noise, "which produces a spurious
`exp(±iπ/2)` phase in `unfold_isdf_operator`", measured on Si Fd-3m as **14 of
64 q at rel err ~0.8**. So any survivor that RETURNS `L` must snap to
FFT-grid integers before `floor`. S2 is safe with the cheap rounding *only*
because it discards `L`, and that exemption has to be written at S2 or the next
person "unifies" them and reintroduces the phase.

**Blast radius.**

- S4 is a **pure deletion** across a package boundary (`src/centroid` →
  the service) and needs one new export. Zero numerical risk: integer in,
  integer out. Do this one first.
- S3 is three call sites, one of them jax — behaviour-preserving only if the
  `wrap` flag is threaded correctly; the three `kmeans_isdf` sites are the risk.
- S1/S2/S5 **should not be touched**. They are already one function each; the
  "duplication" between them is the direction/rounding distinction the 4 eV hex
  gap is named after. Consolidating them is how that bug returns.

**So the honest target is 12 → 5 by deleting 7 restatements, not by unifying 5
survivors into 1** — and 4 of the 7 are in a module (`kmeans_isdf.py`) this
page did not previously name.

**Not attempted here.** The item was scoped as "~6 copies, upgrade twice"; it
is twelve across three packages with a measured rounding hazard between two of
them.

### `epsreader.py:136` is a DEFECT, independent of any of this

`src/file_io/epsreader.py:136` `unfold_eps_comps` is a fifth `G' = S·G − G₀`
that no consolidation reaches, and it should be decided on its own:

- Its own comment at `:126` says **"NO SUPPORT FOR TAU (FRAC TRANS)
  CURRENTLY"** — it is known-wrong on every non-symmorphic deck.
- It has **no in-tree caller** (only `misc/archived_tests/`), yet `EPSReader`
  is re-exported from `src/file_io/__init__.py:41`, so it is public surface.
- It was registered **nowhere** until 2026-08-15.

The decision owed is **delete or fix**, not "consolidate later". Deleting a
re-exported public method is a surface change; fixing a method with no caller
is speculative. Either is a one-commit answer and neither depends on the
r-grid work.

### One documentation defect found while counting

`orbit_syms.py:18-20` names `charge_density._symmetrise_density` and
`symmetry_maps.py:339-345` as the canonical real-space convention. The former
is the **known-broken bench copy** (`KNOWN_FAILURES.md`: wrong matrix, no τ
phase, "fix is deletion… not repair"); the latter is a flat module that no
longer exists since the service split. So the canonical convention doc points
a reader at a broken G-space routine using the transposed matrix.

**Density symmetrisation**: canonical `gw/qsgw_density.py symmetrise_density`
(pure gather over the pull-back table). A known-broken duplicate in
`tests/bench/charge_density.py` — no τ phase, transposed G convention —
registered.

---

## Where a single-backend upgrade propagates, and where it doesn't

| Operation | One backend? | What silently won't follow |
|---|---|---|
| SU(2) from 3×3 | **yes** | nothing — clean |
| TRS spinor augmentation | **yes** | nothing live |
| ψ unfold sequencing | **yes: `SymMaps` methods** | host/device kernels consume the same typed factors |
| Pauli-channel mix | **yes: typed Cartesian action** | nothing live |
| G rotation + umklapp | **3 live** (was miscounted as 4) | `vcoul/bgw_parity.py` (blocked) and **`file_io/epsreader.py:136`**, which was unregistered and is τ-blind by its own admission |
| τ phase (G-space) | **yes** | but the 2π convention splits across 4 dividers |
| TRS conjugation predicate | 2 semantics, **no default, all 13 call sites explicit** | nothing — AST-swept |
| r-grid image / source map | **12 expressions** (was miscounted as ~6) | forward/pull-back must be upgraded together — AND the four in `centroid/kmeans_isdf.py`, the module that GENERATES the centroid sets |
| Density symmetrisation | 2 (**one broken**) | the bench copy |
| Cartesian-index rotation | **yes: `cartesian_action`** | nothing live |

---

## File k-bases — what is stored where

**The stamping model to copy is `kin_ion.h5`**: per-dataset `k_storage` /
`k_storage_version` / `n_sym_spatial`, constants owned in
`file_io/kin_ion.py`, **absent ⇒ `"full"`**, and the unfold happens at the
reader through the single adapter. That default is load-bearing: four older
committed fixtures were computed independently at every full-BZ k and do *not*
satisfy the star relation (max|Δ| up to 7.8 Ry), so treating them as
compressible would move physics.

| Artifact | k/q basis today | Notes |
|---|---|---|
| `eqp0/eqp1.dat`, `sigma_diag.dat`, `eqp_g0w0.dat`, sigma-freq debug | **wedge** | all `.dat` outputs; coordinates on every block |
| `sigma_mnk.h5` Σ datasets | **wedge**, stamped | carries `irr_idx_k`/`sym_idx_k` + star-spread attrs |
| `kin_ion.h5` | **wedge**, stamped | the reference implementation |
| `zeta_q.h5` | **q-IBZ** by default | no `q_storage` attr — basis inferred from shape |
| `mpa_*.h5` | **q-IBZ** always | most complete stamping in the tree (rank-checked v2) |
| `WFN.h5` / `WFN_qp.h5` | **IBZ** | BGW's own header *is* the declaration |
| `isdf_tensors.h5` `V_qmunu`/`W0_qmunu` | **computed q parents**, stamped | The parent route retires explicit `restart_q_storage=full`; naturally unreduced q sets and historical full-q readers remain |
| `isdf_tensors.h5` `psi_parent_y` and `_mun` | **raw file parents**, canonical logical centroids | `psi_parent_k_rows` authenticates the row map; transverse counterparts carry the second centroid family |
| Historical `isdf_tensors.h5` `psi_full_y` | **full BZ** | Legacy consumers may read it; current GW does not write it. Parent-to-child transport is not a row gather |
| `qp_wfn_rotations.h5` | **file wedge when the writer can prove it**, stamped | `qp_rotations_k_storage` default `"auto"`; the writer runs the reader's round trip and keeps the wedge only if it reproduces the arrays |
| `dipole.h5` | **full BZ** | NOT convertible by a gather — measured, below |
| `v_q_bispinor.h5` | **q-IBZ**, per-dataset stamps | seven unique tiles, two centroid families; `photon_blocks_full_q` restores one Lorentz block |

### The three rows that are NOT one operation

`kin_ion` and Σ move onto a wedge through `star_broadcast`,
which is a **pure row gather plus `conj` on the time-reversed rows**. That
works because they are SCALAR operators: each commutes with every space-group
operation and with time reversal, so a star holds one matrix.

The historical full-BZ forms of these artifacts are not scalars, and each fails the
gather in its own way. **Measured 2026-08-16** on the committed fixtures
(`reports/wedge_storage_migration_2026-08-16/artifacts/probe_star_relation.py`):

| quantity | what the index is | plain gather | the rule that works |
|---|---|---|---|
| `deltaE` | none (energies) | **exact** | gather |
| `dipole_cart` | Cartesian component of **v̂** | **rel 2.0 — 200 % wrong** | `cartesian_action(..., axial=False, time_odd=True)` on the component axis plus `conj` on antiunitary rows |
| `psi_full_y` | ψ on the centroid grid | **rel 1.640 — 164 % wrong** | centroid permutation + L-phase + spinor rotation + the `unfold_psi` TRS rule |
| `U_mnk` | eigenvector gauge | run-dependent | *there is no rule* — see `qp_wfn_rotations.h5` below |

`dipole_cart` is the sharp one, because both halves of the failure are
measurable separately and both are ~100 % of the signal:

* **si_cohsex_debug** (64 k → 8, 48 spatial ops, **0 time-reversed rows**):
  plain gather `max|Δ| = 4.800535` against a scale of 2.400268 — **rel 2.000**.
  With the inverse axial table applied untransposed, 3.687563. With the
  explicit forward polar action, **4.000962e-15**. Exact.
* **gnppm_debug** (9 k, 4 of 9 rows time-reversed): transposed-`R` + `conj`
  gives spatial rows **0.000000e+00** and TRS rows **2.836690** — again
  rel 2.000, which is what `+v` against `−v` looks like. Adding **−1 on the
  antiunitary rows** takes the whole array to **2.703570e-15**.

So the complete rule is `d_a(gk) = −1^{TRS} · R_forward[s]_{ab} ·
conj^{TRS}(d_b(k))`, and the sign is there because **v̂ is odd under time
reversal** while `kin_ion` is even. The live G-space Hartree matrix follows
the scalar-operator star relation but is not stored. Production consumers now
obtain this matrix as `cartesian_action(..., axial=False, time_odd=True)`;
they do not choose the transpose or sign independently.

The one-shot scalar and packed head consumer,
`qsgw_head.read_authenticated_dipole_velocity`, reads only `kirr_fullids`
and the active chi band window after authenticating the existing full-BZ
file. `unfold_file_wedge_polar_matrix` restores the velocity for the shared
head and parent-star wing kernels. The file indexing and provenance stay
unchanged; no `deltaE` payload is read by this consumer.

#### …and the SAME sign applies to the QSGW velocity — derived 2026-08-22

The row above measured the parity of the **bare** term. The head lane
differentiates that velocity and adds two more terms,
`v^Q = v^DFT + d_k Σ − i[A, Σ]`, whose parity was documented nowhere — and a
wrong sign there is silent on a TRS-broken deck, where the identity does not
hold anyway. Derivation, and the convention, now live in the
`src/gw/qsgw_head.py` module docstring (eq. 1–2). In the LORRAX full-BZ gauge
`u_n(−k) = Θ u_n(k)`, antiunitarity gives `O_mn(−k) = s·conj(O_mn(k))`
elementwise with no transpose, for any operator with `Θ O Θ⁻¹ = s O`. Then:

| term | parity | why |
|---|---|---|
| `H`, `Σ`, `kin_ion`, `V_H` | **even** | scalar operators, `s = +1` |
| `d_k Σ`, `d_k H` | **odd** | differentiation flips it: `(d_i M)(−k) = −conj((d_i M)(k))` for even `M` |
| `A_i` (Berry connection) | **even** | the explicit `i` in `A = i⟨u|∂u⟩` cancels the derivative's flip |
| `−i[A_i, Σ]` | **odd** | conjugating the commutator of two even objects, with the leading `−i` |

**All three terms of `v^Q` therefore carry the same odd parity, so
`v^Q_i(−k) = −conj(v^Q_i(k))`** — the correction never flips the sign, which
is what makes a sign error in it visible rather than absorbed.

`gw.qsgw_head.trs_velocity_parity_residual` measures it, and
`report_trs_velocity_parity` is the verdict. **The verdict statistic is the
band TRACE**, because `tr v_i(k)` is invariant under any unitary mixing inside
the retained window and therefore survives both of the two ways the
elementwise form breaks: (a) `k` and `−k` reached from their IBZ parent by
unrelated spatial rows sit in gauges related by a little-group rotation — the
same "one consistent row per orbit" question the q axis settles; (b) with
`Θ² = −1` the partner of band `n` at `−k` is its **Kramers** partner and the
label inside a degenerate doublet is gauge-arbitrary. The elementwise number
is returned beside it as a strictly stronger diagnostic, never as the verdict.
Its stated blindness: a trace cannot see a parity error whose band matrix is
traceless, including a sign flip confined to the off-diagonal transition
sector. The gate takes no verdict at all when the density measurement says
time reversal is broken, or when no measurement is available — an unmeasured
system is not a TRS system.

`psi_full_y` was measured on a real run's `isdf_tensors_144.h5`
(`si_cohsex_debug` fast deck, 64 k → 8): a plain gather gives
`max|Δ| = 3.634856e-02` against a scale of `2.2e-02`, **rel 1.640**. So the
`star_broadcast` route is closed for ψ, which is the concrete form of "the
option is not a gather". The two follow-up hypotheses tried in the same probe
(the centroid permutation, and the same permutation on `|ψ|`) are
**inconclusive rather than negative**: the only `sym_perm` on that file is the
**q**-wedge table from `V_qmunu__qirr`, addressed by `sym_idx_q`, and the
probe addressed it by `sym_idx_k`. A fair test needs the k-side centroid
permutation, which that historical probe did not build. The current
`CentroidKUnfoldPlan` supplies it through the symmetry service; BSE selected
parent-face reads use that plan after WFN/centroid/row authentication.

**`cohsex_debug` does not reproduce this** (spatial rows 5.317e-02, TRS rows
2.759753) and is **not** evidence against the rule: that deck's own
`kin_ion.h5` is 8.04 Ry away from its star relation, i.e. the fixture predates
the current symmetry map. Judge the rule on the two decks whose fixtures ARE
star-consistent.

**Historical dipole scope.** The measurement above did not convert
`dipole.h5`. Its original blocker was the then-unresolved TT rotation audit.
The four-current action is now resolved by the typed service described in
§§1–2; the deleted `mix_channels_by_proper_rotation` is no longer an owner.
That resolution does not itself migrate dipole storage or certify its
Cartesian-index unfold.

### `v_q_bispinor.h5` — q parents and two centroid families

The writer persists seven unique q-IBZ tiles in canonical centroid order.
Each dataset has its own `QirrTables` and closure verdict: CC uses the charge
family, TT the current family. The existing format service already supports
per-dataset table groups. Unstamped legacy full-q V files require a fresh run.
`BispinorVqReader` packs each endpoint at the file seam and returns q parents.

The photon body and Dyson solve keep the leading q-IBZ axis.
`StaticPhotonResponse` retains the measured `QgridTrsPolicy` and both family
plans. `photon_blocks_full_q` transports source centroid blocks through the
scalar symmetry service, then applies `mix_lorentz_blocks` with the polar,
time-odd Lorentz action. A selected output avoids a nine-block full-q result.
The Si SOC P4 V gate resolves eight parents to64 q rows: worst TT difference
from full-q main is3.34e-13; omitting the Lorentz mix gives0.46–2.0.
Evidence: sandbox `runs/Si/100_bisp_parent_route_2026-09-05/30_v_lorentz_restore_p4`.
The W, Sigma and portable-restart gates remain part of the campaign acceptance.

### `absent ⇒ full` — re-measured, and the range is wider than recorded

The four committed `kin_ion.h5` fixtures whose rows do not satisfy the star
relation, through the file-wedge round trip, 2026-08-16:

| fixture | `nk_tot` → file wedge | `max|Δ|` |
|---|---|---|
| `gnppm_debug` | 9 → 9 | **31.05 Ry** |
| `bispinor_debug` | 9 → 9 | **12.44 Ry** |
| `cohsex_debug` | 9 → 4 | **8.04 Ry** |
| `si_cohsex_debug` | 64 → 8 | 2.0e-03 Ry |
| `si_bse_debug` | 64 → 8 | 0 (exact) |
| `hbn_cohsex_debug` | 18 → 18 | 0 (exact) |

This page previously said "up to 7.8 Ry". The worst is **31.05 Ry**, four
times that. The conclusion is unchanged and stronger: a file without a
`k_storage` stamp is full-BZ, full stop.

---

## Non-closed centroid sets in the tree: KEEP, and why

Four non-closed sets remain, deliberately. **Do not re-open the sweep.**

| set | worst | why it stays |
|---|---|---|
| `si_cohsex_debug/centroids_frac_960.txt` | 1.318e-01 | `test_qgrid_symmetry_resolution.py`'s `_OPEN_SET`, paired against the 144 set as `_CLOSED_SET` — the pair *is* the thing under test |
| `si_bse_debug/centroids_frac_480.txt` | 1.718e-01 | measured specimen pinned by `test_symmetry_maps_closure.py` and `..._qgrid_resolution.py` (47/48 ops); the deck itself already uses the closed twin |
| `cohsex_debug/centroids_frac_60.txt` | 2.762e-01 | `test_star_offdiag_gate.py` asserts its consequence as a fact; also the deck behind ~12 test files incl. `conftest.py` |
| `bispinor_debug/centroids_frac_256.txt` | 1.436e-01 | owner ruling: KEEP unchanged; the same parent route uses a trivial `SymMaps` view and loader-unfolded full-k parents; regenerating would change the frozen reference |

These are what the closure machinery is *tested against*: deleting them deletes
the tests that establish closure behaviour.

**And closure is not what people think it is on these decks.** Measured
2026-08-15: an orbit-closed 960-point set for the Si production deck
(`centroids_frac_960_orbitclosed.txt`, `closed=True` at 1.000e-06 on 48 ops,
same count, same deck) moved the within-star Σ spread only 2.611 → 1.964 meV at
the compared bands and 41.34 → 39.88 meV over the full window (**both figures
are the PER-BAND metric AT A SLICED BAND EDGE, and are WITHDRAWN as evidence
about closure — see "THE SI STAR SPREAD IS A BAND-SLICING ARTIFACT" below;
both arms go to 0.0000 at a clean edge**) — while making
BerkeleyGW agreement ~35× worse (sigTOT MAE 0.4329 → 14.9426 meV). Forcing
closure did not fix the symptom it was believed to cause.

## THE SI STAR SPREAD IS A BAND-SLICING ARTIFACT — and the slice is in the DECK

**Read this before quoting any star-spread number on this page.** Two lanes
measured this on 2026-08-15 and the answers only look contradictory. Both are
right; they varied different things.

### The deck's `nband` edge is the cause, and it slices at ZERO gap

`si_cohsex_debug` runs `nband = 60` on a WFN carrying **62** bands. Measured
with `boundary_min_gaps` on the FULL mean field:

| edge | min gap over k | clean? |
|---|---|---|
| 8 | 2540.408 meV | yes |
| 16 | 283.348 | yes |
| 20 | 228.290 | yes |
| **24** | **0.000000** | **no** |
| 28 | 393.797 | yes |
| 36 | 156.992 | yes |
| 40 | 818.241 | yes |
| **60 — the deck's own edge** | **0.000000** | **no** |

Holding the centroid set, `zeta_rcond` and P fixed and moving ONLY that edge,
the within-star Σ spread goes to **exactly zero** (the other lane's
measurement):

| `nband` | edge gap | sigSX | sigCOH | sigTOT | V_H |
|---|---|---|---|---|---|
| 60 | **0 meV — slices** | 0.0270 | 1.9570 | 1.9430 | 0.0990 |
| 40 | 818 meV clean | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| 36 | 157 meV clean | **0.0000** | **0.0000** | **0.0000** | **0.0000** |

**So the spread is a truncated-multiplet artifact in the ζ / Σ BAND SUM, not
a property of the centroid set and not an artifact of the measurement.** The
measuring instrument in that experiment uses no symmetry code and returns
exact zeros on the same files at a clean edge.

### THE TRAP THAT MADE THIS LOOK CLEAN — do not repeat it

`boundary_min_gaps` returns `+inf` at `b = nb` by construction: within the
array it is handed, the outer boundary cuts nothing. **So applied to an
already-truncated window it CANNOT see the truncation that produced it.** On
the 60-band Σ window it reports edge 60 as `+inf`, i.e. clean, while the same
function on the 62-band mean field reports **0.000000 meV**. Measured, both
numbers, same deck, same session.

The first lane's analysis varied the REPORTING cut (the max over the first
`n` entries of a fixed 60-band per-band vector) and correctly found that
restricting to clean cuts changes nothing — 8, 16, 40 are already clean and
only 24 is split. That conclusion is right about the reporting cut and says
nothing about the deck's edge, which is the one that matters.

**Rule: give `boundary_min_gaps` the FULL mean field, never the window you
are about to slice out of it.**  **FIXED 2026-08-15**: `is_full_spectrum` is
now a required keyword with no default, and a window's outer boundaries come
back `nan` — neither `> tol` nor `<= tol`, so they cannot be certified either
way and must be asked about the full spectrum.

`snap_cut_to_clean_boundary` is **not** part of this and must not be
re-implemented: it arrives with `feat/band-extrapolation-sampling-2026-08-15`
(`81edc49c`), a separate worktree.

### The per-band spread is a sharp instrument, now that this is known

At a clean band edge the per-band star spread reads **exactly 0.0000**, not
"small". That makes it a discriminating check rather than a noisy one: any
non-zero value is either a sliced edge or a real defect, and the two are
separated by one `boundary_min_gaps` call on the mean field.

### The degeneracy structure, and the subspace-invariant twin

Within a degenerate multiplet any unitary mixing of the subspace is an
equally valid eigenbasis, so a per-band `Re Σ_bb` is not symmetry-invariant
while the TRACE over the multiplet is. On the Si 60-band window **60 of 60
bands sit inside a multiplet** (groups of 4, 4, 8, 8, 8, 8 and one of 20),
tolerance-insensitive from 1 meV to 13.6 µeV. Measured per group at the
sliced `nband = 60`:

| bands | size | per-band max | multiplet trace | ratio |
|---|---|---|---|---|
| 0–7 | 8 | 0.980 meV | 0.134 | 7.3× |
| 8–15 | 8 | 2.611 | 0.593 | 4.4× |
| 16–19 | 4 | 4.821 | 2.302 | 2.1× |
| 20–27 | 8 | 7.267 | 2.835 | 2.6× |
| 28–35 | 8 | 10.020 | 2.909 | 3.4× |
| 36–39 | 4 | 9.471 | 6.734 | 1.4× |
| **40–59** | **20** | **41.338** | **3.604** | **11.5×** |

So even at the sliced edge the headline 41.338 meV is ~91 % band-label gauge
in its own block, and the worst invariant residual is 6.734 meV. The
diagnostic is emitted as `star_spread_multiplet_ev` /
`star_spread_multiplet_ev_per_band` beside the per-band row
(`gw_output._star_spread_over_multiplets`). The per-band row STAYS — it is
what the historical figures and the BerkeleyGW gate threshold are quoted in.

**WITHDRAWN: "orbit mode buys a tighter within-star spread."** Both arms of
that comparison were measuring the same sliced edge, and both go to 0.0000 at
a clean one. The orbit-closed-960 vs literal-960 numbers quoted below
(2.611 → 1.964 and 41.34 → 39.88) are therefore **not** evidence about
closure; they are two measurements of the same slicing artifact.

### The conditioning synthesis — hand this to whoever picks up the count question

Three independent measurements in this tree point at ONE mechanism, and it is
**conditioning, not geometry**. **ITEM 1 IS WITHDRAWN — read the section
immediately above first.** Its "~40 meV" is a sliced band edge, not a
centroid-count effect, so the synthesis now rests on items 2 and 3 alone and
should be re-argued rather than quoted.

1. ~~The Si deck carries ~537–588 G per k, so 960 centroids is ~1.7×
   over-complete and 144 is ~0.25×. The 144 set measures a star spread of
   exactly 0.000; both 960 sets measure ~40 meV, closed or not.~~
   **WITHDRAWN 2026-08-15.** The ~40 meV is the deck's `nband = 60` edge
   slicing a multiplet at 0.000 meV gap, not the centroid count: hold the
   centroid set fixed and move only the edge to 40 or 36 and every Σ channel
   goes to **exactly 0.0000**. This item was the synthesis's first leg and it
   does not survive.
2. A ledger row records **1776 centroids on a 588-G deck (3.0× over-complete)
   dropping 300+ modes per q at κ ≈ 1e10 and producing a 100 eV Σ_c with no
   refusal.**
3. The `zeta_rcond` ladder drifts sigTOT **0.054 → 1.021 → 37.218 meV** as rcond
   loosens 1e-8 → 1e-6 → 1e-4.

Over-complete centroid sets make the ζ fit ill-conditioned; the fit noise breaks
the symmetry the quadrature should respect. That explains why forcing closure
did not help, and it implies a **real tension**: low count buys symmetry and
loses accuracy, high count does the reverse. The count sweep that would test it
is an open owner question, not authorised.

## GENERAL PROPERTY: a wedge-stored array indexed by a full-BZ index

**It is silently wrong wherever the index is below `nk_red`, and only raises
above it.** Measured with h5py against an 8-row dataset on the real Si map
`kirr_to_kfull = [0, 1, 2, 5, 6, 7, 10, 27]`: indices 10 and 27 raise
`IndexError`; **0, 1, 2, 5, 6, 7 — six of eight — return the wrong row with no
error**. On the full BZ, every index `< nk_red` (8 of 64 on Si) is silently
wrong.

This governs the DESIGN of every wedge move, not just one file:

- "A stale consumer will `IndexError`" is **false** for the majority of
  lookups. Do not rely on it.
- A `k_storage` stamp protects in-tree readers and does nothing for
  out-of-tree ones, which by definition do not read it.
- The only mitigation loud in *all* cases is **renaming the datasets** when the
  storage basis changes: a stale reader then gets `KeyError` immediately.

The text writers moved earlier in this branch are not exposed to this, because
they are parsed by block rather than indexed by k — the hazard is specific to
array-indexed formats.

## `qp_wfn_rotations.h5` → wedge: LANDED 2026-08-16, on a proof rather than a claim

The 2026-08-15 investigation below said DROPPED and it was **half right for a
reason worth keeping**. What follows first is what changed; the original text
is kept underneath because two of its three answers still stand.

**The lossiness argument is PATH-SPECIFIC, not general.** Re-read
2026-08-16, `sc_iteration.py:998-1003` is now a `vh.rebuild` timing comment
and the quoted sentence lives at `gw_output.py:476` — where it is about the
**one-shot** dump only. Under `config.sc_on_ibz`, which **defaults to `True`**
(`gw_config.py:888`), `final_qp_eigenstates` runs its `eigh` on the STAR
WEDGE (`sc_iteration.py:3100-3101`) and `_loop_arrays_on_full_bz`
(`:3135-3153`) obtains the full-BZ rows by `KStarMap.broadcast` — a gather
with TRS conjugation. On the default SC path the off-wedge rows **are**
reconstructible, by the very function that produced them.

**So the discriminator is the run, not the file format**, and the writer now
asks it instead of assuming either answer.
`file_io.qp_wfn.write_qp_rotations_h5` reduces to the file wedge, unfolds
straight back through `kin_ion.broadcast_ibz_to_full_bz` — the reader's own
composition — and keeps the wedge **only when the reconstruction reproduces
the arrays exactly**. `auto` falls back to full-BZ storage naming the array
that failed and by how much; `ibz` refuses instead of falling back; `full` is
byte-for-byte the old file, attrs included. A stamped file is one whose
reconstruction was checked by the process that wrote it.

**The wedge is the FILE wedge**, which is the k-set `kirr_to_kfull` already
addressed, so `kirr_to_kfull` becomes `arange(nk_red)` and both in-tree
consumers (`postprocess.rotate_wfn_to_qp:174`, `gw.eqp_bgw:986`) stay correct
with no change; the old table is kept as `kirr_to_kfull_in_full_bz`.

**One claim below is FALSE and was load-bearing: "only wedge rows are ever
read."** `tests/test_invariance_gates.py:223-230` reads the WHOLE
`E_qp_nk_rydberg` and asserts it elementwise against a frozen `(9, 46)`
full-BZ `.npy` (`gnppm_debug/eqp_rotations_fixedpoint_ref.npy`). It survives
only because `gnppm_debug`'s file wedge is 9 of 9 — the deck does not reduce —
so the shape does not move. Had that gate run on `si_cohsex_debug` (64 → 8) it
would have tripped its own shape assert. A second reader,
`tests/multi_device/sigma_omega_layout_ab.py:459-462`, also compares whole
arrays. Neither is in `src/`, and both were missed by the survey.

### The original 2026-08-15 text, kept — two of its three answers still stand

**1. Are the non-wedge `U_mnk` rows computed, or broadcast?  COMPUTED.**
Both producers do `E_qp_ry, U_qp = eigh(state.H_qp_dft)`
(`sc_iteration.py:998-1003`) over the **full-BZ** k axis — `gw_output.py:468`
says so outright: "when Σ is unfolded to the full BZ the eigh runs on
nk_full > wfn.nkpts". So every full-BZ row is an independent diagonalisation
of that k's own H, not a gather from the wedge. **Storing only the wedge is
therefore LOSSY, not redundancy removal** — and the discarded rows are not
reconstructible by any gather, because an eigenvector is defined up to a phase
and, inside a degenerate multiplet, up to a unitary mixing. Defensible while
nothing reads them (verified: nothing does), but it is a different change and
must be described as one.

> **SUPERSEDED 2026-08-16 — true of the one-shot path, false of the default
> SC path.** See above. The *caution* it expresses is right and is why the
> writer proves the round trip rather than trusting a mode flag.

**2. Does a stale reader of a newly-written file fail loudly?  NO.**
Measured with h5py on the real Si map `kirr_to_kfull = [0,1,2,5,6,7,10,27]`
against an 8-row dataset: indices 10 and 27 raise `IndexError`, but
**0, 1, 2, 5, 6, 7 — six of eight — silently return the wrong row.** On the
full BZ every index `< nk_red` (8 of 64) is silently wrong. "It will
IndexError" is false for the majority of lookups. The only mitigation that is
loud in *all* cases is renaming the datasets when wedge-stored, which gives a
stale reader `KeyError` immediately; the `k_storage` stamp does not help an
out-of-tree consumer, which by definition does not read it.

**3. How big is the saving, measured?  3.58 MB → ~0.47 MB.**
On the Si production fixture: `U_mnk {64, 60, 60}` complex128 = 3.52 MB, whole
file 3,757,812 bytes. The wedge form is `{8, 60, 60}` = 0.44 MB, whole file
≈ 0.47 MB — a **7.6× reduction saving ~3.1 MB**. The survey's "184 MB → 24 MB"
is ~50× larger than anything measurable here and was not reproduced.

### `restart_q_storage`: VERIFIED, unlike the two above

Checked 2026-08-15 because the survey had by then been wrong twice on one
entry. **This claim holds.**

The c2406 anchor run (`lorrax_mos2_12x12/run_A_c2406_b400_AF`) no longer exists
on disk, so this is DERIVED rather than disk-measured — but the derivation rests
on a convention verified empirically on a real file, and on the service's own
symmetry algebra:

- **Convention, measured**: `V_qmunu {64, 960, 960}` and `W0_qmunu {64, 960,
  960}`, complex128, shape `(nq, μ, μ)`, on the Si production fixture. Predicted
  file size from those shapes reproduces the measured 2,013,798,432 bytes to
  99.6% (the residual is HDF5 metadata and the smaller datasets).
- **Anchor size, arithmetic**: at `nq = 144`, `μ = 2406`, complex128 —
  `V_qmunu` = 13,337,478,144 B = **13.34 GB**, and `V + W0` =
  **26.67 GB**. The survey's "26.7 GB" is exact, not rounded up.
- **Reduction, computed by the service**: `find_irreducible_bz_points` on a
  12×12×1 mesh under a closed 6mm point group (order 12, closure verified)
  gives **19 irreducible q of 144 — 7.6×**, with and without TRS. So the
  wedge form is **3.51 GB**, matching the claimed "~3.5 GB".

**The 13 GB transient** behind "V's unfold after the Dyson solve" is the same
arithmetic: one full-BZ `V_qmunu` at this anchor is 13.34 GB.

So the I/O section's headline number is sound and the `_MunuSlabPlan` blocker
is worth measuring. Note what was NOT established: that the on-disk file at
that anchor actually was 26.7 GB (the run is gone), and that a production deck
still uses μ = 2406.

### `_MunuSlabPlan`: UNBLOCKED AND LANDED, 2026-08-15

The refusal is gone. The sharded BSE reader takes a q-wedge restart, unfolds
it through the same `file_io.tagged_arrays._unfold_wedge` the GW leg has used
since `536cbac9`, and `si_bse_debug`'s `restart_q_storage` pin is removed.

**Acceptance, `si_bse_debug` GW+BSE end to end at P=4, `auto` vs `full`:**

| | `full` | `auto` (wedge) |
|---|---|---|
| `isdf_tensors_480.h5` | 541,335,584 B | **130,299,936 B** (4.15×, −392 MB) |
| BSE lowest-8 eigenvalues | — | **bit-identical** |
| `eqp0`/`eqp1`/`sigma_diag` | — | **byte-identical** |

The log line to look for is
`BSE-sharded: q-WEDGE restart, unfolding 8 -> 64 q through the symmetry service`.

**Γ needs no collective.** Γ is its own orbit parent, so the single-q `V_q0`
read stays ONE hyperslab at Γ's wedge row. That is asserted against the file's
own tables — identity permutation, zero lattice wrap, not time-reversed — never
assumed, and any other single-q read on a wedge refuses by name.

**What still refuses**, and it is now about reconstructibility rather than
cost: a q extent below the k-grid with no unfold tables is a truncated or
mis-stamped file, not a wedge. Re-deriving the tables from *this* run's `sym`
is deliberately not offered — a table that reconstructs the tensor must be the
table that deconstructed it. An OVERSIZED q extent gets a message that does
**not** mention `restart_q_storage`, because no setting of it produces or fixes
that; an existing cell pins exactly this.

**The trap that would have flattered the result**, carried into the code
comment and not only the report: without `JAX_ENABLE_X64=1` JAX silently
truncates complex128 to complex64, halving the bytes actually moved while the
predicted-size arithmetic still counts 16 B/element — a clean 2× inflation of
the measured bandwidth with only a `UserWarning` to show for it.

Remaining, and neither is cost: the single-q route for a non-Γ q on a wedge
(refused by name, no in-tree caller), and `DESIGN_restart_consolidation.md`,
which four sites defer to and **is still not in the repository**.

### The original blocker, kept for the record: what it said, and what settled it

Audited by reading, 2026-08-15. **The refusal is at
`src/bse/bse_loading.py:530-556`, and its own comment says in capitals
`THE REASON IS COST, NOT CORRECTNESS`** and that the price "has never been
measured on a real interconnect". Corroborated independently at
`tests/known_failures/2026-08-11-munu-layout-fact-stated-twice.md:49`.

**Everything the wedge path needs already exists.** The writer, the pre-unfold
capture, the per-dataset `q_storage` / `qirr_closure_verdict` stamp, the
sharded `unfold_isdf_operator` — which takes and returns *exactly* the
`P(None,'x','y')` spec `_MunuSlabPlan.request` already builds — and a public
wedge-unfolding sharded reader (`tagged_arrays.read_munu_tensor_from_h5:1153`)
that the GW leg uses in production. `tagged_arrays.py:935-948` states the
premise directly: the double-gather argument "is true of SlabIO and the
conclusion does not follow", because the collective happens *after* the read,
in jax.

**Four things block it, and only one is the cost:**

1. The ~3-line insertion of the unfold between the read and the μ-major
   transpose in `_slabio_read_munu`.
2. A decision on the single-q `V_q0` route — `plan.request(q_index=0)` selects
   one flat q by hyperslab offset, and on a wedge that q lives at
   `irr_idx_q[0]` under `sym_idx_q[0]`, i.e. a per-q reconstruction, which the
   tree has ruled unavailable.
3. **`DESIGN_restart_consolidation.md`, the design every one of these sites
   defers to, IS NOT IN THE REPOSITORY.** Cited at `bse_loading.py:540`,
   `gw/restart_q_storage.py:527`, `file_io/tagged_arrays.py:945` and twice in
   the known-failures note; `find . -iname '*restart_consolidation*'` is empty.
4. The timing leg — for which **no in-tree deck drives a wedge through
   `_MunuSlabPlan` on real bytes at all.** The three cells that touch the
   refusal (`test_restart_qirr_consumers.py:400,419,433`) build the plan from
   a bare shape tuple: no HDF5, no SlabIO, no mesh. `restart_q_storage_ab.sh`
   greps for the refusal rather than timing it. The only committed
   `unfold_isdf_operator` numbers are correctness (bit-identity at 1×1/2×2/4×1
   emulated, plus four real CPU processes) — never wall time.

**The deck size does not matter, and that is why this was measurable.** The
unfold moves `C · nq_full · μ² · 16` bytes over the interconnect; the wedge
saves reading `(nq_full − nq_ibz) · μ² · 16` bytes off disk. **μ² cancels.**
The wedge wins iff

    B_net  >  [ nq_full / (nq_full − nq_ibz) ] · B_disk

and at the 7.6× reduction of the μ = 2406 anchor the bracket is **1.152**. So
the verdict is a bandwidth ratio, not a size question. Nobody needs to build
a μ = 2406 deck to settle it.

### MEASURED 2026-08-15 — and the cost basis does not survive

`unfold_isdf_operator` timed on **4 × A100, a real 2×2 device mesh over
NVLink**, complex128 (`JAX_ENABLE_X64=1`; without it JAX silently truncates to
complex64 and halves the bytes, which inflates the answer 2×), 5 reps after a
warm-up call, at the μ = 2406 anchor's own q geometry (144 q → 19, 7.6×):

| μ | full-BZ tensor | unfold | **B_net** | **B_net / B_disk** | verdict |
|---|---|---|---|---|---|
| 2048 | 9.000 GiB | 0.157 s | **57.4 GiB/s** | **19.7×** | wedge |
| 1024 | 2.250 GiB | 0.041 s | 54.3 GiB/s | 18.6× | wedge |
| 512 | 0.562 GiB | 0.015 s | 38.5 GiB/s | 13.2× | wedge |
| 256 | 0.141 GiB | 0.007 s | 20.7 GiB/s | 7.1× | wedge |

against `B_disk = 2.919 GiB/s` — the tree's own committed SlabIO phdf5 figure
at 16 ranks (`bse_loading.py:455-458`) — and a bracket of **1.152**.

**The wedge wins by 6–17× at every size tested**, and by 17× at μ = 2048,
which is within 15 % of the anchor's μ. Reproducible to ~1.5 % with the sizes
run in reverse order (that ordering check matters: run first, μ = 256 measured
6.4 GiB/s against 20.7 GiB/s run last — the difference is warm-up, and a
single-size run would have reported the low number).

Even granting disk **10× faster** than the committed figure, the μ = 2048
margin is still 1.9×. The cost argument does not hold on this hardware.

**Scope, stated because it bounds the conclusion.** Single node, four GPUs,
NVLink — which IS the geometry `_MunuSlabPlan` runs at P = 4, but a
multi-node BSE would cross Slingshot instead and is NOT covered. The shapes
are synthetic (a valid per-op cyclic permutation, zero umklapp wrap): this
measures TRANSPORT, and the table's contents change which element lands where,
not how many bytes move. `B_disk` is quoted from the tree, not re-measured.
And this is the cost only — items 1–3 above (the insertion, the `V_q0` route,
the missing design doc) are untouched, so **this does not by itself authorise
lifting the refusal.** What it removes is the stated reason for it.

**Two defects found while auditing the refusal:**

- The refusal's *message* contradicts its own comment. The comment says "cost,
  not correctness"; the string handed to the operator (`:548-552`) says the
  transport **"cannot unfold it"** — an inability claim the tree elsewhere
  states is false. The test that pins the message
  (`test_restart_qirr_consumers.py:400-417`) only checks for `"wedge"` and
  `"restart_q_storage=full"`, so it does not hold the reason honest.
- `gw_config.py:943-948` still tells deck authors "the GW restart reader
  refuses it too". That stopped being true at `536cbac9`; the GW reader has
  unfolded since. Three more places repeat the refuted double-gather premise
  as fact: `tests/multi_device/README.md:132-136`,
  `tests/multi_device/restart_q_storage_ab.py:39-45`, and
  `tests/regression/si_bse_debug/README.md:229-236`.

### Calibration on the ranked I/O list

The survey was wrong about `E_qp_nk_hartree` having "no reader at all" — it has
a **gated** one (`sigma_omega_layout_ab.py:154`, consumed at `:461` with
`required=sc`; `U_mnk` is the report-only one). It was wrong about the file
size by ~50× **on the fixture available** — though `U_mnk` scales as `nb²`, so
184 MB is not impossible on a production deck with far more bands; record it as
unreproducible here rather than as simply false.

The 26.7 GB restart figure and the 13 GB transient have since been CHECKED and
hold (above). So the survey's record is: two claims wrong on the
`qp_wfn_rotations` entry, two claims right on the `restart_q_storage` entry.
Treat every remaining unverified assertion — any "no reader" claim especially —
as a **lead to check, not a fact to act on**, and check it on the deck the
claim names rather than a convenient small one.

## The question to ask before any further wedge move

*Is there a diagnostic that exists only while the redundant copies are alive?*

For V, W0, ψ and `enk_full` the answer is **no** — their closure is measured at
generation time and stamped as `qirr_closure_verdict`.

For Σ the answer was **yes**: the star spread is the worst per-band max−min of
Re diag Σ between members of one star. (It was long believed to *measure*
centroid non-closure; measured 2026-08-15, it does not — an orbit-closed 960-set
for the Si deck moves it only 2.611 → 1.964 meV. It is a real symmetry
diagnostic whose driver on that deck is still unidentified.) Unfolding a wedge back through the service is a *gather*, so every member
would equal its parent and the spread would read 0.000 **by construction** — a
fake green, worse than no check. It is therefore measured upstream on the
full-BZ arrays and recorded into the wedge file's header, **per band**, because
the band scope belongs to the consumer (the comparison's fixture) and not to the
producer (the deck's sigma window).

## Bispinor convention audit (PHYS, source b8e036a8)

The following table is adopted verbatim from the independent PHYS report; its source locations and statuses are as-of that audit. Integration closures and changed conventions are recorded below the table.


PROVED means equality at the explicitly tested seam/fixture within the tolerance above. FLAGGED means source-derived, model-limited, or missing an independent numerical witness. Oracle names below omit the `test_` prefix and refer to the new module unless marked existing.

| Convention | Site at b8e036a8 | Status | Oracle / remaining scope |
|---|---|---|---|
| Raw/isometric hσ·p lift; positive h | bispinor_init:237 | PROVED | literal_gamma_lift_and_all_96_spatial_trs_rows |
| Γ² Hermitian but Γ²ᵀ=−Γ²; other Γ transpose signs | gamma_matrices | PROVED | literal_gamma_lift_and_all_96_spatial_trs_rows; independent Pauli |
| det(S) on small block | maps:1913 | PROVED | all96 row oracle |
| iσ_y K on both blocks; conjugated U†ΓU for TR | maps:1913 | PROVED | all96 row oracle |
| Current polar/time-odd; charge even | maps:3090,3115 | PROVED | all96 rows; nonzero_ct_covariance_recomputed_from_spinors |
| Local +k·L phase and K acting on phase | maps:1051 | PROVED | vertex_after_unfold_equals_mixed_vertices_before_unfold; parent χ/Σ glide |
| Reciprocal exp(−iSG·tnp), sphere/local gauge | maps:2061 | FLAGGED | existing parent-face gate only; no new sphere oracle |
| Vertex after unfold and complex coefficient conjugation | centroid_k_unfold:121 | PROVED | vertex_after_unfold_equals_mixed_vertices_before_unfold |
| Pair ψ†Γψ and signed C²=−Q² | core:1593 | PROVED (signed) | isdf_current_signed_normal_matrix_against_literal_pair_gram |
| Same C/Z sign cancels | gamma_double_contract; core:3280 | PROVED at algebra seam | signed_isdf_rhs_cancels_in_unregularized_fit |
| Full Z host-store/FFT sign cancellation | core:2985,3280–3315 | FLAGGED | end-to-end Z oracle missing |
| Positive ridge becomes Q−δI for Γ² | core:5148 | PROVED limitation | positive_ridge_moves_negative_gram_toward_zero; public certification/deck effect unpriced |
| χ left/right conjugation, Γ not Γᵀ, k±q order | w_isdf:416,661–666 | PROVED | all_16_chi_blocks_nonzero_ct_literal_both_orientations |
| Reverse χ ordered orientation, not 2×forward | w_isdf:98,1815 | PROVED | same non-TR χ oracle |
| Raw1/√Nₖ and downstream spin/file prefactor | w_isdf raw kernel, solve_w | PROVED for nspin1/file2 | χ + packed_dyson_order_prefactor_hermiticity_and_bare_limit; other caller conventions source-only |
| Parent χ equals explicit full children | w_isdf parent face kernel | PROVED on toy | parent_chi_equals_literal_full_k_for_all_16_blocks |
| TT Π(q)−Π(0), exact zero at Γ | w_isdf:1918 | PROVED subtraction | q_star_unfold_all_blocks_ward_contact_and_daggers |
| Γ little-group completeness of contact | w_isdf contact placement; head_correction:1343 | FLAGGED | BISP-HEAD obligation |
| Single negative TT metric and transverse projector | v_q_bispinor:270,335 | PROVED | bare_tiles_metric_sign_and_complex_hermitian_companions |
| Scalar v/Ω and full ζ†vζ tile pipeline | v_q_bispinor builder/accumulator | FLAGGED whole pipeline | per-G tensor proved; scalar quadrature/full tile accumulation shared by existing references |
| Reverse stored tile = conjugate centroid transpose | v_q_bispinor:815,831 | PROVED | complex reader companion oracle |
| Packed matrix permutation and I−Dχ order | photon_layout; w_isdf:1345,1548 | PROVED | noncommuting packed Dyson oracle; CPU LU backend |
| W_AB†=W_BA, including complex CT | solve_w; photon_blocks_full_q | PROVED | packed Dyson and q-star oracles |
| Scalar unfold conjugates once; Λ⊗Λ does not conjugate | w_isdf:2003; maps:1867 | PROVED | q_star_unfold_all_blocks_ward_contact_and_daggers |
| Nonzero CT/TC polar/TR covariance | maps:1867,3115 | PROVED | nonzero_ct_covariance_recomputed_from_spinors |
| G=ψ f ψ†; right Γ not Γᵀ | greens_function_kernel:176 | PROVED | parent_sigma_all_vertices_q_convolution_and_projection; transpose red twin |
| Σ convolution minus, k−q and1/Nₖ | cohsex_sigma:233 | PROVED | all16 parent Sigma oracle |
| COH factor −0.5 multiplies already-negative convolution | photon_sigma:182 | PROVED at kernel seam | all16 Sigma factor oracle |
| Wrapper V/W/W−V and occupied/sum windows | photon_sigma:161–182 | FLAGGED numerical assembly | source-consistent; independent complete wrapper test missing |
| Bare TT W=D gives X=SX, COH=0 | solve_w; cohsex_sigma:747–748 | PROVED limit + source assembly | packed bare-limit oracle; physical static total SX+COH |
| Dynamic current correction reported in X | sigma_dispatch:1190–1204 | FLAGGED dynamic route | source audit only, no dynamic driver oracle |
| Parent-band bra conjugated, ket unconjugated | photon_sigma:111 | PROVED | all16 parent Sigma literal projection |
| Final antiunitary band transpose after sector sum | maps:3872; photon_sigma:235 | PROVED static toy | full_band_unfold_matches_literal_sigma_on_symmetric_complete_toy |
| Screened production parent/full-k self-energy | complete driver | FLAGGED | no new fixed-main comparison |
| Density signed weights, real ψ†Γψ, TR signs | get_DFT_mtxels:165,263 | PROVED local density | density_all_currents_signed_weights_and_time_reversal |
| Density caller Ω/k factors, complete SCF feedback | qsgw_density:361 and callers | FLAGGED | local density proof does not cover every caller |
| Periodic direct TT −8πP_T/G², G0=0 | dft_operators:174; kin_ion_io:893 | PROVED | periodic_transverse_hartree_sign_projector_and_zero_mode |
| Hall −iεσ and CT/TC conjugate pairing | head_correction:162 | PROVED declared model | existing photon_head_sign_oracle, rerun final leg |
| Head S+YWZ/Ω, WZ transpose and YW orientation | head_correction:1343,1526–1556 | PROVED local declared formulas | existing head-sign/moment suite; shared sampler limitation |
| Full Ward/gauge completeness, missing current head terms | static_gauge_response module | FLAGGED model limitation | not supplied by the no-pair paramagnetic approximation |

### Integration closures (2026-09-06)

The verbatim PHYS table above describes its audit base. The end-to-end Z host-store/FFT flag is closed by `test_signed_current_z_host_store_fft_cancels_in_normal_solve`: literal pair sums prove both signed normal inputs and their cancellation. The wrapper and dynamic-X flags are closed at their actual assembly/dispatch boundaries by the additional literal oracles; these do not replace whole-driver gates.

`head_correction._photon_q0_factor_orbit` now transports each complete rank-four factor pair through `SymMaps.active_symmetry_rows` and averages its outer products. The typed Lorentz action mixes charge/current indices; `apply_band_matrix_symmetry` owns antiunitary conjugation; centroid pullbacks come from the parent plans. At Gamma no Bloch return phase remains. Averaging factors separately is wrong and has an explicit negative oracle. Both physical insertion and opt-in attribution consume the same factor orbits. The carrier retains original factors plus family-plan metadata, never a second centroid-square body.

MoS2 actual TT completion covariance improves0.556871776 to2.30561e-11; the strict1e-11 probe remains failed, while the supplied Cartesian orthogonality defect is2.695311e-11. This input-precision limit is stated rather than hidden. Bare/screened CC were also noncovariant at1.1094e-6/1.1021e-6; projection brings them below6e-16. Claim1201 prices this change at max4.647µeV and spread6.471µeV in both QP files versus labelled incumbent fixed-main. Full Ward/gauge completeness and omitted current head terms remain model limitations.

The PHYS fixed-positive-ridge limitation is repaired for the equal-current
fit schedule: `_transverse_lu_ridge` owns sign(Re tr C)δ in all four local/
distributed, hoisted/fused preparations, preserving C=sQ and Z=sRHS together.
The extended `positive_ridge_moves_negative_gram_toward_zero` oracle retains
the legacy6.0 red value and verifies the corrected0.54545 positive-Gram
solution for both input signs. Default `resolve_linalg` selects ridge for
both local and distributed layouts; charge `zeta_ridge` is separate. The
existing LU conditioning refusal is still applied to the corrected factor.
Fresh Mo202/Si235 match all90/256 QP and printed sector rows to their
Gamma-only/Si controls; CPU233 passes24tests. No general-indefinite
regularization or full gauge-completeness claim follows.


The final antiunitary band transpose is also proved for complex dynamic
frequency sums by `test_deferred_band_unfold_preserves_complex_frequency_weighted_tau`:
three complex nodes, two complex frequency weights, non-Hermitian off-diagonal
outputs, and a wrong-conjugation red twin. The transpose follows the frequency
sum. Historical Si/Mo tau captures had zero explicit HLO collectives after
the typed Green partner placement repair. The final route unfolds endpoint
faces before one Green GEMM and deletes that partner branch; native distributed
GEMM traffic remains, with the final capture recorded in the campaign report.

The production self-energy audit flag is complemented by matched-store,
matched-native-rule Si/Mo parent versus fixed-main gates (claims1239–1247),
with covariant contact and Gamma-completion prices separated in the campaign
report. P16 fresh/restart and exact seven-file replay are claims1254/1257/1258.
These do not certify the flagged complete-SCF or omitted-current model terms.

BSE selected-face transport is a file-boundary consumer of the same action:
claim1275 compares every P4 X/Y selected-face shard to the P1 canonical
reference bit-for-bit, including399-to400 zero padding. V/W remain distributed
on both processor axes. Full parent-row coverage does not imply an identity
action: the authenticated file may still use time-reversed rows.

Nonclosed centroid admission uses the same parent route with the group
restricted by `SymMaps.trivial_view()`. Its identity k/q tables describe the
computational carrier, while the original `WfnLoader.symmetry()` describes
raw file rows and their authenticated G-sphere action. Mixing those domains
causes an out-of-bounds file-energy/Hartree read or inconsistent rotation
metadata; file-output reductions explicitly use the loader's map. This is
a restriction of the computational group, not a revised physical TRS
measurement. See [the binding admission ruling](decisions.md).

| Convention | Measured class | Evidence |
|---|---|---|
| Fresh-fit mesh regrouping | Same-source P4/P16 canonical C192, distributed rank truncation at rcond1e-8 and ridge0: C_q first differs by4.3368225e-19, C+ by0.18697155, and ζ by1.70111944e-10 normalized (strict1e-13 FAIL). QP maximum0.001000004µeV; all90 sector rows printed exact. Class: floating-point normal-equation/factor regrouping, with amplification in the factor. Changed-C amplification and eigensolver grouping are not separately isolated; this is not sole-eigensolver attribution. | Sandbox claim1339; JID57988457 lx-Xg4-142358-1108522-5997 / lx-Xg4-142601-1133511-9005; DEV111/sub_12_acceptance_p16/charge_C_q.txt, charge_C_pinv.txt, charge_zeta.txt. |
