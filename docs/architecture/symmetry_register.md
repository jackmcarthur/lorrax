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

**Bispinor (4-component) rotation does not exist.** `get_spinor_rotations`'s
docstring promises 4×4 for four-component states; the code unconditionally
allocates 2×2. **This is the upgrade path the register exists to protect**: the
SU(2) construction and the TRS augmentation each have exactly one
implementation, and both named unfolds share one backend, so adding a
4-component branch is a change in a few named functions rather than a search.

**Bispinor is explicitly OUT OF SCOPE** (owner ruling, 2026-08-15). This entry
is a note for whoever does that work, not a task. In particular
`tests/regression/bispinor_debug/centroids_frac_256.txt` is **not** to be
regenerated — it stays exactly as it is, keep-with-justification, recorded
under the non-closed-set dispositions below.

## 2. Proper-rotation channel mixing (the bispinor-adjacent one)

`maps.py` `mix_channels_by_proper_rotation`, one live caller
(`gw/v_q_bispinor.py`). `R_proper` built once in `SymMaps.__init__`.

Two **recorded, unreconciled** convention disagreements against the offline
fixture `reports/bispinor_ibz_2026-05-16/cri3_R_proper.npz`: it is transposed
row-for-row, and its TRS half carries `−R` where live code carries `+R`. The
live choice is justified only for the tiles currently stored — a future
off-diagonal `(0,i)` tile would need the −1.

## 3. Wavefunction / orbital rotation

**ψ coefficient transform.** Order is `conj (if TRS) → τ phase → U`.
Two implementations of that *sequencing* — host `maps.py unfold_psi`, device
`wfn_loader/loader.py` — which **agree**, and both import the *factors*
(`tau_phase_row`, `spinor_rotation_for_sym_row`) rather than rebuilding them.
This is the one genuine duplication in rotation math and it is minimal: two
lines of ordering, no math.

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

### The one deck that runs it, and the one that cannot

`tests/regression/gnppm_debug/gnppm_sc.in` is the **only** committed deck with
`qp_solver = self_consistent`. Before it, every deck was `one_shot_dft`, which
is why this rotted invisibly: flipping the default would have changed no suite
result.

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
| ψ unfold sequencing | 2 (agree) | order-of-ops must be edited twice; factors are shared |
| Proper-rotation channel mix | **yes** | the CrI₃ fixture is transposed + TRS-sign-flipped |
| G rotation + umklapp | **3 live** (was miscounted as 4) | `vcoul/bgw_parity.py` (blocked) and **`file_io/epsreader.py:136`**, which was unregistered and is τ-blind by its own admission |
| τ phase (G-space) | **yes** | but the 2π convention splits across 4 dividers |
| TRS conjugation predicate | 2 semantics, **no default, all 13 call sites explicit** | nothing — AST-swept |
| r-grid image / source map | **12 expressions** (was miscounted as ~6) | forward/pull-back must be upgraded together — AND the four in `centroid/kmeans_isdf.py`, the module that GENERATES the centroid sets |
| Density symmetrisation | 2 (**one broken**) | the bench copy |
| Cartesian-index rotation | `R_cart_forward` named but **unused**; the one live site uses `R_cart` untransposed | — |

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
| `isdf_tensors.h5` `V_qmunu`/`W0_qmunu` | **full BZ by default** | wedge machinery exists and is tested; default is `"full"` |
| `isdf_tensors.h5` `psi_full_y` | **full BZ, no option** | |
| `qp_wfn_rotations.h5` | **full BZ** | only wedge rows are ever read |
| `dipole.h5` | **full BZ** | |
| `v_q_bispinor.h5` | **full BZ always** | unfolds before writing |

---

## Non-closed centroid sets in the tree: KEEP, and why

Four non-closed sets remain, deliberately. **Do not re-open the sweep.**

| set | worst | why it stays |
|---|---|---|
| `si_cohsex_debug/centroids_frac_960.txt` | 1.318e-01 | `test_qgrid_symmetry_resolution.py`'s `_OPEN_SET`, paired against the 144 set as `_CLOSED_SET` — the pair *is* the thing under test |
| `si_bse_debug/centroids_frac_480.txt` | 1.718e-01 | measured specimen pinned by `test_symmetry_maps_closure.py` and `..._qgrid_resolution.py` (47/48 ops); the deck itself already uses the closed twin |
| `cohsex_debug/centroids_frac_60.txt` | 2.762e-01 | `test_star_offdiag_gate.py` asserts its consequence as a fact; also the deck behind ~12 test files incl. `conftest.py` |
| `bispinor_debug/centroids_frac_256.txt` | 1.436e-01 | the documented full-BZ fallback the bispinor fixture exercises; regenerating changes a frozen reference — **open owner question** |

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

## `qp_wfn_rotations.h5` → wedge: DROPPED

Investigated 2026-08-15, **not implemented, and not to be re-opened off the
survey's ranking**.  Lossy, silently corrupting for six of eight in-range
indices, and worth ~3 MB. Recorded because each answer
contradicts what the I/O survey asserted, and because the same three questions
apply to every remaining wedge candidate.

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

### The `_MunuSlabPlan` blocker: what it says, and what would settle it

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
