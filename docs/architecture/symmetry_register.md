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
implementation, so adding a 4-component branch is a change in two named
functions, not a search.

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

**G rotation + umklapp** (`G' = sym_mats_k @ G − kg0`): **four implementations.**
Canonical `wfn_loader/loader.py`; umklapp `maps.py get_umklapp_vector`.
Independent copy in `services/vcoul/src/vcoul/bgw_parity.py` (registered, not
fixed — the service returns no `kg0`); two more in `misc/archived_tests/`.

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
twice**. Three more copies of `Rinv·r + τ mod 1` live in the same file, plus a
byte-identical FFT-grid gather in `centroid/charge_density.py`.

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
| G rotation + umklapp | **4** | `vcoul/bgw_parity.py` |
| τ phase (G-space) | **yes** | but the 2π convention splits across 4 dividers |
| TRS conjugation predicate | 2 semantics, **no default, all 13 call sites explicit** | nothing — AST-swept |
| r-grid image / source map | **~6 expressions** | forward/pull-back must be upgraded together |
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
the compared bands and 41.34 → 39.88 meV over the full window — while making
BerkeleyGW agreement ~35× worse (sigTOT MAE 0.4329 → 14.9426 meV). Forcing
closure did not fix the symptom it was believed to cause.

### The conditioning synthesis — hand this to whoever picks up the count question

Three independent measurements in this tree point at ONE mechanism, and it is
**conditioning, not geometry**:

1. The Si deck carries ~537–588 G per k, so **960 centroids is ~1.7×
   over-complete** and 144 is ~0.25×. The 144 set measures a star spread of
   exactly 0.000; both 960 sets measure ~40 meV, closed or not.
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
