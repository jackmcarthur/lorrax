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

## 5. Time reversal — the most fragmented operation

Five predicates, of which **two are genuinely different and the difference is
priced at 183.61 eV**:

- `trs_reference="star_row"` → XOR of two flags (`_star_conj_flags`) — for
  operands that came from `star_select`.
- `trs_reference="ibz_slab"` → the member's own flag — for a wedge slab read
  verbatim off disk.

They disagree on 6 of 9 k-points on `cohsex_debug`, with the **real diagonal
left exactly intact** — which is why nothing caught it for a month. The other
three spellings (`sym_idx >= n_sym_spatial` at ψ level, in three places) are
consistent.

`tests/multi_device/star_invariance_gate.py` uses the **wrong one** — registered.

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
| TRS conjugation predicate | 5 spellings, **2 semantics** | `star_invariance_gate.py` uses the wrong one |
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
