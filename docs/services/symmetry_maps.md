# symmetry_maps — k-grid reduction, star maps, unfolds, and the TRS measurement

`services/symmetry_maps/`. Independently installable (`pyproject.toml`,
src-layout); depends on `lxkit` + jax + numpy and nothing else in LORRAX.
`import symmetry_maps` does import jax — the unfold machinery is written in
it — but the TRS *measurement* (`check_density_symmetries`) keeps jax lazy
and runs on stdlib + numpy until its FFT step.

## Purpose

One door for everything symmetry: the IBZ ⇄ full-BZ tables (`SymMaps`), the
k-star index map for band-index quantities (`KStarMap`, `star_select` /
`star_broadcast` / `star_spread`), the sharded q-axis unfolds (`unfold_v_q`,
`unfold_v_q_bispinor_lorentz`), the ψ-unfold antiunitary rule (`unfold_psi`,
`trs_augment_U`, `tau_phase_row`), the real-space orbit machinery
(`compute_centroid_sym_perm` and friends), and the time-reversal
MEASUREMENT (`check_density_symmetries`).

Before this service the same machinery was three modules in two packages —
`common/symmetry_maps.py`, `centroid/orbit_syms.py`,
`common/density_symmetry_check.py` — consumed by 40 files, and its worst
failure class was invisible by construction: a conjugation predicate
applied against the wrong reference row costs **183.61 eV** in off-diagonal
Σ while every diagonal observable — norm, hermiticity, trace, electron
count, the eqp columns — stays *exactly* unchanged (`27cc885`). Nothing in
the suite could see it because nothing in the suite looked off the
diagonal. The service exists so that predicate lives in ONE place
(`_star_conj_flags`, the XOR), is read by four entry points, and is pinned
by tests that CONSTRUCT the disagreement rather than assume it.

The service is the door. `import symmetry_maps` and use top-level names;
`from symmetry_maps.maps import …` from outside the package is a layering
failure that `tests/test_layering.py` fails on (with a red twin). Since
2026-08-07 `upward_edges()` skips an edge whose target is a service's
top-level package — the door rule made structural, not an exception entry —
so reaching *past* the door is the thing that still flags.

## API

| name | what it is |
|---|---|
| `SymMaps(wfn)` | The eager table builder. Symmetry ops (spatial + TRS-augmented halves), `irr_idx_k`/`sym_idx_k`, the q maps, `R_cart` / `R_cart_forward` / `R_proper`, `U_spinor`, umklapp vectors. Reads 11 header attributes of `wfn`, including the MEASURED `trs_holds`. |
| `KStarMap(irr_idx, sym_idx, n_sym_spatial)` / `.from_sym(sym, nss)` | Band-index IBZ ⇄ full BZ. Bundles the three arrays that always travel together so no call site can supply two of the three. `.identity(nk)` is the no-reduction map, so a driver reads the same whether or not symmetry is in use. `select` / `broadcast` / `spread` / `spread_rel`. |
| `star_select(A_full, irr_idx_k)` | Keep one row per star — the FIRST occurrence, in full-BZ order, never `np.unique`'s ascending label order. |
| `star_broadcast(A_irr, irr, sidx, nss, *, trs_reference='star_row')` | IBZ → full BZ with the conjugation predicate. Two legal values, one per operand flavour; an unknown value RAISES rather than defaulting. The default is `'star_row'` — see Contract for why the one `'ibz_slab'` caller passes it as a literal anyway. |
| `star_spread(A_full, irr, sidx, nss)` | The one diagnostic that sees a gauge or conjugation mismatch. Hermiticity, norms and electron counts all survive one. |
| `unfold_v_q(V_q_ibz, *, irr_idx, sym_idx, sym_perm, L_table, q_irr_frac, mesh_xy, n_sym_spatial)` | Sharded centroid double-gather + umklapp L-phase + TRS conj. `shard_map` + paired `all_to_all`, 1× single-tile peak memory per rank. |
| `unfold_v_q_bispinor_lorentz(...)` | The 3-vector Lorentz mixing on the bispinor TT block. Its `R_proper_table` operand is in a convention the §A5 formula compensates for — see Antipatterns. |
| `unfold_psi(cnk_kbar, *, sym_idx, g_kbar, sym_mats_k, translations, U_spinor_spatial)` | The (★) ψ derivation: spinor rotation, τ phase, G-list negation, TRS conjugation. Hard-raises unless `len(sym_mats_k) == 2·len(U_spinor_spatial)`. |
| `slice_q_full_to_ibz` | Full-BZ → IBZ q-axis gather, sharding-preserving and jit-cached. |
| `trs_augment_U`, `tau_phase_row`, `kgrid_shift_map`, `find_irreducible_bz_points` | The pure-numpy primitives. `find_irreducible_bz_points`' anchored branch reproduces `find_symmetry_ops_simple`'s op-selection policy bit-for-bit — deliberately (see Contract). |
| `compute_centroid_sym_perm`, `compute_rgrid_sym_perm`, `build_real_space_syms`, `orbit_images`, `canonicalize_orbit`, `unfold_orbit_unique_with_id`, `recover_symmorphic_density_point_group` | Real-space orbit machinery. `compute_centroid_sym_perm(validate=True)` REFUSES a non-orbit-closed centroid set and names the regeneration fix. |
| `centroid_set_hash` | Canonical, order-sensitive centroid identity: exact FFT-grid indices (`g:`) when a grid is supplied, wrapped fractional coordinates (`f:`) otherwise. |
| `check_density_symmetries`, `cached_density_symmetry_check`, `DensitySymmetryReport`, `trs_check_mode` | The TRS measurement — the only MEASUREMENT, as opposed to inference, in the symmetry stack. Quadrature is injectable (`valence_density_fn`, `spin_degeneracy_fn`); `None` preserves the lazy `psp.get_DFT_mtxels` import verbatim for in-tree callers. |

`SymMaps.validate_kgrid_unfolding` is public surface on the class and is
kept rather than deleted, with the case where it returns FALSE constructed
(a corrupted table) — a check that cannot fail is not a check.

## Contract

* **`_star_conj_flags` is the single source of conjugation truth.** One
  predicate — `trs(member) XOR trs(reference_row)` — read by four entry
  points: `star_broadcast`'s `star_row` branch, `star_spread` via
  `_spread_tables`, and `KStarMap` twice. No site inside the package
  re-derives it, and nothing outside imports it. The hand-rolls that remain
  in `src/` are registered to their owners, not copied in here.
* **`trs_reference` names the operand flavour, and the one caller that needs
  the non-default value says so in the source.** `"star_row"` means
  `A_irr`'s rows are the values at the KEPT FULL-BZ rows (what
  `star_select` returns), so the conjugation is the XOR. `"ibz_slab"` means
  `A_irr` is the raw IBZ slab, read with no symmetry operation applied, so
  every row is TRS-false by construction and the predicate is the member's
  own flag. The two coincide only while every star's first full-BZ row is
  spatial, which is a property of the op-selection policy and not of the
  physics. Getting the pairing wrong costs ~0.4–0.6 relative on real decks
  (measured: gnppm `kin_ion` 3.7e-16 right vs 5.6e-01 wrong) and is
  entirely off-diagonal in Σ. The parameter HAS a default (`"star_row"`,
  right for what `star_select` hands back), so a caller on the other
  flavour is one omitted keyword away from the 183.61 eV — which is why
  `gw/kin_ion_io.py` passes the literal and
  `tests/test_kin_ion_star_broadcast.py` asserts by AST that it is a string
  CONSTANT, not a variable and not a conditional. An unknown spelling
  raises with both legal values named; it never falls through.
* **The op-selection policy is bit-frozen.** `find_symmetry_ops_simple` has
  no `break` — the HIGHEST matching `ikbar` wins, then the lowest sym — and
  `find_irreducible_bz_points`' anchored branch shadows it bit-for-bit.
  Changing either moves eqp by up to 15.9 eV in the V_H column downstream
  (measured, `3e002f2`) and is an OWNER decision. Two tripwires fail loudly
  if it drifts: the four-deck `(irr_idx_k, sym_idx_k)` bit-equality test,
  and the cohsex TRS-first-row precondition assertion (which FAILS rather
  than skips).
* **Refusals are part of the API.** TRS-disallowed construction names
  `noinv=.true.` and the `LORRAX_TRS_CHECK=0` escape hatch; the
  orbit-closure refusal names the regeneration fix; `unfold_v_q`'s four
  shape refusals exist because `promise_in_bounds` gathers clip SILENTLY on
  an out-of-bounds index, so an unrefused shape is a wrong answer rather
  than an error.
* **`nspinor = 2` means NONCOLLINEAR, not spin-orbit.** `SymMaps` and
  `unfold_psi` branch on the spinor axis, never on SOC.
* **The TRS check is deliberately non-circular.** It uses no `unfold_psi`,
  no `SymMaps`, no `U_spinor`, no τ: it compares two real-space densities.
  A bug in the τ phase, in `U_spinor`, or in the umklapp vector cannot move
  its verdict. That is what makes it a measurement.
* Env surface: `LORRAX_TRS_CHECK` (`1` | `0` | `strict`), `LORRAX_TRS_TOL`,
  `LORRAX_TRS_SPATIAL_TOL`, `LORRAX_TRS_MAX_K` — all in
  `docs/dev/env_vars.md`. Env grants and tunes the measurement; it never
  selects a symmetry convention.

## Backends

Pure jax + numpy. No vendor library, no `.so`, no `dlopen` anywhere in the
package — the whole "which backend" question that `distrib_la` exists to
answer does not arise here, and this section is short because of it.

The mesh-touching paths (`unfold_v_q`, `slice_q_full_to_ibz`, the star
helpers on device operands) go through the package's PRIVATE `_shard_map`,
a ruling-3 copy of `common/shard_map.py`. It PROBES for the two spellings
(`jax.shard_map`, `jax.experimental.shard_map`), announces which one it
took, and REFUSES rather than degrades when a jax has neither — every
distributed kernel in the tree is written in `shard_map`, so there is no
meaningful fallback to fall back to. Exercised on both stacks this branch
runs: jax 0.9.1 in the WSL venv and jax 0.7.0 in the Perlmutter container,
which announces `using jax.shard_map (jax 0.7.0)` on every rank of the
four-process legs. Consolidation into `lxkit.jax_compat` retires this copy
and `distrib_la`'s together, post-wave.

Operand placement is the real dispatch: host operands take numpy fast paths
(`_take_rows`, `_broadcast_rows`), device operands get cached jits with
explicit out-shardings, and a sharded `jax.Array` never crosses to the host
to be indexed.

## Tests

`services/symmetry_maps/tests`, markers `services` + `symmetry_maps`.
**165 cells, and every one of them runs on a laptop in 18 seconds** —
164 passed, 1 xfailed, nothing skipped.

| tier | file | cells | needs |
|---|---|---|---|
| L-a star contract | `test_symmetry_maps_star_contract.py` | 32 | nothing |
| L-a algebra + primitives | `test_symmetry_maps_algebra.py` | 31 | nothing |
| L-a `R_cart` / `R_proper` contract | `test_symmetry_maps_r_cart.py` | 17 | nothing |
| L-a+ deck tables | `test_symmetry_maps_deck_tables.py` | 34 | h5py + the four in-tree WFN headers |
| L-b emulated mesh | `test_symmetry_maps_emulated_mesh.py` | 17 | `XLA_FLAGS` set by the SERVICE conftest; **skips**, never asserts, below 4 devices |
| L-c real multi-process | `test_symmetry_maps_multiproc.py` | 11 | `srun -n 4`; shared `check_*(mesh, …)` bodies + a `__main__` CLI (`_CLI_CELLS`) |
| import isolation | `test_symmetry_maps_import_isolation.py` | 9 | `python -S` subprocess; `sys.modules` AND `sys.path` asserted, plus a red twin and a with-lorrax-still-passes |
| skip honesty | `test_symmetry_maps_skip_honesty.py` | 14 | a machine profile. Four MUST rows — the four in-tree decks, the `cohsex_debug` density fixture, `h5py`, and four forceable devices. ABSENT = skip, PRESENT-AND-BROKEN = **FAIL** |

What the star tier does that the old gate did not:

* **Hand-typed, production-confirmed tables only.** gnppm's and cohsex's,
  re-derived from `SymMaps(wfn)` on 2026-08-07 and committed as
  `tests/data/star_tables_e9340d1.json`. A table DERIVED from a generated
  grid cannot carry these tests — see Antipatterns — and that trap is
  itself pinned by a meta-test asserting a lex-min-derived grid has ZERO
  TRS-first stars.
* **Every table-driven cell carries the anti-tautology assertion.** The
  `star_row`/`ibz_slab` disagreement count must be 8 (gnppm) resp. 6
  (cohsex) or the test FAILS. A test that would pass on a deck where the
  two predicates agree is not testing the predicate.
* **`SymMaps` on h5py header stubs of all four in-tree decks**, bit-compared
  to the committed tables, with a WfnLoader parity arm that re-derives them
  through the PRODUCTION loader. That arm skips on a standalone install
  (there is no lorrax to import `file_io` from — the quarantine working),
  and skip-honesty makes that legal only because the allowlist NAMES the
  leg that runs it: the monorepo run from the checkout root. A skip with no
  covering leg named is evaporated coverage and fails the gate.
* **Hostile geometry is mandatory**: `n_rmu % (Px·Py) != 0` must refuse, and
  the red twin constructs the case.

Integration tests stay on the lorrax side, because what they pin is the
call site: `tests/test_kin_ion_star_broadcast.py` pins the 183.61 eV class
end-to-end against the committed `kin_ion.h5` references (including an AST
assertion that the call site passes the `"ibz_slab"` literal), and
`tests/test_star_offdiag_gate.py` is the off-diagonal-sensitive symmetry
gate that the diagonal BGW-anchor metric structurally cannot be.

Run it standalone: `pytest services/symmetry_maps/tests` (never loads
`tests/conftest.py`). From the monorepo: `pytest -m symmetry_maps`.
Deselect: `--no-services` / `--only-service=symmetry_maps`, never a second
`-m` (`pyproject` sets `addopts = "-m 'not extra'"` and an explicit `-m`
REPLACES it, silently re-enabling 26 deselected suites).

## Performance

**Baselines are `services/symmetry_maps/bench/baselines/`**, written by
`bench/bench_symmetry_maps.py` in `distrib_la`'s claims format (op,
backend, shape, mesh, seconds per row; nodes, jobid, machine per file) —
`wsl1x1.json` (`SymMaps` construction on all four decks, the four star
entry points host **and** device at nk=64/nb=60, `unfold_v_q` at 1×1) and
`wsl2x2.json` (`unfold_v_q` on an emulated 2×2); the Perlmutter legs are
run separately against the same schema, and the WSL rows are an
under-load band, not a floor — the driver's module note has the numbers.
Everything below is measured and is quoted, not re-run.

**Suite cost**, both machines, HEAD `5daf979`:

| leg | result | wall |
|---|---|---|
| WSL venv, jax 0.9.1, `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1` | 164 passed / 1 xfailed | 18.2 s |
| Perlmutter container, jax 0.7.0, `lx test` | 163 passed / 1 skipped / 1 xfailed | 42.58 s |

The one skip is the WSL-kernel row in skip-honesty, correctly refusing to
assert about a machine it is not on; the xfail is the sharded-NaN
reduction (see `tests/KNOWN_FAILURES.md`). The two legs differ by one cell and by
2.3× in wall time; where that time goes has not been measured and this page
does not guess. The same Perlmutter leg read 39.83 s one commit earlier
(`1e90726`) on identical cell counts, so quote the band, not a single
second.

**`SymMaps(wfn)` construction**, before/after the dead parent-map drop
(`1e90726`; 9 runs per deck, best-of, one process per arm, uncontended,
`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`):

| deck | before (ms) | after (ms) | speedup |
|---|---|---|---|
| bispinor_debug | 1.74 | 1.33 | 1.30× |
| cohsex_debug | 2.11 | 1.44 | 1.46× |
| gnppm_debug | 1.75 | 1.32 | 1.33× |
| si_cohsex_debug | 78.28 | 44.27 | 1.77× |

Si is where the retired loop cost something real — 64 full k × 96 sym rows
× 8 IBZ k in python, 34.0 ms of the 78.3. `(irr_idx_k, sym_idx_k)` is
bit-identical before and after on all four decks; this sits on the 15.9 eV
op-selection axis, so the tables were re-derived rather than assumed.

**Real four-process legs** (`srun -n 4`, Perlmutter, jax 0.7.0,
`JAX_PLATFORMS=cpu`), at 2×2 and 4×1 — the first time these
bodies ran on genuinely separate processes rather than an emulated mesh:

| cell | 2×2 | 4×1 |
|---|---|---|
| `unfold_v_q` vs the hand reference | 4.22e-17 rel | 9.43e-17 rel |
| hostile extent (`n_rmu=10`) | refused | refused |
| star select/broadcast/spread, device vs host | bit-identical, sharding kept | bit-identical, sharding kept |
| `spread_rel` on a NaN-poisoned sharded operand | `nan` (propagates) | `nan` (propagates) |

That last row is load-bearing and is NOT what the emulated mesh returns —
see the KNOWN-issues register.

`spread_rel` is the caller-facing spread: one reduction and one 16-byte
transfer for a device operand, against two full host readbacks for the
naive form. The index tables (`irr_idx_k`, `sym_idx_k`, the row tables
built from them) are `n_k` integers and stay on the host; the array operand
is `(n_k, nb, nb)` complex128 — 9.2 GB at nk=144/nb=2000, four times per SC
iteration — and is the thing every helper is written not to move.

## Antipatterns

* **Rotating a Cartesian index with `sym.R_cart[s]` untransposed.**
  `R_cart` is the INVERSE Cartesian rotation, because `mtrx` is the inverse
  real-space rotation while `mtrx.T` is what acts on k and G. Norms,
  hermiticity and traces all survive the mistake. Use `R_cart_forward` for
  anything rotating a rank ≥ 1 Cartesian index. Two consumers are exempt
  and each says why in place: `get_spinor_rotations` (its transposed
  Shepperd form cancels the inversion) and `unfold_v_q_bispinor_lorentz`
  (its §A5 contraction compensates internally, and since 2026-08-07 its
  docstring — not a comment 800 lines away — says so).
* **Re-deriving the conjugation predicate.** `sidx >= n_sym_spatial` is the
  member's OWN flag, correct only for the raw-IBZ-slab operand flavour.
  Against a star-row reference it INVERTS the rule for every star whose
  first member is a time-reversal row — on gnppm that is 4 of 5 stars and
  8 of 9 rows.
* **Using a diagonal min/max spread as a symmetry gate.** The BGW-anchor
  `_star_spread` in `tests/harness.py` is a real-diagonal metric;
  conjugating an entire star member moves it by EXACTLY 0.0 (measured live
  on the cohsex fixture: 1.2130460739135742 both ways). It is the right
  check for anchor agreement and the wrong one for conjugation errors —
  that is `tests/test_star_offdiag_gate.py`'s job, and the red twin asserts
  both halves at once.
* **Deriving star-test tables from a generated grid.** Lex-min orbit
  representatives are always spatial, so a derived grid has no TRS first
  row and every discriminating test silently becomes a tautology. This is
  the trap that makes a green symmetry suite worthless.
* **Testing symmetry on Si only.** Si has 0 TRS rows at all 64 k; every
  antiunitary branch is dead there. A suite green on Si alone proves
  nothing whatever about time reversal.
* **Regenerating a centroid set to make the orbit-closure refusal pass.**
  The production sets' non-closure is a measured, owner-scoped fact
  (`centroids_frac_960`: 2.611 meV star spread; cohsex's
  `centroids_frac_60`: spatial star relations broken at 1.8e-01–3.9e-01 in
  Σ_SX and V_H while the TRS-conjugation relations hold at ≤ 7e-4).
  Regenerating means re-freezing the BerkeleyGW anchor. Tests that need a
  non-closed set construct one SYNTHETICALLY, by dropping a centroid from a
  closed one.
* **Reading the theory out of this page.** It is the service's contract,
  not the conventions. `docs/theory/symmetry.md` is the consolidated
  derivation — BGW `mtrx`/τ conventions, the ψ-unfold algebra, the
  (α, L) decomposition — and this page deliberately does not duplicate it.

Cites: `docs/theory/symmetry.md` (the conventions),
`docs/architecture/layers.md` (R3's dissolution and the service-door
carve-out), `tests/KNOWN_FAILURES.md` (the two open rows this service
owns), and the commit messages `27cc885` / `3e002f2` / `f7ef931` /
`061f8a3` for the measured history — with one erratum, recorded here
because history is not rewritten: `27cc885`'s message claims gnppm and
bispinor have all-singleton stars and therefore an identically-zero XOR.
Re-derived 2026-08-07 from `SymMaps(wfn)`, both decks have 5 stars
`[1,2,2,2,2]` with 4 TRS first rows and 8/9 predicate disagreement — the
STRONGEST in-tree discriminators, not no-ops. The fix that commit landed is
unaffected; the corrected tables live in
`services/symmetry_maps/tests/data/star_tables_e9340d1.json`.
