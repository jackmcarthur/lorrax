# symmetry_maps — k-grid reduction, star maps, unfolds, and the 2c TRS check

`services/symmetry_maps/` is the one door for crystal symmetry: the
IBZ ⇄ full-BZ tables (`SymMaps`), the band-index star map (`KStarMap`,
`star_*`), the sharded q-axis unfolds, the ψ-unfold antiunitary rule, the
real-space orbit machinery, the q_irr restart store, and the time-reversal
measurement. It is independently installable (src-layout); it depends on
`lxkit`, jax and numpy (h5py lazily, inside the two q_irr file functions) and
on nothing in LORRAX. Import top-level names only: `from symmetry_maps.maps
import …` from outside fails `tests/test_layering.py`.

The conventions (BGW `mtrx`/τ, the ψ-unfold algebra, the (α, L)
decomposition) are derived in [`theory/symmetry.md`](../theory/symmetry.md);
the cross-object conventions and design reasoning are in
[`architecture/symmetry_register.md`](../architecture/symmetry_register.md).
This page is the service contract.

## API

| name | contract |
|---|---|
| `SymMaps(wfn)` | The canonical operation source. Requires `wfn.trs_holds`, which becomes `trs_allowed`; a missing verdict refuses, and `allow_trs=` refuses by name. `operation_rows` returns reciprocal rotation, Seitz translation and the antiunitary bit; `cartesian_action(rows, *, axial, time_odd)`, `lorentz_action`, `fft_grid_pullback`, `spinor_action`, `reciprocal_phase` and `unfold_wavefunction` act through the same typed rows. `validate_kgrid_unfolding` checks the tables. |
| `read_qe_symmetry_receipt`, `bind_qe_symmetry_receipt`, `resolve_qe_symmetry_binding`, `discover_qe_schema_paths`, `qe_xml_seitz_to_bgw` | Bounded `data-file-schema.xml` reader and WFN authentication: matrix orientation, Seitz translation, k rows/grid, spinor count, per-operation antiunitary bits, and QE's pure-TR k-reduction permission. |
| `build_spatial_operator_tables(wfn)` | `mtrx.T`, translation, Cartesian and spinor tables without a k-map, so the 2c check can measure a reduced WFN that `SymMaps` would refuse. |
| `KStarMap(irr_idx, sym_idx, n_sym_spatial)`, `.from_sym(sym, nss)`, `.identity(nk)` | The three star arrays bundled so no caller supplies two of three. `select`, `broadcast`, `spread`, `spread_rel`. |
| `star_select(A_full, irr_idx_k)` | One row per star: the **first** occurrence in full-BZ order, never ascending label order. |
| `star_broadcast(A_irr, irr, sidx, nss, irr_labels=None, *, trs_reference, trs_rule='conj')` | IBZ → full BZ with conjugation (or transpose) on time-reversed rows. `trs_reference` is required: `'star_row'` or `'ibz_slab'` (§ Contract). |
| `star_spread(A_full, irr, sidx, nss)` | The residual that sees a gauge or conjugation mismatch; norms, hermiticity and electron counts do not. |
| `unfold_file_wedge_to_full_bz`, `unfold_star_wedge_to_full_bz`, `unfold_file_wedge_polar_matrix`, `unfold_file_wedge_band_operator(sym, data, *, trs_rule)`, `reduce_full_bz_to_file_wedge`, `star_tables_of` | The named unfolds, taking a `SymMaps` so a driver never holds index tables. The FILE wedge (`wfn.kpoints`, `nk_red`, what BerkeleyGW files use) and the STAR wedge (one row per orbit) differ in size on most decks. `unfold_file_wedge_band_operator` takes `trs_rule='conj'` for an observable or `'transpose'` for an operator that transforms like G (Σ_{Θk,mn} = Σ_{k,nm}). |
| `directed_edge_orbit_table`, `q_stencil_orbit_table`, `apply_band_matrix_symmetry` | Pure-array edge and q-stencil orbit tables and the one band-matrix symmetry action, § [Directed band-matrix edges](#directed-band-matrix-edges). |
| `unfold_isdf_operator(V_q_ibz, *, irr_idx, sym_idx, sym_perm, L_table, q_irr_frac, mesh_xy, n_sym_spatial, trs_rule='conj', ...)` | Centroid-indexed operator IBZ → full BZ: double centroid gather, umklapp L-phase, antiunitary conjugation. `shard_map` with paired `all_to_all`; one single-tile peak per rank. |
| `unfold_operator_local(...)` | The manual-mode body of the same unfold for a caller already inside an `('x','y')` `shard_map`: one local parent tile in, the rank's full-k tile out. Endpoint tables are runtime operands, so changed tables reuse one executable. Both endpoints must be orbit-packed (the caller's plan certifies it); `left_mesh_axis`/`right_mesh_axis` name the axis whose table shard this rank holds, `None` meaning already local. |
| `unfold_isdf_one_leg`, `isdf_one_leg_source_slots`, `unfold_spin_centroid_operator`, `open_spin_block_coefficient(U, a, b)` | One-leg (parent-G relabelling, scalar/polar transport) and spin-operator unfolds; `open_spin_block_coefficient` gives one block `coef[k,c,d] = U_k[a,c]·conj(U_k[b,d])` of `U O U†`. |
| `mix_lorentz_blocks(blocks, *, sym, sym_idx, mesh_xy, keys=None)` | Mixes charge/current sectors by Λ⊗Λ from `SymMaps.lorentz_action`; callers supply no rotation convention. `sym_idx` is host metadata. |
| `unfold_wavefunction_local(psi_parent_local, *, irr_idx, sym_idx, k_irr_frac, local_perm, L_table, spin_action_full, n_sym_spatial, spin_axis, mu_axis, mesh_axis=None)` | Children on one μ-local slab: $\psi_{g\bar k,a}(\mu) = \sum_c U_g[a,c]\,T_g\!\left[e^{2\pi i \bar k\cdot L_{g,\mu}}\psi_{\bar k,c}(\alpha_g\mu)\right]$, $T_g$ conjugating on antiunitary rows; the spin table is per full-k row. |
| `certify_endpoint_locality`, `endpoint_panel_cost`, `unfold_endpoint_panel` | Bounded endpoint-panel unfolds, § [Endpoint panels](#endpoint-panels). |
| `unfold_psi`, `spinor_rotation_for_sym_row`, `apply_spinor_rotation`, `tau_phase_row`, `tau_phase_row_jax`, `unfold_reciprocal_carriers` | Pure-array ψ(G) unfold: spinor rotation, τ phase, G-list negation, TRS conjugation. `unfold_psi` refuses unless `len(sym_mats_k) == 2·len(U_spinor_spatial)`. |
| `slice_q_full_to_ibz` | Full-BZ → IBZ q gather, sharding-preserving, jit-cached. |
| `kgrid_shift_map`, `bgw_signed_q_representative`, `bgw_integer_q_to_fractional`, `q_negation_index`, `common_uniform_grid_indices`, `find_irreducible_bz_points`, `map_full_kpoints_to_irreducible` | Pure-NumPy grid algebra. The mapping routines share the highest-parent / lowest-operation rule; `map_full_kpoints_to_irreducible` returns a coverage mask so incomplete metadata refuses before an index table is used. |
| `real_space_action_tables`, `centroid_source_map_and_wrap`, `fft_grid_pullback_perm`, `grid_point_image_perm`, `orbit_images`, `canonicalize_orbit`, `unfold_orbit_unique_with_id`, `permutation_orbit_labels`, `real_space_orbit_labels`, `r_action_forward`, `snap_to_grid_and_split_wrap`, `project_polar_fft_field` | Real-space orbits. `centroid_source_map_and_wrap` returns a SOURCE map plus lattice wrap; `fft_grid_pullback_perm` returns a PULL-BACK permutation; they point in opposite directions. `real_space_orbit_labels(sym_matrices, translations, fft_grid)` builds orbit labels one operation at a time in O(n_rtot) host memory. |
| `recover_atomic_space_group`, `recover_symmorphic_density_point_group` | Centroid-only Seitz rows from lattice, positions and species; they do not authorize electronic reductions. |
| `verify_centroid_orbit_closure`, `CentroidClosureVerdict`, `resolve_qgrid_symmetry`, `QgridSymmetryResolution` | Orbit closure as a measurement (by how much, on which operations), and the one q-grid decision (verdict → `"ibz"`/`"full_bz"` → tables → reason). The resolution composes its announcement; the process running the deck prints it. |
| `write_qirr_tensor`, `read_tensor`, `read_tables`, `allocate_qirr_placeholder`, `QirrTables`, `QirrHeader`, `validate_qirr_tables`, … | q_irr restart tensors: the pre-unfold wedge on disk with its tables, unfolded on read. The writer refuses a non-closed centroid set; the reader refuses version, hash or table drift and an unpersisted placeholder; a file without attrs reads as full BZ. |
| `check_spinor_reference_trs`, `check_density_symmetries`, `cached_density_symmetry_check`, `DensitySymmetryReport`, `occupation_operator_residual`, `trs_check_mode` | The 2c time-reversal measurement, § Contract. |
| `build_qgrid_trs_policy(*, trs_measured, irr_idx_q, sym_idx_q, q_irr_full_idx, kgrid, n_sym_spatial, active_symmetry_rows=None, ...) -> QgridTrsPolicy` | The only q-axis consumer of the TRS verdict. `trs_measured` is keyword-only with no default. |
| `QgridTrsPolicy.measure_covariance`, `little_group_covariance_residual` | The little-group covariance the unfold assumes, measured with the same authorized row, centroid permutation, umklapp phase and conjugation; `nan` when no non-identity little-group operation exists. |
| `project_little_group_operator(operator, *, transposed_partner, ...)` | Average over all authorized unitary and antiunitary stabilizers of each q. Returns the average and its transpose at `P(None,'x','y')`, one operation per loop step, with volume-preserving `all_to_all` for nonlocal permutations; each matrix stays ≤ `b·M·M/P` entries per rank. |
| `self_negative_q_mask(q_full_idx, *, kgrid)`, `minus_q_parent_partners`, `trs_pair_coherent_unfold_sym_idx`, `trs_project_self_negative_q_rows` | The one-element orbits of q → −q (every TRIM of an even mesh, Γ alone on an odd one), where the fixed-q Θ projector acts, and the pair-coherent row map. |

`unfold_v_q`, `trs_augment_U`, `compute_centroid_sym_perm`,
`compute_rgrid_sym_perm` and `build_real_space_syms` are call-through
aliases of `unfold_isdf_operator`, `spinor_rotation_for_sym_row`,
`centroid_source_map_and_wrap`, `fft_grid_pullback_perm` and
`real_space_action_tables` (`symmetry_maps.RENAMES`). Use the primary names.

## Contract

* **The time-reversal verdict is measured once and consumed everywhere.**
  `WfnLoader` authenticates the QE `data-file-schema.xml`; the
  occupied-density check measures the two-component DFT state; the only
  executable verdict is `WfnLoader.trs_holds` → `SymMaps.trs_allowed`.
  Missing or inconclusive evidence disables global TR; it never defaults to
  true. Every consumer (q-grid policy, W gates, GN probe, MPA contour, QSGW
  velocity parity) reads `SymMaps.trs_allowed`, and none accepts an override.
  The run record prints `QE schema`, `Stored QE type`, `DFT 2c TRS`,
  `Global TRS` and the active operation rows. The MPA ordered-orientation
  equation is owned by [Multipole frequency integration](../theory/THEORY_mpa_implementation.md#21-ordered-orientations-when-time-reversal-is-broken).
* **The 2c check never uses an antiunitary-generated state as evidence.** Raw
  `k/−k` pairs and TRIM closure are direct evidence. With only a spatial
  partner, the check uses the canonical spatial unfold and labels the result
  conditional; a mismatch then disables antiunitary unfolding without being
  attributed to TRS alone. The metric is the occupied one-particle-subspace
  residual in G space, invariant to band phases and to rotations within
  degenerate blocks. TRIM-only or absent evidence is inconclusive.
* **One measurement per WFN, across processes.** A completed measurement
  (passing or broken) is stamped in `lxkit.user_cache_dir("wfn_trs")`, never
  beside the WFN. The key is the resolved path, size, `mtime_ns` and inode,
  a SHA-256 of the header arrays and G lists the verdict reads (the band
  energies stand in for the coefficients), the algorithm version, and
  `(tol, max_k, nocc)`. A hit skips the coefficient reads, replays the
  on/strict policy and prints the stamp path. A check that raised is never
  stamped. Processes that share the check's collective agree on hit or miss
  by one all-gather, and rank 0 writes. There is no dial.
* **Env surface.** `LORRAX_TRS_CHECK` takes `1`/`on` (default) or `strict`
  (a broken or inconclusive verdict refuses); `0`/`off` refuses.
  `LORRAX_TRS_TOL` and `LORRAX_TRS_MAX_K` tune the measurement. The
  environment never grants a symmetry convention;
  [`docs/dev/env_vars.md`](../dev/env_vars.md) owns the definitions.
* **`_star_conj_flags` is the single conjugation predicate:**
  `trs(member) XOR trs(reference_row)`. It is read by `star_broadcast`'s
  `'star_row'` branch, by `star_spread`, and twice by `KStarMap`. Nothing
  outside the package imports it and nothing inside re-derives it.
* **`trs_reference` names the operand flavour.** `'star_row'`: `A_irr` rows are
  values at the kept full-BZ rows (what `star_select` returns), so the
  predicate is the XOR. `'ibz_slab'`: `A_irr` is the raw IBZ slab with no
  operation applied, so every reference row is TRS-false and the predicate is
  the member's own flag (`sym_idx >= n_sym_spatial`). The two agree only when
  every star's first full-BZ row is spatial. Choosing the wrong one leaves
  every diagonal observable unchanged and corrupts off-diagonal Σ (by
  183.61 eV on a real deck), so the argument is required, an unknown value
  raises with both legal values named, and the raw-slab callers
  (`file_io.kin_ion.broadcast_ibz_to_full_bz`,
  `unfold_file_wedge_band_operator`) pass the literal.
* **The op-selection policy is frozen.** `SymMaps.find_symmetry_ops_simple`
  takes the highest matching irreducible k, then the lowest symmetry index;
  `find_irreducible_bz_points`' anchored branch reproduces it bit for bit.
  Changing it moves eqp by up to 15.9 eV (V_H column) and is an owner
  decision. The tripwire is the bit-equality of `(irr_idx_k, sym_idx_k)` on
  the four in-tree decks against
  `services/symmetry_maps/tests/data/star_tables_e9340d1.json`.
* **Translations: one array, two conventions.** `SymMaps.translations` is raw
  BGW `tnp` (= 2π·τ). G-space consumes it undivided (`tau_phase_row`); every
  real-space `orbit_syms` entry point divides by 2π. Passing one function's
  argument to the other is a 2π error no shape or dtype check catches.
  `verify_centroid_orbit_closure` takes an exclusive `tnp=`/`tau=` keyword
  pair for that reason.
* **Refusals.**
  - A stored k/symmetry table must cover every point of `kgrid`; missing rows
    never fall back to Γ or identity. `ntran = 1` takes the fast full-grid
    path only when all grid points are stored.
  - With TRS disallowed, a WFN whose reduction needs time reversal refuses
    and names the fix: regenerate with `noinv=.true.`.
  - `centroid_source_map_and_wrap` refuses a non-closed centroid set and
    names regeneration as the fix.
  - `unfold_isdf_operator` refuses every table/shape mismatch before tracing,
    because an out-of-bounds `promise_in_bounds` gather clips silently.
* **`nspinor = 2` means noncollinear, not spin-orbit.** `SymMaps` and
  `unfold_psi` branch on the spinor axis, never on SOC.
* **The q axis consumes the verdict and the QE row typing.** With
  `trs_measured=True`, `QgridTrsPolicy` enables pair-coherent q/−q rows and
  the fixed-q Θ projector. With `False` it disables both but keeps
  individually authenticated magnetic antiunitary rows, and refuses every
  unauthorized row; q and −q are then independent irreducible parents.
* **`V_{−q} = conj(V_q)` is not a TRS statement.** The pair densities at −q are
  the conjugates of those at +q with bra and ket relabelled, for any mean
  field, and $v(\lvert q+G\rvert)$ is real and even. The reciprocity gate stays
  armed on a magnet; there it is an independent measurement because q and −q
  are solved separately.
* **Little-group covariance is an assumption of the unfold, and it is
  measured.** `unfold_isdf_operator` presumes each stored parent tile is
  invariant under its own little group: exact for the continuum operator,
  approximate for a finite ISDF fit. `measure_covariance` reports the
  residual so a fit error is diagnosed as a fit error; the pair-coherent row
  map prevents it from becoming a reciprocity error.
* **Storage boundary.** The service owns star membership and symmetry actions,
  not the quadrature meaning of a WFN's stored k rows: `SymMaps` does not
  reinterpret `kweights` (the centroid sampling metric decides between
  full-BZ and IBZ storage).

### Directed band-matrix edges

`directed_edge_orbit_table` takes only arrays from the canonical point map:
`kgrid`, the WFN `shift` in mesh-index units, `sym_mats_k`,
`irr_idx_k`/`sym_idx_k`, and the raw-source rows `kirr_fullids`. A stored
link has layout `(n_source_k, n_source_step, ..., n_band_x, n_band_y)`. The
returned dense fields have layout `(n_k_full, n_target_step)` and index that
array with `source_row`/`source_direction`; `reverse`, `antiunitary`,
`sym_idx` and both stored/oriented endpoint pairs make every action explicit.

For a source link $M(k_0, k_1)$, `apply_band_matrix_symmetry` implements

$$M(gk_0, gk_1) = B_g(k_0)\, M(k_0, k_1)\, B_g(k_1)^\dagger .$$

An antiunitary row conjugates $M$; it does not transpose a non-Hermitian
link. A reverse edge first adjoints $M$ and swaps the endpoint sewings.
`component_mix[..., out, in]` optionally mixes a component axis after the band
action; Cartesian callers take it from `SymMaps.cartesian_action`.
Translations and nonsymmorphic phases belong in the endpoint sewing matrices,
not in the edge table. Identity sewings reproduce `star_broadcast` exactly.

Every symmetry row must be an affine permutation of `(n + shift)/kgrid` and
every stored direction a signed elementary-step permutation. An operation
that maps an elementary step to a multi-step combination (a C3 on some grids)
refuses and names the fix: the direction basis must be closed under the point
group, or a multi-hop orbit precomputed. There is no nearest-direction,
clipped-index or last-write-wins fallback. The host table costs
`O(n_k·n_sym + n_k·n_target·n_source_step)` small-integer work; endpoint
sewing adds only the two distributed band-space products, with no full-band
gather or wavefunction dependency.

### q-stencil orbits

`q_stencil_orbit_table(*, kgrid, sym_mats_k, irr_idx_q, sym_idx_q, seed_steps, n_sym_spatial, active_symmetry_rows)`
closes a set of integer q-step seeds under the authorized
`SymMaps.sym_mats_k` operations, groups the result with `irr_idx_q`, and
returns the symmetry-inequivalent source q rows plus a complete
target-to-source action table and a target-to-seed mask (so a caller keeps
shell labels without putting its policy in this service). It never acts on
a transition matrix at fixed k: sum the finite-q response over the full k grid
at a stored source q, then unfold the resulting scalar/vector/tensor with
`apply_band_matrix_symmetry` (Cartesian wings take
`sym.cartesian_action(target_sym_idx, ...)` as `component_mix`). It has no
second TR switch. Distinct steps that alias modulo the mesh refuse; there is
no clipped or nearest-q fallback.

### Endpoint panels

`unfold_endpoint_panel(factor_face, *, irr_idx, sym_idx, q_irr_frac, source_perm, L_table, spin_action_full, n_sym_spatial, active_mask, mesh, mesh_axis, max_live_bytes)`
unfolds a bounded child-q panel of a `[parent, μ, spin, K]` factor whose μ
tiles `mesh_axis` and K the other axis (`P(None,'x',None,'y')` or
`P(None,'y',None,'x')`, like the two wavefunction faces). It returns
`(child_face, cost)` in the same layout. `certify_endpoint_locality` checks
that each `source_perm` row is a permutation preserving `active_mask` and
reports whether the map is shard-local; a nonlocal map costs `P_axis − 1`
collective permutes per panel per call on a ring that sends one local child
panel at a time. `endpoint_panel_cost` returns analytical per-rank bounds
(`2·parent + 6·child` bytes plus metadata, and the ring traffic), not a
compiled peak; the caller admits `max_live_bytes` against all other live
stages. Phase, spin and antiunitary actions delegate to
`unfold_wavefunction_local`. For repeated calls, bind the immutable metadata
in an outer `jit`.

## Backends

Pure jax and numpy: no vendor library and no `.so`. Mesh-touching paths go
through the package's private `_shard_map`, which picks `jax.shard_map` or
`jax.experimental.shard_map` and refuses on a jax with neither. Host operands
take numpy paths; device operands take cached jits with explicit output
shardings, and a sharded `jax.Array` is never pulled to the host to be
indexed. The star index tables are `n_k` host integers; the operand
(`(n_k, nb, nb)` complex128, 9.2 GB at nk = 144, nb = 2000) is what the helpers
are written not to move. `spread_rel` on a device operand costs one reduction
and one 16-byte transfer.

## Tests

`services/symmetry_maps/tests` (markers `services`, `symmetry_maps`) runs on a
laptop: `pytest services/symmetry_maps/tests`, or `pytest -m symmetry_maps`
from the monorepo (deselect with `--no-services` /
`--only-service=symmetry_maps`, never a second `-m`, which replaces
`addopts = "-m 'not extra'"`).

| tier | file | needs |
|---|---|---|
| star contract, algebra, typed actions | `test_symmetry_maps_star_contract.py`, `test_symmetry_maps_algebra.py`, `test_typed_representation_actions.py`, `test_symmetry_maps_r_cart.py` | nothing |
| deck tables | `test_symmetry_maps_deck_tables.py` | h5py and the four in-tree WFN headers |
| emulated mesh | `test_symmetry_maps_emulated_mesh.py` | four forced CPU devices; skips below four |
| real multi-process | `test_symmetry_maps_multiproc.py` (`check_*` bodies plus a `_CLI_CELLS` CLI) | one process per device |
| import isolation, skip honesty | `test_symmetry_maps_import_isolation.py`, `test_symmetry_maps_skip_honesty.py` | `python -S`; a machine profile (absent skips, present-and-broken fails) |

* Star tests use hand-verified production tables
  (`tests/data/star_tables_e9340d1.json`), never tables derived from a
  generated grid, and each table-driven cell asserts that `'star_row'` and
  `'ibz_slab'` disagree on the expected number of rows (8 on gnppm, 6 on
  cohsex) so it cannot pass as a tautology.
* Hostile geometry is mandatory: `n_rmu % (Px·Py) ≠ 0` must refuse.
* `spread_rel` on a NaN-poisoned sharded operand returns `nan` on real
  processes; the emulated-mesh result differs (`tests/KNOWN_FAILURES.md`).

## Antipatterns

* **Reading `R_cart` or reconstructing determinant/TR signs in a driver.** Use
  `SymMaps.cartesian_action`; it owns orientation, polar versus axial parity
  and the antiunitary time sign.
* **Re-deriving the conjugation predicate.** `sidx >= n_sym_spatial` is correct
  only for the raw-slab flavour; against a star-row reference it inverts the
  rule for every star whose first member is time-reversed (4 of 5 stars on
  gnppm).
* **Using a diagonal spread as a symmetry gate.** Conjugating a Hermitian star
  member leaves its real diagonal exactly unchanged; only an off-diagonal
  metric (`star_spread`) sees a conjugation error.
* **Deriving star-test tables from a generated grid.** Lex-min orbit
  representatives are always spatial, so a derived grid has no TRS-first row
  and every discriminating test becomes a tautology.
* **Testing symmetry on Si only.** Si has no TRS rows at its 64 k; every
  antiunitary branch is dead there.
* **Regenerating a centroid set to make the orbit-closure refusal pass.** The
  production sets' non-closure is measured and owner-scoped; regenerating
  re-freezes the BerkeleyGW anchor. Tests that need a non-closed set build one
  by dropping a centroid from a closed set.
