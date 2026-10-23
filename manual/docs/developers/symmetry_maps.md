# symmetry_maps

`import symmetry_maps` Crystal symmetry for LORRAX: IBZ ⇄ full-grid tables, k stars, the transport of ψ and of centroid-basis operators between wedge and full grid, spinor rotations, centroid orbits, and the time-reversal verdict.
{: .page-summary }

LORRAX computes on wedges and reconstructs the full grid by symmetry. Wavefunctions are stored only at the k rows of `WFN.h5`; $\zeta$, $V_\bq$ and $W_\bq$ are computed only on the irreducible $\bq$ wedge; Σ and other band matrices are written on the WFN's k rows. Every full-grid object is a parent transformed by one space-group operation $\{R_s|\boldsymbol\tau_s\}$, possibly composed with time reversal Θ ([Symmetry](../theory/symmetry.md)). Because the centroids $\br_\mu$ are FFT-grid points selected in closed orbits, each operation permutes them up to a lattice vector, $M_s(\br_\mu-\boldsymbol\tau_s)=\br_{\alpha_s(\mu)}+\mathbf L_{s\mu}$, and a two-point operator in the centroid basis is transported by a double gather with wrap phases, $X_{S_s\bar{\mathbf p}}(\mu,\nu) = e^{2\pi i\bar{\mathbf p}\cdot(\mathbf L_{s\mu}-\mathbf L_{s\nu})}\,\mathcal U_s\,X_{\bar{\mathbf p}}\big(\alpha_s(\mu),\alpha_s(\nu)\big)\,\mathcal U_s^\dagger$, with conjugation (static Hermitian operators) or transposition (operators at a fixed complex time or frequency) on antiunitary rows ([Symmetry §4](../theory/symmetry.md#sym-two-point)). `symmetry_maps` builds the tables ($\bar\bk$, $s$, $\alpha_s$, $\mathbf L_s$, $\mathcal U_s$) from the WFN header and applies them: to plane-wave coefficients in the loader, to band-index matrices (Σ, eigenvector rotations) on k stars, and to centroid-indexed operators ($G$, $\chi^0$, $V$, $W$) on the $\bq$ axis, sharded over the device mesh. It also decides whether time reversal may be used at all, from QE's schema and a measurement on the occupied states. A developer adding a new two-point quantity (an electron–phonon vertex, a strain derivative) computes it on the wedge and unfolds it here; a developer adding a new driver reads operations from `SymMaps` instead of reconstructing rotation, determinant or time-reversal signs.
{: .lead }

<div class="glance" markdown>

Import
:   `import symmetry_maps`; top-level names only. A direct-library caller runs `ffi._services.ensure_on_path()` first.

Depends on
:   lxkit, JAX, numpy; h5py lazily (q-wedge storage only). Nothing from LORRAX.

Devices
:   Tables are host numpy. Operator unfolds run in `shard_map` on a 2-D `('x', 'y')` mesh with centroid-matrix tiles at `P(None, 'x', 'y')`; ψ and band-matrix helpers follow their operand (numpy in, numpy out; `jax.Array` in, cached jit out).

Used by
:   `wfn_loader`, `gw.qgrid_symmetry`, `gw.isdf_fitting`, `gw.v_q_g_flat`, `gw.w_isdf`, `gw.sc_iteration`, `file_io.restart_bundle`, `file_io.tagged_arrays`, `file_io.kin_ion`, `centroid.*` ([below](#used-by))

</div>

## Conventions {: #conventions }

| Object | Convention |
|---|---|
| Operation rows | Rows `0 … n_t−1` are the WFN's spatial operations; rows `n_t … 2n_t−1` are the same operations composed with Θ (antiunitary). `n_sym_spatial` = $n_t$ = `len(sym_matrices)` everywhere |
| `sym_matrices` | `(n_t, 3, 3)` int, BerkeleyGW `mtrx` in crystal coordinates. Real-space action $\br' = \mathtt{mtrx}^{-1}\br + \boldsymbol\tau$ |
| `sym_mats_k` | `(2n_t, 3, 3)` int, reciprocal action: `mtrx.T` on spatial rows, `-mtrx.T` on Θ rows. $\bk = S_s\bar\bk$ up to a reciprocal lattice vector |
| `translations` | `(n_t, 3)` float, BerkeleyGW `tnp` $= 2\pi\boldsymbol\tau$ with $\boldsymbol\tau$ in lattice fractions. G-space functions take it as is; real-space functions divide by $2\pi$ |
| k, q coordinates | crystal (fractional) floats, or integers on `kgrid` |
| `sym_perm`, `L_table` | `(n_rows, N_μ)` int source map $\alpha_s$ and `(n_rows, N_μ, 3)` lattice wraps $\mathbf L_s$, from `centroid_source_map_and_wrap` |
| `irr_idx`, `sym_idx` | `(n_full,)` int: parent row and operation row of every full-grid point |
| File wedge | the k rows of `WFN.h5` (`nk_red` rows, `wfn.kpoints`); what BerkeleyGW files and LORRAX text outputs are indexed by |
| Star wedge | one row per k orbit, the first full-grid member; can be smaller than the file wedge |
| `nspinor = 2` | noncollinear, with or without spin–orbit coupling; code branches on the spinor axis, never on SOC |

## SymMaps {: #symmaps }

```python
sym = SymMaps(wfn)          # or wfn.symmetry() on a WfnLoader
```

`wfn` is a [`WfnLoader`](wfn_loader.md) or any object with its `mf_header` attributes and an explicit boolean `trs_holds`. A missing verdict raises `GATE SymMaps_needs_measured_trs`; the retired `allow_trs=` argument raises `GATE retired_SymMaps_allow_trs`. Construction builds the k and q parent tables eagerly (host numpy). A stored k table must cover every point of `kgrid`; a full-grid k that no permitted operation reaches raises, and a WFN whose reduction needs time reversal when it is not allowed raises and names the fix (regenerate with `noinv = .true.`).

| Attribute | Shape | Meaning |
|---|---|---|
| `trs_allowed` | bool | the measured verdict; the only switch for antiunitary rows |
| `sym_matrices`, `translations`, `sym_mats_k` | see [conventions](#conventions) | |
| `R_cart` | `(2n_t, 3, 3)` | Cartesian rotations, Θ rows negated |
| `U_spinor` | `(n_t, 2, 2)` complex | SU(2) images of the proper rotations $\det(R)R$ |
| `active_symmetry_rows` | `(n,)` int | rows this instance may select |
| `unfolded_kpts` | `(nk_tot, 3)` | full-grid k, crystal |
| `irr_idx_k`, `sym_idx_k` | `(nk_tot,)` | parent WFN row and operation row of each full-grid k |
| `kirr_fullids` | `(nk_red,)` | full-grid row of each WFN row |
| `nk_tot`, `nk_red` | int | full-grid and WFN k counts |
| `kvecs_asints` | `(nk_tot, 3)` int | full-grid k on `kgrid` |
| `irr_idx_q`, `sym_idx_q` | `(nq,)` | irreducible-$\bq$ parent and operation row of each full-grid $\bq$ |
| `q_irr_full_idx` | `(nq_irr,)` | full-grid row of each irreducible $\bq$ |
| `q_irr_kgrid_int` | `(nq_irr, 3)` int | irreducible $\bq$ on `kgrid`; `bgw_integer_q_to_fractional` gives `q_irr_frac` |
| `qe_symmetry_binding`, `operation_typing_source` | | QE schema receipt and whether operation types came from it |
| `kq_map`, `kqfull_map` | | $\bk-\bq$ lookups against the reduced and the full $\bq$ list, built on first access |

| Method | Returns |
|---|---|
| `operation_rows(rows)` | `(sym_mats_k[rows], translations[rows % n_t], rows >= n_t)` |
| `cartesian_action(rows, *, axial, time_odd)` | `(n, 3, 3)` action on a Cartesian index: polar or axial (times $\det R$), time-even or time-odd (negated on Θ rows). Both flags are required bools |
| `lorentz_action(rows)` | `(n, 4, 4)` $\mathrm{diag}(1, R_{\text{polar, time-odd}})$ for charge and current indices |
| `spinor_action(rows, *, nspinor)` | `(n, n_s, n_s)`: 1, $U_s$ or $\mathrm{diag}(U_s,\det R\,U_s)$; $i\sigma_y\overline{U_s}$ on Θ rows |
| `reciprocal_phase(row, carriers)` | `(n_G,)` $e^{-i(S\bG)\cdot\mathbf t}$, or `None` when $\boldsymbol\tau = 0$ |
| `fft_grid_pullback(rows, fft_grid, *, validate=True, cache=True)` | `(n, n_rtot)` read-only pull-back permutations of the FFT grid |
| `unfold_wavefunction(coefficients, *, row, g_parent)` | the child coefficients ([Wavefunctions](#psi)) |
| `validate_kgrid_unfolding(wfn)`, `validate_atomic_symmetries(wfn)` | lists of failures (empty when consistent) |
| `trivial_view()` | a copy restricted to the identity, over loader-unfolded full-grid parents |

`SpatialOperatorTables` and `build_spatial_operator_tables(wfn)` give `sym_matrices`, `sym_mats_k`, `translations`, `R_cart`, `U_spinor` without any k map, so a WFN that `SymMaps` refuses can still be measured.

## Time reversal {: #trs }

The verdict is measured once per WFN and consumed everywhere. For a two-component WFN, time reversal holds only if QE's schema types no operation as antiunitary and the SCF absolute magnetization is below $10^{-4}\,\mu_B$/cell; when QE says nonmagnetic, the loader also checks the occupied one-particle density operator at raw $\bk/-\bk$ pairs, spatially unfolded partners and TRIM points ([Symmetry §2](../theory/symmetry.md#sym-trs)). The verdict lands on `WfnLoader.trs_holds` and becomes `SymMaps.trs_allowed`; no consumer accepts an override.

| Name | Meaning |
|---|---|
| `check_spinor_reference_trs(loader, *, tol_trs=None, tol_spatial=None, max_k=None, nocc=None, ...)` | the measurement; returns `DensitySymmetryReport` (`trs_holds`, `subspace_residual`, `evidence_counts`, `conclusive`, `summary()`, …). `check_density_symmetries` is the same function |
| `cached_density_symmetry_check(loader)` | the loader's entry point: process cache, then an on-disk stamp keyed on file identity and content hash |
| `occupation_operator_residual(overlap, target_occupations, source_occupations)` | `(residual, …)` distance between a target and a time-reversed density operator |
| `trs_check_mode()` | `"on"` or `"strict"` from `LORRAX_TRS_CHECK`; `0`/`off` raises `GATE retired_LORRAX_TRS_CHECK_off` |
| `read_qe_symmetry_receipt(path) -> QESymmetryReceipt` | the bounded symmetry and k-sampling subset of a QE `data-file-schema.xml` |
| `bind_qe_symmetry_receipt(wfn, receipt) -> QESymmetryBinding` | authenticates a receipt against a WFN (orientation, Seitz translations, k rows, spinor count) and returns per-operation `antiunitary` bits |
| `resolve_qe_symmetry_binding(wfn, *, wfn_path, schema=None)` | `(binding or None, diagnostic)`; an explicit `schema` that does not match raises |
| `discover_qe_schema_paths(*wfn_paths)` | candidate schemas beside the WFN and in nearby `*.save/` directories |
| `qe_xml_seitz_to_bgw(rotation, fractional_translation)` | one QE translation in BerkeleyGW `tnp` units |

`LORRAX_TRS_CHECK` (`1`/`on` default, or `strict`: a broken or inconclusive verdict raises), `LORRAX_TRS_TOL` (default $10^{-6}$) and `LORRAX_TRS_MAX_K` (default 12) tune the measurement. With `strict` unset, a raw-pair failure when QE reports zero moment raises `GATE trs_qe_nonmagnetic_wfn_consistent`; other failures turn time reversal off with a warning.

## Grid algebra {: #grid-algebra }

Pure numpy, host.

| Name | Signature | Returns |
|---|---|---|
| `kgrid_shift_map` | `(nkx, nky, nkz, q_off)` | C-order index of $\bk+\bq_{\mathrm{off}}$ for every grid k, and the umklapp $\bG$ |
| `bgw_integer_q_to_fractional` | `(q_int_kgrid, kgrid)` | `(n, 3)` fractional $\bq$ with BerkeleyGW's wrap (wrap first, then divide) |
| `bgw_signed_q_representative` | `(qvec)` | BerkeleyGW's signed representative of fractional rows |
| `q_negation_index` | `(kgrid)` | `(nq,)` permutation $\mathrm{index}(\bq)\to\mathrm{index}(-\bq)$ |
| `common_uniform_grid_indices` | `(grid_a, grid_b)` | aligned C-order rows shared by two unshifted grids |
| `find_irreducible_bz_points` | `(full_kgrid_int, sym_mats_k, *, irr_kgrid_int=None)` | parent and operation for each full-grid point (derive the wedge, or anchor to a given one) |
| `map_full_kpoints_to_irreducible` | `(kpoints, sym_mats_k, full_kpoints, *, tol=1e-6)` | parent, operation and a coverage mask; incomplete coverage should refuse before use |

The parent and operation choice is frozen: highest matching parent, then lowest operation index. Changing it moves reference quasiparticle energies.

## k stars and band-index objects {: #stars }

A band-index quantity $A_{\bk}$ (a self-energy matrix, an eigenvector rotation, kinetic and ionic matrix elements) moves between wedge and full grid by indexing plus conjugation on time-reversed rows; no centroid gather is involved.

```python
star_broadcast(A_irr, irr_idx_k, sym_idx_k, n_sym_spatial, irr_labels=None, *,
               trs_reference, trs_rule="conj", rows=None)
unfold_file_wedge_to_full_bz(sym, data, *, rows=None)       # (nk_red, ...) -> (nk_tot, ...)
unfold_star_wedge_to_full_bz(sym, data)                     # star wedge -> (nk_tot, ...)
reduce_full_bz_to_file_wedge(sym, data)                     # (nk_tot, ...) -> (nk_red, ...)
```

| Name | Meaning |
|---|---|
| `star_select(A_full, irr_idx_k)` | one row per star, the **first** occurrence in full-grid order |
| `star_broadcast(...)` | wedge to full grid. `trs_reference="star_row"` when `A_irr` holds values at the kept full-grid rows (what `star_select` returns), `"ibz_slab"` when it is the raw WFN-row slab. `trs_rule="conj"` for an observable, `"transpose"` for an operator that transforms like $G$ ($\Sigma_{\Theta\bk,mn}=\Sigma_{\bk,nm}$). `rows` returns only those full-grid rows |
| `star_spread(A_full, irr_idx_k, sym_idx_k, n_sym_spatial)` | max residual of `A_full` against its own star; sees gauge and conjugation errors that norms and diagonals do not |
| `unfold_file_wedge_polar_matrix(sym, data, *, component_axis=-3, time_odd=True, rows=None)` | a Cartesian-vector band matrix (velocity, position with `time_odd=False`), rotating the component axis |
| `unfold_file_wedge_band_operator(sym, data, *, trs_rule)` | the band matrix of a k-diagonal operator; `trs_rule` required |
| `star_tables_of(sym)` | `(irr_idx_k, sym_idx_k, n_sym_spatial)` |
| `star_wedge_rows(sym)`, `star_wedge_tables(sym)` | WFN rows and tables for a slab computed on the star wedge |
| `KStarMap(irr_idx, sym_idx, n_sym_spatial, labels=None)` | the three arrays bundled: `.from_sym(sym, n_sym_spatial)`, `.identity(n_k)`, `select`, `broadcast`, `spread`, `spread_rel`, `rows`, `take`, `nk_full`, `nk_irr` |

Operands may be host numpy or sharded `jax.Array`; device operands are never pulled to the host. From `src/file_io/kin_ion.py` (a raw WFN-row slab):

```python
return symmetry_maps.star_broadcast(
    A_irr, parent, np.asarray(sym_idx_k), int(n_sym_spatial),
    irr_labels=np.arange(n_rows, dtype=np.int32),
    trs_reference="ibz_slab")
```

From `src/gw/gw_jax.py`: `KStarMap.from_sym(sym, int(wfn.ntran))`.

### Directed edges and q stencils {: #edges }

| Name | Meaning |
|---|---|
| `directed_edge_orbit_table(*, kgrid, kgrid_shift, sym_mats_k, irr_idx_k, sym_idx_k, source_full_ids, source_steps, n_sym_spatial, target_steps=None)` | full-grid lookup for links $M(\bk_0,\bk_0+\boldsymbol\delta)$ stored on the wedge: source row, direction, `reverse`, `antiunitary`, `sym_idx` per target edge |
| `apply_band_matrix_symmetry(matrix, *, antiunitary=False, reverse=False, sewing_start=None, sewing_end=None, component_mix=None, component_axis=-3)` | $M(g\bk_0,g\bk_1)=B_g(\bk_0)\,M(\bk_0,\bk_1)\,B_g(\bk_1)^\dagger$; conjugation on antiunitary rows, adjoint for reversed edges; `component_mix[..., out, in]` from `SymMaps.cartesian_action` |
| `q_stencil_orbit_table(*, kgrid, sym_mats_k, irr_idx_q, sym_idx_q, seed_steps, n_sym_spatial, active_symmetry_rows)` | closes integer $\bq$-step seeds under the authorized operations and returns the inequivalent source steps with a target-to-source table |

Every operation must map stored directions onto signed elementary steps; one that does not (a C3 on some grids) raises and names the fix. There is no nearest-direction fallback.

## Wavefunctions {: #psi }

$$
c_{\bk}(S_s\bG-\mathbf g_0) = \mathcal U_s\,e^{-i(S_s\bG)\cdot\mathbf t_s}\,c_{\bar\bk}(\bG),\qquad
c_{\bk}(-S_s\bG-\mathbf g_0) = \mathcal U_{n_t+s}\,e^{+i(S_s\bG)\cdot\mathbf t_s}\,\overline{c_{\bar\bk}(\bG)}
$$

| Name | Signature | Returns |
|---|---|---|
| `unfold_psi` | `(cnk_kbar, *, sym_idx, g_kbar, sym_mats_k, translations, U_spinor_spatial)` | child coefficients `(nb, n_s, ngk)` in the parent's slot order (the child Miller indices come from `unfold_reciprocal_carriers`). Raises unless `len(sym_mats_k) == 2 * len(U_spinor_spatial)` |
| `spinor_rotation_for_sym_row` | `(U_spinor_spatial, sym_idx, n_sym_spatial, *, nspinor=2, R_cart=None)` | `(n_s, n_s)` or `(n, n_s, n_s)` spinor factor; `nspinor=4` needs `R_cart` |
| `apply_spinor_rotation` | `(U, coeff_last)` | `U` applied to a trailing spinor axis of extent 1 or 2, numpy or jax |
| `tau_phase_row` | `(sym_mat_k, tau, g_kbar)` | `(ngk,)` $e^{-i(S\bG)\cdot\mathbf t}$ with `tau` = `tnp`; `None` when zero |
| `tau_phase_row_jax` | same | device form, never `None` |
| `unfold_reciprocal_carriers` | `(sym_mat_k, g_parent, umklapp)` | child Miller indices |
| `unfold_wavefunction_local` | `(psi_parent_local, *, irr_idx, sym_idx, k_irr_frac, local_perm, L_table, spin_action_full, n_sym_spatial, spin_axis, mu_axis, mesh_axis=None)` | children at the centroids on one μ-local slab: $\psi_{g\bar\bk,a}(\mu)=\sum_c U_g[a,c]\,T_g[e^{2\pi i\bar\bk\cdot\mathbf L_{g\mu}}\psi_{\bar\bk,c}(\alpha_g\mu)]$ |

The loader omits the band-independent factor $e^{-2\pi i(S_s\bar\bk)\cdot\boldsymbol\tau_s}$; it is a gauge and cancels in every $G_\bk$ and every same-k band matrix.

Bounded child panels of a `[parent, μ, spin, K]` factor at `P(None, 'x', None, 'y')` (or with the axes swapped) go through `unfold_endpoint_panel(factor_face, *, irr_idx, sym_idx, q_irr_frac, source_perm, L_table, spin_action_full, n_sym_spatial, active_mask, mesh, mesh_axis, max_live_bytes) -> (child_face, cost)`. `certify_endpoint_locality(source_perm, *, mesh, mesh_axis, active_mask=None)` checks the maps and reports whether routing is shard-local; `endpoint_panel_cost(shape, n_children, *, mesh, mesh_axis, dtype)` gives analytical per-rank byte and ring-traffic bounds.

## Centroid-basis operators on the q axis {: #operators }

### unfold_isdf_operator {: #unfold-isdf-operator }

```python
unfold_isdf_operator(V_q_ibz, *, irr_idx, sym_idx, sym_perm, L_table, q_irr_frac, mesh_xy,
                     n_sym_spatial, trs_rule="conj", right_sym_perm=None, right_L_table=None,
                     left_logical_extent=None, right_logical_extent=None, trs_pair_q_ibz=None,
                     axis_local_sym_perm=None, right_axis_local_sym_perm=None) -> jax.Array
```

$$
V[\bq,\mu,\nu] = C_s\Big[e^{2\pi i\,\bar\bq\cdot(\mathbf L_{s\mu}-\mathbf L_{s\nu})}\,V_{\mathrm{ibz}}\big[p,\alpha_s(\mu),\alpha_s(\nu)\big]\Big],\qquad p=\texttt{irr\_idx}[\bq],\ s=\texttt{sym\_idx}[\bq]
$$

| Argument | Shape, meaning |
|---|---|
| `V_q_ibz` | `(n_q_ibz, n_left, n_right)` `jax.Array` at `P(None, 'x', 'y')` |
| `irr_idx`, `sym_idx` | `(n_q_full,)` int, host (`SymMaps.irr_idx_q`, `sym_idx_q`, or a policy's `unfold_sym_idx`) |
| `sym_perm`, `L_table` | `(n_rows, n_left)`, `(n_rows, n_left, 3)`; `n_rows = 2 n_sym_spatial` when any Θ row is used |
| `q_irr_frac` | `(n_q_ibz, 3)` crystal |
| `trs_rule` | `"conj"` (static Hermitian: $C_s$ conjugates on Θ rows) or `"pair_transpose"` (fixed complex time or frequency: $C_s$ takes the transposed partner, `trs_pair_q_ibz` `(n_q_ibz, n_right, n_left)` for a rectangular operator) |
| `right_sym_perm`, `right_L_table` | the right endpoint's tables when it is a different centroid family; both or neither |
| `*_logical_extent` | physical extent inside a padded centroid axis; the maps must keep padding on padding |
| `axis_local_sym_perm`, `right_axis_local_sym_perm` | shard-local source maps for orbit-packed endpoints, avoiding `all_to_all` |

Returns `(n_q_full, n_left, n_right)` at `P(None, 'x', 'y')`, one single-tile peak per rank. Every table or shape mismatch raises `ValueError` before tracing (an out-of-range gather would clip silently). An identity unfold returns the input. The transport assumes each parent tile is invariant under its own little group; `QgridTrsPolicy.measure_covariance` measures that.

From `src/file_io/restart_bundle.py`, unfolding a stored wedge tensor:

```python
t = tables.padded(int(n_rmu_pad))
return unfold_isdf_operator(
    A, irr_idx=t.irr_idx_q, sym_idx=t.sym_idx_q, sym_perm=t.sym_perm,
    L_table=t.L_table, q_irr_frac=t.q_irr_frac, mesh_xy=mesh_xy,
    n_sym_spatial=int(t.n_sym_spatial))
```

### Related operator transports {: #operators-related }

| Name | Meaning |
|---|---|
| `unfold_operator_local(operator_parent_local, *, irr_idx, sym_idx, q_irr_frac, left_local_perm, left_L_table, right_local_perm, right_L_table, n_sym_spatial, trs_rule="conj", transposed_parent_local=None, left_mesh_axis="x", right_mesh_axis="y", transposed_output=False)` | the same transport for a caller already inside an `('x','y')` `shard_map`: one local parent tile in, this rank's child tile out; tables are runtime operands |
| `unfold_spin_centroid_operator(operator_ibz, *, irr_idx, sym_idx, sym_perm, L_table, k_irr_frac, spin_action_full, n_sym_spatial, mesh_xy, ...)` | open-spin operators `(nk_parent, μ, s, ν, s)` at `P(None, 'x', None, 'y', None)`: the centroid transport plus $U(\cdot)U^\dagger$ on the spin indices |
| `open_spin_block_coefficient(spin_action_full, a, b)` | `coef[k, c, d] = U_k[a, c]·conj(U_k[b, d])`, one spin block of $U O U^\dagger$ |
| `unfold_isdf_one_leg(zeta_ibz, *, sym, sym_idx, sym_perm, L_table, q_irr_frac, kgrid, mesh_xy, component_action, ...)` | one ISDF Fourier leg ($\zeta$-like, one centroid index and one G index) from the $\bq$ wedge to the full grid, with parent-G relabelling and scalar or polar component transport |
| `isdf_one_leg_source_slots(gvec_components, *, sym, sym_idx, q_irr_frac, kgrid)` | parent G-sphere slot of each full-grid $\bG = 0$ coefficient |
| `mix_lorentz_blocks(blocks, *, sym, sym_idx, mesh_xy, keys=None)` | mixes charge/current sectors by $\Lambda\otimes\Lambda$ from `SymMaps.lorentz_action` |
| `select_current_basis(sym)`, `current_component_action(sym, rows, basis_rows)`, `CIRCULAR_CURRENT_ROWS` | the current-component basis (circular about $z$ or Cartesian) in which every operation is monomial, and its action |
| `slice_q_full_to_ibz(arr_full, q_irr_full_idx, *, out_sharding=None)` | full grid to wedge rows, sharding-preserving |
| `QirrOperator` | an interaction held on its wedge with the tables that define the full zone: `whole_zone`, `restrict`, `at_rows`, `unfold(mesh_xy)`, `load_tables(mesh_xy)`, `at_minus_q`, … |
| `unfold_load_tables(...) -> UnfoldLoadTables`, `local_unfold_load_tables`, `apply_unfold_load_tables_local`, `device_load_tables(t, mesh_xy) -> DeviceLoadTables`, `DEVICE_LOAD_SPECS`, `umklapp_phase(q_per, L)` | the transport as gather tables, for consumers that fuse the unfold into their own k-convolution kernel instead of materializing the full grid |

## Real-space orbits and centroid closure {: #orbits }

| Name | Meaning |
|---|---|
| `centroid_source_map_and_wrap(r_mu_fft_idx, sym_matrices, translations, fft_grid, *, validate=True, extend_trs=False, required_rows=None)` | `(sym_perm, L_table)`: **source** map $\alpha_s$ and wraps with $\mathtt{mtrx}(\mathbf x_\mu-\boldsymbol\tau)=\mathbf x_{\alpha(\mu)}+\mathbf L_\mu$. `r_mu_fft_idx` `(N_μ, 3)` int grid indices; `translations` is `tnp`. `extend_trs=True` appends the Θ rows (`2n_t` rows). Raises on a non-closed set |
| `fft_grid_pullback_perm(sym_matrices, translations, fft_grid, *, validate=True)` | `(n_t, n_rtot)` **pull-back** permutation over the whole FFT grid, `perm[s, r_new] = r_old` |
| `grid_point_image_perm(fft_grid, M)` | FFT-grid permutation of a symmorphic integer operation |
| `real_space_action_tables(wfn, sym, validate=True, *, charge_density=None)` | `(R, Rinv, tau)` for real-space orbit construction, `tau` in lattice fractions |
| `orbit_images(reps, Rinv, tau)`, `canonicalize_orbit(reps, Rinv, tau)`, `unfold_orbit_unique_with_id(reps_np, Rinv, tau, tol=1e-6)` | images, lexicographic representative, and distinct images with orbit ids |
| `r_action_forward(points, Rinv, tau, *, wrap)`, `r_action_forward_one(...)` | $\br'=R^{-1}\br+\boldsymbol\tau$ for a stack or one operation; `wrap` is required |
| `snap_to_grid_and_split_wrap(images_frac, fft_grid)` | grid index and lattice wrap, rounding before splitting |
| `permutation_orbit_labels(permutations)` | orbit labels from a stack of permutations |
| `project_polar_fft_field(field, sym) -> PolarFFTFieldProjection` | symmetrizes a `(3, nx, ny, nz)` polar field and returns the residuals |
| `recover_atomic_space_group(avec, atom_crys, atom_types, *, tol=1e-6)`, `recover_symmorphic_density_point_group(avec, charge_density, *, tol_rho=1e-3)` | groups for closing a centroid set; they never authorize electronic reductions |
| `verify_centroid_orbit_closure(centroids_frac, sym_matrices, *, tnp=None, tau=None, fft_grid=None, tol=CLOSURE_TOL_DEFAULT) -> CentroidClosureVerdict` | closure as a measurement: `closed`, `worst_residual`, `violating_ops`, `residual_by_op`, `centroid_hash`, `raise_if_not_closed()`. Exactly one of `tnp=` / `tau=` |
| `resolve_qgrid_symmetry(r_mu_fft_idx, sym_matrices, *, tnp=None, tau=None, fft_grid, extend_trs=True, tol=CLOSURE_TOL_DEFAULT, context="", required_rows=None) -> QgridSymmetryResolution` | the one $\bq$-grid decision: `mode` (`"ibz"`/`"full_bz"`), `use_ibz`, `sym_perm`, `L_table`, `reason`, `announcement()`; on fallback the message ends with `FULL_BZ_CONSEQUENCE` |

`CLOSURE_TOL_DEFAULT` is $10^{-5}$ (fractional coordinates). From `src/gw/qgrid_symmetry.py`:

```python
res = resolve_qgrid_symmetry(
    np.asarray(jax.device_get(centroid_indices), dtype=np.int32),
    np.asarray(sym.sym_matrices[:n_tran]),
    tnp=tnp[:n_tran],
    fft_grid=np.asarray(fft_grid, dtype=np.int32),
    extend_trs=True,
    required_rows=np.asarray(sym.sym_idx_q),
    context=context,
)
```

## q-axis time-reversal policy {: #qgrid-trs }

```python
build_qgrid_trs_policy(*, trs_measured, irr_idx_q, sym_idx_q, q_irr_full_idx, kgrid,
                       n_sym_spatial, active_symmetry_rows=None, context="") -> QgridTrsPolicy
```

`trs_measured` is keyword-only with no default; pass `sym.trs_allowed`. With it true, the $\bq$ and $-\bq$ children of one parent use the same spatial row (`unfold_sym_idx`), and at self-negative $\bq$ the projector $A\mapsto(A+\overline A)/2$ applies. With it false both are off and $\bq$, $-\bq$ are independent parents.

| Member | Meaning |
|---|---|
| `QgridTrsPolicy.unfold_sym_idx` | `(nq,)` operation rows to pass to `unfold_isdf_operator` |
| `.self_negative_q` | mask of $\bq\equiv-\bq$ rows |
| `.project_fixed_q(operator, q_full_idx, *, transposed_partner=None, measure=True)` | `(op, removed_rel)`: the Θ projection at self-negative rows and the size of what it removed |
| `.measure_covariance(V_ibz, **kwargs)` | little-group covariance residual of the stored tiles |
| `little_group_covariance_residual(V_ibz, *, q_irr_frac, q_irr_full_idx, sym_mats_k, sym_perm, L_table, kgrid, n_sym_spatial, ...)` | the same statistic as a function; `nan` when no non-identity stabilizer exists |
| `project_little_group_operator(operator, *, transposed_partner, q_full_idx, q_irr_frac, sym_mats_k, sym_perm, L_table, active_symmetry_rows, kgrid, n_sym_spatial, mesh, active_mask=None)` | average over every authorized stabilizer of each $\bq$; returns the average and its transpose at `P(None, 'x', 'y')` |
| `little_group_projection_is_identity(...)` | whether that average is the identity on the given rows |
| `self_negative_q_mask(q_full_idx, *, kgrid)`, `minus_q_parent_partners(...)`, `trs_pair_coherent_unfold_sym_idx(...)`, `trs_project_self_negative_q_rows(...)` | the pieces the policy composes |

From `src/gw/qgrid_symmetry.py`:

```python
policy = build_qgrid_trs_policy(
    trs_measured=bool(sym.trs_allowed),
    irr_idx_q=irr_idx_q, sym_idx_q=sym_idx_q,
    q_irr_full_idx=sym.q_irr_full_idx,
    kgrid=tuple(kgrid), n_sym_spatial=int(n_sym_spatial),
    active_symmetry_rows=np.asarray(sym.active_symmetry_rows, dtype=np.int32),
    context=str(context))
```

## q-wedge tensor storage {: #qirr }

Centroid-basis tensors (for example $V_\bq$ and $W_0$ in the ISDF tensor file) can be stored on the $\bq$ wedge with the tables that unfold them, in a group `<name>__qirr`. The format is versioned (`QIRR_FORMAT_VERSION = 1`, tensor rank `QIRR_RANK_BY_VERSION`, attribute `QIRR_VERSION_ATTR`, suffix `QIRR_TABLE_SUFFIX`).

| Name | Meaning |
|---|---|
| `QirrTables(irr_idx_q, sym_idx_q, q_irr_frac, sym_perm, L_table, n_sym_spatial)` | exactly the `unfold_isdf_operator` arguments; `n_q_full`, `n_q_ibz`, `n_mu`, `padded(n)`, `logical(n)`, `digest()` |
| `write_qirr_tensor(dest, name, X_ibz, *, tables, closure_verdict=None, centroids_frac=None, ...)` | writes the wedge tensor and its tables; refuses a non-closed centroid set (pass a verdict or the centroids) |
| `stamp_qirr_tensor(dest, name, *, tables, closure_verdict, ...)` | makes an already-written dataset (written collectively elsewhere) a wedge tensor: tables and attributes only |
| `allocate_qirr_placeholder(dest, name, shape, *, tables, ...)` | an empty dataset marked not ready |
| `read_tensor(src, name, *, mesh_xy=None, unfold=True, n_mu_padded=None, require_persisted=True, expect_centroid_hash=None, expect_table_hash=None, metadata_only=False)` | `(array, QirrHeader)`, unfolded to the full grid when stored on a wedge; refuses version, hash or table drift and an unpersisted placeholder; a file without attributes reads as full grid |
| `read_tables(src, name)`, `dataset_q_storage(ds)` | the stored tables, and `"ibz"` or `"full"` for an open dataset without reading data |
| `validate_qirr_tables(tables, n_q_on_disk, n_mu)` | consistency checks; returns the shape verdict |
| `QirrHeader` | what the reader found: `q_storage`, `format_version`, `n_q_on_disk`, `n_q_full`, `n_mu`, hashes, `closure_verdict`, `data_ready`, `provenance` |
| `QirrDest`, `qirr_attr_str`, `qirr_generator_commit` | open-or-use an h5 group, read a string attribute, and the source commit stamped as provenance |

From `src/file_io/tagged_arrays.py`: `stamp_qirr_tensor(f, "V_qmunu", tables=tables, ...)`.

## Deprecated names {: #deprecated }

These resolve to the primary function and will be removed. `RENAMES` maps old to new; `RETIREMENT_GATE` states the removal condition.

| Old | Use |
|---|---|
| `unfold_v_q` | `unfold_isdf_operator` |
| `trs_augment_U` | `spinor_rotation_for_sym_row` |
| `compute_centroid_sym_perm` | `centroid_source_map_and_wrap` |
| `compute_rgrid_sym_perm` | `fft_grid_pullback_perm` |
| `build_real_space_syms` | `real_space_action_tables` |
| `check_density_symmetries` | `check_spinor_reference_trs` |

## Invariants and gotchas {: #gotchas }

- **Source map versus pull-back.** `centroid_source_map_and_wrap` returns where each target centroid comes from plus a lattice wrap; `fft_grid_pullback_perm` returns a pull-back over the FFT grid. They point in opposite directions, and $\alpha_s^{-1}$ is the table of $g_s^{-1}$, which differs for order-3 operations.
- **`tnp` is $2\pi\boldsymbol\tau$.** G-space functions (`tau_phase_row`) take it undivided; real-space functions divide by $2\pi$. Passing one convention to the other is an error no shape check catches; `verify_centroid_orbit_closure` makes you name it.
- **`trs_reference` is not optional.** `"star_row"` versus `"ibz_slab"` agree only when every star's first full-grid row is spatial. The wrong one leaves diagonals unchanged and corrupts off-diagonal Σ.
- **Use `cartesian_action`.** Do not read `R_cart` and rebuild determinant or time-reversal signs in a driver.
- **Off-diagonal checks only.** A conjugation error leaves the real diagonal of a Hermitian matrix unchanged; check with `star_spread`.
- **Two wedges.** The file wedge (`nk_red`) and the star wedge differ on most inputs; `unfold_file_wedge_to_full_bz` refuses a star-wedge operand and names the right function.
- **$V_{-\bq}=\overline{V_\bq}$ is not a time-reversal statement.** It holds for any mean field and stays armed on magnets.
- **Do not regenerate centroids to pass the closure check.** Measure with `verify_centroid_orbit_closure`; `resolve_qgrid_symmetry` falls back to the full grid and announces it once.
- **Test on more than Si.** Si has no time-reversed rows among its 64 k, so every antiunitary branch is untested there; tables derived from a generated grid have spatial first members only.

## Where LORRAX uses it {: #used-by }

| File | Purpose |
|---|---|
| `services/wfn_loader` | ψ unfold, parent-star streaming, the time-reversal measurement |
| `src/gw/qgrid_symmetry.py` | the $\bq$-grid resolution and time-reversal policy, announced once per run |
| `src/gw/isdf_fitting.py`, `src/isdf/core.py`, `src/isdf/galerkin.py` | wedge $\bq$ rows of the ζ fit; centroid maps |
| `src/gw/v_q_g_flat.py`, `src/gw/v_q_bispinor.py`, `src/gw/w_isdf.py`, `src/gw/screening.py` | $V_\bq$ and $W_\bq$ on the wedge, unfolded with `unfold_isdf_operator` |
| `src/gw/centroid_k_unfold.py`, `src/common/centroid_basis.py`, `src/gw/tangential_screening.py` | centroid source maps for $G_\bk$ and residues |
| `src/file_io/restart_bundle.py`, `src/file_io/tagged_arrays.py`, `src/file_io/mpa_store.py` | q-wedge storage in the ISDF tensor file and MPA stores |
| `src/file_io/kin_ion.py`, `src/file_io/sigma_output.py`, `src/gw/sc_iteration.py` | band matrices between the WFN rows and the full grid |
| `src/centroid/` | orbit-closed centroid selection, symmetrized charge density |
| `src/psp/get_DFT_mtxels.py`, `src/gw/qsgw_density.py` | FFT-grid pull-backs for symmetrizing densities |
| `src/bse/` | wedge-to-full transport of W and quasiparticle rotations for the BSE |
