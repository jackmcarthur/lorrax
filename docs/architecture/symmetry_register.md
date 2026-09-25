# The symmetry register

This page lists every symmetry table a LORRAX run builds: who builds it, its
shape, what it means, which kernel consumes it, and what refuses. The action
the tables encode is derived in [Symmetry](../theory/symmetry.md). Signatures
and the time-reversal verdict are on the
[`symmetry_maps` service page](../services/symmetry_maps.md). Every table and
every action comes from `services/symmetry_maps`, bound to a centroid basis by
`gw.centroid_k_unfold`. No driver spells its own rotation, sign, wrap or
conjugation rule.

Notation: \(n_t\) spatial operations (`wfn.ntran`), \(N_k\) full-grid k
(`nk_tot`), \(n_{\rm red}\) stored k (`nk_red`), \(M\) centroids, an
\(s\times s\) process mesh, \(P=s^2\).

## 1. Operation tables

`SymMaps(wfn)`, reached as `WfnLoader.symmetry()`, builds these once from the
WFN header, the QE schema receipt and the time-reversal verdict that the
loader attaches.

| attribute | shape, dtype | contents |
|---|---|---|
| `sym_matrices` | \((n_t,3,3)\) int | `mtrx` (\(M_s\)), raw |
| `translations` | \((n_t,3)\) float | `tnp` \(=2\pi\boldsymbol\tau_s\), cut to \(n_t\) rows |
| `sym_mats_k` | \((2n_t,3,3)\) int | \([M^{\mathsf T};-M^{\mathsf T}]\); rows \(\ge n_t\) are antiunitary |
| `R_cart` | \((2n_t,3,3)\) float | Cartesian image of \(M_s\); rows \(\ge n_t\) negated |
| `U_spinor` | \((n_t,2,2)\) complex | SU(2), spatial rows only |
| `active_symmetry_rows` | \((n_{\rm act},)\) int32 | rows the k and q searches may select |
| `trs_allowed` | bool | `wfn.trs_holds`, the measured global verdict |
| `qe_antiunitary_rows`, `operation_typing_source` | | receipt provenance: `"qe-schema"` or `"wfn-fallback"` |

`active_symmetry_rows` holds all \(2n_t\) rows when global time reversal
holds. Otherwise it holds the QE-typed row of each operation, or only the
unitary half when there is no receipt. The table keeps its \(2n_t\) rows
either way; only the search set shrinks.

Consumers read the typed accessors, never `R_cart` or a determinant:

| accessor | returns |
|---|---|
| `operation_rows(rows)` | \(S\) (with the antiunitary minus), spatial \(\mathbf t\), antiunitary bit |
| `spinor_action(rows, nspinor=)` | \((n,n_s,n_s)\): \(1\), \(U\), \(i\sigma_y\bar U\), or the four-component blocks |
| `cartesian_action(rows, axial=, time_odd=)` | forward Cartesian \((n,3,3)\); both flags required |
| `lorentz_action(rows)` | \((n,4,4)\) \(=\operatorname{diag}(1,\) polar time-odd \(R)\) |
| `reciprocal_phase(row, carriers)` | \(e^{-i(S\mathbf K)\cdot\mathbf t}\), or `None` when \(\mathbf t\approx0\) |
| `unfold_wavefunction(c, row=, g_parent=)` | plane-wave child coefficients, host |
| `fft_grid_pullback(rows, fft_grid)` | \((n,N_r)\) grid pull-back permutation, read-only, cached per process |

## 2. k and q maps

| attribute | shape | contents |
|---|---|---|
| `unfolded_kpts` | \((N_k,3)\) | uniform grid from `kgrid` and `shift`, wrapped to \([0,1)\) |
| `kirr_fullids` | \((n_{\rm red},)\) | full row equal to WFN row \(i\), matched to \(10^{-6}\) |
| `irr_idx_k`, `sym_idx_k` | \((N_k,)\) | parent WFN row \(p\) and operation row \(s\): \(\mathbf k=S_s\bar{\mathbf k}_p+\mathbf g_0\) |
| `parent_k_domain` | `"ibz"` or `"full_bz"` | the loader's k set for parents, `wfn.kvecs(k=...)`; `"full_bz"` only on the trivial view (§7) |
| `kvecs_asints` | \((N_k,3)\) int | C-order integer mesh; q labels |
| `irr_idx_q`, `sym_idx_q` | \((N_k,)\) | q parent and row |
| `q_irr_kgrid_int`, `q_irr_full_idx` | \((N_q^{\rm irr},3)\), \((N_q^{\rm irr},)\) | q parents and their full rows |
| `kq_map`, `kqfull_map` | \((N_k,\cdot)\) | full row of \(\mathbf k-\mathbf q\) |

**Selection rules.** The k map (`map_full_kpoints_to_irreducible`) compares
\(S\bar{\mathbf k}\) with each full-grid point in fractional coordinates, at
tolerance \(10^{-6}\), over the authorized rows. Among matches the highest
stored row wins, then the lowest operation row. The rule is frozen: changing
it moves results and is an owner decision
([service contract](../services/symmetry_maps.md#contract)). The q map
(`find_irreducible_bz_points`) is integer arithmetic on the unshifted mesh.
Each orbit's parent is its lexicographically smallest member, reached by the
lowest row. The umklapp \(\mathbf g_0=\mathbf k-S\bar{\mathbf k}\) is BGW's
`kg0` (`get_umklapp_vector`), and child G vectors are
\(S\mathbf G-\mathbf g_0\) (`unfold_reciprocal_carriers`).

q-axis unfolds take their rows from the q policy (§5), not from `sym_idx_q`.

## 3. Centroid tables

`centroid_source_map_and_wrap(r_idx, M[:n_t], t[:n_t], fft_grid,
extend_trs=True, required_rows=)` builds

| table | shape, dtype | contents |
|---|---|---|
| `sym_perm` | \((2n_t,M)\) int32 | \(\alpha_s(\mu)\), the pull-back source; rows \([n_t,2n_t)\) repeat rows \([0,n_t)\) |
| `L_table` | \((2n_t,M,3)\) int8 | \(\mathbf L_{s\mu}\) |

Kernels gather with `sym_perm[s]` directly; none inverts it.

**Rounding.** `snap_to_grid_and_split_wrap` is the one place an image becomes
a (grid index, wrap) pair: round to grid integers, then floor-divide.
Flooring the raw float would turn a zero wrap into −1 on negative rounding
noise and put a spurious \(e^{\pm i\pi/2}\) on the affected q.
`fft_grid_pullback_perm` uses the same split and discards \(\mathbf L\).

**Availability.** With `required_rows`, a row that is not a permutation comes
back as all −1 with zero wraps, and a required row that fails refuses.
Without it, any failed row refuses. The refusal names the operation and the
centroid, and the fix is orbit-aware k-means. Two centroids landing on one
image means \(N\boldsymbol\tau\) is not integral.

**Closure verdict.** `load_centroid_basis` measures closure under every
spatial row of `sym` (`verify_centroid_orbit_closure`) and records
`orbit_closed`. The driver switches to the trivial view (§7) when any
centroid set of the run is not closed. The q-wedge decision
(`resolve_qgrid_symmetry`) applies the same closure test and uses the full q
grid when it fails.

**Padding.** Packed tables (§4) already have the packed extent. Canonical
tables for a suffix-padded carrier get an identity tail on `sym_perm` and a
zero tail on `L_table`, baked once in `gw.v_q_g_flat._resolve_ibz_q_list`.
`unfold_isdf_operator` then requires an exact extent match and maps that keep
logical rows logical and pad rows pad, because its `promise_in_bounds`
gathers clip an out-of-range index silently.

## 4. Orbit-packed layout and the axis-local certificate

`PackedCentroidBasis.build(centroid_indices, sym, fft_grid, mesh)` owns the
run's centroid order (`meta.mu_basis`):

1. Orbits are the connected components of the available `sym_perm` rows
   (`permutation_orbit_labels`).
2. `build_square_grouped_shard_layout(groups, (s, s))` places whole orbits on
   \(s\) shards, largest first onto the least-loaded shard, with rows in
   canonical order inside each orbit. The shard size is the largest load
   rounded up to a multiple of \(s\), so the packed extent is divisible by
   \(s^2\). A non-square mesh refuses.
3. With no nontrivial orbit (identity group, a non-closed set, or every
   centroid its own orbit) the layout is the identity with the canonical
   suffix pad.

The layout carries `packed_to_canonical` (−1 in pad slots),
`canonical_to_packed` and `active_mask`. Pad slots are fixed points with zero
wrap and exact-zero values. A dense μ solve runs at the full packed extent
([Mesh-padded axes](padding.md#orbit-packed-runtime-centroids)).

**Packed tables.** `layout.axis.pack_permutations_host(sym_perm)` conjugates
\(\alpha\) into packed order and refuses a map that leaves its shard. The
owner-local offsets are the packed map modulo the shard size
(`CentroidKUnfoldPlan.centroid_local_perm`).

**Certificate.** `certify_endpoint_locality(source_perm, mesh=, mesh_axis=,
active_mask=)` checks the actual maps: each row a permutation of the carrier,
the carrier divisible by the axis, and pad slots mapped among themselves. It
returns `is_local`, `crossing_count` and `local_perm`. `unfold_isdf_operator`
re-certifies supplied local offsets against the global tables before it
compiles the collective-free kernel, and refuses on any disagreement.
`unfold_operator_local` runs inside the caller's `shard_map` and cannot check
under `jit`; there the caller's plan owns the certificate.

**I/O seam.** Readers pack and writers unpack. `pack_axis`, `unpack_axis`,
`pack_operator` and `unpack_operator` use one all-to-all round trip per
sharded axis and never an all-gather. Nothing between the seams converts.
Files hold canonical order at the logical extent.

## 5. The parent plan and the q policy

**`CentroidKUnfoldPlan`.** `gw.centroid_k_unfold.build_centroid_k_unfold_plan(sym,
centroid_indices, fft_grid, mesh, nspinor=, parent_k_frac=, layout=)` binds
the k tables to one centroid family. The driver passes
`parent_k_frac = wfn.kvecs(k=sym.parent_k_domain)` and
`layout = meta.mu_basis.layout`. Each family, charge and current, has its own
plan.

| field | shape | contents |
|---|---|---|
| `irr_idx`, `sym_idx` | \((N_k,)\) | `sym.irr_idx_k`, `sym.sym_idx_k` |
| `sym_perm`, `L_table` | \((2n_t,M_{\rm packed})\), \((2n_t,M_{\rm packed},3)\) | packed; unavailable rows −1 |
| `k_parent_frac` | \((n_{\rm parent},3)\) | raw WFN k, used only for Bloch phases |
| `spin_action_full` | \((N_k,n_s,n_s)\) | `sym.spinor_action(sym_idx_k)`, one per full row |
| `parent_full_rows` | \((n_{\rm parent},)\) | `kirr_fullids`: the full rows at which Σ is projected on raw parents |
| `n_sym_spatial`, `nspinor` | | \(n_t\); \(n_s\in\{1,2,4\}\) |

The builder refuses a non-square mesh, \(n_s\notin\{1,2,4\}\), a layout built
for a different centroid count, `irr_idx_k` outside the parent table, and any
row that `sym_idx_k` selects without a centroid permutation.
`parent_rows(x)` takes star-invariant scalars (energies, occupations,
weights) to the parent axis. It must not be used for wavefunctions.

**`QgridTrsPolicy`.** Built only through
`gw.qgrid_symmetry.qgrid_trs_policy_for(sym=, irr_idx_q=, sym_idx_q=, kgrid=,
n_sym_spatial=, context=)`, which reads `trs_measured` from
`sym.trs_allowed`. It supplies `unfold_sym_idx` \((N_q,)\), the rows every q
unfold uses, plus `project_fixed_q` and `measure_covariance`: the
pair-coherent rows, fixed-q projector and covariance residual of
[theory §5](../theory/symmetry.md#5-interpolation-vectors-and-the-q-axis).
With time reversal broken the rows are the table's own and no Θ step runs.
Refusals:

* `GATE trs_pair_unfold_map`: q and −q have different parents under a
  time-reversal verdict;
* `GATE trs_active_rows`: a selected row is outside the active set;
* `GATE trs_measured_vs_tables`: an antiunitary table row sits beside a
  broken verdict.

## 6. Transport kernels

| kernel | operand → result | antiunitary rule | route | callers |
|---|---|---|---|---|
| `unfold_isdf_operator` | \((N^{\rm irr},n_L,n_R)\) `P(None,'x','y')` → \((N,n_L,n_R)\), same spec | `conj` (default) or `pair_transpose` | axis-local when `axis_local_sym_perm` is supplied; otherwise two all-to-alls per endpoint axis, at most one tile per rank, extents divisible by \(P\) | V (`v_q_g_flat`), RPA and ladder W (`screening`, `screening_bse`), q-parent restart readers, MPA pole fields (`pair_transpose`, zero wrap on Ω), photon blocks (axis-local, rectangular) |
| `unfold_spin_centroid_operator` | \((n_{\rm parent},M,n_s,M,n_s)\) `P(None,'x',None,'y',None)` → \((N_k,\dots)\) | `pair_transpose` over the merged (μ, s) endpoint, partner from `operator_transpose` | axis-local only; spin action after the gather | `CentroidKUnfoldPlan.unfold_operator` ← `build_G` (χ₀ Green tiles, COHSEX, photon and shared-pole Σ) |
| `unfold_load_tables` (`plan.unfold_load_tables()`) | plan tables → host `UnfoldLoadTables`: `row`, `trs` \((N_k)\); `lsrc`, `rsrc` \((N_k, M n_s)\) int32 shard-local sources (−1 = zero); `mph`, `nph` \((N_k, M n_s)\) complex128 umklapp phases with the TRS rule applied; `spin` \((N_k,n_s,n_s)\); `n_parent`; `mesh_shape` | `pair_transpose`: an antiunitary k reads the partner row | the `unfold_spin_centroid_operator` action as load tables; `local_unfold_load_tables` slices them per rank inside the `shard_map`. Refuses a source map that crosses a shard, parent rows outside \([0,n_{\rm parent})\) and a broken logical/padded split | the Σ τ sweep and static SX/RI through `ffi.fft.make_kconv_klead_unfold` (mathdx mode 7, which stores only the parent rows the projector reads); `apply_unfold_load_tables_local` is the XLA composition (cpu leg) |
| `unfold_operator_local` | one rank's tile inside the caller's `('x','y')` `shard_map` | `conj` or `pair_transpose` (`transposed_parent_local`) | local; tables are traced operands | shared-pole Σ (`mpa.sigma`) |
| `isdf.core.parent_projector_kconv` | parent pair projectors → full-zone \(Z_q\) | `conj` | the `unfold_operator_local` action fused into the native k-convolution (`ffi.fft.make_fused_conv_kparent`), fed the plan's owner-local tables and `open_spin_block_coefficient` | ζ fit |
| `isdf.zeta_mubatch.typed_child_G_tables(plan, ...)` | → parent slot, phase \(e^{-2\pi i(\bar{\mathbf k}+\mathbf G)\cdot\mathbf t}\), antiunitary flag per child G slot; \(\mathbf t=M\boldsymbol\tau\) rounded to the grid | conjugate | host tables; the Fourier image of the centroid action | route-G ζ fit |
| `unfold_wavefunction_local` | \((n_{\rm parent},\dots)\) slab with μ and spin axes → \((N_k,\dots)\) | \(\mathcal T\), then \(\mathcal U\) | local | `CentroidKUnfoldPlan.unfold_face` ← χ₀ τ chain (`w_isdf`), response bank, restart and BSE parent faces |
| `unfold_endpoint_panel` | \([n_{\rm parent},M,n_s,K]\), μ on one mesh axis and K on the other | through `unfold_wavefunction_local` | local, or a ring of \(s-1\) `ppermute`s for a nonlocal map; `endpoint_panel_cost` prices it against `max_live_bytes` | shared-pole panels (`mpa.sigma`, `mpa.sector_sigma`) |
| `unfold_isdf_one_leg` | ζ \((N_q^{\rm irr},M,n_G)\) `P(None,('x','y'),None)`, or preselected \((N_q^{\rm irr},M)\) → \((N_q,M)\) `P(None,'x')`; polar \((3,N_q,M)\) | conjugate the whole leg | gather and phases | head columns (`v_q_g_flat`) |
| `mix_lorentz_blocks` | dict of charge/current blocks | none: Λ is real | after scalar transport | `w_isdf.photon_blocks_full_q` |
| `unfold_psi`, loader `k="full_bz"` | \((n_b,n_s,n_G)\), host | plane-wave rule | host | trivial-view parents; full-grid readers of the loader |
| `gw_output.sigma_table_to_file_wedge(time_ordered_diagonal=True)` | star-wedge diagonal of the dynamic Σ_c, \((N_k^{\rm star},n_b)\) or \((n_\omega,N_k^{\rm star},n_b)\), host → file wedge | `pair_transpose`, which on a diagonal is a copy: the time-ordered Σ_c is symmetric, Σ(r,r′)=Σ(r′,r), so it does not commute with Θ and ⟨Θm|Σ|Θn⟩ = Σ_nm. `conj` flipped Im Σ_c on every time-reversed k | host | SC `sigma_diag` and eqp assembly |

In `unfold_spin_centroid_operator` the spin action runs as a CUDA FFI kernel
(`SpinRotateCentroidCudaFfi`: one thread per \((k,\mu,\nu)\) spin block, in
place) for complex128 with \(n_s\in\{2,4\}\) on GPU, and as two JAX
contractions otherwise. A provider without the handler refuses.

`unfold_isdf_operator`, and every kernel built on it, refuses:

* a row outside its table;
* a selected map that is not a bijection;
* antiunitary rows with a table that does not have \(2n_t\) rows;
* an operand whose extent does not match its tables;
* maps that do not preserve the logical/pad split;
* a rectangular `pair_transpose` without its reversed partner.

**Cost.** Axis-local transport is a pure gather: \(16\,N M_LM_R/P\) bytes read
and written per rank at complex128, and no messages. The global route moves
the tile through four volume-preserving all-to-alls. Executables are cached
by table content and mesh (axis names, device ids, shape), so V and W with the
same tables share one.

## 7. The trivial view

`SymMaps.trivial_view()` returns a shallow copy restricted to the identity.
The driver calls it when any centroid set of the run is not orbit-closed
(`gw_jax._load_system_inputs`), before any packed basis or q policy exists.
The BSE parent-face reader in `restart_bundle` does the same, and also when
the saved parent rows are the full grid of a reduced WFN.

| field on the view | value |
|---|---|
| `sym_matrices`, `translations`, `U_spinor` | the identity row |
| `sym_mats_k`, `R_cart` | \([I,-I]\) |
| `active_symmetry_rows` | `[0]` |
| `trs_allowed` | the source's measured verdict |
| `qe_operation_antiunitary`, `qe_antiunitary_rows` | `[False]`, empty |
| `irr_idx_k`, `kirr_fullids` | `arange(N_k)`; `sym_idx_k` zero; `nk_red = N_k` |
| `irr_idx_q`, `q_irr_full_idx` | `arange(N_k)`; `sym_idx_q` zero |
| `parent_k_domain` | `"full_bz"` |

Parents are the loader's full-grid rows, realized by the plane-wave action of
the original `SymMaps`. The loader keeps that original for G spheres, file
energies and file-wedge output. The view refuses unless exactly one identity
row exists.

The view selects no antiunitary action, but time reversal is a property of the
Hamiltonian, so `trs_allowed` keeps the measured verdict: on a time-reversal
symmetric crystal response stores stay unordered and the full shared-pole head
runs. Each q is its own parent with the identity row, so no q is paired with
−q through a table. The resolvent ladders (`screening_diagrams = w_bse` or
`w_rpa_resolvent`) solve on the WFN's reduced q wedge and refuse the view at
startup (`GATE resolvent_ladder_trivial_view`).

The in-tree non-closed sets are deliberate test specimens and are not
regenerated: `si_cohsex_debug/centroids_frac_960.txt`,
`si_bse_debug/centroids_frac_480.txt`, `cohsex_debug/centroids_frac_60.txt`
and `bispinor_debug/centroids_frac_256.txt`.

## 8. There are two different IBZs: file wedge and star wedge

A band-index quantity (energies, \(\Sigma_{mn}\), \(H^{\rm QP}\), U) needs no
centroid transport: rows are gathered under the band-matrix rule of
[theory §4](../theory/symmetry.md#4-two-point-operators-g-v-w). Two reduced k
sets exist:

| wedge | rows | length | addressed by | used by |
|---|---|---|---|---|
| file wedge | `wfn.kpoints`, the raw parents | `nk_red` | `kirr_fullids` | `.dat` outputs, `kin_ion.h5`, `sigma_mnk.h5`, Σ projection |
| star wedge | first full-grid row of each orbit (`star_select`) | number of parents in use | `irr_idx_k` labels | the QSGW loop's H, E and U under `sc_on_ibz` (`KStarMap`) |

They coincide when every stored k lies in a distinct orbit (Si: 64 → 8 and
8). They differ when the WFN stores two k of one orbit (`cohsex_debug`: 9 → 4
and 3; `gnppm_debug` and `bispinor_debug`: 9 → 9 and 5). Equal lengths do not
imply equal row order.

| direction | function | antiunitary predicate |
|---|---|---|
| file → full | `unfold_file_wedge_to_full_bz`, `unfold_file_wedge_band_operator(trs_rule=)`, `unfold_file_wedge_polar_matrix` | the member's own row (`trs_reference="ibz_slab"`) |
| star → full | `unfold_star_wedge_to_full_bz`, `KStarMap.broadcast` | XOR of the member's and the star reference's rows (`"star_row"`) |
| full → file | `reduce_full_bz_to_file_wedge` | none: row selection |
| full → star | `star_select`, `KStarMap.select` | none |

`star_broadcast(..., trs_reference=, trs_rule="conj"|"transpose")` is the one
backend, and `trs_reference` has no default. There is no star → file
operation; go star → full → file. Reducing to the file wedge and unfolding
back is not the identity when one stored k is an image of another: the row
with no children is replaced by its partner's image.

`star_spread` measures how consistent a full-grid array is with its own star
relation. It cannot see an error applied uniformly to a whole orbit, and a
wrong conjugation predicate conjugates entire stars. Conjugation also leaves a
Hermitian diagonal unchanged. Such errors show only against independently
computed full-grid values.

An array stored on a wedge but indexed with a full-grid index returns a wrong
row silently for every index below `nk_red`. Readers check the storage stamp
(§9) before indexing.

Links between two k (parallel-transport overlaps) use the directed-edge table
on the [service page](../services/symmetry_maps.md#directed-band-matrix-edges).
The table exists only when every operation maps an elementary mesh step to ±
an elementary step (`file_io.parallel_transport.link_symmetry_reduction_applies`).
That holds for simple-cubic, tetragonal and orthorhombic primitive cells and
fails for bcc and fcc primitive cells, whose links are streamed on the full
grid.

## 9. What each file stores

| file or dataset | k or q basis | declared by |
|---|---|---|
| `eqp0.dat`, `eqp1.dat`, `sigma_diag.dat` | file wedge | coordinates on each block |
| `sigma_mnk.h5` | file wedge | `kin_ion` stamp contract, plus star-spread attributes measured on the full grid before reduction |
| `kin_ion.h5` | file wedge | per-dataset `k_storage` and version, `irr_idx_k`, `sym_idx_k`; no attribute means full |
| `qp_wfn_rotations.h5` | file wedge only when the writer's own round trip reproduces the arrays, otherwise full (`qp_rotations_k_storage = auto`) | `k_storage` |
| `isdf_tensors.h5` `V_qmunu`, `W0_qmunu` | q parents when reduced (`restart_q_storage = auto`) | per-dataset `q_storage` and the reconstruction tables |
| `isdf_tensors.h5` `psi_parent_y*` | raw WFN parents, canonical centroids | `psi_parent_k_rows`; a `psi_full_*` dataset refuses |
| `zeta_q.h5` | q parents or full | q extent |
| `mpa_*.h5` | q parents | tables in the file |
| `v_q_bispinor.h5` | q parents, seven unique tiles | per-dataset tables, one set per centroid family |
| `dipole.h5` | full grid | none |
| `WFN.h5`, `WFN_qp.h5` | file wedge | BGW header |

A q-parent tensor carries its own tables in canonical centroid order at the
logical extent: `irr_idx_q`, `sym_idx_q`, `q_irr_frac`, `sym_perm`, `L_table`
and `n_sym_spatial` (`symmetry_maps.qirr_store`). The writer refuses a
non-closed centroid set. The reader refuses a digest mismatch, a rank that
does not match the stamped version, and a `q_storage` that contradicts the q
extent. It never re-derives tables from the current run.

## 10. Refusals

| where | condition | fix |
|---|---|---|
| `SymMaps` | `allow_trs=` passed (`GATE retired_SymMaps_allow_trs`) | construct from `WfnLoader`; the verdict is measured |
| `SymMaps` | `wfn.trs_holds` not a boolean (`GATE SymMaps_needs_measured_trs`) | same |
| `SymMaps` | `kgrid` not three positive extents, or more stored k than grid points | fix the WFN header |
| `SymMaps` | \(n_t=1\) full-grid file whose operation is not the identity, or whose k do not cover the mesh | fix the WFN |
| `SymMaps` | a stored k off the grid of its own `kgrid`/`shift` | fix the k list or `kgrid`/`shift` |
| `SymMaps` | a full-grid k unreachable with the authorized rows | QE-typed: the schema's operations are incomplete. Time reversal broken: regenerate with `noinv=.true.`. Otherwise regenerate with a consistent header or a full grid. Never a Γ or identity substitute |
| `centroid_source_map_and_wrap` | a required operation does not permute the centroids | orbit-aware k-means |
| `pack_permutations_host`, `certify_endpoint_locality` | a map crosses a shard | build the layout from the same orbits |
| layout, plan | non-square mesh | a square process count |
| `unfold_*` | §6 | |
| `QgridTrsPolicy` | §5 | |
| q-parent reader | §9 | regenerate the file |
| `unfold_psi` | `len(sym_mats_k) != 2·len(U_spinor_spatial)` | pass the augmented table |

## 11. Function contracts

The service's functions point here for their contract. This is what each one
guarantees; the signatures are in the code.

**Grids and q labels.**

* `kgrid_shift_map(nkx, nky, nkz, q_off)` → `kpq_index` \((N_k,)\), `G_umk`
  \((N_k,3)\) int32: the C-order row of \(\mathbf k+\mathbf q_{\rm off}\) and
  its per-axis floor-division wrap. `arr[kpq_index]` equals `jnp.roll` by
  \(-\mathbf q_{\rm off}\) on the C-order grid. `G_umk` drives
  \(e^{-2\pi i\mathbf G_{\rm umk}\cdot\mathbf s_\mu}\) on the cell-periodic
  \(u\).
* `bgw_signed_q_representative(q)`: rows in \([-1/2,1)\); components above
  \(1/2\) wrap negative and \(+1/2\) stays.
* `bgw_integer_q_to_fractional(q_int, kgrid)`: \(q>k/2\mapsto q-k\); the
  even-grid half point stays positive.
* `q_negation_index(kgrid)`: the C-order permutation \(\mathbf q\mapsto-\mathbf q\).
* `common_uniform_grid_indices(a, b)`: aligned C-order rows of the points two
  unshifted grids share (per-axis \(\gcd\)), integer only.
* `slice_q_full_to_ibz(arr, q_irr_full_idx, out_sharding=None)`: row gather
  on axis 0; the trailing sharding is kept.

**Planners.**

* `find_irreducible_bz_points(full_int, sym_mats_k, irr_kgrid_int=None)` →
  `(irr_idx, sym_idx, irr_out)`, integer. Derived parents are the smallest
  orbit members (q side). With an anchored list, the highest-row,
  lowest-operation rule applies (k side). A point with no preimage refuses.
* `map_full_kpoints_to_irreducible(kpoints, sym_mats_k, full, tol=1e-6)` →
  `(parent, op, matched)`: fractional coordinates, the same tie rule. `matched`
  lets the caller refuse. It never decides time reversal.
* `build_spatial_operator_tables(wfn)`: `mtrx`, `mtrx.T`, `tnp`, Cartesian and
  spinor tables with no k map, so the two-component reference check can
  measure a WFN that `SymMaps` would refuse.
* `SymMaps.__init__(wfn, *, allow_trs=None)`: builds §1 and §2; refusals in
  §10.
* `SymMaps.create_kpoint_symmetry_map(wfn)`: the uniform grid, nothing else.
* `SymMaps.q_irr_is_full_identity` (property): the q parents are exactly the
  ordered full table, which is stronger than equal counts.
* `SymMaps.get_kminusq_map`, `_get_kminusq_index_map`: \(\mathbf k-\mathbf q\)
  rows by quantized periodic lookup, \(O(N_kN_q)\).
* `SymMaps.get_umklapp_vector(...)`: BGW `kg0`; refuses a non-integral
  difference.
* `SymMaps.find_qpoint_index(q, tol)`: the row in the full q table; raises if
  absent.

**Operation representations.**

* `SymMaps.syms_crystal_to_cartesian(wfn)`:
  \(R_{\rm cart}=a^{\mathsf T}M(a^{\mathsf T})^{-1}\), the Cartesian image of
  `mtrx`, which is the inverse rotation; rows \(\ge n_t\) negated. The SU(2)
  extraction uses the transposed Shepperd form, so the two inversions cancel.
  A Cartesian index uses `cartesian_action`, which returns the forward
  rotation.
* `SymMaps.get_spinor_rotations(wfn, R)`: \((n,2,2)\) SU(2) by
  Markley–Shepperd quaternions; improper rows are made proper by \(R\to-R\).
* `SymMaps.operation_rows`, `cartesian_action`, `lorentz_action`,
  `spinor_action`, `reciprocal_phase`, `unfold_wavefunction`: §1.
* `spinor_rotation_for_sym_row(U, rows, n_t, nspinor=, R_cart=)`: the table
  in [theory §3](../theory/symmetry.md#3-wavefunctions); `nspinor=4` requires
  `R_cart`.
* `apply_spinor_rotation(U, c)`: \(n_s\in\{1,2\}\), written as two
  multiply-adds because a \(K=2\) GEMM fails XLA's sharding autotuner at
  production shapes. NumPy in, NumPy out.
* `tau_phase_row(S, t, K)` → \(e^{-i(S\mathbf K)\cdot\mathbf t}\) or `None`.
  `tau_phase_row_jax` returns ones instead of `None`.
* `unfold_reciprocal_carriers(S, G, g0)` → \(S\mathbf G-\mathbf g_0\).
* `unfold_psi(c, *, sym_idx, g_kbar, sym_mats_k, translations,
  U_spinor_spatial)`: \((n_b,n_s,n_G)\) in and out, on the parent G axis.
  Conjugation comes before the phase. It refuses unless
  `len(sym_mats_k) == 2·len(U_spinor_spatial)`, and reads \(n_s\) from the
  array.
* `open_spin_block_coefficient(U, a, b)` →
  \(c[k,c,d]=U_k[a,c]\overline{U_k[b,d]}\): one output spin block of
  \(UOU^\dagger\), for consumers that hold one block at a time.
* `_rotate_open_spin_centroid_operator(spatial, spin)`: \(UOU^\dagger\) by two
  local contractions over the resident spin axes.

**Transport internals.**

* `_apply_unfold_phase_and_trs_local`: wrap phases and the antiunitary rule
  on one local tile. The transpose rule conjugates the other endpoint's phase
  because its operand is the partner.
* `_get_unfold_isdf_operator_jit`: the executable is cached on table bytes,
  extents, rule and mesh; the tables are embedded as NumPy constants.
* `isdf_one_leg_source_slots(gvec, *, sym, sym_idx, q_irr_frac, kgrid)`: the
  parent sphere slot of each child's \(\mathbf G'=0\) coefficient. A missing or
  duplicated \(\mathbf G_p\) refuses (`GATE isdf_one_leg_parent_g`). A carrier
  that holds these slots for each parent gives the same one-leg result as the
  whole sphere.
* `_get_unfold_isdf_one_leg_jit`: slots, τ phases, q and wraps are runtime
  operands, so tied head columns share one executable.

**Star map.**

* `_star_row_order(irr_idx_k)` → `(rows, labels)`: the first full-grid row of
  each label, in full-grid order. Select and broadcast both address this
  order, never the order of `np.unique`.
* `_star_conj_flags`: the XOR predicate, the only one in the package.
* `_spread_tables`, `_star_stats`: members, references and flags; `(worst,
  scale)` from one compiled reduction.
* `_broadcast_rows(A, take, trs, transpose=)`: gather, then conjugate or
  band-transpose the flagged rows through `apply_band_matrix_symmetry`.
* `_row_out_sharding`, `_scalar_out_sharding`, `_jit_with`: a row gather keeps
  the operand's sharding when axis 0 is replicated, the spread result is
  replicated `P()`, and `out_shardings` is passed only when it is set.
* `star_select`, `star_broadcast`, `star_spread`; `star_tables_of(sym)` →
  `(irr_idx_k, sym_idx_k, len(sym_mats_k)//2)`, public so that wedge writers
  stamp the same \(n_t\).
* `unfold_file_wedge_to_full_bz`, `unfold_file_wedge_band_operator`,
  `unfold_file_wedge_polar_matrix`, `reduce_full_bz_to_file_wedge`,
  `unfold_star_wedge_to_full_bz`: §8. Each file-wedge unfold refuses an operand
  whose leading extent is not `nk_red`. The polar variant applies the forward
  polar time-odd Cartesian action; translation phases cancel between an
  equal-k bra and ket.
* `KStarMap(irr, sym, n)`: the three arrays that must travel together.
  `identity(n_k)` is the no-reduction map; `select`, `broadcast` (star-row
  rule), `spread`, and `spread_rel` (one reduction, one 16-byte transfer).
