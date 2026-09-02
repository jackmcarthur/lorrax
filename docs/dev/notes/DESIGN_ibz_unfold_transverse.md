# Design: IBZ-local unfold for the transverse four-current path

Status: design only.  The shipping parent-k acceleration remains charge-only;
`bispinor = true` uses full-k wavefunction storage and prints
`GATE parent_k_green_bispinor_vector_unfold_unimplemented` when the charge-only
path would otherwise be selected.

## Service action and representation

The Cartesian-vector action already belongs to `symmetry_maps`; no GW-local
rotation may be added.  For a q stencil, call `q_stencil_orbit_table` with
`SymMaps.active_symmetry_rows`, then unfold the source operator with
`apply_band_matrix_symmetry`.  Its `component_mix` must come from

```python
sym.cartesian_action(target_sym_idx, axial=False, time_odd=True)
```

and its `antiunitary` flag must be the table's `target_antiunitary`.  This is
the polar, time-odd action of a current.  For a band matrix joining distinct
endpoints, first use `directed_edge_orbit_table`; pass both endpoint sewing
matrices to `apply_band_matrix_symmetry`, because the current vertex joins
`k` and `k-q` and nonsymmorphic phases do not cancel.  The equal-k shortcut
`unfold_file_wedge_polar_matrix` is therefore insufficient for screening.

For an ISDF one-leg coefficient, the matching service primitive is
`unfold_isdf_one_leg(..., action="polar", source=a)`.  It combines the same
typed Cartesian column with the centroid pullback, q/G relabel, translation
phase, and antiunitary conjugation.  A transverse implementation should use
that primitive for vector-valued zeta data and the q-stencil/band-matrix pair
above for band operators; it must not copy either action into `gw`.

## Objects that must move together

1. **`zeta_T`.**  The three files `zeta_q_mu{1,2,3}.h5` are one polar-vector
   object, not three unrelated scalar wedges.  Store one authenticated parent-q
   vector and unfold all three components together with
   `unfold_isdf_one_leg(action="polar")`.  The component mixing must precede
   construction of CT/TC/TT tiles; scalar centroid permutation and phase alone
   are incomplete.

2. **The current bundle.**  The transverse-centroid wavefunctions and the
   gamma/current vertices must retain raw parent rows plus the band-space
   sewings for both `k` and `k-q`.  The present `ParentGreenCarrier` contains
   only the scalar charge operands and a point-local spin/centroid action.  A
   transverse carrier must bind the vector component axis and the two endpoint
   actions in one authenticated plan; selecting parent rows from an already
   unfolded full-k wavefunction is not equivalent.

3. **The sixteen-block response.**  Treat `chi^{IJ}`, `I,J in {0,x,y,z}`, as
   one Lorentz-block operator.  CC is scalar, CT and TC each carry one polar
   action, and TT carries two.  A practical service call can flatten `(I,J)`
   to one 16-component axis and pass the product representation
   `diag(1,R) tensor diag(1,R)` as `component_mix` to
   `apply_band_matrix_symmetry`; antiunitary conjugation is applied once.
   Unfold only after the full parent-k sum for each source q, then restore the
   canonical centroid basis before `photon_layout.pack_photon_operator`, the
   Ward subtraction, Dyson solve, head completion, or Sigma consumer.

## Saving and acceptance boundary

The saving is confined to the Green-function band contractions.  The current
implementation builds two G matrices per tau node for each of 16 blocks, so
the k-batched work is proportional to `2 * 16 * N_tau * n_k_full`.  Contracting
raw WFN parents changes this to `2 * 16 * N_tau * n_k_parent`, saving
`2 * 16 * N_tau * (n_k_full - n_k_parent)` per-k GEMMs; the ideal contraction
speedup is `n_k_full / n_k_parent` (8x for a 64-to-8 wedge).  q FFTs, component
mixing, packed assembly, the distributed Dyson solve, and Sigma are unchanged,
so this ratio is not an end-to-end speedup claim.

Landing requires a P=4 negative/positive pair and value parity against the
full-k path for all 16 blocks, including a nontrivial rotation, an antiunitary
row, distinct `k`/`k-q` sewings, CT/TC orientation, TT Ward subtraction, and
the final packed Sigma.  Until those checks exist, fallback is the correctness
path.
