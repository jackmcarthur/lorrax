# Appendix D — File formats

BerkeleyGW formats (`WFN.h5`, `eqp*.dat`, `vcoul`, `eigenvectors.h5`) follow the
published BGW specifications and are not restated. LORRAX-native files:

| File | Layout |
|---|---|
| `centroids_frac_<N>.txt` | fractional coordinates, one per line; header comments record the seed, weight (`density: scalar` or `current`), and orbit-closure provenance |
| ζ store (per-q) | flat-q dataset `(n_q, n_r, n_μ)`, μ innermost (contiguous for both the fit's r-chunk writes and $V_q$'s per-q reads); per-q G-sphere, padded, on the `zeta_cutoff` sphere |
| `isdf_tensors_<n_rmu>.h5` | tagged-array restart bundle: centroid ψ values, $V_q$, fit metadata, and the provenance keys the restart check validates (§10.3) |
| `sigma_mnk.h5` | $\Sigma_{ij\mathbf{k}}(\omega)$: complex datasets indexed (k, i, j, ω) plus the ω grid and Fermi reference |
| `sigma_diag.dat` | text, per (k, band): the Σ decomposition of the active mode |
| `qp_rotations.h5` | per-k quasiparticle unitaries $U_{ij}$ and QP energies |
| `kin_ion.h5`, `dipole.h5` | kinetic+ionic and velocity matrix elements over the band window, k-resolved |

<!-- TODO(verify): dataset-level names/shapes table generated from the writers
     (file_io/sigma_output.py, wfn_writer.py, tagged_arrays.py) before
     publication. -->
