"""QP wavefunction rotation matrix I/O.

Two writers live here:

* :func:`write_qp_rotations_h5` — small ``(U, E_qp)`` companion file used
  by tools that want to apply the QP rotation themselves.
* :func:`write_qp_wfn_h5` — full BGW-compatible ``WFN.h5`` with ψ already
  rotated and energies replaced.  This is the canonical "QP WFN" output
  consumed by downstream BSE / restart paths that just want a WFN.h5
  drop-in replacement.
"""
import numpy as np
import h5py


def write_qp_rotations_h5(
    filepath: str,
    U_mnk: np.ndarray,
    E_qp_nk: np.ndarray,
    band_start: int,
    band_stop: int,
    kpoints_crys: np.ndarray,
    nkx: int,
    nky: int,
    nkz: int,
    kpoints_reduced: np.ndarray = None,
    kirr_to_kfull: np.ndarray = None,
):
    """Write QP rotation matrices and eigenvalues to HDF5 file.
    
    This file can be used to postprocess WFN.h5 → WFN_qp.h5 by rotating
    the G-vector coefficients and replacing eigenvalues.
    
    Args:
        filepath: Output path for the h5 file
        U_mnk: Unitary matrices (nk, nb, nb) where U[k,m,n] = <m_DFT|n_QP>
               To rotate coefficients: c_qp_n(G) = Σ_m U[k,m,n] c_dft_m(G)
        E_qp_nk: QP eigenvalues (nk, nb) in Hartree atomic units
        band_start: First band index (0-based) included in the calculation
        band_stop: One past last band index included
        kpoints_crys: Full k-mesh in crystal coordinates (nk, 3)
        nkx, nky, nkz: k-mesh dimensions
        kpoints_reduced: Reduced k-points from WFN.h5 (nk_red, 3), optional
        kirr_to_kfull: Mapping from reduced k-point index to full zone index, optional
    
    For postprocessing WFN.h5 → WFN_qp.h5:
        1. Load WFN.h5 coefficients for bands [band_start:band_stop]
        2. For each k-point k:
           c_qp[n, G] = Σ_m U[k, m, n] * c_dft[m, G]  (matrix form: c_qp = U^T @ c_dft)
        3. Replace eigenvalues with E_qp_nk (convert to Rydberg if needed)
        4. Write rotated coefficients back to WFN_qp.h5
    """
    with h5py.File(filepath, 'w') as f:
        # Main data
        f.create_dataset('U_mnk', data=U_mnk, dtype=np.complex128)
        f.create_dataset('E_qp_nk_hartree', data=E_qp_nk, dtype=np.float64)
        f.create_dataset('E_qp_nk_rydberg', data=E_qp_nk * 2.0, dtype=np.float64)  # Also save in Ry
        
        # Metadata
        f.create_dataset('band_range', data=np.array([band_start, band_stop], dtype=np.int32))
        f.create_dataset('kpoints_crys', data=kpoints_crys, dtype=np.float64)
        f.create_dataset('kgrid', data=np.array([nkx, nky, nkz], dtype=np.int32))
        
        # Optional: reduced k-points and mapping for easy WFN.h5 lookup
        if kpoints_reduced is not None:
            f.create_dataset('kpoints_reduced', data=kpoints_reduced, dtype=np.float64)
        if kirr_to_kfull is not None:
            f.create_dataset('kirr_to_kfull', data=kirr_to_kfull, dtype=np.int32)
        
        # Attributes for documentation
        f.attrs['description'] = (
            'QP rotation data for transforming DFT wavefunctions to QP basis. '
            'U_mnk[k,m,n] = <m_DFT|n_QP>. '
            'To rotate: c_qp[n,G] = sum_m U[k,m,n] * c_dft[m,G] (i.e. c_qp = U^T @ c_dft)'
        )
        f.attrs['energy_units'] = 'E_qp_nk_hartree in Hartree, E_qp_nk_rydberg in Rydberg'
        f.attrs['band_convention'] = '0-based indexing; bands [band_start, band_stop) were computed'
        if kirr_to_kfull is not None:
            f.attrs['mapping_description'] = (
                'kirr_to_kfull[ik_red] gives the index into kpoints_crys/U_mnk/E_qp_nk '
                'for the reduced k-point ik_red from WFN.h5'
            )


# ---------------------------------------------------------------------------
# Full WFN.h5 with rotated ψ + replaced energies
# ---------------------------------------------------------------------------

def write_qp_wfn_h5(
    output_path: str,
    wfn,                                    # WFNReader (also serves as `crystal` for the writer)
    U_kmn: np.ndarray,                      # (nk, nb_active, nb_active)  ⟨DFT_m | QP_n⟩
    enk_active_qp_ry: np.ndarray,           # (nk, nb_active)             E_QP for active block, Ry
    band_start: int,
    band_stop: int,
) -> None:
    """Write a BGW-compatible WFN.h5 with QP-rotated ψ and replaced energies.

    For each k:
      * Bands ``[band_start, band_stop)`` (the "active" block):
          ``c_qp[n, s, G] = Σ_m U[k, m, n] · c_dft[m, s, G]``
          ``E[n] ← enk_active_qp_ry[k, n - band_start]``
      * All other bands keep their DFT coefficients and DFT energies
        unchanged — the SC iteration only touched the active subspace.

    The ``wfn`` argument is a :class:`~file_io.wfnreader.WFNReader` and
    is reused as the ``crystal`` source for :class:`WFNWriter` (it
    exposes the same ``nspin``, ``nspinor``, ``nelec``, ``ecutwfc``,
    ``ecutrho``, ``fft_grid``, ``avec``, ``bdot``, … attributes the
    writer reads).

    Notes
    -----
    Symmetry & k-mesh: the rotation is on the irreducible-k WFN —
    output is on the same irreducible-k grid as the input, with the
    ``mtrx`` / ``tnp`` blocks copied through.  Symmetry-equivalent
    full-zone wavefunctions are reconstructed by downstream consumers
    (BSE, etc.) via the same maps as for the input WFN.

    Spinors: handled identically per (k, s) — the rotation is in band
    space and does not mix spinor components.

    Occupations: ``ifmin`` / ``ifmax`` are inherited from the source
    WFN via :class:`WFNWriter` (which sets ``ifmax = nelec`` on every
    k).  Safe whenever the SC iteration preserves the overall
    valence/conduction band ordering — which holds for insulators with
    QP shifts smaller than the gap.  For metals or near-gap-closure
    systems, an SC update can permute occupied vs. empty bands; the
    output ``occ`` array would then be wrong and would need to be
    recomputed from the QP energies + a fresh midgap E_F before
    handing the file to a downstream consumer that trusts ``occ``.
    """
    from .wfn_writer import WFNWriter

    nb_active = int(band_stop - band_start)
    if U_kmn.shape != (wfn.nkpts, nb_active, nb_active):
        raise ValueError(
            f"write_qp_wfn_h5: U shape {U_kmn.shape} inconsistent with "
            f"(nk={wfn.nkpts}, nb_active={nb_active}).")
    if enk_active_qp_ry.shape != (wfn.nkpts, nb_active):
        raise ValueError(
            f"write_qp_wfn_h5: enk_active_qp_ry shape "
            f"{enk_active_qp_ry.shape} inconsistent with "
            f"(nk={wfn.nkpts}, nb_active={nb_active}).")

    # All-band IBZ coefficients + per-k G-vectors via the unified loader.
    # qp_wfn is a one-shot host-mutate-write path; everything stays on
    # the host (sharding=None forces replicated → host numpy view).
    enk_full_ry = np.array(wfn.energies[0], dtype=np.float64).copy()  # (nk, nbands)
    enk_full_ry[:, band_start:band_stop] = np.asarray(
        enk_active_qp_ry, dtype=np.float64)

    # The top-level WfnLoader carries the device mesh; ``.load`` is a
    # collective on the phdf5 backend, so it MUST be called by every rank
    # even though only rank-0 writes the file.  We open a fresh
    # mesh-less WfnLoader (eager backend) here so the rank-0 write does
    # not need the other ranks at all — qp_wfn is a one-shot
    # end-of-run dump and the re-slurp cost is paid once.
    from .wfn_loader import WfnLoader
    loader = WfnLoader(wfn._filename)
    psi_all = np.asarray(loader.load(
        bands=(0, int(wfn.nbands)), k="ibz", sharding=None))   # (nk, nb, ns, ngkmax)
    gvecs_full = loader.gvecs(k="ibz")                          # (nk, ngkmax, 3)
    ngk_v = loader.ngk_valid(k="ibz")                           # (nk,)
    gvecs_per_k = [gvecs_full[ik, : int(ngk_v[ik])]
                   for ik in range(int(wfn.nkpts))]
    loader.close()

    with WFNWriter(
        output_path, wfn,
        kpoints=np.asarray(wfn.kpoints, dtype=np.float64),
        weights=np.asarray(wfn.kweights, dtype=np.float64),
        kgrid=tuple(int(x) for x in wfn.kgrid),
        nbands=int(wfn.nbands),
        gvecs_per_k=gvecs_per_k,
        nosym=False,
        shift=tuple(float(x) for x in wfn.shift),
    ) as w:
        for ik in range(int(wfn.nkpts)):
            n = int(ngk_v[ik])
            c_all_dft = psi_all[ik, :, :, :n].copy()           # (nbands, ns, ngk)
            c_active_dft = c_all_dft[band_start:band_stop]      # view
            c_all_dft[band_start:band_stop] = np.einsum(
                "mn,msg->nsg", U_kmn[ik], c_active_dft, optimize=True)
            w.write_k(ik, enk_full_ry[ik], c_all_dft)


