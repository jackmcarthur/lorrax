"""QP wavefunction rotation matrix I/O."""
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

