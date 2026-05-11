# TODO(fractional translations): the per-k readers here (``get_cnk``,
# ``get_cnk_batch``, ``get_gvec_nk``) return raw irreducible-zone data
# with no unfolding, rotation, or τ phase applied.  All symmetry unfolding
# (including the non-symmorphic fractional-translation phase and spinor
# rotation) lives in ``common.symmetry_maps.SymMaps.get_cnk_fullzone``;
# G-vectors are unfolded by ``SymMaps.get_gvecs_kfull`` which intentionally
# omits τ because scalar functions of |q+G| don't need it.  If future
# callers combine raw coefficients with unfolded G-indices directly (i.e.
# not via ``get_cnk_fullzone``), the fractional-translation phase must be
# threaded in explicitly to stay consistent with BGW/gmap.f90.
import h5py as h5
import numpy as np

from .mf_header import read_mf_header_from_file

class WFNReader:
    def __init__(self, filename):
        """Initialize WFNReader with WFN file."""
        self._filename = filename
        self._file = h5.File(filename, 'r')

        # mf_header — read via the shared module so the on-disk schema
        # lives in one place (file_io/mf_header.py).  Every attribute
        # this class historically exposed is mirrored 1:1 from
        # :class:`MfHeader`.
        hdr = read_mf_header_from_file(self._file)
        self.version = hdr.version
        self.flavor = hdr.flavor
        # Current workflows assume two-component Pauli spinors.  Support for the
        # four-component formalism under development will hook in here once the
        # wavefunction files expose the extra components.  Metadata related to
        # CTSP grids will also be parsed here when available in future files.
        self.nspin = hdr.nspin
        self.nspinor = hdr.nspinor
        self.nkpts = hdr.nkpts
        self.nbands = hdr.nbands
        self.ngkmax = hdr.ngkmax
        self.ecutwfc = hdr.ecutwfc
        self.kgrid = hdr.kgrid
        self.shift = hdr.shift
        self.ngk = hdr.ngk
        self.ifmin = hdr.ifmin
        self.ifmax = hdr.ifmax
        self.kweights = hdr.kweights
        self.kpoints = hdr.kpoints
        self.energies = hdr.energies
        self.occs = hdr.occs
        self.ng = hdr.ng
        self.ecutrho = hdr.ecutrho
        self.fft_grid = hdr.fft_grid
        self.ntran = hdr.ntran
        self.cell_symmetry = hdr.cell_symmetry
        self.sym_matrices = hdr.sym_matrices
        self.translations = hdr.translations
        self.cell_volume = hdr.cell_volume
        self.recip_volume = hdr.recip_volume
        self.alat = hdr.alat
        self.blat = hdr.blat
        self.nat = hdr.nat
        self.avec = hdr.avec
        self.bvec = hdr.bvec
        self.adot = hdr.adot
        self.bdot = hdr.bdot
        self.atom_types = hdr.atom_types
        self.atom_positions = hdr.atom_positions

        # Number of occupied bands (ifmax is 1-based occupied-band index in WFN files).
        if np.size(self.ifmax) > 0:
            self.nelec = int(np.max(self.ifmax))
        else:
            self.nelec = int(np.sum(self.occs[0, 0] > 0.5))

        nband = int(self.energies.shape[-1])
        occ_idx = max(0, min(self.nelec - 1, nband - 1))
        self.vbm = float(np.max(self.energies[:, :, occ_idx]))
        if occ_idx + 1 < nband:
            self.cbm = float(np.min(self.energies[:, :, occ_idx + 1]))
            self.efermi = 0.5 * (self.vbm + self.cbm)
        else:
            self.cbm = float(self.vbm)
            self.efermi = float(self.vbm)

        # /wfns/* — not part of mf_header; WFN-specific datasets.
        self.gvecs = self._file['wfns/gvecs'][()]
        self.coeffs = self._file['wfns/coeffs'][:]

        # this is one correct way to get the atomic positions in crystal coordinates (they're in units of alat, cartesian)
        self.atom_crys = np.einsum('ij,kj->ki',np.linalg.inv(self.avec).T, self.atom_positions)

        # Calculate k-point starts
        self.kpt_starts = np.zeros(self.nkpts, dtype=np.int64)
        for ik in range(1, self.nkpts):
            self.kpt_starts[ik] = self.kpt_starts[ik-1] + self.ngk[ik-1]

        # Band norms from G-space coefficients: ||c_n||² = Σ_{s,G} |c_{n,s}(G)|²
        # For deterministic bands: ~1.0.  For pseudobands: sqrt(n_eff/k).
        # Computed at k=0; consistent across k-points for uniform grids.
        start0 = int(self.kpt_starts[0])
        end0 = start0 + int(self.ngk[0])
        c0 = self.coeffs[:, :, start0:end0, :]  # (nbands, nspinor, ngk0, 2)
        self.band_norms = np.sqrt(
            np.sum(c0[:, :, :, 0]**2 + c0[:, :, :, 1]**2, axis=(1, 2))
        )  # (nbands,)
        # Clamp zero norms to 1.0 so division is a no-op for zero-weight bands
        # (their coefficients are already zero, so /1.0 keeps them zero).
        self.band_norms = np.where(self.band_norms > 0.5, self.band_norms, 1.0)

    def __del__(self):
        f = getattr(self, "_file", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass

    def get_cnk(self, ik, ib):
        """Get complex coefficients for both spinor components of a wavefunction.
        
        Args:
            ik (int): k-point index
            ib (int): band index
            
        Returns:
            np.ndarray: Complex coefficients array of shape (2, ngk[ik]) for both spinor components
        """
        start = self.kpt_starts[ik]
        end = start + self.ngk[ik]
        nkg = self.ngk[ik]
        cnk = np.zeros((2, nkg), dtype=np.complex128)
        cnk[0, :] = self.coeffs[ib, 0, start:end, 0] + 1j * self.coeffs[ib, 0, start:end, 1]
        cnk[1, :] = self.coeffs[ib, 1, start:end, 0] + 1j * self.coeffs[ib, 1, start:end, 1]
        return cnk

    def get_cnk_batch(self, ik, band_indices):
        """Get complex coefficients for multiple bands at once (vectorized).
        
        Args:
            ik (int): k-point index
            band_indices: array-like of band indices
            
        Returns:
            np.ndarray: Complex coefficients of shape (nb, 2, ngk[ik])
        """
        start = self.kpt_starts[ik]
        end = start + self.ngk[ik]
        band_indices = np.asarray(band_indices, dtype=np.int64)
        # coeffs shape: (mnband, nspinor, ngktot, 2) where last dim is real/imag
        raw = self.coeffs[band_indices, :, start:end, :]  # (nb, 2, ngk, 2)
        return raw[..., 0] + 1j * raw[..., 1]  # (nb, 2, ngk)
    
    def get_gvec_nk(self, ik):
        """Get G-vectors for a specific k-point.
        
        Args:
            ik (int): k-point index
            
        Returns:
            np.ndarray: G-vectors array of shape (ngk[ik], 3) in Fortran order,
                       where each row is a G-vector [Gx, Gy, Gz]
        """
        # Add debug prints
        #print(f"Getting G-vectors for k-point {ik}")
        # Get start and end indices for this k-point
        start = self.kpt_starts[ik]
        end = start + self.ngk[ik]
        
        # Get G-vectors from file
        gvecs = self.gvecs[start:end,:]
        
        #print(f"Retrieved G-vectors shape: {gvecs.shape}")
        
        # Return in shape (ngk[ik], 3) in Fortran order
        return gvecs
    
    def get_syms_from_kgridlog(self,kgridfname):
        matrices = [np.eye(3, dtype=np.int32)]
        parsing = False
        
        with open(kgridfname, 'r') as f:
            for line in f:
                if "symmetries that reduce the k-points" in line:
                    parsing = True
                    continue
                
                if parsing and line.strip():
                    # Check if line starts with 'r' followed by numbers
                    if line.strip().startswith('r'):
                        # Extract the matrix elements
                        parts = line.split('=')[1].strip().split()
                        if len(parts) != 9:
                            continue
                        
                        # Convert to integers and reshape to 3x3
                        matrix = np.array([int(x) for x in parts]).reshape(3, 3)
                        matrices.append(matrix)
                    else:
                        # Stop parsing if we hit a line that doesn't match format
                        parsing = False
        
        return np.array(matrices, dtype=np.int32)


    def get_cnk_fullzone_gpu(self,ik,ib):
        pass

# if name == main, load WFN.h5 with wfnreader and print shape and dtype of get_gvec_nk[0]
if __name__ == "__main__":
    # Load WFN.h5 file
    wfn = WFNReader("WFN.h5")
    
    # Get G-vectors for first k-point
    gvecs = wfn.get_gvec_nk(0)
    
    # Print shape and dtype
    print(f"Shape of G-vectors for k-point 0: {gvecs.shape}")
    print(f"Dtype of G-vectors: {gvecs.dtype}")
