import h5py as h5
import numpy as np

class WFNReader:
    def __init__(self, filename):
        """Initialize WFNReader with WFN file."""
        self._filename = filename
        self._file = h5.File(filename, 'r')
        
        # Read all header information from mf_header
        # Version and flavor
        self.version = self._file['mf_header/versionnumber'][()]
        self.flavor = self._file['mf_header/flavor'][()]
        
        # Kpoints group
        self.nspin = self._file['mf_header/kpoints/nspin'][()]
        self.nspinor = self._file['mf_header/kpoints/nspinor'][()]
        # Current workflows assume two-component Pauli spinors.  Support for the
        # four-component formalism under development will hook in here once the
        # wavefunction files expose the extra components.  Metadata related to
        # CTSP grids will also be parsed here when available in future files.
        self.nkpts = self._file['mf_header/kpoints/nrk'][()]  # nrk = number of k-points
        self.nbands = self._file['mf_header/kpoints/mnband'][()]  # mnband = number of bands
        self.ngkmax = self._file['mf_header/kpoints/ngkmax'][()]
        self.ecutwfc = self._file['mf_header/kpoints/ecutwfc'][()]
        self.kgrid = self._file['mf_header/kpoints/kgrid'][:]
        self.shift = self._file['mf_header/kpoints/shift'][:]
        self.ngk = self._file['mf_header/kpoints/ngk'][:]
        self.ifmin = self._file['mf_header/kpoints/ifmin'][:]
        self.ifmax = self._file['mf_header/kpoints/ifmax'][:]
        self.kweights = self._file['mf_header/kpoints/w'][:]
        self.kpoints = self._file['mf_header/kpoints/rk'][:]
        self.energies = self._file['mf_header/kpoints/el'][:]
        self.occs = self._file['mf_header/kpoints/occ'][:]

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
        
        # Gspace group
        self.ng = self._file['mf_header/gspace/ng'][()]
        self.ecutrho = self._file['mf_header/gspace/ecutrho'][()]
        self.fft_grid = self._file['mf_header/gspace/FFTgrid'][:]
        self.fft_grid = np.asarray(self.fft_grid, dtype=np.int32)

        self.gvecs = self._file['wfns/gvecs'][()]
        self.coeffs = self._file['wfns/coeffs'][:]
        
        # Symmetry group
        self.ntran = self._file['mf_header/symmetry/ntran'][()]
        self.cell_symmetry = self._file['mf_header/symmetry/cell_symmetry'][()]
        self.sym_matrices = self._file['mf_header/symmetry/mtrx'][:]
        self.translations = self._file['mf_header/symmetry/tnp'][:]
        
        # preprocessing sym_matrices, which always have length 48 and fortran order
        #self.sym_matrices = np.asarray(self.sym_matrices[:self.ntran])#.transpose(0,2,1))

        # Crystal group
        self.cell_volume = self._file['mf_header/crystal/celvol'][()]
        self.recip_volume = self._file['mf_header/crystal/recvol'][()]
        self.alat = self._file['mf_header/crystal/alat'][()]
        self.blat = self._file['mf_header/crystal/blat'][()]
        self.nat = self._file['mf_header/crystal/nat'][()]
        self.avec = self._file['mf_header/crystal/avec'][:]
        self.bvec = self._file['mf_header/crystal/bvec'][:]
        self.adot = self._file['mf_header/crystal/adot'][:]
        self.bdot = self._file['mf_header/crystal/bdot'][:]
        self.atom_types = self._file['mf_header/crystal/atyp'][:]
        self.atom_positions = self._file['mf_header/crystal/apos'][:]
        # this is one correct way to get the atomic positions in crystal coordinates (they're in units of alat, cartesian)
        self.atom_crys = np.einsum('ij,kj->ki',np.linalg.inv(self.avec).T, self.atom_positions)

        
        # Calculate k-point starts
        self.kpt_starts = np.zeros(self.nkpts, dtype=np.int64)
        for ik in range(1, self.nkpts):
            self.kpt_starts[ik] = self.kpt_starts[ik-1] + self.ngk[ik-1]

    def __del__(self):
        f = getattr(self, "_file", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass

    def get_cnk(self, ik, ib):
        """Get raw irreducible-zone coefficients for one band at one k-point.

        Args:
            ik (int): Irreducible-zone k-point index stored in the WFN file
            ib (int): band index

        Returns:
            np.ndarray: Complex coefficients of shape (2, ngk[ik])

        Notes:
            This returns the coefficients exactly as stored in ``WFN.h5``.
            It does not apply symmetry unfolding, fractional-translation
            phases, spinor rotations, or time-reversal conjugation.  Call
            ``get_cnk_fullzone`` when pairing coefficients with full-BZ
            G-vectors from ``get_gvecs_kfull``.
        """
        start = self.kpt_starts[ik]
        end = start + self.ngk[ik]
        nkg = self.ngk[ik]
        cnk = np.zeros((2, nkg), dtype=np.complex128)
        cnk[0, :] = self.coeffs[ib, 0, start:end, 0] + 1j * self.coeffs[ib, 0, start:end, 1]
        cnk[1, :] = self.coeffs[ib, 1, start:end, 0] + 1j * self.coeffs[ib, 1, start:end, 1]
        return cnk

    def get_cnk_batch(self, ik, band_indices):
        """Get raw irreducible-zone coefficients for multiple bands.

        Args:
            ik (int): Irreducible-zone k-point index stored in the WFN file
            band_indices: array-like of band indices

        Returns:
            np.ndarray: Complex coefficients of shape (nb, 2, ngk[ik])

        Notes:
            This is the batched analogue of ``get_cnk`` and has the same
            symmetry caveat: the returned coefficients are still on the raw
            irreducible-zone G-list with no non-symmorphic phase applied.
        """
        start = self.kpt_starts[ik]
        end = start + self.ngk[ik]
        band_indices = np.asarray(band_indices, dtype=np.int64)
        # coeffs shape: (mnband, nspinor, ngktot, 2) where last dim is real/imag
        raw = self.coeffs[band_indices, :, start:end, :]  # (nb, 2, ngk, 2)
        return raw[..., 0] + 1j * raw[..., 1]  # (nb, 2, ngk)
    
    def get_gvec_nk(self, ik):
        """Get the raw irreducible-zone G-list for one stored k-point.

        Args:
            ik (int): Irreducible-zone k-point index stored in the WFN file

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

    def get_gvecs_kfull(self, sym, ik_full):
        """Return full-BZ G-vectors for ``ik_full`` via ``SymMaps``.

        This applies only the integer rotation+umklapp needed to align the
        unfolded G-list with the target full-zone k-point.  Fractional
        translations do not modify the integer G-indices themselves; the
        corresponding non-symmorphic phase is applied by
        ``get_cnk_fullzone`` / ``get_cnk_fullzone_batch``.
        """
        return sym.get_gvecs_kfull(self, ik_full)

    def get_cnk_fullzone(self, sym, ik_full, ib):
        """Return phase-correct unfolded coefficients for one full-BZ k-point.

        Delegating through ``SymMaps`` keeps the coefficients consistent with
        ``get_gvecs_kfull`` by applying any non-symmorphic fractional
        translation phase, spinor rotation, and time-reversal conjugation.
        """
        return sym.get_cnk_fullzone(self, ib, ik_full)

    def get_cnk_fullzone_batch(self, sym, ik_full, band_indices):
        """Return phase-correct unfolded coefficients for multiple bands."""
        return sym.get_cnk_fullzone_batch(self, band_indices, ik_full)
    
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
