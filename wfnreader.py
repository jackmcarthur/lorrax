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

        self.nelec = int(np.sum(self.occs[0,0]))
        # in-gap chemical potential
        self.efermi = 0.5 * (np.amin(self.energies[0,:,self.nelec]) - np.amax(self.energies[0,:,self.nelec-1]))
        self.cbm = np.amax(self.energies[0,:,self.nelec-1])
        self.vbm = np.amin(self.energies[0,:,self.nelec])
        
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
        """Clean up by closing the file when the object is destroyed."""
        if hasattr(self, '_file') and self._file is not None:
            self._file.close()

    def get_cnk(self, ik, ib):
        """Get complex coefficients for both spinor components of a wavefunction.
        
        Args:
            ik (int): k-point index
            ib (int): band index
            
        Returns:
            np.ndarray: Complex coefficients array of shape (ngk[ik], 2) for both spinor components,
                       in Fortran order
        """
        # Get start and end indices for this k-point
        start = self.kpt_starts[ik]
        end = start + self.ngk[ik]
        nkg = self.ngk[ik]
        
        # Initialize complex array for both spinor components
        cnk = np.zeros((2,nkg), dtype=np.complex128,)
        
        # Get coefficients from file and transpose to Python order
        # Original: (2, ngktot, nspin*nspinor, mnband) in Fortran order
        #coeffs = np.array(self._file['wfns/coeffs'])
        
        #print(f"Shape of coeffs array: {coeffs.shape}")
        #print(f"Shape of cnk array: {cnk.shape}")
        #print(f"Start index: {start}, End index: {end}")
        #print(f"Number of G-vectors for this k-point (ngk[ik]): {nkg}")
        
        # First spinor component
        cnk[0,:] = self.coeffs[ib, 0, start:end, 0] + 1j * self.coeffs[ib, 0, start:end, 1]
        
        # Second spinor component
        cnk[1,:] = self.coeffs[ib, 1, start:end, 0] + 1j * self.coeffs[ib, 1, start:end, 1]
        
        return cnk
    
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
