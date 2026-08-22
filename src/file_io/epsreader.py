"""BerkeleyGW ``epsmat.h5`` / ``eps0mat.h5`` reader.

WHAT IT DOES NOT DO, and this is the point of the class: it does not copy
``mats/matrix`` into host memory.  Until 2026-08-22 the constructor read
``mats/matrix``, ``mats/matrix-diagonal`` and the three optional subspace
matrices with ``[:]``, i.e. every q and every frequency, ON EVERY RANK,
before the caller had asked for anything.  The only in-tree consumer
(``gw.head_correction.resolve_head_sample``) then used **six numbers** out
of it — ``matrix[0,0,0,0,0,{0,1}]``, the q->0 head.  At a production
``epsmat.h5`` that is tens of GB per rank for a complex scalar.

The datasets are now held as **h5py dataset handles** and sliced in the
accessors, which is what the accessors were already written to do
(``self.matrix[iq, imatrix, ifreq, :nmtx, :nmtx, 0]`` is a hyperslab
request when ``self.matrix`` is a handle and a no-op view when it is a
copy).  So the change is a lifetime change, not an interface one — with
one consequence a caller must know:

**OWNERSHIP.**  The handles are only valid while the file is open.  Use
the context manager, or call :meth:`close` when done::

    with EPSReader(path) as eps:
        head = eps.epshead

``__del__`` still closes, so existing call sites that drop the object keep
working; what changes is that a slice taken AFTER an explicit ``close()``
raises instead of returning stale memory.

THIS FILE IS BerkeleyGW'S, NOT SlabIO'S.  ``epsmat.h5`` is written by
BerkeleyGW and never by the phdf5 transport, so no ``file_io.hdf5_owner``
cohabitation arises here and serial h5py is the right reader.  The defect
this docstring records was per-rank REPLICATION, and that is what is
fixed; a q/frequency-sharded reader is a separate piece of work and would
need SlabIO plus a shape contract the BGW format does not currently state.
"""
import h5py
import numpy as np

class EPSReader:
    def __init__(self, filename):
        """Initialize EPSMATReader with epsmat.h5 file."""
        self._filename = filename
        self._file = h5py.File(filename, 'r')
        
        # Read eps_header information
        # Version and flavor
        self.version = self._file['eps_header/versionnumber'][()]
        self.flavor = self._file['eps_header/flavor'][()]
        
        # Parameters group
        params = self._file['eps_header/params']
        self.matrix_type = params['matrix_type'][()]  # 0=epsilon^-1, 1=epsilon, 2=chi0
        self.has_advanced = params['has_advanced'][()]
        self.nmatrix = params['nmatrix'][()]
        self.matrix_flavor = params['matrix_flavor'][()]
        self.icutv = params['icutv'][()]
        self.ecuts = params['ecuts'][()]
        self.nband = params['nband'][()]
        self.efermi = params['efermi'][()]
        
        # Optional parameters
        self.subsampling = params['subsampling'][()] if 'subsampling' in params else False
        self.subspace = params['subspace'][()] if 'subspace' in params else False
        
        # Q-points group
        qpoints = self._file['eps_header/qpoints']
        self.nq = qpoints['nq'][()]
        self.qpts = qpoints['qpts'][:]
        self.qgrid = qpoints['qgrid'][:]
        self.qpt_done = qpoints['qpt_done'][:]
        
        # Frequencies group
        freqs = self._file['eps_header/freqs']
        self.freq_dep = freqs['freq_dep'][()]
        self.nfreq = freqs['nfreq'][()]
        self.nfreq_imag = freqs['nfreq_imag'][()]
        self.freqs = freqs['freqs'][:]
        
        # G-space group
        gspace = self._file['eps_header/gspace']
        self.nmtx = gspace['nmtx'][:]
        self.nmtx_max = gspace['nmtx_max'][()]
        self.ekin = gspace['ekin'][:]
        self.gind_eps2rho = np.array(gspace['gind_eps2rho'][:]-1, dtype=np.int32) # -1 because of fortran indexing
        self.gind_rho2eps = np.array(gspace['gind_rho2eps'][:]-1, dtype=np.int32)
        self.vcoul = gspace['vcoul'][:]

        self.gvec_ind_max = int(np.amax(self.gind_eps2rho))
        self.comps = self._file['mf_header/gspace/components'][:self.gvec_ind_max,:]
        

        # Subspace group (if exists)
        if 'subspace' in self._file['eps_header']:
            subspace = self._file['eps_header/subspace']
            self.keep_full_eps_static = subspace['keep_full_eps_static'][()]
            self.matrix_in_subspace_basis = subspace['matrix_in_subspace_basis'][()]
            self.eps_eigenvalue_cutoff = subspace['eps_eigenvalue_cutoff'][()]
            self.neig_max = subspace['neig_max'][()]
            self.neig = subspace['neig'][:]
        
        # Matrix elements — HANDLES, not copies.  See the module docstring:
        # `[:]` here was tens of GB per rank at a production epsmat.h5, read
        # before the caller had asked for anything, for a consumer that wants
        # six numbers.  Every accessor below already slices, so a handle
        # turns each of them into an HDF5 hyperslab read of exactly the block
        # asked for.
        self.matrix = self._file['mats/matrix']
        self.matrix_diagonal = self._file['mats/matrix-diagonal']

        # you should only really want this for eps0. TODO: frequency dep.
        # Two scalar reads, not a whole-array indexing of a resident copy.
        self.epshead = complex(
            self.matrix[0, 0, 0, 0, 0, 0]) + 1j * complex(
            self.matrix[0, 0, 0, 0, 0, 1])

        # Optional matrix elements if using subspace approximation
        if self.subspace:
            if 'matrix_subspace' in self._file['mats']:
                self.matrix_subspace = self._file['mats/matrix_subspace']
            if 'matrix_eigenvec' in self._file['mats']:
                self.matrix_eigenvec = self._file['mats/matrix_eigenvec']
            if 'matrix_fulleps0' in self._file['mats']:
                self.matrix_fulleps0 = self._file['mats/matrix_fulleps0']

    # ------------------------------------------------------------------
    # Lifetime.  The matrix attributes are h5py handles now, so the file
    # has to outlive them and the caller has to be able to say when.
    # ------------------------------------------------------------------
    def close(self):
        """Close the file.  Idempotent; slices taken after it raise."""
        f = getattr(self, '_file', None)
        if f is not None:
            self._file = None
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        """Clean up by closing the file when the object is destroyed."""
        # Kept so call sites that just drop the object still release the
        # file; `close()` is what a caller uses when the moment matters.
        try:
            self.close()
        except Exception:              # interpreter teardown, h5py gone
            pass


    def get_eps_matrix(self, iq, ifreq=0, imatrix=0):
        """Get the epsilon matrix for a specific q-point and frequency.
        
        Args:
            iq (int): Q-point index
            ifreq (int): Frequency index (default=0 for static)
            imatrix (int): Matrix index (default=0)
            
        Returns:
            np.ndarray: Complex epsilon matrix of shape (nmtx[iq], nmtx[iq])
        """
        # int() because these indices now reach h5py's selection parser
        # rather than numpy's: a numpy scalar bound is accepted by numpy
        # everywhere and by h5py only in some versions.
        iq, imatrix, ifreq = int(iq), int(imatrix), int(ifreq)
        nmtx_q = int(self.nmtx[iq])
        block = self.matrix[iq, imatrix, ifreq, :nmtx_q, :nmtx_q, :]
        return block[..., 0] + 1j * block[..., 1]
    
    def get_eps_minus_delta_matrix(self, iq, ifreq=0, imatrix=0):
        """Get the epsilon matrix for a specific q-point and frequency.
        
        Args:
            iq (int): Q-point index
            ifreq (int): Frequency index (default=0 for static)
            imatrix (int): Matrix index (default=0)
            
        Returns:
            np.ndarray: Complex epsilon matrix of shape (nmtx[iq], nmtx[iq])
        """
        nmtx_q = int(self.nmtx[int(iq)])
        mat = self.get_eps_matrix(iq, ifreq=ifreq, imatrix=imatrix)
        mat.flat[::nmtx_q+1] -= 1.0  # Subtracts 1 from diagonal elements in-place
        return mat
    
    # ``unfold_eps_comps`` WAS HERE AND IS DELETED (2026-08-16).
    #
    # It was a fifth independent ``G' = S·G − G_umklapp`` in this tree, found
    # by re-counting the symmetry register's "four implementations" claim, and
    # registered nowhere until 2026-08-15.  Two facts decided it:
    #
    #   * its own comment said "NO SUPPORT FOR TAU (FRAC TRANS) CURRENTLY",
    #     i.e. it was known-wrong on every non-symmorphic deck, and nothing
    #     outside this file said so;
    #   * it had NO live caller.  The only references were in
    #     ``misc/archived_tests/cohsex_noisdf.py``, which cannot be imported
    #     at all — it does ``from wfnreader import WFNReader`` and
    #     ``from gpu_utils import cp``, and neither module exists anywhere in
    #     the tree.
    #
    # DELETED RATHER THAN FIXED because adding τ support to a method with no
    # caller is speculative and untestable, while leaving a τ-blind G-rotation
    # in a re-exported class is an invitation to use it.  Same disposition the
    # register records for the other known-broken rotation copy: "fix is
    # deletion in favour of the canonical, not repair."
    #
    # If you need this, the canonical G rotation + umklapp is
    # ``wfn_loader/loader.py`` with ``symmetry_maps.SymMaps.get_umklapp_vector``
    # — and it handles τ.

    def get_eps_diagonal(self, iq):
        """Get the static diagonal elements for a specific q-point.
        
        Args:
            iq (int): Q-point index
            
        Returns:
            np.ndarray: Complex diagonal elements
        """
        iq = int(iq)
        diag = self.matrix_diagonal[:, :int(self.nmtx[iq]), iq]
        return diag[0] + 1j * diag[1]

if __name__ == "__main__":
    # Example usage
    eps = EPSReader("epsmat.h5")
    print(f"Number of q-points: {eps.nq}")
    print(f"Q-point grid: {eps.qgrid}")
    print(f"Number of frequencies: {eps.nfreq}")
    
    # Get epsilon matrix for first q-point
    eps_q0 = eps.get_eps_matrix(0)
    print(f"Shape of epsilon matrix for q=0: {eps_q0.shape}") 