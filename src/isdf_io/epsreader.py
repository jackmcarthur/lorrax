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
        
        # Matrix elements
        self.matrix = self._file['mats/matrix'][:]
        self.matrix_diagonal = self._file['mats/matrix-diagonal'][:]

        # you should only really want this for eps0. TODO: frequency dep.
        self.epshead = self.matrix[0,0,0,0,0,0] + 1j * self.matrix[0,0,0,0,0,1]
        
        # Optional matrix elements if using subspace approximation
        if self.subspace:
            if 'matrix_subspace' in self._file['mats']:
                self.matrix_subspace = self._file['mats/matrix_subspace'][:]
            if 'matrix_eigenvec' in self._file['mats']:
                self.matrix_eigenvec = self._file['mats/matrix_eigenvec'][:]
            if 'matrix_fulleps0' in self._file['mats']:
                self.matrix_fulleps0 = self._file['mats/matrix_fulleps0'][:]

    def __del__(self):
        """Clean up by closing the file when the object is destroyed."""
        if hasattr(self, '_file') and self._file is not None:
            self._file.close()
            
    def get_eps_matrix(self, iq, ifreq=0, imatrix=0):
        """Get the epsilon matrix for a specific q-point and frequency.
        
        Args:
            iq (int): Q-point index
            ifreq (int): Frequency index (default=0 for static)
            imatrix (int): Matrix index (default=0)
            
        Returns:
            np.ndarray: Complex epsilon matrix of shape (nmtx[iq], nmtx[iq])
        """
        nmtx_q = self.nmtx[iq]
        mat = self.matrix[iq, imatrix, ifreq, :nmtx_q, :nmtx_q,0] + 1j * self.matrix[iq, imatrix, ifreq, :nmtx_q, :nmtx_q,1]
        return mat
    
    def get_eps_minus_delta_matrix(self, iq, ifreq=0, imatrix=0):
        """Get the epsilon matrix for a specific q-point and frequency.
        
        Args:
            iq (int): Q-point index
            ifreq (int): Frequency index (default=0 for static)
            imatrix (int): Matrix index (default=0)
            
        Returns:
            np.ndarray: Complex epsilon matrix of shape (nmtx[iq], nmtx[iq])
        """
        nmtx_q = self.nmtx[iq]
        mat = self.matrix[iq, imatrix, ifreq, :nmtx_q, :nmtx_q,0] + 1j * self.matrix[iq, imatrix, ifreq, :nmtx_q, :nmtx_q,1]
        mat.flat[::nmtx_q+1] -= 1.0  # Subtracts 1 from diagonal elements in-place
        return mat
    
    def unfold_eps_comps(self, iqbar, S, Gq):
        # get the components Gtilde in order to do sum_GG' M*_q1(Gtilde) epsinv_GG'(q_1) M_q1(Gtilde)

        #assert isinstance(S, np.ndarray) and S.shape == (3, 3), f"S must be a 3x3 numpy array, got shape {S.shape}"
        #assert isinstance(Gq, np.ndarray) and Gq.shape == (3,), f"Gq must be a 3-element numpy array, got shape {Gq.shape}"

        # consider q_1 = qbar{S|tau} + Gq, then (see Deslippe 2012 section 7.2)
        # epsinv_GG'(q_1) = exp(-i(G-G')tau) epsinv_[(G+Gq)Sinv,(G'+Gq)Sinv](qbar)
        # therefore, sum_GG' M*_q1(G) epsinv_GG'(q_1) M_q1(G') = sum_GG' M*_q1(GS-Gq) epsinv_GG'(qbar) M_q1(GS-Gq)
        # NO SUPPORT FOR TAU (FRAC TRANS) CURRENTLY

        # iqbar must be the index of the q-point *in the epsilon file*

        #Sinv = np.linalg.inv(S)
        # note gind_eps2rho is much longer than actual mtx size.
        G_comps_qbar = np.zeros((self.nmtx[iqbar],3),dtype=np.int32)
        G_comps_qbar = self.comps[self.gind_eps2rho[iqbar,:self.nmtx[iqbar]],:]
        #G_comps_q1 = np.matmul(G_comps_qbar+Gq[:,np.newaxis],Sinv)
        G_comps_q1 = np.einsum('ij,kj->ki',S.astype(np.int32),G_comps_qbar) - Gq[np.newaxis,:]

        return G_comps_q1


    def get_eps_diagonal(self, iq):
        """Get the static diagonal elements for a specific q-point.
        
        Args:
            iq (int): Q-point index
            
        Returns:
            np.ndarray: Complex diagonal elements
        """
        diag = self.matrix_diagonal[:, :self.nmtx[iq], iq]
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