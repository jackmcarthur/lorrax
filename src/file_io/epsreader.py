import h5py
import numpy as np

from common.units import RYD_TO_EV

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

        # Backward-compatible static spelling.  Dynamic head consumers must
        # call ``get_epsinv_head`` with the frequency they actually need;
        # silently reusing this zero-frequency slot at i*omega_p collapses a
        # two-point GN fit onto one sample and produces an arbitrarily large
        # pole.
        self.epshead = self.get_epsinv_head(0.0 + 0.0j)
        
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

    def _frequency_points_ev(self):
        """Return the stored complex-frequency axis in eV.

        BerkeleyGW HDF5 files store ``freqs`` as ``(nfreq, 2)`` real/imag
        pairs.  Accept a native complex vector too so the reader has one
        normalized representation rather than teaching every consumer the
        on-disk spelling.
        """
        raw = np.asarray(self.freqs)
        if np.iscomplexobj(raw):
            points = raw.reshape(-1).astype(np.complex128)
        elif raw.ndim == 2 and raw.shape[1] == 2:
            points = (raw[:, 0] + 1j * raw[:, 1]).astype(np.complex128)
        else:
            points = raw.reshape(-1).astype(np.float64).astype(np.complex128)
        if points.size != int(self.nfreq):
            raise ValueError(
                f"eps frequency axis has {points.size} entries but nfreq="
                f"{int(self.nfreq)}")
        return points

    def frequency_index(self, omega_ry, *, rtol=1.0e-6, atol_ev=1.0e-8):
        """Return the stored index matching complex ``omega_ry``.

        Frequencies in ``eps0mat.h5`` are in eV while GWJAX's head resolver
        works in Ry.  The tolerance covers the rounded physical constants in
        BerkeleyGW files (the shipped ``2 i Ry`` point differs from the
        current conversion constant by about 1e-6 eV) without ever selecting
        a genuinely different frequency.
        """
        target_ev = complex(omega_ry) * RYD_TO_EV
        points = self._frequency_points_ev()
        distance = np.abs(points - target_ev)
        index = int(np.argmin(distance))
        tolerance = float(atol_ev) + float(rtol) * max(1.0, abs(target_ev))
        if float(distance[index]) > tolerance:
            available = ", ".join(f"{z.real:g}{z.imag:+g}i" for z in points)
            raise ValueError(
                f"eps0mat carries no frequency matching {target_ev.real:g}"
                f"{target_ev.imag:+g}i eV (nearest distance "
                f"{float(distance[index]):.3e} eV, tolerance "
                f"{tolerance:.3e} eV); available: [{available}]")
        return index

    def get_epsinv_head(self, omega_ry=0.0 + 0.0j):
        """Return ``epsilon^-1_00(omega)`` at q=Gamma.

        The scalar is meaningful as a screened-Coulomb head only when the
        file declares ``matrix_type = 0`` (epsilon inverse).  Refuse the
        other BGW matrix types rather than interpreting epsilon or chi0 as
        epsilon inverse merely because they occupy the same HDF5 slot.
        """
        if int(self.matrix_type) != 0:
            raise ValueError(
                f"eps head requires matrix_type=0 (epsilon inverse), got "
                f"{int(self.matrix_type)}")
        ifreq = self.frequency_index(omega_ry)
        return complex(
            self.matrix[0, 0, ifreq, 0, 0, 0]
            + 1j * self.matrix[0, 0, ifreq, 0, 0, 1]
        )
    
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
