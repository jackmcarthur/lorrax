import numpy as np
import jax.numpy as jnp
from .wfnreader import WFNReader


class SymMaps:
    def __init__(self, wfn):
        """
        Initialize symmetry mappings for a given WFN file.
        class variables are:
        dict: irk_to_k_map[irk] = [k1, k2, k3, ...], kpt id's that map to irk
        dict: irk_sym_map[irk] = [sym1, sym2, sym3, ...], sym op sym_matrices[sym1] maps irk to ik
        U_spinor[sym_idx] is the spinor rotation matrix for the sym_idx-th symmetry operation.
        The matrices are currently 2x2 Pauli-spinor rotations; upcoming work
        will expand this to the 4-component formalism used in relativistic
        treatments.
        R_grid[sym_idx] is the corresponding list of symmetry operations in the WFN file
        u_{n,Rk,a}(G) = U_spinor_{a,b} u_{n,k,b}(Rinv G)
        
        Args:
            wfn: WFNReader instance
        """
        # Create k-point mappings
        #self.sym_matrices, self.sym_mats_k = self.get_syms_from_kgridlog('kgrid.log')

        # get symmetry matrices from wfn file
        try:
            ntran = int(getattr(wfn, 'ntran', 1))
        except Exception:
            ntran = 1
        if ntran <= 1:
            # Trivial identity-only symmetry path
            self.sym_matrices = np.eye(3, dtype=np.int32)[None, :, :]
            self.sym_mats_k = self.sym_matrices.transpose(0, 2, 1).copy()

            # In no-symmetry case, unfolded grid equals irreducible grid
            self.unfolded_kpts = np.asarray(wfn.kpoints, dtype=float)
            self.kpoint_map = np.arange(self.unfolded_kpts.shape[0], dtype=np.int32)
            self.kpoint_map_ibz_ids = self.kpoint_map.copy()

            # Maps: each full k maps to itself; only identity symmetry
            self.irk_to_k_map = np.arange(self.unfolded_kpts.shape[0], dtype=np.int32)
            self.irk_sym_map = np.zeros(self.unfolded_kpts.shape[0], dtype=np.int32)

            self.nk_tot = int(self.unfolded_kpts.shape[0])
            self.nk_red = int(getattr(wfn, 'nkpts', self.nk_tot))

            # kirr_fullids: identity mapping
            self.kirr_fullids = np.arange(self.nk_red, dtype=np.int32)

            # Rotation matrices and spinor (identity)
            self.R_grid = np.eye(3, dtype=np.int32)[None, :, :]
            self.Rinv_grid = self.R_grid.copy()
            self.R_cart = self.R_grid.astype(float)
            self.U_spinor = np.eye(2, dtype=complex)[None, :, :]

            # k−q maps on the unfolded (which equals reduced) grid
            self.kq_map = self.get_kminusq_map(wfn, self.unfolded_kpts)
            self.kqfull_map = self.get_kminusqfull_map(wfn, self.unfolded_kpts)
            self.kfull_symmap = np.zeros((self.nk_tot, 1), dtype=np.int32)

            # Integer k-grid vectors and q enumerations
            kx, ky, kz = np.meshgrid(np.arange(wfn.kgrid[0]),
                            np.arange(wfn.kgrid[1]),
                            np.arange(wfn.kgrid[2]),
                            indexing='ij')
            self.kvecs_asints = np.stack([kx.flatten(), ky.flatten(), kz.flatten()], axis=1)
            qpt_vecs = self.kvecs_asints[:, None, :] - self.kvecs_asints[None, :, :]
            self.all_unfolded_qpts = np.unique(qpt_vecs.reshape(-1, 3), axis=0)
            self.all_unfolded_qpt_ids = np.zeros((len(self.kvecs_asints), len(self.kvecs_asints)), dtype=np.int32)
            for i, q in enumerate(self.all_unfolded_qpts):
                mask = (qpt_vecs == q).all(axis=2)
                self.all_unfolded_qpt_ids[mask] = i
            return

        self.sym_matrices = wfn.sym_matrices[:wfn.ntran] # these apply to real space coords as sym_matrices[i] @ [rx,ry,rz]
        self.sym_mats_k = self.sym_matrices[:wfn.ntran].transpose(0,2,1).copy()  # these apply to k-points as sym_mats_k[i] @ [kx,ky,kz]

        # get the list of full zone k-points and the map from k_full to k_irr
        self.kpoint_map, self.unfolded_kpts = self.create_kpoint_symmetry_map(wfn)

        # change the map from "k_full points indexed by full grid position" to "k_full points indexed by irr. k-point position"
        self.kpoint_map_ibz_ids = self.kpoint_map_irrbz_ids(wfn, self.unfolded_kpts)

        self.irk_to_k_map, self.irk_sym_map = self.find_symmetry_ops_simple(wfn, self.kpoint_map, self.unfolded_kpts)
        
        
        self.nk_tot = int(self.unfolded_kpts.shape[0])
        self.nk_red = int(wfn.nkpts)

        # Create mapping from irreducible k-points to full BZ indices
        self.kirr_fullids = np.zeros(self.nk_red, dtype=np.int32)
        for kirr in range(self.nk_red):
            matches = np.where(self.irk_to_k_map == kirr)[0]
            if matches.size == 0:
                # Fallback: identity mapping if not found
                self.kirr_fullids[kirr] = kirr
            else:
                self.kirr_fullids[kirr] = matches[0]

        # useful maps:
        # k (full zone) to kbar 
        # k,q (both full zone) to k-q (full zone)
        
        # Get rotation matrices and their spinor representations
        self.R_grid = np.rint(self.sym_matrices).astype(np.int32)
        self.Rinv_grid = np.rint(np.linalg.inv(self.R_grid)).astype(np.int32)

        
        self.R_cart = self.syms_crystal_to_cartesian(wfn)
        self.U_spinor = self.get_spinor_rotations(wfn, self.R_cart)
        self.kq_map = self.get_kminusq_map(wfn, self.unfolded_kpts)
        self.kqfull_map = self.get_kminusqfull_map(wfn, self.unfolded_kpts)
        self.kfull_symmap = self.get_kfull_symmap(wfn, self.unfolded_kpts)


        # the above kq maps are for inputting some k and some q and getting k-q in the 1BZ, but it is actually necessary to store W_q on q outside 1BZ
        # As such, the following functions are for inputting some k and some k' and getting the relevant q outside 1BZ
        kx, ky, kz = np.meshgrid(np.arange(wfn.kgrid[0]), 
                        np.arange(wfn.kgrid[1]), 
                        np.arange(wfn.kgrid[2]), 
                        indexing='ij')
        self.kvecs_asints = np.stack([kx.flatten(), ky.flatten(), kz.flatten()], axis=1) # kpoints * kgrid (kpoints as integers)

        # Generate q-vectors using broadcasting
        qpt_vecs = self.kvecs_asints[:, None, :] - self.kvecs_asints[None, :, :]  # Automatic broadcasting

        # Find unique q-vectors (already vectorized)
        self.all_unfolded_qpts = np.unique(qpt_vecs.reshape(-1, 3), axis=0)

        # Generate indices using vectorized operations
        self.all_unfolded_qpt_ids = np.zeros((len(self.kvecs_asints), len(self.kvecs_asints)), dtype=np.int32)
        # This is still a loop but operates on whole arrays at once
        for i, q in enumerate(self.all_unfolded_qpts):
            mask = (qpt_vecs == q).all(axis=2)
            self.all_unfolded_qpt_ids[mask] = i


    def get_qpt_id_from_kkp(self, kidx, kpidx):
        # meant to return the unique q idx of kp-k, so that sym.all_unfolded_qpts[qpt_id] = kp-k
        kpminkvec = self.kvecs_asints[kpidx] - self.kvecs_asints[kidx]
        return np.where(np.all(self.all_unfolded_qpts == kpminkvec, axis=1))[0][0]

        

    def get_syms_from_kgridlog(self,kgridfname):
        # return the identity + the set of sym_matrices that unfold the k-points
        # if \psi_nk(S^-1r) = psi_n(S^-1T.k)(r), Skbar + G_S = k
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
        symmats = np.array(matrices, dtype=np.int32)
        symmatskvecs = np.array([np.linalg.inv(mat).T for mat in symmats],dtype=np.int32) # correct crystal coord form to act on k
        
        return symmats, symmatskvecs

    def create_kpoint_symmetry_map(self, wfn):
        """
        Read k-point mapping from kgrid.log file.
        Converts from 1-based to 0-based indexing for kpts.
        
        Args:
            wfn (WfnReader): WFN reader object
            
        Returns:
            tuple: (kpoint_map, full_kpoints)
                - kpoint_map: Array mapping each k-point to its irreducible k-point (full zone)
                - full_kpoints: Array of all k-points in the full grid
        """
        kpoint_map = []
        parsing = False
        
        # Generate full k-point grid
        kx = np.linspace(0, 1, wfn.kgrid[0], endpoint=False)
        ky = np.linspace(0, 1, wfn.kgrid[1], endpoint=False)
        kz = np.linspace(0, 1, wfn.kgrid[2], endpoint=False)
        
        # Apply shift
        kx += wfn.shift[0]/wfn.kgrid[0]
        ky += wfn.shift[1]/wfn.kgrid[1]
        kz += wfn.shift[2]/wfn.kgrid[2]
        
        # Create full k-point grid
        kpts_mesh = np.meshgrid(kx, ky, kz, indexing='ij')
        full_kpoints = np.stack([k.flatten() for k in kpts_mesh]).T

        # Map each full k-point to its symmetry operation
        kpoint_map = np.zeros(len(full_kpoints), dtype=np.int32)
        
        for kfull_idx in range(len(full_kpoints)):
            k_found = False
            for i, sym_mat in enumerate(self.sym_mats_k):
                # Apply symmetry operation to k-point
                k_transformed = sym_mat @ full_kpoints[kfull_idx]
                # Wrap to first BZ
                k_transformed = k_transformed % 1.0
                # Replace values close to 1 with 0
                k_transformed[k_transformed > 0.999] = 0.0
                
                # Check if transformed k-point matches any k-point in wfn.kpoints
                for j, k in enumerate(wfn.kpoints):
                    if np.allclose(k_transformed, k, atol=1e-6):
                        kpoint_map[kfull_idx] = i
                        k_found = True
                        break
                
                if k_found:
                    break
            
            if not k_found:
                raise ValueError(f"No symmetry operation found for k-point {full_kpoints[kfull_idx]}")
        
        return kpoint_map, full_kpoints
    
    def kpoint_map_irrbz_ids(self, wfn, full_kpts):
        irr_kpts = wfn.kpoints

        kpoint_map_irrbz_ids = np.zeros_like(self.kpoint_map)
        for i, idx in enumerate(self.kpoint_map):
            target_kpt = full_kpts[idx]  # Get the k-point this maps to
            # Find this k-point's index in irr_kpts
            irr_idx = np.argmin(np.sum(np.abs(irr_kpts - target_kpt), axis=1))
            kpoint_map_irrbz_ids[i] = irr_idx

        return kpoint_map_irrbz_ids
        
    def find_symmetry_ops_simple(self, wfn, kpoint_map, full_kpts):
        irk_to_k_map = np.zeros(full_kpts.shape[0], dtype=np.int32)
        irk_sym_map = np.zeros(full_kpts.shape[0], dtype=np.int32)
        # all symmetries applied to the irr k-points: shape (nkbar, nsym, 3)
        Skbar = np.einsum('ijk,lk->lij', self.sym_mats_k, wfn.kpoints)
        Skbar = Skbar % 1.0
        Skbar = np.where(Skbar > 0.99999, 0.0, Skbar)

        # find the symmetry operations that map the irr k-points to the full k-points
        for ikfull, kfull in enumerate(full_kpts):
            for ikbar in range(wfn.nkpts):
                # Compare each component within tolerance
                diffs = np.abs(Skbar[ikbar] - kfull)
                matches = np.where(np.all(diffs < 1e-6, axis=1))[0]
                if len(matches) > 0:
                    irk_to_k_map[ikfull] = ikbar
                    irk_sym_map[ikfull] = matches[0]

        return irk_to_k_map, irk_sym_map

    def syms_crystal_to_cartesian(self, wfn):
        """
        Convert symmetry matrices from crystal to cartesian coordinates.
        
        Args:
            sym_matrices_crys (numpy.ndarray): Symmetry matrices in crystal coords (nsym, 3, 3)
            
        Returns:
            numpy.ndarray: Symmetry matrices in cartesian coordinates (nsym, 3, 3)
        """
        # Get blat and bvec from WFNReader
        B_T = np.asarray(wfn.bvec)
        
        # Calculate (B^T)^-1
        B_T_inv = np.linalg.inv(B_T)
        
        # Convert each symmetry matrix
        # NOT SURE IF THESE SHOULD BE SYM_MATS_K OR SYM_MATS TODO
        sym_matrices_cart = np.einsum('ij,njk,kl->nil', B_T_inv, self.sym_mats_k, B_T)
        sym_matrices_cart = np.around(sym_matrices_cart, decimals=10)
        
        return sym_matrices_cart

    def get_spinor_rotations(self, wfn, sym_matrices_cart):
        """
        Converts a list of rotation matrices to their spinor representations using Markley's modification
        of Shepperd's algorithm (aka quaternion representation, see Brad Barker's dissertation).

        When the wavefunction files store four-component states these routines will
        compute the corresponding 4x4 spinor rotation matrices.
        
        Parameters:
        sym_matrices (numpy.ndarray): Array of 3x3 rotation matrices with shape (nsym, 3, 3)

        Returns:
        numpy.ndarray: Array of spinor matrices with shape (nsym, 2, 2) of complex type
        """
        nsym = len(sym_matrices_cart)
        spinor_matrices = np.zeros((nsym, 2, 2), dtype=complex)
        
        # Add Pauli matrices (moved outside the loop since they're constant)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        for isym, R in enumerate(sym_matrices_cart):
            # Construct the symmetric 4x4 matrix Q
            Q = np.zeros((4, 4))
            Q[0, 0] = R[0, 0] + R[1, 1] + R[2, 2]
            Q[0, 1] = Q[1, 0] = R[1, 2] - R[2, 1]
            Q[0, 2] = Q[2, 0] = R[2, 0] - R[0, 2]
            Q[0, 3] = Q[3, 0] = R[0, 1] - R[1, 0]
            
            Q[1, 1] = R[0, 0] - R[1, 1] - R[2, 2]
            Q[1, 2] = Q[2, 1] = R[0, 1] + R[1, 0]
            Q[1, 3] = Q[3, 1] = R[0, 2] + R[2, 0]
            
            Q[2, 2] = -R[0, 0] + R[1, 1] - R[2, 2]
            Q[2, 3] = Q[3, 2] = R[1, 2] + R[2, 1]
            
            Q[3, 3] = -R[0, 0] - R[1, 1] + R[2, 2]

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(Q)
            
            # The quaternion is the eigenvector corresponding to the largest eigenvalue
            q = eigenvectors[:, np.argmax(eigenvalues)]
            q = q / np.linalg.norm(q)  # Normalize
            
            # Quaternion components
            q0, q1, q2, q3 = q
            
            # Compute the angle
            theta = 2 * np.arccos(q0)
            
            # Handle axis calculation
            sin_theta_over_2 = np.sqrt(1 - q0**2)
            if sin_theta_over_2 < 1e-8 or np.isclose(theta, 0) or np.isclose(theta, 2 * np.pi):
                theta = 0.0
                n = np.array([1.0, 0.0, 0.0])
            elif np.isclose(theta, np.pi):
                axis = np.array([q1, q2, q3])
                n = axis / np.linalg.norm(axis)
            else:
                n = np.array([q1, q2, q3]) / sin_theta_over_2
                n = n / np.linalg.norm(n)
            
            # Calculate spinor matrix components
            cos_half_theta = np.cos(theta/2)
            sin_half_theta = np.sin(theta/2)
            
            # Construct spinor matrix
            spinor = cos_half_theta * np.eye(2, dtype=complex)
            spinor -= 1j * sin_half_theta * (
                n[0] * sigma_x +
                n[1] * sigma_y +
                n[2] * sigma_z
            )
            
            spinor_matrices[isym] = spinor
        
        return spinor_matrices

    def get_kminusq_map(self, wfn, full_kpts):
        """Create mapping between k and k-q points in the full k-point grid.
        
        Args:
            wfn: WFNReader instance
            full_kpts: Array of all k-points in the full grid
            
        Returns:
            numpy.ndarray: kq_map[ik,iq] = index of k-q in full k-point grid,
                          where ik is index in full grid, iq is index in reduced grid
        """
        # Initialize mapping array
        nk_full = len(full_kpts)
        nk_red = wfn.nkpts
        kq_map = np.zeros((nk_full, nk_red), dtype=np.int32)
        
        # Get reduced k-points
        reduced_kpts = np.asarray(wfn.kpoints)
        
        # For each full k-point and each reduced q-point
        for ik in range(nk_full):
            k = full_kpts[ik]
            for iq in range(nk_red):
                q = reduced_kpts[iq]
                
                # Calculate k-q and wrap to first BZ
                kminusq = k - q
                kminusq = kminusq % 1.0  # Wrap to [0,1)
                kminusq = np.where(kminusq > 0.99999, 0.0, kminusq)
                
                # Find which full k-point this maps to
                diffs = np.abs(full_kpts - kminusq[None, :])
                diffs = np.sum(diffs, axis=1)  # Sum over coordinates
                min_diff = np.min(diffs)
                
                if min_diff > 1e-8:
                    raise ValueError(f"k-q point {kminusq} not found in k-point grid")
                
                kq_idx = np.argmin(diffs)
                if kq_idx >= nk_full:
                    raise ValueError(f"Invalid k-q mapping: {kq_idx} >= {nk_full}")
                    
                kq_map[ik, iq] = kq_idx
        
        return kq_map

    def get_kminusqfull_map(self, wfn, full_kpts):
        # Initialize mapping array
        nk_full = len(full_kpts)
        nk_red = wfn.nkpts
        kq_map = np.zeros((nk_full, nk_full), dtype=np.int32)
        
        # For each full k-point and each reduced q-point
        for ik in range(nk_full):
            k = full_kpts[ik]
            for iq in range(nk_full):
                q = full_kpts[iq]
                
                # Calculate k-q and wrap to first BZ
                kminusq = k - q
                kminusq = kminusq % 1.0  # Wrap to [0,1)
                kminusq = np.where(kminusq > 0.99999, 0.0, kminusq)
                
                # Find which full k-point this maps to
                diffs = np.abs(full_kpts - kminusq[None, :])
                diffs = np.sum(diffs, axis=1)  # Sum over coordinates
                min_diff = np.min(diffs)
                
                if min_diff > 1e-8:
                    raise ValueError(f"k-q point {kminusq} not found in k-point grid")
                
                kq_idx = np.argmin(diffs)
                if kq_idx >= nk_full:
                    raise ValueError(f"Invalid k-q mapping: {kq_idx} >= {nk_full}")
                    
                kq_map[ik, iq] = kq_idx
        
        return kq_map
    
    def get_kfull_symmap(self, wfn, full_kpts):
        nk_full = len(full_kpts)
        n_sym = self.sym_mats_k.shape[0]
        kfull_symmap = np.zeros((nk_full, n_sym), dtype=np.int32)

        # For each k-point in the full grid
        for ik in range(nk_full):
            # Apply all symmetry operations to this k-point
            k_sym = np.einsum('ijk,k->ij', self.sym_mats_k, full_kpts[ik])
            k_sym = k_sym % 1.0  # Wrap to first BZ
            k_sym = np.where(k_sym > 0.99999, 0.0, k_sym)
            
            # For each symmetry operation
            for isym in range(n_sym):
                # Find which full k-point this maps to
                diffs = np.abs(full_kpts - k_sym[isym][None, :])
                diffs = np.sum(diffs, axis=1)  # Sum over coordinates
                min_diff = np.min(diffs)
                
                if min_diff > 1e-8:
                    raise ValueError(f"Symmetry-transformed k-point {k_sym[isym]} not found in k-point grid")
                
                kfull_symmap[ik, isym] = np.argmin(diffs)
        
        return kfull_symmap

    def get_gvecs_kfull(self,wfn,nk):
        # nb: band index
        # nk: index of k in sym.unfolded_kpts
        # relationship: u_n(kbar{S|tau}) (G) = u_nkbar(G{S|tau} - G_S), apparently..
        # (S.T@G = G@S)

        sym_idx = self.irk_sym_map[nk]
        kbar_idx = self.irk_to_k_map[nk]
        sym_krep = self.sym_mats_k[sym_idx] # note we apply with no changes
        
        #wfn_kG = wfn.get_cnk(kbar_idx,nb)
        k_gvecs = wfn.get_gvec_nk(kbar_idx)

        q_full = sym_krep @ wfn.kpoints[kbar_idx] #+ 0.001
        q_inzone = q_full%1.0
        q_inzone[q_inzone>0.9999] = 0.0
        Gkk = q_inzone - q_full
        Gkk = Gkk.astype(int) 

        k_gvecs_rot = np.einsum('ij,kj->ki', sym_krep.astype(np.int32), k_gvecs) # kgrid.x says sym 9 maps k to kbar
        k_gvecs_rot -= Gkk
        #wfn_kG = np.einsum('jk,kl->jl', self.U_spinor[sym_idx], wfn_kG)
        #wfn_kGgrid = np.zeros((2,*wfn.fft_grid),dtype=np.complex128)
        #for ispin in range(2):
        #    fftbox[ispin,k_gvecs_rot[:,0],k_gvecs_rot[:,1],k_gvecs_rot[:,2]] = wfn_kG[ispin]
        return k_gvecs_rot
    
    def get_cnk_fullzone(self,wfn,nb,nk):
        sym_idx = self.irk_sym_map[nk]
        kbar_idx = self.irk_to_k_map[nk]
        sym_krep = self.sym_mats_k[sym_idx] # note we apply with no changes

        wfn_kG = wfn.get_cnk(kbar_idx,nb)
        wfn_kG = np.einsum('jk,kl->jl', self.U_spinor[sym_idx], wfn_kG)
        return wfn_kG

    def find_qpoint_index(self, q_ext, tol=1e-6):
        """Find index of q-point in unfolded k-points list.

        Args:
            q_ext: Vector of length 3 (crystal coordinates)
            tol: Tolerance for floating point comparison

        Returns:
            Index of matching q-point, or raises ValueError if not found
        """
        # Get fractional part of q_ext
        q_frac = q_ext % 1.0
        diffs = jnp.abs(self.unfolded_kpts - q_frac[None, :])
        # Sum over coordinates and find minimum difference
        total_diffs = jnp.sum(diffs, axis=1)
        min_diff = jnp.min(total_diffs)

        if min_diff > tol:
            raise ValueError(f"No matching q-point found within tolerance {tol}")

        return jnp.argmin(total_diffs)
