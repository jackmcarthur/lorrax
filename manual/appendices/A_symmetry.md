# Appendix A — Symmetry conventions and unfolding

LORRAX inherits BerkeleyGW's conventions from `WFN.h5`. With $S$ the rotation
(`mtrx`) and $\boldsymbol\tau$ the fractional translation, the real-space action is
$\mathbf r' = S^{-1}\mathbf r + \boldsymbol\tau$; the composition law is
$\tau_c = S_a^{-1}\tau_b + \tau_a \pmod 1$ for $S_c = S_a S_b$; reciprocal vectors
transform forward, $\mathbf q' = S\mathbf q$. Time reversal augments the spatial
group, doubling the operation list for the IBZ maps. `SymMaps`
(`common/symmetry_maps.py`) holds the tables: full-zone ↔ irreducible-wedge index
maps, the operation taking each point to its representative, and the TRS half.

Three objects unfold, all through the same tables:

Wavefunctions, spatial operation $s$:
$\psi_{\rm full}(S\mathbf G + \mathbf{kg}_0) =
e^{-i(S\mathbf G)\cdot\boldsymbol\tau_{\rm raw}}\, U_{\rm spinor}(S)\,
\psi_{\bar k}(\mathbf G)$; TRS-augmented operations conjugate and apply
$i\sigma_y U^*_{\rm spinor}(S)$ with the opposite phase sign.

Centroids: an orbit-closed set (§5.4) satisfies
$S(\mathbf x_\mu - \boldsymbol\tau) = \mathbf x_{\alpha(\mu)} + \mathbf L_\mu$ with
integer lattice wrap $\mathbf L_\mu$; the permutation $\alpha$ and wrap table are
built once, with images snapped to FFT-grid integers before flooring (skipping the
snap breaks non-symmorphic groups silently).

Coulomb matrices:
$V_{\rm full}[q,\mu,\nu] = e^{2\pi i\, \mathbf q_{\rm irr}\cdot(\mathbf L_\mu - \mathbf L_\nu)}\,
V_{\rm ibz}[i(q), \alpha(\mu), \alpha(\nu)]$ for spatial operations, and
$V_{\rm full}[T\mathbf q] = V_{\rm ibz}[i(q), \alpha(\nu), \alpha(\mu)]$ for the
TRS half (Hermiticity).

What is and is not reduced. The ζ solve, $V_q$, and the bare exchange ride the IBZ
cascade (a ~6–12× reduction at hexagonal symmetry), gated on orbit closure of the
centroid set. The quasi-density accumulation of §5.3 runs over the full k-lattice,
and the screening and Σ q-loops evaluate all momentum transfers (the lattice-FFT
formulation produces them simultaneously, §4.3); quasiparticle energies themselves
are never symmetrized beyond the degeneracy averaging of §4.5. Transverse bispinor
quantities are excluded from the cascade entirely (§8.3).

Validation recipe: run matched sym/nosym QE+LORRAX pairs and diff; symmorphic
crystals cannot detect τ-phase errors, so any change to this machinery must be
gated on a non-symmorphic system.
