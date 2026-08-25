# Uniform DFT gauge vertices

This page owns the sign, units, validity boundary, and nonlocal-projector
algebra used by `psp.dft_operators.static_gauge_vertices_matrix_k`.

The returned neutral Hamiltonian derivatives are

$$
\Gamma_a = +\frac{\partial H}{\partial K_a}, \qquad
\Lambda_{ab} = +\frac{\partial^2 H}{\partial K_a\partial K_b}.
$$

Here Cartesian $K$ is in bohr$^{-1}$, so $\Gamma$ is in Ry bohr and
$\Lambda$ is in Ry bohr$^2$.  Electron-charge signs and factors of $1/c$
are not included; the caller applies them once according to its explicit
minimal-coupling convention.

For fixed bra and ket state blocks, the kinetic pieces are

$$
\Gamma^{\mathrm{kin}}_{a,mn}
=2\langle\psi_m|(K+G)_a|\psi_n\rangle,
$$

$$
\Lambda^{\mathrm{kin}}_{ab,mn}
=2\delta_{ab}\langle\psi_m|\psi_n\rangle.
$$

A local multiplicative potential is invariant under a uniform momentum
shift and contributes neither derivative.  For the norm-conserving nonlocal
operator, let

$$
c_{Rsn}=\langle Z_{Rs}(k)|\psi_{ns}\rangle, \qquad
(V_{\mathrm{NL}})_{mn}=c^\dagger_{m} E c_n.
$$

`VNLSetup` is the sole owner of $Z$'s flattened projector metadata, SOC
matrix $E$, and reduced radial tables $G_l$, $G_l'$, and $G_l''$.  The contact
setup is explicit (`build_vnl_setup(..., compute_contact=True)`); ordinary
Hamiltonian, NSCF, and dipole setup does not build the extra $l+2$ Bessel
table.  A contact requested from a setup without $G_l''$ refuses.

Projector differentiation is fused with the band-block contraction.  The
normal path returns only $c$, $\partial_a c$, and $\partial_a\partial_b c$;
it never returns or retains a full $\partial_a\partial_b Z(G)$ array.  The
nonlocal contact closes in coefficient space with exactly four terms:

$$
\begin{aligned}
\partial_a\partial_b(c_L^\dagger E c_R)={}&
(\partial_a\partial_b c_L)^\dagger E c_R
+(\partial_a c_L)^\dagger E(\partial_b c_R)\\
&+(\partial_b c_L)^\dagger E(\partial_a c_R)
+c_L^\dagger E(\partial_a\partial_b c_R).
\end{aligned}
$$

The bra and ket band extents may differ, which is the low-memory
valence-by-conduction closure.  Coefficient carriers are deliberately
in-process objects: closure requires the exact same `VNLKData` instance, so
setup, SOC mode, $k$, G labels, and padding mask cannot silently drift.
Wavefunctions are held fixed while the operator is differentiated;
eigenstate drift belongs to the Sternheimer response.

## Regular radial Hessian

With $F_l$ the projector Hankel transform and $G_l=F_l/q^l$, define
$H_{l+1}$ from $\beta(r)j_{l+1}(qr)$ and $J_{l+2}$ from
$r\beta(r)j_{l+2}(qr)$.  Spherical-Bessel recurrence gives, for $q>0$,

$$
G_l''(q)=\frac{J_{l+2}(q)}{q^l}
-\frac{H_{l+1}(q)}{q^{l+1}}.
$$

The apparent singularity is not evaluated at the origin.  The exact regular
moment is used instead:

$$
G_l''(0)=-\frac{\int dr\,\beta(r)r^{l+3}}
{(2l+3)!!}.
$$

The exact regular $G_l(0)$ moment is also stored.  This matters for the
$l=1$ and $l=2$ solid-harmonic derivatives even when the radial $G_l''$ term
itself is not the only contribution.

## Validity boundary

Exact $q=0$ is implemented.  A finite-wavevector nonlocal current/contact
requires a selected Wilson-line path for $V_{\mathrm{NL}}(r,r')$.  No such
path is bound in LORRAX, so every public derivative and explicit apply door
shares the fail-closed `EM-VERTEX-FINITE-Q-WILSON` boundary.  Endpoint
averaging is not substituted because it does not define the contact or, by
itself, enforce a finite-$q$ Ward identity.

Canonical four-component bispinors are accepted only in
`[large_up, large_down, small_up, small_down]` order; the DFT VNL acts on the
first two physical components.  Every other spin-layout mismatch refuses.
