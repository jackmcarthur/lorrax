# Time-dependent adiabatic GW: the ISDF interaction model

The time-dependent adiabatic GW equation of motion (TD-aGW) propagates the
band-basis density matrix on the full k grid. Its self-energy update is the
static-COHSEX change linear in the density-matrix change. LORRAX applies that
update on the ISDF basis with the BSE kernel's own W term and a G-space
Hartree term (`tdagw.model`).

## Equation of motion

With $\rho_{\mathbf k,nm}=\langle c^\dagger_{m\mathbf k}c_{n\mathbf k}\rangle$
(row $n$, column $m$) and $G^<_{\mathbf k}=i\rho_{\mathbf k}$,

$$
i\partial_t\rho_{\mathbf k}=[H_{\mathbf k}(t),\rho_{\mathbf k}]
-i\,\Gamma\circ(\rho_{\mathbf k}-\rho^{\rm eq}_{\mathbf k}),\qquad
H_{\mathbf k}=\mathrm{diag}(\varepsilon^{\rm QP}_{\mathbf k})
+e\mathbf E(t)\cdot\mathbf r_{\mathbf k}+\Delta\Sigma_{\mathbf k}[\delta\rho],
$$

$\delta\rho=\rho-\rho^{\rm eq}$, $\rho^{\rm eq}_{\mathbf k,nm}=f_{n\mathbf k}\delta_{nm}$
with $f=1$ on every valence band of the window. The static COH term does not
depend on $\rho$ and sits in $\varepsilon^{\rm QP}$. The update is linear,

$$
\Delta\Sigma_{\mathbf k,nm}=\sum_{\mathbf k'n'm'}K_{\mathbf knm,\mathbf k'n'm'}\,\delta\rho_{\mathbf k',n'm'},
\qquad K=K^{H}-K^{W}.
$$

## The kernel on the ISDF basis

With $\psi_{n\mathbf k,s}(\mu)\equiv\psi_{n\mathbf k,s}(\mathbf r_\mu)$ the
Bloch function at the centroids, the pair amplitude
$M_{\mathbf k,nm}(\mu)=\sum_s\psi^*_{n\mathbf k,s}(\mu)\psi_{m\mathbf k,s}(\mu)$
and $N_k$ the number of k points,

$$
K^{H}_{\mathbf knm,\mathbf k'n'm'}=\frac{w_s}{N_k}\sum_{\mu\nu}M_{\mathbf k,nm}(\mu)\,
\bar V_{\mu\nu}\,M^*_{\mathbf k',n'm'}(\nu),\qquad
\bar V_{\mu\nu}=\sum_{\mathbf G\neq0}\tilde\zeta^*_{0\mu}(\mathbf G)\,v(\mathbf G)\,\tilde\zeta_{0\nu}(\mathbf G),
$$

$$
K^{W}_{\mathbf knm,\mathbf k'n'm'}=\frac{1}{N_k}\sum_{\mu\nu ts}
\psi^*_{n\mathbf k,t}(\mu)\,\psi_{n'\mathbf k',t}(\mu)\,W_{\mu\nu}(\mathbf k-\mathbf k')\,
\psi^*_{m'\mathbf k',s}(\nu)\,\psi_{m\mathbf k,s}(\nu).
$$

$w_s=2$ for a spin-restricted scalar run and $1$ for spinors
(`bse_preconditioner.exchange_spin_weight`): a scalar run stores one spin
channel of $\delta\rho$, and the Hartree field sees both, while the screened
exchange couples equal spins only. $W(\mathbf q)$ is the static screened
interaction $W(\omega=0)$ of the GW restart with its q = 0 head; $\tilde\zeta_0$
is the q = 0 interpolation vector of `zeta_q.h5` and $v$ the dimensionally
truncated Coulomb kernel of the producer ([ζ and V](isdf-zeta-vq.md)).

**G = 0 is excluded from the Hartree term.** The macroscopic field of the
induced polarization is the external $\mathbf E(t)$, so $v(\mathbf G=0)$ is
zero at q = 0. The BSE's q = 0 exchange tile carries in addition the rank-one
cell-averaged head $(v_{\rm head}/\Omega)\,g_0^*g_0$, $g_0(\mu)=\tilde\zeta_{0\mu}(\mathbf G=0)$
([BSE](../architecture/bse.md#the-screened-interaction-w0)); the TD model does
not. $W$ keeps its head $w_{\rm head}$ in both.

**Hermiticity.** For Hermitian $\delta\rho_{\mathbf k}$ the centroid density
$\delta n_\nu$ is real, so $\Delta V^H_{\mathbf k}$ is Hermitian when $\bar V$ is
real symmetric: $\zeta_0(\mathbf r)$ real, $\tilde\zeta_0(-\mathbf G)=\tilde\zeta^*_0(\mathbf G)$,
which the conjugation-closed q = 0 fit gives, and $v(-\mathbf G)=v(\mathbf G)$.
$\Delta\Sigma^W_{\mathbf k}$ is Hermitian when every $W(\mathbf q)$ is a
Hermitian matrix in $(\mu,\nu)$, as the static response is.

## Algorithm

One application, per density matrix of a batch:

1. encode $T_{\mathbf k',ts}(\mu,\nu)=\sum_{ab}\psi_{a\mathbf k',t}(\mu)\,\delta\rho_{\mathbf k',ab}\,\psi^*_{b\mathbf k',s}(\nu)$;
2. convolve $U_{\mathbf k}=\sum_{\mathbf k'}W(\mathbf k-\mathbf k')\,T_{\mathbf k'}
   =\sqrt{N_k}\,\mathrm{FFT}_k\big(W_R\cdot\mathrm{IFFT}_kT\big)$ (ortho norms, $W_R$ the screened tile in R space);
3. decode $\Delta\Sigma^W_{\mathbf k,nm}=N_k^{-1}\sum\psi^*_{n\mathbf k,t}(\mu)\psi_{m\mathbf k,s}(\nu)U_{\mathbf k,ts}(\mu,\nu)$;
4. Hartree: $\delta n_\nu=\sum_{\mathbf k nm}M^*_{\mathbf k,nm}(\nu)\,\delta\rho_{\mathbf k,nm}$,
   $\delta n(\mathbf G)=\sum_\nu\tilde\zeta_{0\nu}(\mathbf G)\,\delta n_\nu$,
   $U_\mu=\sum_{\mathbf G\neq0}\tilde\zeta^*_{0\mu}(\mathbf G)v(\mathbf G)\,\delta n(\mathbf G)$,
   $\Delta V^H_{\mathbf k,nm}=(w_s/N_k)\sum_\mu M_{\mathbf k,nm}(\mu)U_\mu$.

Steps 1–3 are the screened direct term of the TDA BSE matvec
(`bse_stack_matvec.make_w_term`) with both orbital legs the whole TD band
window and $X_{nm\mathbf k}=\delta\rho_{\mathbf k,nm}$: the BSE's
$A=D+w_xV-W$ restricted to $c\to n$, $v\to m$. The resonant block of $K$ is
therefore $A-D$ and its $(cv,v'c')$ block the coupling $B$, so a linear-response
TD-aGW spectrum and the full (non-TDA) BSE share their poles once $V$ carries
no head. Step 4 is the BSE exchange term with $V_{q0}$ factored through
$\tilde\zeta_0(\mathbf G)$; no $N_\mu^2$ Coulomb tile is formed.

## Units, normalization and the TDGW kernel file

LORRAX works in Ry. $\psi(\mathbf r_\mu)$ is sampled exactly as the GW run
samples it (ortho FFT norm, Bloch phase from the WFN's own k table), so
$M\bar VM^\dagger/N_k$ and $K^W$ carry the normalization of BerkeleyGW's
kernel: $\delta\rho_{\mathbf k}$ per k point enters as a BSE trial vector,
with no further factor of $N_k$.

TDGW's `kernel.h5` (`td/io.py:read_kernel`) holds
$-K$ in Ry, read as $K^{\rm TDGW}=-\,\texttt{kernel}/2$ Ha; the dataset is
`kernel[k', k, n, m, n', m', re/im]` $=\langle\mathbf knm|K|\mathbf k'n'm'\rangle$
with the band axes in file order $(i_{\rm vbm}, i_{\rm vbm}-1,\dots,
i_{\rm vbm}+1, i_{\rm vbm}+2,\dots)$, and `bse_header/params/nblocks = 4`
selects every block. `tdagw.kernel_export` writes $K$ in that convention.

## Cost

Per rank on a $\sqrt P\times\sqrt P$ mesh, with $n_b$ the window, $n_s$ the
spinor extent, $N_G$ the q = 0 ζ sphere and complex128 throughout:

| object | bytes |
|---|---|
| $\psi$ on μ (x) and on ν (y) | $2\cdot16\,N_kn_bn_sN_\mu/\sqrt P$ |
| $M$, transition layout | $16\,N_kn_b^2N_\mu/P$ |
| $W_R$ | $16\,N_kN_\mu^2/P$ |
| $\tilde\zeta_0$, $v$ on G (x·y) | $16\,N_\mu N_G/P$ |
| $T$, one trial (only where $T$ is stored) | $16\,N_kn_s^2N_\mu^2/P$ |
| $\delta\rho$ block, gathered | $16\,BN_kn_b^2$ |

The W term costs one k-convolution of an $(n_sN_\mu)^2/P$ tile per density
matrix, $O(N_kN_\mu^2\log N_k/P)$, plus the encode and decode,
$O(N_kn_bn_s^2N_\mu^2/P)$; the Hartree term is $O(N_kn_b^2N_\mu/P+N_\mu N_G/P)$
with two psums of a $(B,N_\mu)$ vector.
