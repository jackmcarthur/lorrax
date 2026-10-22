# Time-dependent adiabatic GW: the ISDF interaction model

The time-dependent adiabatic GW equation of motion (TD-aGW) propagates the
band-basis density matrix on the full k grid. Its self-energy update is the
static-COHSEX change linear in the density-matrix change. LORRAX applies that
update on the ISDF basis with the BSE kernel's own W term and a G-space
Hartree term (`tdagw.model`). The steps from a GW run to a drive are on
[TD-aGW end to end](../how-to/td-agw-workflow.md).

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

## Loading a model

`tdagw.model.load_td_model(deck, mesh, bands=(b_lo, b_hi))` reads W (with its
q = 0 head) and, by default, ψ through the BSE restart loader
(`bse_loading.load_bse_data_from_restart_sharded`), so a deck with
`bse_k_grid` delivers the densified ψ and W. ψ come instead from TDGW's WFN
with `wfn_file`. The Hartree operator always comes from the q = 0 row of
`zeta_q.h5`. Four options are off by default, and with all of them off the
loader call is the earlier one:

| option | effect |
|---|---|
| `eqp_file` | the BSE run's `--eqp`. On a `bse_k_grid` deck it enters the htransform, so the fine ψ are the quasiparticle f(H) states of a `bse_jax --eqp` run on that deck and the fine ε are QP. On a coarse deck ψ do not change and the file only sets the window energies |
| `velocity_file` | the coarse `dipole.h5` (`--dipole`), carried through the htransform; read on the densified route only, refused elsewhere and by a loader that has no velocity |
| `velocity_full_window` | the velocity over every window pair (vv', cc' and cv); refused until the loader offers it |
| `keep_window` | `model["window"]`: the loader's ε (Ry, window ascending), `n_occ`, `n_val`, `n_cond`, dipoles and where the energies came from; on a coarse deck with `eqp_file` the ε are re-sliced as `bse_jax --eqp` does them |

The densified route reproduces the BSE. On CrI3 30×30 (`bse_k_grid` on a
μ_S = 714 downfold, bands [88, 144), the 6000-centroid run's eqp) the lowest
three roots of D + K, solved by block Davidson through `apply` with the BSE's
rank-one exchange head added, match `bse_jax` on the same deck to 0.004 meV.
A wrong ψ row or k label would move them by tens of meV.

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
With `nblocks = 1` TDGW's TDA reader takes only the $(cv, c'v')$ block of the
same dataset and propagates $\Delta\Sigma_{cv}=K_{cv,c'v'}\,\delta\rho_{c'v'}$,
$\Delta\Sigma_{vc}=\Delta\Sigma_{cv}^\dagger$, every other block zero; the
exporter's and the bridge's `--tda` produce exactly that. At TDGW 41a5c34 a
native run can use only that file: its extended reader builds the kernel
without the run's parallel k axis, and the first update refuses at any rank
count.

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

## Outlook: instantaneous dynamical Σ(t)

This section is a design; none of it is implemented. The kernel above
linearizes static COHSEX about $\rho^{\rm eq}$ with W frozen. An
instantaneous dynamical self-energy keeps the equation of motion and its
time-local form (no memory integral) but builds Σ from the state at time t.
That adds dynamical screening and the response of the screening to the
excited carriers (band-gap renormalization, Pauli blocking of W), which the
static kernel leaves out.

### Coherences and populations

Write $\delta\rho=\delta\rho^{\rm coh}+\delta\rho^{\rm pop}$ in the
instantaneous quasiparticle basis $U_{\mathbf k}(t)$ of the field-free
Hamiltonian $H^0_{\mathbf k}(t)=\mathrm{diag}(\varepsilon^{\rm QP}_{\mathbf k})+\Delta\Sigma_{\mathbf k}(t)$:
the populations $f_{i\mathbf k}=(U^\dagger_{\mathbf k}\rho_{\mathbf k}U_{\mathbf k})_{ii}$
form the diagonal part and the rest is the coherent part. The update becomes

$$
\Delta\Sigma_{\mathbf k}(t)=\big(K\,\delta\rho^{\rm coh}\big)_{\mathbf k}
+\Sigma^{\rm dyn}_{\mathbf k}[f(t)]-\Sigma^{\rm dyn}_{\mathbf k}[f^{\rm eq}].
$$

The coherent part keeps the static kernel and `apply`, so the excitonic
polarization is unchanged. The population part comes from a refreshed
dynamical Σ, whose own static linearization in $\delta f$ takes the place of
$K\,\delta\rho^{\rm pop}$, so no term is counted twice. In a weak field
$\delta f=O(E^2)$ while $\delta\rho^{\rm coh}=O(E)$, so linear response, and
with it the BSE equivalence of the kernel section, is exactly the present
model. The field term $e\mathbf E\cdot\mathbf r$ stays out of $H^0$: the
basis then follows the carriers, not the optical cycle.

`apply(drho)` thus becomes one member of a pluggable
`dsigma(state) -> ΔΣ`, where `state` carries $\rho_{\mathbf k}(t)$,
$H^0_{\mathbf k}(t)$ and a cache. With $\Sigma^{\rm dyn}\equiv$ static COHSEX
and no split it reduces to the present model.

### One refresh of $\Sigma^{\rm dyn}[f]$

The QSGW map ([self-consistency §1](../self_consistency.md#1-the-map)) does
the same steps with an aufbau occupation. Here the occupations are the TD
populations:

1. diagonalize $H^0_{\mathbf k}=U\,\mathrm{diag}(E)\,U^\dagger$ in the window and
   rotate the centroid orbitals once from the fixed basis,
   $\tilde\psi_{i\mathbf k}(\mu)=\sum_m U_{mi}\psi_{m\mathbf k}(\mu)$;
   out-of-window bands keep $f=1$ or $0$;
2. $G_{\mathbf k}(\mu,\nu;\tau_j)$ at the response rule's times from
   $(\tilde\psi, E, f)$: the particle branch weighted by $1-f$, the hole branch
   by $f$, so a partly filled state enters both;
3. $\chi^0_q$ and $\partial_s\chi^0_q$ at the supports: one k-convolution of
   $G(\tau_j)\,G(-\tau_j)$ per time (the W term's FFT path), projected by the
   rule's rows (`minimax.response_group_rules`);
4. $W_q=[1-v_q\chi^0_q]^{-1}v_q$ at the supports, then either the tangential
   pencil per parent q (poles $\Omega_j(t)$, factors $b_j(t)$;
   [tangential model](tangential-w-model.md)) or the two-residue GN-PPM fit
   at $\omega=0$ and $i\omega_p$ ([GN-PPM](gn-ppm-nonhermitian.md)), on the
   ordered route, because the magnet and a helicity-selective drive both break
   time reversal;
5. $\Sigma_c(\omega)$ on the window from the pole model, and $\Sigma_x[f]$;
6. the static operator at the instantaneous energies,
   $\Sigma_{ij}=\tfrac12[\Sigma_{ij}(E_i)+\Sigma_{ij}(E_j)]^{\rm h}$
   ([on-shell Σ](sigma-on-shell.md)), rotated back to the fixed basis.

Excited carriers open intraband transitions with $f_i(1-f_j)\neq0$ and small
$\Delta$, so the state is transiently metallic. The response rule must then be
planned with the metal's occupation envelope and the q = 0 intraband (Drude)
head ([self-consistency §7](../self_consistency.md#7-metals)), not the
insulator's interval.

### Cached and recomputed

| object | held | rebuilt |
|---|---|---|
| $r_\mu$, $\tilde\zeta_q(\mathbf G)$, $V(q)$, the heads | whole run (the restart) | — |
| window ψ at the centroids, fixed basis | whole run | rotated each refresh |
| out-of-window $G(\tau_j)$ | while the response rule is held | with the rule |
| response rules, supports, Σ rules | under the QSGW hold tests ([self-consistency §6](../self_consistency.md#tangential-w-with-retained-quadrature)) | when the populated transition interval leaves the plan, logged by name |
| $U$, $E$, $f$; window $G(\tau_j)$; $\chi^0$, $W$ and the pole model at the supports; $\Sigma_c$ on the window | — | every refresh |

### Cost

Per step the cost stays one `apply` of the coherent part plus the basis
rotation, $O(N_kn_b^3)$. Measured on 4 A100 at b = 1, one `apply` takes
95 ms at 30×30 (N_μ = 714, 56 bands, 21.7 GB/device peak) and 309 ms at
12×12 (N_μ = 3000, 60 bands, 34.8 GB/device peak). A refresh costs $N_\tau$ k-convolutions for
$\chi^0$, each the size of one `apply`'s convolution, plus
$O(N_qN_sN_\mu^3/P)$ for the W solves at $N_s$ supports, which dominates at
production $N_\mu$, plus the window Σ, $O(N_qN_{\rm pole}n_bN_\mu^2/P)$.
The refresh is driven by the populations, not the clock: it runs when
$\max|\delta f|$ has moved by a set tolerance since the last one, which
follows the pulse envelope and the carrier relaxation rather than the optical
period.

Checks before use: zero field gives $\Delta\Sigma\equiv0$; a weak field
reproduces the present linear response; and a refresh at thermalized hot
populations equals a one-shot GW of the GW driver at the same occupations.
