# Physics overview

`LORRAX` The physics of the package: the chain from mean field to GW to BSE, the approximations made at each step, and the two ideas that set the scaling.
{: .page-summary }

<div class="section-cards"></div>

## <span class="n">1</span> The chain {: #chain }

LORRAX starts from a plane-wave mean-field calculation, computes GW quasiparticles on top of it, and solves the Bethe–Salpeter equation on top of those.
{: .intro }

### <span class="n">1.1</span> Mean field {: #ch-mf }

The input is a BerkeleyGW-format `WFN.h5`: Kohn–Sham orbitals $\psi_{n\bk}$, energies $E_{n\bk}$ and occupations $f_{n\bk}$ on a uniform $\bk$ grid, from Quantum ESPRESSO with `pw2bgw` or from the [nscf driver](../drivers/nscf/index.md). The one-body operator $T + V^{\mathrm{ion}}$, with $V^{\mathrm{ion}} = V^{\mathrm{loc}} + V^{\mathrm{NL}}$ the pseudopotential of the mean-field run, is written once to `kin_ion.h5` by [kin-ion](../utilities/kin-ion.md). The Hartree potential $V^{\mathrm H}$ is always built live from the current orbitals. The mean-field exchange-correlation potential is never needed: GW replaces it by $\Sigma^{xc}$.

### <span class="n">1.2</span> GW quasiparticles {: #ch-gw }

The [GW driver](../drivers/gw/index.md) builds the GW self-energy $\Sigma^{xc} = \Sigma^x + \Sigma^c(\omega)$ and from it a static Hermitian quasiparticle Hamiltonian in the mean-field basis:

$$
H = T + V^{\mathrm{ion}} + V^{\mathrm H} + \Sigma^{\mathrm{QSGW}},\qquad H_\bk = U_\bk\,\mathrm{diag}(E^{\mathrm{QP}}_\bk)\,U^\dagger_\bk \tag{1.1}
$$

A one-shot run reads $\Sigma^{xc}$ at the mean-field energies; a QSGW run iterates (1.1) to self-consistency ([quasiparticle self-consistency](qsgw.md)). The production calculation is full-frequency QSGW with the shared-pole $W$.

<figure class="flowchart" markdown>
[![Full-frequency GW driver](../assets/figures/g0.png)](../assets/figures/g0.png)
<figcaption><strong>Fig. G0.</strong> The GW driver as a map of the physics: the ISDF basis and $V$ (1), screening $\chi^0\to W$ and the shared-pole model of $W^c$ (2), the self-energy $\Sigma^c = iG^{\gtrless}W^c$ in real time and $\Sigma^x$ (3), and the quasiparticle Hamiltonian with its QSGW loop (4). Each stage links to its own theory page in §6.</figcaption>
</figure>

### <span class="n">1.3</span> BSE excitons {: #ch-bse }

The [BSE driver](../drivers/bse/index.md) solves for neutral excitations in the basis of transitions $|cv\bk\rangle$, with the QP energies of the GW step and the static screened interaction $W(\omega=0)$ from its restart bundle:

$$
\langle cv\bk|H^{\mathrm{BSE}}|c'v'\bk'\rangle = \big(E^{\mathrm{QP}}_{c\bk}-E^{\mathrm{QP}}_{v\bk}\big)\delta_{cc'}\delta_{vv'}\delta_{\bk\bk'} + \langle cv\bk|K^x - K^d|c'v'\bk'\rangle \tag{1.2}
$$

$K^x$ is the exchange kernel from the bare $V$ and $K^d$ the direct kernel from $W(0)$. Exciton energies $\Omega_S$ and amplitudes $A^S_{cv\bk}$ give the absorption spectrum; the [exciton-bands driver](../drivers/exciton-bands/index.md) solves the same problem at finite exciton momentum $\bQ$.

## <span class="n">2</span> Approximations {: #approx }

Each step truncates the exact many-body problem in a stated way. The table lists where each approximation is made and which key, if any, controls it.
{: .intro }

### <span class="n">2.1</span> Hedin's equations truncated at GW {: #ap-gw }

Hedin's equations close the one-particle problem with the self-energy $\Sigma = iGW\Gamma$ and the polarization $P = -iGG\Gamma$. LORRAX sets the vertex $\Gamma = 1$:

$$
\Sigma(1,2) = i\,G(1,2)\,W(1^+,2),\qquad P(1,2) = \chi^0(1,2) = -i\,G(1,2)\,G(2,1) \tag{2.1}
$$

No vertex correction enters Σ or the screening.

### <span class="n">2.2</span> RPA screening {: #ap-rpa }

The screened interaction is the random-phase-approximation Dyson series in the irreducible polarizability $\chi^0$ of independent quasiparticles:

$$
W(z) = \big[1 - V\chi^0(z)\big]^{-1}V,\qquad W^c = W - V \tag{2.2}
$$

In a one-shot run $\chi^0$ is built from the mean-field states; in QSGW from the current QP states.

### <span class="n">2.3</span> Where each approximation sits {: #ap-table }

| Approximation | Where | Control | Page |
|---|---|---|---|
| $\Gamma = 1$ (GW) | Σ and $\chi^0$ | — | this page |
| RPA, $\chi^0$ of independent particles | $W$ | — | this page |
| Frequency model of $W^c$ | Σ | `compute_mode`, `sigma_w_model` | [frequency treatments](frequency-treatments.md) |
| Static Hermitian Σ in $H$ | QP Hamiltonian | `qp_solver` | [QSGW](qsgw.md) |
| Σ read at mean-field energies (one-shot) | QP Hamiltonian | `qp_solver = one_shot_dft` | [QSGW](qsgw.md) |
| Finite band sum of $G$ in $\Sigma^c$, extrapolated | Σ | `use_band_extrapolation` | [band extrapolation](band-extrapolation.md) |
| Finite band sum of $\chi^0$, not extrapolated | $W$ | `number_bands_chi` | [band extrapolation](band-extrapolation.md) |
| Rank-$N_\mu$ pair-density basis | every two-particle object | centroid count | [ISDF](isdf.md) |
| Analytic $\bq\to0$ head | $W$ and Σ at $\bq=0$ | `head_correction` | [the q → 0 head](q0-head.md) |
| Static screening $W(0)$ in the kernel | BSE | — | [BSE theory](../drivers/bse/theory.md) |
| Tamm–Dancoff (resonant block only) | BSE | `--tda` | [BSE theory](../drivers/bse/theory.md) |
| Pseudopotential of the mean-field run | $V^{\mathrm{ion}}$ | the mean-field run | [kin-ion](../utilities/kin-ion.md) |

!!! note
    The BSE driver solves the full problem with resonant and coupling blocks unless `--tda` is given. The production absorption route uses the Tamm–Dancoff approximation, as BerkeleyGW does by default.

## <span class="n">3</span> Space-time, real-axis {: #space-time }

Green's functions, $\chi^0$ and $\Sigma^c$ are products in time and real space, and the self-energy is evaluated on the real frequency axis. These two choices remove every explicit transition-pair sum.
{: .intro }

### <span class="n">3.1</span> Greater and lesser Green's functions {: #st-g }

`greens_function_kernel`
{: .in-code }

Real time, time-ordered, $T = 0$, $\hbar = 1$. On the ISDF points the band sums are

$$
G^>_\bk(\br_\mu,\br_\nu;t) = -i\sum_n(1-f_{n\bk})\,\psi_{n\bk}(\br_\mu)\psi^*_{n\bk}(\br_\nu)\,e^{-iE_{n\bk}t},\qquad
G^<_\bk(\br_\mu,\br_\nu;t) = +i\sum_n f_{n\bk}\,\psi_{n\bk}(\br_\mu)\psi^*_{n\bk}(\br_\nu)\,e^{-iE_{n\bk}t} \tag{3.1}
$$

and $G(t) = G^{\gtrless}(t)$ for $t\gtrless0$. Occupations select and weight a branch; they never redefine a band energy, so the same form serves insulators and metals. $G$ is never stored on the full grid: it exists only at a time node, as these $\psi\psi^*$ products.

### <span class="n">3.2</span> The polarizability as a product in time {: #st-chi }

After IFFT $\bk\to\bR$, the polarizability is a pointwise product of two Green's functions:

$$
\chi^0_\bR(\br_\mu,\br_\nu;t) = -i\,G^{\gtrless}_\bR(\br_\mu,\br_\nu;t)\,G^{\lessgtr}_{-\bR}(\br_\nu,\br_\mu;-t),\qquad t\gtrless0 \tag{3.2}
$$

FFT $\bR\to\bq$ and the transform $X(\omega) = \int dt\,e^{i\omega t-\eta|t|}X(t)$ give $\chi^0_\bq(\omega)$, continued to the complex frequencies $z$ at which $W$ is sampled. In frequency (3.2) is the familiar $\sum_{vc\bk}|M|^2\big[(\omega-\Delta+i\eta)^{-1}-(\omega+\Delta-i\eta)^{-1}\big]$, whose direct sum is quartic in system size. Per time node the product costs two band GEMMs and one lattice FFT, and the code needs a logarithmic number of nodes ([Time–frequency transforms §3](time-frequency.md#exponentials)).

### <span class="n">3.3</span> The self-energy on the real axis {: #st-sigma }

With $W^c$ as a set of poles, $W^c_\bq(\mu,\nu;z) = \sum_j b_{j\bq}(\mu)b^*_{j\bq}(\nu)\big[(z-\Omega_{j\bq})^{-1}-(z+\Omega_{j\bq})^{-1}\big]$ ([normalisation of $b$](shared-pole-w.md#model-form)), its time form is closed and even in $t$, and the correlation self-energy is again a product:

$$
W^c_\bq(\mu,\nu;t) = -i\sum_jb_{j\bq}(\mu)\,b^*_{j\bq}(\nu)\,e^{-i\Omega_{j\bq}|t|},\qquad
\Sigma^c_\bR(\br_\mu,\br_\nu;t) = i\,G^{\gtrless}_\bR(\br_\mu,\br_\nu;t)\,W^c_\bR(\mu,\nu;t),\quad t\gtrless0 \tag{3.3}
$$

The transform to $\omega$ is taken on the real axis directly, with broadening $\eta$ (`sigma_regularization_ev`, 0.25 eV), so no analytic continuation from the imaginary axis is needed. Exchange is the equal-time limit, $\Sigma^x_\bk(\br_\mu,\br_\nu) = i\sum_\bq G^<_{\bk-\bq}(\br_\mu,\br_\nu;t{=}0)\,V_\bq(\mu,\nu)$. The Σ time nodes are on [Frequency treatments §6](frequency-treatments.md#sigma) and their windows and error certificates on [Time–frequency transforms §4](time-frequency.md#windows) and [§9](time-frequency.md#sigma-boxes).

## <span class="n">4</span> ISDF {: #isdf }

Interpolative separable density fitting replaces every pair index by $N_\mu$ real-space points. After the fit no band pair appears again.
{: .intro }

### <span class="n">4.1</span> Pair densities on interpolation points {: #isdf-pairs }

`isdf_fitting`
{: .in-code }

Each pair density is interpolated from its values at the points $\br_\mu$, with vectors $\zeta_{\bq\mu}$ that do not depend on the bands:

$$
\psi^*_{m\bk}(\br)\,\psi_{n,\bk+\bq}(\br) \approx \sum_\mu \zeta_{\bq\mu}(\br)\,\psi^*_{m\bk}(\br_\mu)\,\psi_{n,\bk+\bq}(\br_\mu) \tag{4.1}
$$

$\zeta$ is the least-squares fit over the band windows that $\chi^0$ and Σ consume. The points are chosen by weighted k-means on the band density ([centroid selection](centroid-selection.md)).

### <span class="n">4.2</span> Two-body objects in the ζ basis {: #isdf-v }

The bare Coulomb interaction becomes an $N_\mu\times N_\mu$ matrix at each $\bq$:

$$
V_\bq(\mu,\nu) = \sum_\bG \zeta^*_{\bq\mu}(\bG)\,v(\bq+\bG)\,\zeta_{\bq\nu}(\bG) \tag{4.2}
$$

$\chi^0$, $W$ and $W^c$ are matrices of the same size. Response, screening and self-energy are then $N_\mu\times N_\mu$ algebra with the band sums inside GEMMs ([ISDF and V(q)](isdf.md)).

### <span class="n">4.3</span> Scaling {: #isdf-scaling }

At fixed $N_k$, the band count $N_b$, $N_\mu$ and the grid size all grow linearly with the number of atoms $N$.

| Step | Direct | LORRAX |
|---|---|---|
| $\chi^0$ at one frequency | $N_kN_qN_vN_cN_\mu^2$, $O(N^4)$ | $N_\tau\,(N_kN_bN_\mu^2 + N_\mu^2N_k\log N_k)$, $O(N^3)$ |
| $W$ by Dyson | — | $N_qN_zN_\mu^3$, $O(N^3)$ |
| $\Sigma^c$ | state-by-pole sum | one contraction per time node and window, $O(N^3)$ |

GW costs $O(N^3)$ and BSE $O(N^4)$ in system size.

## <span class="n">5</span> Conventions {: #conventions }

One notation runs through the manual, the figures and the code comments.
{: .intro }

### <span class="n">5.1</span> Labels and indices {: #cv-labels }

Word labels are superscripts and indices are subscripts: $V^{\mathrm H}$, $V^{\mathrm{ion}}$, $V^{\mathrm{NL}}_\bk$, $\Sigma^x$, $\Sigma^c$, $\Sigma^{xc}$, $\Sigma^{\mathrm{QSGW}}$, $E^{\mathrm{QP}}_{n\bk}$. Mean-field energies are plain $E_{n\bk}$. Matrix elements are bare bra-kets, $\langle m\bk|\Sigma^c(\omega)|n\bk\rangle$; the basis is named only where the distinction between mean-field and QP states matters. BSE indices run $cv\bk$.

### <span class="n">5.2</span> ISDF arguments {: #cv-isdf }

Objects sampled at the interpolation points carry $(\br_\mu,\br_\nu)$: $\psi_{n\bk}(\br_\mu)$, $G^{\gtrless}_\bk(\br_\mu,\br_\nu;t)$, $\chi^0_\bR(\br_\mu,\br_\nu;t)$. Two-body objects expanded in the ζ basis carry $(\mu,\nu)$: $V_\bq(\mu,\nu)$, $W_\bq(\mu,\nu;z)$, $b_{j\bq}(\mu)$.

### <span class="n">5.3</span> Transforms, units and dropped factors {: #cv-units }

Reciprocal to real space ($\bk\to\bR$, $\bq\to\bR$, $\bG\to\br$) is an IFFT; real to reciprocal is an FFT. This page drops the $1/N_k$ on $\bk$ and $\bq$ sums and the spin factor 2 in $\chi^0$ from its displayed equations; the theory pages that state a normalization keep $1/N_k$ ([Frequency treatments (1.1)](frequency-treatments.md#frame-time)). The code works in Rydberg; deck keys name their unit (`_ev`, `_ry`), and `eqp0.dat`, `eqp1.dat` and the logs are in eV. The interpolation grids of band interpolation are $\bk_{\mathrm{co}}$ and $\bk_{\mathrm{fi}}$.

## <span class="n">6</span> Where each piece is documented {: #map }

Each fact has one home page. The table follows the chain.
{: .intro }

### <span class="n">6.1</span> Page map {: #map-table }

| Piece | Figure | Page |
|---|---|---|
| Mean-field orbitals on a new $\bk$ set | M1 | [nscf driver](../drivers/nscf/index.md) |
| $T+V^{\mathrm{ion}}$, velocity matrix elements | P1 | [kin-ion](../utilities/kin-ion.md), [dipole](../utilities/dipole.md) |
| Interpolation points | K1 | [centroids](../utilities/centroids.md), [centroid selection](centroid-selection.md) |
| ζ fit and $V_\bq(\mu,\nu)$ | G1 | [ISDF and V(q)](isdf.md) |
| Time and frequency rules | G3, G6 | [time–frequency transforms](time-frequency.md) |
| $\chi^0$, $W$, frequency models of $W^c$ | G5, G7, G8 | [frequency treatments](frequency-treatments.md) |
| Tangential W model | G9, G9b | [Tangential W model](shared-pole-w.md) |
| $\bq\to0$ head | G10 | [the q → 0 head](q0-head.md) |
| Fractional occupations | G14 | [metals](metals.md) |
| Band tail of $\Sigma^c$ | G16 | [band extrapolation](band-extrapolation.md) |
| One-shot and QSGW Hamiltonian | G11, G12 | [quasiparticle self-consistency](qsgw.md) |
| QP velocity in QSGW | — | [velocity operator](velocity-operator.md) |
| Space-group action on ψ, ζ, $V$, $W$ | — | [symmetry](symmetry.md) |
| The whole GW driver | G0 | [GW theory](../drivers/gw/theory.md) |
| Band interpolation | H1 | [htransform](../utilities/htransform.md) |
| $V$ and $W$ on fewer points | D1 | [downfold](../utilities/downfold.md) |
| BSE Hamiltonian, solvers, absorption | B1–B4 | [BSE theory](../drivers/bse/theory.md) |
| Excitons at finite $\bQ$ | X1 | [exciton-bands theory](../drivers/exciton-bands/theory.md) |

## Notation {: #notation }

| Symbol | Meaning |
|---|---|
| $\psi_{n\bk}$, $E_{n\bk}$, $f_{n\bk}$ | mean-field orbitals, energies, occupations |
| $G^{\gtrless}$ | greater and lesser Green's functions |
| $\chi^0$ | independent-particle polarizability |
| $V$, $W$, $W^c = W - V$ | bare and screened Coulomb interaction; correlation part |
| $b_{j\bq}(\mu)$, $\Omega_{j\bq}$ | shared-pole vectors (residue $b_{j\bq}b^\dagger_{j\bq}$) and pole energies of $W^c$ |
| $\Sigma^x$, $\Sigma^c$, $\Sigma^{xc}$ | exchange, correlation and total GW self-energy |
| $\Sigma^{\mathrm{QSGW}}$ | static Hermitian QSGW self-energy |
| $V^{\mathrm{ion}}$, $V^{\mathrm H}$ | pseudopotential; Hartree potential |
| $U_\bk$, $E^{\mathrm{QP}}_{n\bk}$ | QP eigenvectors and energies |
| $N_\mu$, $\br_\mu$, $\zeta_{\bq\mu}(\br)$ | ISDF point count, points, interpolation vectors |
| $\bR$ | lattice vector, Fourier partner of $\bk$ |
| $K^x$, $K^d$ | BSE exchange and direct kernels |
| $A^S_{cv\bk}$, $\Omega_S$ | exciton amplitudes and energies |
| $\eta$ | real-axis broadening |

## References {: #refs }

1. L. Hedin, Phys. Rev. 139, A796 (1965).
2. M. S. Hybertsen and S. G. Louie, Phys. Rev. B 34, 5390 (1986).
3. H. N. Rojas, R. W. Godby and R. J. Needs, Phys. Rev. Lett. 74, 1827 (1995).
4. J. Lu and L. Ying, J. Comput. Phys. 302, 329 (2015).
5. K. Dong, W. Hu and L. Lin, J. Chem. Theory Comput. 14, 1311 (2018).
6. T. Kotani, M. van Schilfgaarde and S. V. Faleev, Phys. Rev. B 76, 165106 (2007).
7. M. Rohlfing and S. G. Louie, Phys. Rev. B 62, 4927 (2000).

