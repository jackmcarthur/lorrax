# TD-aGW with an instantaneous dynamical self-energy

This page owns the design of a time-dependent GW self-energy that is rebuilt
from the instantaneous state during a TD-aGW propagation: the equations, the
split of the Green function into a frozen and an active part, the static limit
that is the present model, what the scheme conserves and what it loses, which
GW components it reuses, the cost per refresh measured from the GW stage
walls, and the refactors it needs. The static-screening member is implemented
(`tdagw.dynsig`); the GN-PPM and shared-pole members are designed here. The
equation of motion, integrators and layout are owned by
[TD-aGW propagation](td_agw_propagation.md); the frozen-W kernel by
[TD-aGW interaction](td_agw.md). Equations in Ry unless stated.

## 1. The scheme {#scheme}

The propagator integrates
$i\dot\rho_k=[H_k(t),\rho_k]+\dots$ with
$H_k=\mathrm{diag}(\varepsilon^{\rm QP}_k)+e\mathbf E\cdot\mathbf r_k+\Delta\Sigma_k(t)$
and reads $\Delta\Sigma$ only through the contract
`apply(params, rho, t) -> dSigma` (`tdagw.self_energy`). The instantaneous
scheme sets

$$
\Delta\Sigma(t)=\Sigma\big[\rho(t)\big]-\Sigma\big[\rho^{\rm eq}\big],\qquad
\Sigma[\rho]=\Sigma^{x}[\rho]+\Sigma^{c}\big[G_{\rho},W_{\rho}\big]+V_H[\rho], \tag{1}
$$

where $G_\rho(\tau)$ is an equilibrium-like Green function built from the
state at time $t$ and $W_\rho$ the RPA screening of that Green function. Every
evaluation is a pure function of $\rho(t)$, so any integrator of the
propagation page may call it; nothing is carried between calls.

### 1.1 The instantaneous state {#state}

The TD window $A=[b_{\rm lo},b_{\rm hi})$ carries $\rho$; every other band of
the GW run's sums is **frozen**: its energy and occupation are the
equilibrium ones. In the window

$$
H^G_{A,k}(t)=\mathrm{diag}\big(\varepsilon^G_{A,k}\big)+\lambda\,\Delta\Sigma^{(0)}_k[\delta\rho]
=U_k\,\mathrm{diag}(e_k)\,U_k^\dagger,\qquad
\tilde\psi_{ik}=\textstyle\sum_m\psi_{mk}U_{k,mi},\qquad
f_{ik}=\mathrm{Re}\,\big(U_k^\dagger\rho_kU_k\big)_{ii}. \tag{2}
$$

$\varepsilon^G$ are the energies the GW run's $G$ used (DFT for
`one_shot_dft`), so that $\Sigma[\rho^{\rm eq}]$ is the run's own $\Sigma$
(§6.1). $\lambda=0$ (`dress = "none"`) keeps $U=1$, $e=\varepsilon^G$;
$\lambda=1$ (`dress = "instantaneous"`) diagonalizes the field-free
$H^G+\Delta\Sigma^{(0)}$ with $\Delta\Sigma^{(0)}=K\,\delta\rho$ the frozen-W
update, a pure function of $\rho$ (a self-consistent dressing would iterate
Eq. 1 inside one call). The field term stays out of $H^G$: the basis follows
the carriers, not the optical cycle.

**The approximation for the coherences.** $\rho$ is not diagonal in the
instantaneous basis. The scheme uses it in two ways:

* **Screening** sees only the populations $f$ of Eq. (2): the coherences
  $(U^\dagger\rho U)_{i\neq j}$ are dropped from $\chi_0$. A non-stationary
  state has no frequency-domain $\chi_0$; the dropped part oscillates at the
  transition frequencies and is the $\delta W/\delta G$ term the BSE also
  drops.
* **Exchange** sees the full equal-time $G^<(t,t)=i\rho$: Generalized
  Kadanoff–Baym ansatz (GKBA) in either ordering,
  $G^<(t,t-s)=ie^{-iHs}\rho$ or $i\rho e^{-iHs}$, gives
  $G(0^-)=\rho$ exactly, so the static screened exchange
  $\Sigma^{SX}=-\rho\,W$ keeps every coherence. For a dynamical $W$ the
  correlation part uses the symmetrized GKBA,
  $g_{ij}(\tau)=\tfrac12(e^{-e_i\tau}+e^{-e_j\tau})\,\tilde\rho_{ij}$, a
  coherence entering at both of its energies (designed, §3.2).

### 1.2 Frozen and active Green functions {#split}

At a minimax node $\tau$, with $x_n=e_n-e_{\rm sep}$ measured from a mid-gap
separator $e_{\rm sep}$ (lower set L: $x<0$, upper set U: $x>0$), the
spin-resolved Green tiles are

$$
G^{h/p}_{X,k}(\mu s,\nu t;\tau)=\sum_{n\in X}w^{h/p}_{nk}(\tau)\,\psi_{nk,s}(\mu)\psi^*_{nk,t}(\nu),\qquad
w^{h}_n=f_ne^{-|x_n|\tau},\; w^{p}_n=(1-f_n)e^{-|x_n|\tau}, \tag{3}
$$

$G=G_{\rm frozen}+G_{\rm active}(t)$. $G_{\rm frozen}$ (hole tile over the
frozen occupied bands, particle tile over the frozen empty ones) is
**composed at every node from the centroid wavefunctions of every band**,
as the GW $\chi_0$ build does; no $G_R(\tau)$ is stored. $G_{\rm active}$
carries the rotated orbitals $\tilde\psi$ and the populations $f$ and costs
$O(N_kn_An_s^2N_\mu^2)$ per tile.

### 1.3 $\chi_0$ with the split {#chi0}

$\chi_{0,q}(\mu,\nu)=\frac{g_s}{N_k}\sum_k\sum_{a\in k,\,b\in k-q}
\frac{f_a-f_b}{E_a-E_b}\rho_{ab}(\mu)\rho_{ab}^*(\nu)$ (Adler–Wiser over
ordered pairs, $\rho_{ab}=\sum_s\psi_{a,s}\psi^*_{b,s}$, $g_s=1$ for
spinors). Pairs across the gap use the GW run's Laplace rule
$1/x\simeq\sum_l\alpha_le^{-x\tau_l}$; with $f_a-f_b=f_a(1-f_b)-(1-f_a)f_b$
they factorize into two products per node, the second being Pauli blocking
(stimulated emission):

$$
\chi_{0,R}=-2\,\mathrm{Re}\sum_l\alpha_l\Big[G^h_L\star G^{p*}_U-G^p_L\star G^{h*}_U\Big]_{R,\tau_l},\qquad
A\star B^*\equiv\sum_{st}A_R(\mu s,\nu t)B^*_R(\mu s,\nu t), \tag{4}
$$

the spin-traced elementwise product of `gw.w_isdf`'s $\chi_0$ kernel
(`Gc_R` $\cdot$ `conj(Gv_R)`), then $\chi_{0,q}=(g_s/\sqrt{N_k})\,\mathrm{FFT}_R$.
At equilibrium $G^p_L=G^h_U=0$ and Eq. (4) is the run's $\chi_0$. With
$G=G_f+G_a$,

$$
\chi_0=\underbrace{\chi_0^{ff}}_{\rm static}+\chi_0^{fa}(t)+\chi_0^{aa}(t),
$$

and only the terms with an active leg change. Regrouping
$G_{\rm eq}=G_f+G_a(0)$ gives the update without any difference of two
equilibrium-sized products:

$$
\delta\chi_0=\big[\chi_0^{fa}(t)-\chi_0^{fa}(0)\big]+\big[\chi_0^{aa}(t)-\chi_0^{aa}(0)\big]
=-2\,\mathrm{Re}\sum_l\alpha_l\Big[G^h_{\rm eq}\star\delta G^{p*}+\delta G^h\star G^{p*}_{\rm eq}
+\delta G^h\star\delta G^{p*}-G^p_L\star G^{h*}_U\Big]+\chi_0^{ss}, \tag{5}
$$

$\delta G^h=G^h_{a,L}(t)-G^h_{a,L}(0)$, $\delta G^p=G^p_{a,U}(t)-G^p_{a,U}(0)$.
$\chi_0^{ss}$ collects the **same-side pairs** whose populations differ
(carrier intraband pairs, active $\times$ frozen on one side of the gap);
their energy order is not fixed across $k$, so they do not factorize on the
Laplace rule. The prototype sums them exactly at $\omega=0$; production takes
the fractional-occupation contour route of
[metallic MPA screening §1](metallic-mpa-screening.md#11-the-identity-and-the-implemented-form),
whose real-time product $A_q(t)=\sum f_a(1-f_b)e^{-i\Delta t}X_{ab}$ is exact
for any real $f$ (the deck-level Fermi–Dirac gate is a deck rule, not a
kernel limit).

### 1.4 W and the self-energy {#w}

$\chi_0^{ff}$ is folded once into the run's own $W_0$ at each support $z_s$:

$$
W(z_s,t)=\big[1-W_0(z_s)\,\delta\chi_0(z_s,t)\big]^{-1}W_0(z_s), \tag{6}
$$

the run's Dyson solve with $W_0$ in the place of $v$ (`gw.w_isdf.solve_w`).
Storing $\chi_0^{ff}(\tau)$ per $q$ instead costs
$N_\tau N_qN_\mu^2\cdot16$ B, the class of $W$ at every node: 767 GB at
12×12, $N_\mu=3000$, 37 nodes, against 41 GB for $W_0$ at the two GN
supports (§7). Recomputing $\chi_0^{ff}$ adds only its products, since
$G_{\rm eq}$ is composed anyway for Eq. (5); folding is cheaper in both.

**Static member** (`screening = "static_rpa"`, implemented). One support,
$z=0$; COHSEX with `gw.cohsex_sigma`'s occupied and band-sum completeness
Greens:

$$
\Delta\Sigma=K\delta\rho+\mathcal W[\delta W]\big(-\rho^{\rm eq}_{\rm tot}-\delta\rho+\tfrac12\mathbb 1\big)\Big|_A,\qquad
\mathcal W[W](X)_{k,ij}=\frac1{N_k}\sum_{k'nm}\psi^*_{ik}\psi_{nk'}\,W_{k-k'}\,\psi^*_{mk'}\psi_{jk}\,X_{k',nm}, \tag{7}
$$

$\mathcal W$ being the BSE W-term (`bse_stack_matvec.make_w_term`) on the
full band set; $K\delta\rho$ is the TD model unchanged.

**GN-PPM member** (designed). Two supports $0$ and $i\omega_p$, Eq. (6) at
each; the per-element pole fit of `ppm_sigma.fit_ppm` (ordered, since the
magnet and a helicity drive break time reversal); $\Sigma^c_{ij}(\omega)$ by
the Σ τ sweep with $G=G_{\rm eq}+\delta G$ (symmetrized GKBA for the
coherences, §1.1); the static operator at the instantaneous energies,
$\Sigma_{ij}=\tfrac12[\Sigma_{ij}(e_i)+\Sigma_{ij}(e_j)]^{\rm h}$
([on-shell Σ](sigma-on-shell.md)), rotated back with $U$.

**Shared-pole member** (designed). $W(z)=\sum_jb_j(q)R_j/(z-\Omega_j)$ with
poles shared per $q$; Eq. (6) at the bank's supports, then the pencil and
constructor of `gw.shared_pole_*`. Measured on 516 the constructor alone is
16.2 s of a 30 s W stage (§7), so it is the expensive member.

**Head.** The prototype keeps the run's $W_h$ (band-diagonal SX/COH shifts,
[static head](four-current-head-corrections.md#static-head)) frozen: $\delta\chi_0$
carries no $q\to0$ head. The head change is $q\cdot\delta S\cdot q$ from the
window's dipoles and populations (the `qsgw_head` route), plus a Drude term
once carriers exist (§4).

## 2. The static limit is the present model {#static-limit}

With $W$ frozen at $W_0(\omega=0)$: $\Sigma^{SX}[\rho]=-\mathcal W[W_0](\rho_{\rm tot})$
is linear in $\rho$ for **any** basis $U$, because $G(0^-)=\rho$ in either
GKBA ordering; $\Sigma^{COH}=\tfrac12\mathcal W[W_0-v](\mathbb 1)$ does not
depend on $\rho$; $V_H$ is linear. Eq. (1) is then exactly
$K\delta\rho=(K^H-K^W)\delta\rho$ of [TD-aGW interaction](td_agw.md), with no
linearization. Measured (§8): the literal Eq. (1) assembled in a rotated
instantaneous basis equals $-K^W\delta\rho$ to $1.4\times10^{-14}$ on the
synthetic bundle and $1.1\times10^{-11}$ on CrI3 516, and dropping the
coherences from $G^<$ misses it by 175% and 1220%; `screening = "frozen"`
through the propagator equals `isdf_model` bitwise on a pulse on both.

With a frozen *dynamical* $W$, the linearization of $\Sigma^c[G_\rho,W_0]$ in
$\delta\rho$ is $K$ plus a dynamical correction that vanishes as the pole
energies $\to\infty$ (static $W$). With $W$ updated, $\delta W\propto\delta f$
and the populations are $O(E^2)$ (`dress = "none"`: $U=1$, $f=\mathrm{diag}\rho$),
so linear response is exactly TD-aGW, the full BSE of the kernel page.
`dress = "instantaneous"` shifts $e$ at $O(E)$ through $K\delta\rho^{\rm coh}$
and changes the screening at $O(E)$: its linear response is not the BSE.

TDMODEL's outlook ([td_agw.md](td_agw.md), outlook) keeps $K\delta\rho^{\rm coh}$
and adds $\Sigma^{\rm dyn}[f]-\Sigma^{\rm dyn}[f^{\rm eq}]$ for the populations.
With `dress = "none"` the present scheme is that split plus one term: the
coherent part is $K\delta\rho$ exactly, the population part is
$\mathcal W[\delta W](-\rho^{\rm eq}_{\rm tot}-\delta\rho^{\rm pop}+\tfrac12)$,
and $-\mathcal W[\delta W](\delta\rho^{\rm coh})$, the screened exchange of the
coherences with the changed $W$ (second order, $\delta f\,\delta\rho^{\rm coh}$),
is the term their split drops.

## 3. Conservation, and what is lost {#conservation}

### 3.1 Kept

* $\mathrm{Tr}\,\rho_k$ per $k$, Hermiticity of $\Delta\Sigma$ (measured
  $\le1.2\times10^{-17}$), unitarity of the commutator flow (the spectrum of $\rho$
  at $\Gamma=0$);
* $\rho=\rho^{\rm eq}$ is a fixed point, $\Delta\Sigma=0$ exactly;
* covariance under a rotation of the window basis: the populations are those
  of the instantaneous basis, not of the band gauge (the red twin that reads
  them in the fixed basis misses the reference by 10%);
* linear response = TD-aGW = BSE (§2).

### 3.2 Lost

* **Time locality.** $\Sigma(t,t')$ is replaced by a function of $\rho(t)$ alone;
  no memory, no non-Markovian dephasing.
* **Φ-derivability.** With $W$ updated, $\Delta\Sigma$ is not $\delta E/\delta\rho$
  of any functional of $\rho$ (the GW Φ functional is a functional of the
  two-time $G$, and Eq. (2) is not its stationarity condition), so the energy
  is not conserved exactly. The frozen-W member is Φ-derivable from
  $\tfrac12\mathrm{Tr}\,\delta\rho K\delta\rho$ and conserves its energy.
* **Damping.** The Hermitian on-shell operator drops $\mathrm{Im}\,\Sigma$:
  no quasiparticle lifetimes, carrier relaxation or impact ionization; these
  need $\Gamma$ or a collision term.
* **Coherence-driven screening** ($\delta W/\delta G$ at the drive frequency),
  by construction (§1.1).

## 4. Excited carriers and static screening {#carriers}

The same-side pairs of Eq. (5) carry $(f_a-f_b)/(E_a-E_b)$ for states at
different $k$ whose energies can be close (4.3 mRy in the synthetic test) and
whose populations are not a function of energy. Their static response is
large, and of either sign for an inverted population. Measured through the
propagator (§8): with the cross-gap terms only, the departure of the
`static_rpa` run from the frozen run scales as $E_0^{2.2}$ ($E_0$ = 2.5 and 5 $\times10^{-4}$ a.u.);
with the exact same-side pairs it grows 77× when $E_0$ doubles from 0.002 to
0.004 a.u. At fixed weak populations $\delta\chi_0$
is linear and $|W_0\delta\chi_0|\approx2\times10^{-3}$, so the growth is the
feedback of a large static carrier screening on the dynamics, not a
singular solve. On CrI3 516 (2v2c, after a 6 fs, $5\times10^8$ V/m pulse along
$x$, $\max|\delta\rho|=0.011$) the cross-gap part changes ΔΣ by
$1.8\times10^{-3}$ of $K\delta\rho$, and the exact same-side sum refuses:
920 same-side entries pair equal energies (within $10^{-6}$ Ry) with different
populations, where $(f_a-f_b)/(E_a-E_b)$ has no $\omega=0$ limit. Equal
energies come with the point group (partners at $k$ and $Rk$, which a
linearly polarized drive populates differently) and with the multiplets the
window closes. **Static ($\omega=0$) screening by non-thermal carriers is not
a usable production model**: the carrier response must be taken at the
frequencies it acts at (the GN-PPM or shared-pole member, with the
fractional contour route for $\chi_0^{ss}$), or the carriers thermalized
first. This is the case for the frequency-dependent push.

## 5. GW components reused, and their layout {#reuse}

| component | where | layout | per-step use |
|---|---|---|---|
| Laplace / imaginary-axis rules | `gw.minimax_screening.solve_laplace_minimax_interval` (:1060), `build_imag_probe_response_rule` (:1417) | host arrays | planned once on the union of the equilibrium and the expected instantaneous interval (`static_rule(margin=)`); held |
| Green tiles at a node | `gw.greens_function_kernel.build_G_tau` (:426) | $(k,\mu_X,s,\nu_Y,t)$ | $G_{\rm eq}$ of every band, every node, every refresh |
| $\chi_0$ products | `gw.w_isdf._get_chi_minimax_kernel_face` (:505), `_fused` (:772) | $(R,\mu_X,\nu_Y)$ | Eq. (5); needs weight rows (§9) |
| fractional $\chi_0$ | `gw.w_isdf.compute_chi0_contour_fractional` (:3349), `compute_chi0_direct_fractional` (:3657) | as above | $\chi_0^{ss}$ |
| Dyson | `gw.w_isdf.solve_w` (:2114), `meta=None` route | $(q,\mu_X,\nu_Y)$ | Eq. (6), unchanged (reused by the prototype) |
| GN-PPM fit | `gw.ppm_sigma.fit_ppm` (:808), `minimax_screening.fit_gn_ppm_from_wc_pair` (:309) | $(q,\mu_X,\nu_Y)$ | array-callable already |
| shared-pole W | `gw.shared_pole_screening.screen_shared_poles` (:443), `shared_pole_constructor.construct_shared_poles` (:57) | bank and model stores on disk | §9 |
| Σ τ sweep | `gw.ppm_tau_kernel._sigma_subtile_kernel` (:283), `_get_sigma_kij_kernel` (:482); `ppm_pipeline.compute_ppm_sigma_pipeline` (:690) | parent Greens $\times$ $W(\tau)$, window projection | GN-PPM / shared-pole Σ |
| static Σ | `gw.cohsex_sigma._make_cohsex_kernels_face` (:400); `bse.bse_stack_matvec.make_w_term` (:470) | $(b,n_X,m_Y,k)$ | Eq. (7) (prototype: `make_w_term` on the full band set) |
| static head | `gw.head_correction.static_head_terms_to_kij` (:1826), `fit_head_ppm` (:1508) | band diagonal | frozen in the prototype; §1.4 |
| frozen-W kernel | `tdagw.model.build_td_agw_apply` | $(b,k,n_X,m_Y)$ | $K\delta\rho$, unchanged |

## 6. Cached and recomputed {#cache}

| object | held for the run | rebuilt per refresh |
|---|---|---|
| $r_\mu$, $\tilde\zeta$, $V_q$, centroid $\psi$ of every band (`psi_full`, $2\cdot16N_kN_bn_sN_\mu/\sqrt P$ B/rank), $\varepsilon^G$, $f^{\rm eq}$ | ✓ | — |
| rules ($\tau_l,\alpha_l$), $e_{\rm sep}$ | ✓ (re-planned only when an instantaneous transition leaves the interval: `GATE dynsig_transition_outside_rule`) | — |
| $W_0(z_s)$ per support (folds $\chi_0^{ff}$), heads $W_h,v_h$ | ✓ | — |
| $G_{\rm frozen}(\tau)$, $G_{\rm eq}(\tau)$ | never stored | composed per node |
| $U,e,f$; $\delta G(\tau)$; $\delta\chi_0(z_s)$; $W(z_s,t)$; pole model; $\Delta\Sigma^{\rm dyn}$ | — | ✓ |

### 6.1 The t = 0 check {#t0}

$\chi_0^{ff}+\chi_0^{fa}(0)+\chi_0^{aa}(0)$, built from separate frozen and
active tiles at the run's own nodes, must give the run's $W_0$ by Dyson with
the run's $V$, and Eq. (7)'s $\Sigma^{SX}+\Sigma^{COH}$ at $\rho^{\rm eq}$
(plus the run's band-diagonal heads) must give the run's `sigma_diag.dat` in
the window. The cross term is not small: $\chi_0^{fa}$ is 47% of
$\max|\chi_0|$ on the synthetic bundle, and dropping it, or counting the
active valence as empty, fails the check by orders of magnitude (§8).

On CrI3 516 (`05_gw_cohsex_bundle`: 6×6 k, 160 bands, $N_\mu$ 978, window
0-based [128, 132), the run's interval [0.110, 6.130] Ry re-solved to 11 nodes,
max error $2.9\times10^{-8}$): $W(0)$ from the split reproduces the run's
`W0_qmunu` to $1.3\times10^{-9}$ (relative, max over all 36 q, q = 0
included); $\chi_0^{fa}$ is 33% and $\chi_0^{aa}$ 6.1% of $\max|\chi_0|$.
Σ^SX + Σ^COH of the window reproduces `sigma_diag.dat` (sigSX, sigCOH; eV, six
decimals; $|\Sigma^{SX}|\le12.9$ eV) to $7.5\times10^{-6}$ eV over 4 bands ×
36 k, and to $7.6\times10^{-6}$ eV from the run's own $W_0$, so the residual is
not $\chi_0$'s; it is consistent with the Ry→eV constant (the check converted with
13.605693122994, the writer with `common.units.RYD_TO_EV` = 13.6056980659,
$3.6\times10^{-7}$ relative = $4.7\times10^{-6}$ eV at 12.9 eV) plus the
file's rounding. Tolerance $10^{-5}$ eV. The twins: $\chi_0^{fa}$ dropped
0.41 eV, the active valence counted empty 0.10 eV, the heads dropped 0.72 eV.

## 7. Cost per refresh {#cost}

One refresh = $\delta\chi_0$ at every support (one $G_{\rm eq}$ composition
per node, the GW $\chi_0$ build), the Dyson solves on the **full** $q$ set (a
general drive breaks the point group; the GW runs below use the IBZ wedge),
the pole model, and the window Σ. From the GW stage walls of the 471 and 516
runs (`gwjax.out` MAJOR-STAGE TIMING; 4 × A100-SXM4-80GB, one rank per GPU,
`linalg = local`, the runs' allocator), with Dyson scaled from the wedge to
the full zone:

| system (bands in the sums) | $N_k$, $N_\mu$ | $\chi_0$ static + probe | Dyson, full $q$ | window Σ | refresh | frozen `apply` |
|---|---|---|---|---|---|---|
| 516 6×6, 2v2c (160), GN-PPM (estimate: no GN run on 516) | 36, 978 | 0.2 + 0.4 s | 0.2 s | 7.0 s (the shared-pole run's sweep) | **~8 s** | 2.95 ms |
| 516 6×6, 2v2c (160), shared-pole (`04`) | 36, 978 | bank 12.5 s + moments 1.3 s | in the bank | 7.0 s | **~37 s** (constructor 16.2 s) | 2.95 ms |
| 471 8×8 (898), N_μ 3000 (`14_bseprod_c3000`) | 64, 3000 | 1.1 + 2.4 s | 2 × 0.2 × 6.4 = 2.6 s | 16.6 + 11.7 s | **~34 s** | — |
| 471 8×8 (898), N_μ 6000 (`15_bseprod_c6000`) | 64, 6000 | 3.7 + 8.7 s | 2 × 0.9 × 6.4 = 11.5 s | 86.4 + 12.4 s | **~123 s** | — |
| 30×30, 42v × 14c (`34_bseprodb_df3000r8_w88_144`) | via the 8×8 parent | as 8×8 | as 8×8 | window on 30×30 | **8×8 + densify** | 95.4 ms |
| 518 12×12 (898), 30v × 30c (`32_bseprodb_k12_c3000`) | 144, 3000 | 2.4 + 14.6 s | 2 × 0.3 × 7.6 = 4.5 s | 46.3 + 13.9 s | **~82 s** | 309.3 ms |

Measured, the static member (`screening="static_rpa"`: Eq. (5) cross-gap,
Dyson on 36 q, Eq. (7) on the window, 160 bands) costs 0.41 s per warm call on
516 at 4 A100, 139× the frozen `apply`; the exact same-side enumeration adds
0.54 s. "Window Σ" is the Σ τ sweep plus the other Σ stages except exchange and
Hartree, which $K\delta\rho$ already carries; the Dyson column scales the
runs' wedge solves (10 q at 8×8, 19 at 12×12) to the full zone. The frozen
`apply` walls are TDPROP's (516 2v2c) and TDMODEL's (30×30, 12×12) at b = 1;
at 30v × 14c on 6×6 k it is 9.0 ms.

The 30×30 W update runs on the 8×8 parent (the 30×30 NSCF has 200 bands, not
the 898 of the sums) and is densified as W already is
(`bse_densify`, srlr/c1); the window populations are folded to the parent grid
through the htransform. **Every-call refresh is unaffordable**: at 80 ΔΣ calls
per fs (cfm4, 50 as, two corrections) the 12×12 refresh is 1.8 h per fs on 4
A100, 265× the frozen `apply`. Memory: with no point-group symmetry the
$G_{\rm eq}$ parent tiles at 12×12 are $144\,(2\cdot3000)^2\cdot16$ B = 83 GB
each (20.7 GB/rank at P4), so the refresh needs P16, k-chunked Greens
(`feat/bse-wterm-chunking`) or a C3-symmetric drive (σ± at normal incidence on
M ∥ z keeps C3: one third).

### 7.1 Refresh policy {#refresh}

$\delta W$ follows the populations, which follow the pulse envelope and the
carrier relaxation, not the optical period. The production design refreshes
Eq. (6) and the pole model when $\max_k|\delta f|$ has moved by a set
tolerance since the last refresh, and in between applies
$\Delta\Sigma=K\delta\rho+\Delta\Sigma^{\rm dyn}_{\rm last}$ (its coherent part
exact, its population part held). The refresh is a host-side decision at a
chunk boundary, so the jitted step stays one program. At one refresh per fs,
a 30 fs 12×12 drive costs 30 × 82 s + 2400 × 0.31 s ≈ 53 min on 4 A100.

## 8. Validation {#validation}

`tests/test_tdagw_dynsig.py` (synthetic 3×2×1 k, 9 bands = 3 frozen
occupied + 2v2c + 2 frozen empty, two spinor components, a NumPy reference
sharing no code with `tdagw.dynsig`; CPU, 2×2 mesh of host devices):
$\chi_0^{ff}+\chi_0^{fa}+\chi_0^{aa}$ = dense $1.4\times10^{-12}$ (twin without
$\chi_0^{fa}$ 0.47); equilibrium COHSEX $5.9\times10^{-16}$; $\delta\chi_0$ of an
excited state $4.7\times10^{-13}$ (twins: no Pauli blocking $4.8\times10^{-3}$,
no same-side pairs 0.81); total ΔΣ $1.5\times10^{-13}$, dressed
$9.4\times10^{-14}$ (twin, fixed-basis populations, 0.10); literal static limit
$1.4\times10^{-14}$ (twin without coherences 1.75); frozen mode on a pulse
= `isdf_model` exactly; `static_rpa` departure $\propto E_0^{2.2}$. The same
script on 4 A100 (one rank per GPU, the runtime's communicator stack)
reproduces every row to round-off.

`tests/tdagw/_dynsig_516_case.py` (CrI3 516 6×6 COHSEX, 2v2c, DFT $H_0$, 4
A100): the loader's window ψ equal the TD model's exactly and the stored $W_0$
equals its $W$ at q ≠ 0 to $7\times10^{-16}$ (at q = 0 the TD model carries its
own head); §6.1 as stated there; a 6 fs pulse (rk4, 20 as, 1200 ΔΣ calls)
with `screening="frozen"` gives P(t) and ρ(t_f) equal to `isdf_model`'s to the
last bit; the literal Eq. (1) in a rotated instantaneous basis at ρ(t_f) equals
$-K^W\delta\rho$ to $1.1\times10^{-11}$ (twin without coherences 12.2); the
static member as in §4 and §7.

## 9. Refactors for a production member {#refactors}

Each item makes a GW stage callable per refresh from in-memory arrays, with no
file I/O and no re-planning (lines at `feat/td-agw-propagator-2026-10-16`
7402a200b):

1. **Weight-row $\chi_0$.** `gw.w_isdf._chi_layout_operands` (:2144) hands the
   kernel val/cond **masks** and `_gap_edges`/`_laplace_chi_args` (:2164,
   :2225) a step partition; `_get_chi_minimax_kernel_face`/`_fused` (:505,
   :772) contract $G_c\cdot G_v^*$ only. Change: take four weight rows
   $(w^h_L,w^p_L,w^h_U,w^p_U)$ per band (Eq. 3) and accumulate the Pauli
   product of Eq. (4) in the same pass; the step partition is the special case.
2. **Rotated active faces.** `greens_function_kernel._build_G_face` (:131)
   refuses a dense `Gij` on faces. Change: none to the kernel — pass the
   window's rotated faces $\psi_AU$ with diagonal weights (Eq. 2), one
   $n_A^2$ rotation per $k$; add an "active rows" carrier so $G_{\rm eq}$ and
   $\delta G$ share one panel exchange.
3. **Same-side pairs.** `compute_chi0_contour_fractional` (:3349) /
   `compute_chi0_direct_fractional` (:3657) take an `OccupationState`
   (`gw.efermi`:871) built by the Fermi–Dirac solver. Change: accept arbitrary
   real $f_{nk}$ on a band support (the window), with the frequency plan from
   the occupation supports (`metallic-mpa-screening.md` §2.2).
4. **Screening entry point.** `gw.screening.compute_screening_model` (:699)
   needs `config`, `meta`, the WFN loader, the head resolver and a run
   directory; `screen_shared_poles` (:443) writes bank files and
   `construct_shared_poles` (:57) writes the model store. Change: a
   `screening_update(W0_at_supports, dchi_at_supports, plan)` returning the
   pole model as arrays; the stores stay the GW driver's.
5. **Σ τ sweep on a window.** `ppm_pipeline.compute_ppm_sigma_pipeline` (:690)
   takes `wfns`, `config`, `meta`, `head_resolver`, `band_slices`, `wfn`,
   `sym`. Change: an array-level entry (`psi_full` faces, $G$ weights at the Σ
   nodes, the pole arrays, the window projectors), planned once; the
   quadrature session (`fixed_quadrature_session`) is the held plan.
6. **Full-band encode, window decode.** `bse_stack_matvec.make_w_term` (:470)
   decodes onto the encode's band set (the prototype decodes 160×160 to keep
   4×4). Change: separate encode (all bands) and decode (window) faces.
7. **Head of $\delta\chi_0$.** `qsgw_head.build_dft_head_response` (:4175)
   builds the head from the DFT state. Change: the window's
   $q\cdot\delta S\cdot q$ from its dipoles and populations, plus the Drude term
   of carriers ([self-consistency §7](../self_consistency.md#7-metals)).
8. **In-jit refusals.** The prototype poisons ΔΣ with NaN and names the gate on
   the host (`dynsig.check_state`); the propagator should return the gate
   flags with each chunk.
