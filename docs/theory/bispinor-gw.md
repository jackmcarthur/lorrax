# Bispinor (four-current) GW

This page owns the theory of bispinor GW: the carrier, the interaction, the
$1/c$ order of every term, and which terms each `bispinor_gw` route keeps.
The Γ-cell heads and the frequency model per channel:
[Four-current heads and frequency](four-current-head-corrections.md).
Producers, shapes, sharding, refusals:
[Four-current wiring](../architecture/four_current_wiring.md). The direct
fields: [Direct Hartree field](hartree.md).

Hartree atomic units, $c=1/\alpha_{\rm FS}$. The code runs in Rydberg; its
one relativistic constant, $\alpha_{\rm FS}/2=1/(2mc)$, is the same number in
both (`common.bispinor_init.HALFALPHA`).

## 1. Carrier {#lift}

A positive-energy Dirac state splits into a large and a small two-spinor
related by kinetic balance:

$$
\Psi=\begin{pmatrix}\psi_L\\\psi_S\end{pmatrix},\qquad
\psi_S=\frac{\alpha_{\rm FS}}{2}\,\boldsymbol\sigma\cdot\mathbf p\,\psi_L+O(c^{-3}).
$$

$\psi_L$ is the noncollinear DFT two-spinor (spin–orbit in the
pseudopotential). $\psi_S$ is built in plane waves,
$\psi_S(\mathbf G)=(\alpha_{\rm FS}/2)\,\boldsymbol\sigma\cdot(\mathbf k+\mathbf G)\,\psi_L(\mathbf G)$,
$\mathbf k+\mathbf G$ Cartesian in bohr⁻¹ (`common.bispinor_init.lift_to_4spinor`).
The lift is raw: $\langle\Psi|\Psi\rangle=1+(\alpha_{\rm FS}^2/4)\langle p^2\rangle$.
Negative-energy states are absent (no-pair).

The stored vertices are $\tilde\gamma^I=\gamma^0\gamma^I$:
$\tilde\gamma^0=1_4$, $\tilde\gamma^i=\alpha^i=\bigl(\begin{smallmatrix}0&\sigma^i\\\sigma^i&0\end{smallmatrix}\bigr)$
(`common.gamma_matrices`). $I=0$ is the charge channel C; $I=1,2,3$ are the
current channels T. Densities are $\Psi^\dagger\tilde\gamma^I\Psi$.

## 2. Sources and propagator {#sources}

For states $m,n$:

$$
\rho_{mn}=\psi_{Lm}^\dagger\psi_{Ln}+\psi_{Sm}^\dagger\psi_{Sn},\qquad
J^i_{mn}=\Psi_m^\dagger\alpha^i\Psi_n
=\tfrac{\alpha_{\rm FS}}{2}\bigl[\psi_{Lm}^\dagger p^i\psi_{Ln}+(p^i\psi_{Lm})^\dagger\psi_{Ln}\bigr]
+\tfrac{\alpha_{\rm FS}}{2}\bigl[\nabla\times(\psi_{Lm}^\dagger\boldsymbol\sigma\psi_{Ln})\bigr]^i+O(c^{-3}).
$$

$\mathbf J$ is the Dirac current over $c$; its two parts are the orbital
current and the spin-magnetization current. In Coulomb gauge, on
$\mathbf K=\mathbf q+\mathbf G$,

$$
D=\begin{pmatrix}v&0\\0&t\end{pmatrix},\qquad
v=\frac{4\pi}{K^2},\qquad
t_{ij}=-v\,P^T_{ij},\qquad
P^T_{ij}=\delta_{ij}-\hat K_i\hat K_j .
$$

$P^T$ is the transverse projector. The minus sign is the spatial metric,
applied once (`vcoul.COULOMB_GAUGE_TT_SIGN`). $D^{0i}=0$ exactly. $t$ is the
static transverse photon, so its exchange is the Breit interaction;
$t(\omega)$ is not built. `sys_dim = 2` truncates $v$ in both blocks. The
bare interaction of two transition sources is

$$
\mathcal V^{\rm bare}_{LR}=\rho_L\,v\,\rho_R+\mathbf J_L\,t\,\mathbf J_R .
$$

## 3. Order in $1/c$ {#counting}

Kinetic balance gives $\psi_S=O(c^{-1})\psi_L$, hence

$$
\rho=\underbrace{\rho^{LL}}_{O(1)}+\underbrace{\rho^{SS}}_{O(c^{-2})},\qquad
J^T\equiv P^T\mathbf J=O(c^{-1}).
$$

The polarization $\Pi$ (the code's $\chi_0$) has sectors CC, CT, TC, TT
($\Pi_{00},\Pi_{0T},\Pi_{T0},\Pi_{TT}$), one source at each end:

$$
\Pi_{00}=O(1),\qquad\Pi_{0T},\Pi_{T0}=O(c^{-1}),\qquad\Pi_{TT}=O(c^{-2}).
$$

$\Pi_{00}$ from the four-component $\rho$ contains its $O(c^{-2})$ part.
Without time reversal the two orientations are not related at the same
arguments, so both are needed. With $W_C=(v^{-1}-\Pi_{00})^{-1}$, block inversion of
$W=D+D\Pi W$ gives (ordered products)

$$
\begin{aligned}
W_{00}&=W_C+W_C\Pi_{0T}\,t\,\Pi_{T0}W_C+O(c^{-4}),\\
W_{0T}&=W_C\Pi_{0T}\,t+O(c^{-3}),\qquad W_{T0}=t\,\Pi_{T0}W_C+O(c^{-3}),\\
W_{TT}&=t+t\bigl[\Pi_{TT}+\Pi_{T0}W_C\Pi_{0T}\bigr]t+O(c^{-4}).
\end{aligned}
$$

$W_{0T}$ is entirely medium-induced ($D_{0T}=0$). Contracting with the
sources adds their orders:

| term | label | order in $\Sigma$ |
|---|---|---:|
| $\rho W_C\rho$ | CC | $c^0$; $\rho^{SS}$ part $c^{-2}$ |
| $J^TtJ^T$ | bare transverse | $c^{-2}$ |
| $\rho W_C\Pi_{0T}tJ^T$, reverse | mixed | $c^{-2}$ |
| $\rho W_C\Pi_{0T}t\Pi_{T0}W_C\rho$ | mixed feedback | $c^{-2}$ |
| $J^Tt[\Pi_{TT}+\Pi_{T0}W_C\Pi_{0T}]tJ^T$ | screened transverse | $c^{-4}$ |

The $c^{-2}$ terms regroup as a bare transverse interaction between
Coulomb-dressed currents,

$$
\mathcal V_{LR}=\rho_LW_C\rho_R+\bigl(J^T_L+\rho_LW_C\Pi_{0T}\bigr)\,t\,\bigl(J^T_R+\Pi_{T0}W_C\rho_R\bigr)+O(c^{-4}),
$$

where $\Pi_{T0}W_C\rho$ is the current the medium carries in the Coulomb
field of the source. Transverse screening starts at $c^{-4}$. This is
complete within RPA at $c^{-2}$, not complete in diagrams.

The expansion needs $\|t[\Pi_{TT}+\Pi_{T0}W_C\Pi_{0T}]\|\ll1$ at fixed
$(\mathbf q,z)$. Near a collective pole or where $t$ compensates the small
vertices at long wavelength, the full denominator
$t^{-1}-\Pi_{TT}-\Pi_{T0}W_C\Pi_{0T}$ is needed.

## 4. Self-energy {#sigma}

$$
\Sigma_{\alpha\beta}(12)=-\sum_{IJ}\tilde\gamma^I_{\alpha\gamma}G_{\gamma\delta}(12)\tilde\gamma^J_{\delta\beta}W^{IJ}(12),
\qquad
\Sigma^B\equiv-\sum_{ij}\alpha^iG\alpha^jt_{ij}.
$$

$\alpha\ldots\delta$ are Dirac components. $\Sigma^B$ is the bare transverse
exchange; it is static and $SX(t)=X(t)$, so it is one exchange contraction in
every compute mode. Every bispinor deck adds the direct fields
$\langle m|v\rho_{\rm occ}+\boldsymbol\alpha\cdot t\mathbf J_{\rm occ}|n\rangle$
([Direct Hartree field](hartree.md)); the second vanishes with time
reversal. As $\alpha_{\rm FS}\to0$ every current term vanishes and
two-component GW is recovered exactly; this is a gate for any change here.

## 5. `bare_transverse` {#bare-transverse}

Two-component GW builds $\rho$ from $\psi_L$ and uses $v$ alone. At $c^{-2}$
it drops two interaction terms of the same order:

1. the small-component charge, $\rho^{LL}v\rho^{SS}+\rho^{SS}v\rho^{LL}$, in
   the Hartree field, the exchange and $\Pi_{00}$ (the CC channel);
2. the transverse exchange $J^TtJ^T$ (Breit: spin-other-orbit, orbit-orbit,
   spin–spin) and its direct part $\boldsymbol\alpha\cdot\mathbf A$.

The four-component carrier brings in (1). Keeping (1) without (2) is
inconsistent at $c^{-2}$. `bare_transverse` adds (2). It is the largest
energy term that two-component GW misses (owner, 2026-09-24): it acts on
every occupied state, its vertex grows with $\mathbf p$ (semicore states feel
it most), and by §3 it is unscreened through $c^{-2}$. It needs no response,
no current Dyson solve and no frequency model.

The route sets $\Pi_{0T}=\Pi_{T0}=\Pi_{TT}=0$ by declaration and never builds
or reads $\chi_{CT}$ or $\chi_{TT}$ (owner rule). Then

$$
W=\begin{pmatrix}W_C&0\\0&t\end{pmatrix},\qquad
\Sigma=-G\,W_C+\Sigma^B+\text{direct fields},
$$

with $W_C$ on the four-component $\rho$ in the run's frequency model.
Adding $\chi_{TT}$ alone would keep a $c^{-4}$ term while $c^{-2}$ terms are
missing. The packed form refuses a nonzero Hall artifact, a CT term
(`GATE packed_bare_transverse_hall_unavailable`).

Omitted terms:

| term | order |
|---|---:|
| $W_{0T}=W_C\Pi_{0T}t$, $W_{T0}$: $\rho W_C\Pi_{0T}tJ^T$ and reverse | $c^{-2}$ |
| $W_C\Pi_{0T}t\Pi_{T0}W_C$ in $W_{00}$ | $c^{-2}$ |
| $\Pi_{TT}$ and $\Pi_{T0}W_C\Pi_{0T}$ in $W_{TT}$ | $c^{-4}$ |
| $t(\omega)$, vertex corrections, negative-energy states | no route |

The two mixed terms are formally the order of $\Sigma^B$. Their size depends
on time reversal:

* **Time-reversal invariant.** A static scalar potential induces no current,
  so $\Pi_{T0}(\mathbf q,0)=0$ and the omitted terms are purely dynamical.
  The whole static screened-minus-bare current difference on MoS₂ 3×3 is
  $1.2\times10^{-8}$ eV (CLAIMS 581).
* **Time-reversal broken (magnets).** $\Pi_{T0}(\mathbf q,0)\neq0$. A scalar
  potential induces spin density whenever the spin channels respond
  differently, and that density carries a magnetization current: with
  $p_\Pi=(\Pi_\uparrow-\Pi_\downarrow)/(\Pi_\uparrow+\Pi_\downarrow)$, strong
  screening gives $W_C\Pi_{s^zn}\to-p_\Pi$, so
  $J_{\rm ind}\sim-p_\Pi\,(i\alpha_{\rm FS}/2)(\mathbf q\times\hat{\mathbf m})\rho$,
  comparable to the spin-magnetic part of $\Sigma^B$ in a half-metal. A Hall
  medium adds $\delta\mathbf j=-i\sigma_H(\hat{\mathbf z}\times\mathbf q)\phi$,
  also CT. These cases need `full_shared_pole`.

## 6. Routes {#routes}

`bispinor_gw` (`gw.gw_config.BispinorGWMode`) is orthogonal to
`compute_mode`; all routes use the raw lift. Admission:
[wiring](../architecture/four_current_wiring.md#routes-and-predicates).

| term | order | `bare_transverse` | `full_static_cohsex` | `full_shared_pole` |
|---|---:|---|---|---|
| $\rho W_C\rho$ | $c^0$, $c^{-2}$ | run's frequency model ($v$ under `x_only`) | `cohsex`: packed $W_{00}(0)$; GN/HL: scalar $W_C(\omega)$ | full frequency |
| $\Sigma^B$, direct fields | $c^{-2}$ | yes | yes | yes |
| mixed | $c^{-2}$ | no | $\omega=0$ | yes |
| mixed feedback in $W_{00}$ | $c^{-2}$ | no | `cohsex` only, $\omega=0$ | yes |
| screened transverse | $c^{-4}$ | no | $\omega=0$ | yes |

`full_static_cohsex` builds sixteen no-pair $\chi_0^{IJ}(\omega=0)$ (TT
Ward-subtracted, $\Pi(q)-\Pi(0)$) and one packed Dyson solve.
`full_shared_pole` solves the four-current Dyson equation at each bank
frequency and fits CC, CT/TC and TT poles separately
([shared-pole model](../architecture/shared_pole_model.md)).
`bare_transverse` runs as exchange only (`x_only`), incumbent (any compute
mode), packed bare (`sys_dim = 2` one-shot COHSEX/GN/HL, contracted as
$\mathrm{diag}(W_C,t)$) or shared-pole hybrid (`mpa`, `sigma_w_model =
shared_pole`); the terms are the same, and packed and incumbent agree byte
for byte with the head off (CLAIMS 581).

| equation | code |
|---|---|
| lift, vertices | `common.bispinor_init.lift_to_4spinor`, `common.gamma_matrices` |
| ζ per channel | `gw.isdf_fitting.fit_zeta_to_h5` |
| $v$, $t$ tiles | `gw.v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5` |
| $\Pi_{00}$, $W_C$ | `gw.screening.compute_screening_model` → `gw.w_isdf.compute_chi0`, `solve_w` |
| $\Pi_{IJ}(0)$, packed $W$ | `gw.w_isdf.compute_static_photon_response` |
| CC/CT/TC/TT bank | `gw.response_bank.compute_photon_bank`, `gw.shared_pole_sectors.construct_sector_poles` |
| $\Sigma^B$ | `gw.sigma_x_bispinor.compute_sigma_x_bispinor` |
| packed $\Sigma$ | `gw.photon_sigma.compute_static_photon_sigma` |
| sector $\Sigma$ | `gw.mpa.sector_sigma.compute_sector_sigma` |
| direct fields | `gw.hartree.direct_field_matrices` |

`sigma_diag.dat` splits $\Sigma_{xc}$ into `sigCC`, `sigCT` and `sigTT`; on
`bare_transverse`, `sigCT` is zero and `sigTT` is $\Sigma^B$.

## 7. ISDF {#isdf}

Each channel is fitted separately,
$\rho^I_{mn}(\mathbf r)\approx\sum_\lambda\zeta^I_\lambda(\mathbf r)\rho^I_{mn}(\mathbf r_\lambda)$,
on two centroid sets: `centroids_file` for $I=0$ and
`centroids_file_current` for $I=1,2,3$
(`kmeans_cli --density-mode current`). The current k-means weight is

$$
m_J(\mathbf r)=\Bigl[\sum_{k,i}w_k\,\mathrm{Tr}\bigl(D_{L,k}\alpha^iD_{R,k}\alpha^i\bigr)/\alpha_{\rm FS}^2\Bigr]^{1/2},\qquad
D_{B,k}=\sum_{n\in B}|\Psi_{nk}\rangle\langle\Psi_{nk}|,
$$

and the charge weight has $1$ for $\alpha^i$. Current fit matrices are
Hermitian indefinite (ridged pivoted LU); the charge one is semidefinite
([ζ fit by μ-batches](../architecture/zeta_fit_mubatch.md)).

## 8. Which route

| material | route |
|---|---|
| time-reversal invariant | `bare_transverse` |
| magnet, half-metal, Hall or Chern state | `full_shared_pole` (`mpa`, `shared_pole`, `head_correction = no_local_fields`) |
| static slab cross-check of current screening | `full_static_cohsex` |
| exchange-only check | `bare_transverse`, `x_only` |
| two-component reference | `bispinor = false` |

Production keys: [Production QSGW](../how-to/production-qsgw.md).

## 9. Open

1. No Dirac–Coulomb–Breit reference certifies the absolute $\Sigma^B$.
2. The dynamical mixed terms on a time-reversal-invariant material are unmeasured.
