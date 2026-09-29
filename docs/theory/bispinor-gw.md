# Bispinor (four-current) GW

This page owns the theory of LORRAX's bispinor GW: what a bispinor is here,
the interaction it adds, the $1/c$ counting of every term, which terms each
`bispinor_gw` route keeps, and why `bare_transverse` is the default. The
method is new and unpublished, so the page assumes no outside reference.

Other pages own the rest. The $\mathbf q\to0$ (Γ-cell) head of every channel
and the frequency model each channel carries:
[Four-current heads and frequency](four-current-head-corrections.md). Each
producer, object, shape, sharding and refusal:
[Four-current wiring](../architecture/four_current_wiring.md). The
transverse direct field: [Direct Hartree field](hartree.md). Deck keys:
[input reference](../input_reference.md).

Units on this page are Hartree atomic units, so $c=1/\alpha_{\rm FS}\approx137$.
The code works in Rydberg. The one relativistic constant it uses, the
kinetic-balance factor $\alpha_{\rm FS}/2=1/(2mc)$, is the same number in
both systems (`common.bispinor_init.HALFALPHA`).

## 1. The bispinor {#lift}

A Dirac electron has a four-component wavefunction. Split it into two
two-component halves, the large component $\psi_L$ and the small
component $\psi_S$:

$$
\Psi=\begin{pmatrix}\psi_L\\ \psi_S\end{pmatrix},\qquad
\psi_S=\frac{\boldsymbol\sigma\cdot\mathbf p}{2mc}\,\psi_L+O(c^{-3})
=\frac{\alpha_{\rm FS}}{2}\,\boldsymbol\sigma\cdot\mathbf p\,\psi_L+O(c^{-3}).
$$

The second relation is kinetic balance. It holds for positive-energy
(electron) states. $\boldsymbol\sigma$ are the Pauli matrices and
$\mathbf p=-i\nabla$. "Bispinor" in this code means this pair of
two-spinors.

LORRAX does not solve the Dirac equation. It reads the two-component spinors
of a noncollinear DFT calculation (QE, spin–orbit coupling in the
pseudopotential) as $\psi_L$, and builds $\psi_S$ from them in plane waves:

$$
\psi_{S,n\mathbf k}(\mathbf G)=\frac{\alpha_{\rm FS}}{2}\,
\boldsymbol\sigma\cdot(\mathbf k+\mathbf G)\,\psi_{L,n\mathbf k}(\mathbf G),
$$

with $\mathbf k+\mathbf G$ Cartesian in bohr⁻¹
(`common.bispinor_init.lift_to_4spinor`, called by the `WfnLoader`). This is
the raw lift: $\Psi$ is not renormalized, so
$\langle\Psi|\Psi\rangle=1+(\alpha_{\rm FS}^2/4)\langle p^2\rangle$. Only
positive-energy states exist in the calculation (the no-pair
approximation). The lift is first order in $\mathbf p/c$; higher-order
kinetic balance is not built.

**Vertices.** With $\gamma^\mu$ the Dirac matrices, the code stores
$\tilde\gamma^\mu\equiv\gamma^0\gamma^\mu$: $\tilde\gamma^0=1_4$ and
$\tilde\gamma^i=\alpha^i=\begin{pmatrix}0&\sigma^i\\\sigma^i&0\end{pmatrix}$
(`common.gamma_matrices`). A Lorentz index $I\in\{0,1,2,3\}$ labels the
charge channel C ($I=0$) and the three Cartesian current channels T
($I=1,2,3$). Every density is written $\Psi^\dagger\tilde\gamma^I\Psi$,
never with $\bar\Psi$.

## 2. Sources, propagator and the interaction {#sources}

**Transition sources.** For two states $m,n$ (band and k labels implied):

$$
\rho_{mn}(\mathbf r)=\Psi_m^\dagger\Psi_n
=\psi_{Lm}^\dagger\psi_{Ln}+\psi_{Sm}^\dagger\psi_{Sn},
\qquad
J^i_{mn}(\mathbf r)=\Psi_m^\dagger\alpha^i\Psi_n
=\psi_{Lm}^\dagger\sigma^i\psi_{Sn}+\psi_{Sm}^\dagger\sigma^i\psi_{Ln}.
$$

$\rho$ is the charge density. $\mathbf J$ is the Dirac current divided by
$c$, i.e. the current in the units where it couples to the vector potential
with no further factor. Inserting kinetic balance splits $\mathbf J$
(Gordon decomposition) into an orbital and a spin part:

$$
\mathbf J_{mn}=\frac{\alpha_{\rm FS}}{2}\Big[\psi_{Lm}^\dagger(\mathbf p\,\psi_{Ln})
+(\mathbf p\,\psi_{Lm})^\dagger\psi_{Ln}\Big]
+\frac{\alpha_{\rm FS}}{2}\,\nabla\times\big(\psi_{Lm}^\dagger\boldsymbol\sigma\,\psi_{Ln}\big)
+O(c^{-3}).
$$

The first bracket is the convection (orbital) current. The second is the
spin-magnetization current; in reciprocal space it is
$(i\alpha_{\rm FS}/2)\,\mathbf q\times\mathbf s(\mathbf q)$.

**Coulomb gauge.** Eliminating the photon field in Coulomb gauge
($\nabla\cdot\mathbf A=0$) leaves two instantaneous interactions and no
mixed one. On the plane wave $\mathbf K=\mathbf q+\mathbf G$:

$$
D=\begin{pmatrix}v&0\\0&t\end{pmatrix},\qquad
v(\mathbf K)=\frac{4\pi}{K^2},\qquad
t_{ij}(\mathbf K)=-\,v(\mathbf K)\,P^T_{ij}(\hat{\mathbf K}),\qquad
P^T_{ij}=\delta_{ij}-\hat K_i\hat K_j .
$$

$P^T$ is the Coulomb-gauge transverse projector: it keeps the part of a
vector field perpendicular to $\mathbf K$. The minus sign in $t$ is the
spatial metric; the code applies it once, in the propagator builder
(`vcoul.COULOMB_GAUGE_TT_SIGN`). $D^{0i}=0$ exactly in this gauge. Both $v$
and $t$ are $O(c^0)$. $t$ is the static (zero-frequency) transverse photon,
so its exchange is the Breit interaction; retardation, $t(\omega)$, is not
built. `sys_dim = 2` replaces $v$ by the slab-truncated kernel in both
blocks.

**Interaction between two electrons.** For left and right transition
sources $(\rho_L,\mathbf J_L)$ and $(\rho_R,\mathbf J_R)$ the bare
interaction is

$$
\mathcal V^{\rm bare}_{LR}=\rho_L\,v\,\rho_R+\mathbf J_L\,t\,\mathbf J_R .
$$

The first term is Coulomb. The second is the magnetic interaction of the two
currents (Gaunt plus the gauge term, together the Breit interaction).

## 3. The $1/c$ counting {#counting}

Kinetic balance gives $\psi_S=O(c^{-1})\psi_L$. Therefore, between
positive-energy states,

$$
\rho_{mn}=\underbrace{\psi_{Lm}^\dagger\psi_{Ln}}_{O(1)}
+\underbrace{\psi_{Sm}^\dagger\psi_{Sn}}_{O(c^{-2})},
\qquad
J^T_{mn}=O(c^{-1}).
$$

$J^T=P^T\mathbf J$ is the transverse current. Only $J^T$ couples to $t$,
because $t$ contains $P^T$.

The polarization $\Pi$ (the code's $\chi_0$) is the independent-particle
response of the sources. It has four blocks, named by their two endpoints:
CC ($\Pi_{00}$), CT ($\Pi_{0T}$), TC ($\Pi_{T0}$) and TT ($\Pi_{TT}$). Each
block carries one source at each end, so

$$
\Pi_{00}=O(1),\qquad \Pi_{0T},\ \Pi_{T0}=O(c^{-1}),\qquad \Pi_{TT}=O(c^{-2}).
$$

$\Pi_{00}$ built from four-component $\rho$ already contains the
$O(c^{-2})$ small-component correction. CT and TC are both needed: without
time reversal they are not equal at the same arguments.

**Screened interaction.** $W=D+D\,\Pi\,W$. Define the charge-only screened
interaction

$$
W_C=(v^{-1}-\Pi_{00})^{-1}.
$$

Block inversion, keeping the Coulomb screening to all orders, gives (ordered
operator products, nothing assumed to commute; $t^{-1}$ is taken on the
transverse subspace):

$$
\begin{aligned}
W_{00}&=W_C+W_C\Pi_{0T}\,t\,\Pi_{T0}W_C+O(c^{-4}),\\
W_{0T}&=W_C\Pi_{0T}\,t+O(c^{-3}),\qquad W_{T0}=t\,\Pi_{T0}W_C+O(c^{-3}),\\
W_{TT}&=t+t\big[\Pi_{TT}+\Pi_{T0}W_C\Pi_{0T}\big]t+O(c^{-4}).
\end{aligned}
$$

$D_{0T}=0$, so all of $W_{0T}$ is induced by the medium. The code names
$W$'s blocks by the same sector labels: $W_{00}$ is CC, $W_{0T}$ CT, $W_{T0}$
TC and $W_{TT}$ TT.

**Order after contraction.** The self-energy puts one source at each end of
$W$. That adds the sources' own powers of $1/c$:

| term | name on this page | order |
|---|---|---:|
| $\rho\,W_C\,\rho$, $\rho$ four-component | charge (CC) | $c^0$, with an $O(c^{-2})$ small-component part |
| $J^T\,t\,J^T$ | bare transverse | $c^{-2}$ |
| $\rho\,W_C\Pi_{0T}\,t\,J^T$ and its reverse | mixed | $c^{-2}$ |
| $\rho\,W_C\Pi_{0T}\,t\,\Pi_{T0}W_C\,\rho$ | mixed feedback into $W_{00}$ | $c^{-2}$ |
| $J^T\,t\big[\Pi_{TT}+\Pi_{T0}W_C\Pi_{0T}\big]t\,J^T$ | screened transverse | $c^{-4}$ |

The three $c^{-2}$ interaction terms regroup into one bare transverse
interaction between Coulomb-dressed currents:

$$
\mathcal V_{LR}=\rho_LW_C\rho_R
+\big(J^T_L+\rho_LW_C\Pi_{0T}\big)\,t\,\big(J^T_R+\Pi_{T0}W_C\rho_R\big)
+O(c^{-4}).
$$

$J_{\rm ind}=\Pi_{T0}W_C\rho$ is the current the medium carries in response
to the Coulomb field of the charge source. So at $c^{-2}$ the transverse
propagator is bare. Transverse screening ($\Pi_{TT}$) starts at $c^{-4}$.
What screening does at $c^{-2}$ is dress the current source.

This is the complete set of screened-interaction terms within RPA (GW with
the bubble $\Pi$) at this order. It is not every many-body diagram at this
relativistic order.

## 4. The self-energy {#sigma}

With $G$ the Green's function of the four-component states, the GW
self-energy carries one vertex at each end:

$$
\Sigma_{\alpha\beta}(12)=-\sum_{I,J}\tilde\gamma^I_{\alpha\gamma}\,
G_{\gamma\delta}(12)\,\tilde\gamma^J_{\delta\beta}\,W^{IJ}(12),
$$

with $\alpha\ldots\delta$ the four Dirac components. The blocks give

$$
\Sigma=\Sigma^{CC}+\Sigma^{CT}+\Sigma^{TC}+\Sigma^{TT},\qquad
\Sigma^{CC}=-G\,W_{00},\qquad
\Sigma^{B}\equiv-\sum_{ij}\alpha^iG\alpha^j\,t_{ij}.
$$

$\Sigma^B$ is the bare transverse exchange: the $J^T t J^T$ term of §3 with
occupied states only. It is frequency independent, and a static term's
screened-exchange and exchange parts are equal
($SX(t)=X(t)$, $COH(t-t)=0$), so in every compute mode it is one exchange
contraction.

The direct (Hartree) fields come with the self-energy on every bispinor
deck: $V_H=v\,\rho_{\rm occ}$ and $\mathbf A=t\,\mathbf J_{\rm occ}$, with
band matrix $\langle m|V_H+\boldsymbol\alpha\cdot\mathbf A|n\rangle$
([Direct Hartree field](hartree.md)). $\mathbf A$ vanishes when the ground
state carries no current, for example with time reversal.

$\Sigma^B$ and every current term vanish as $\alpha_{\rm FS}\to0$, since
$\psi_S\to0$. The calculation then returns to two-component GW exactly.
This limit is a gate for any change to the layer.

## 5. What two-component GW misses, and the case for `bare_transverse` {#bare-transverse}

A two-component GW calculation builds $\rho$ from $\psi_L$ alone and uses
$v$ alone. Spin–orbit coupling is present, but only in the one-body
Hamiltonian. At order $c^{-2}$ the electron–electron interaction has two
corrections that it drops:

1. **The small-component charge.** The four-component
   $\rho=\rho^{LL}+\rho^{SS}$ changes every charge-channel term by
   $\rho^{LL}v\,\rho^{SS}+\rho^{SS}v\,\rho^{LL}=O(c^{-2})$: the Hartree
   field, the bare exchange and, through $\Pi_{00}$, the screening. This is
   the longitudinal–longitudinal (CC) channel.
2. **The transverse interaction.** $J^T\,t\,J^T=O(c^{-2})$: the Breit
   exchange (in the Pauli reduction, spin-other-orbit, orbit-orbit and
   spin–spin), and its direct counterpart $\boldsymbol\alpha\cdot\mathbf A$.

The two are the same order. A four-component charge carrier brings in the
first automatically. Keeping it while dropping the second is not consistent
at $c^{-2}$. `bare_transverse` adds the second.

**Why the bare transverse term is the one to add.**

* **It is the largest energy contribution that two-component GW misses**
  (owner, 2026-09-24). It acts as exchange on every occupied state. Its
  vertex grows with the momentum in the state, so deep and semicore states,
  where $\mathbf p$ is large, feel it most.
* **Screening does not reduce it at this order.** The charge exchange is
  screened, $v\to W_C$. The transverse propagator is not: by §3 its
  screening, $\Pi_{TT}$ and $\Pi_{T0}W_C\Pi_{0T}$ inside $W_{TT}$, enters
  $\Sigma$ at $c^{-4}$. The bare $t$ is the correct transverse propagator
  through $c^{-2}$.
* **It needs no response function.** No $\chi_{CT}$, no $\chi_{TT}$, no
  current Dyson solve and no frequency model. It costs the bare TT tiles
  (six unique, plus CC; `gw.v_q_bispinor`) and one exchange contraction.

**What `bare_transverse` is, exactly.** It sets $\Pi_{0T}=\Pi_{T0}=\Pi_{TT}=0$
by declaration. The Dyson equation is then block diagonal:

$$
W=\begin{pmatrix}W_C&0\\0&t\end{pmatrix},\qquad
\Sigma=-G\,W_C\;+\;\Sigma^B\;(+\ \text{direct fields}).
$$

$W_C$ uses the four-component $\rho$ and the run's frequency model
(static, GN/HL plasmon pole, or full-frequency shared pole). It never builds
or reads $\chi_{CT}$ or $\chi_{TT}$. Adding $\chi_{TT}$ alone would keep a
$c^{-4}$ term while the $c^{-2}$ mixed terms are missing. $\chi_{CT}$ is the
mixed channel itself, which belongs to the full routes (§6). The packed form refuses a nonzero
Hall artifact for the same reason: the Hall term is a CT response
(`GATE packed_bare_transverse_hall_unavailable`).

**What it omits, and at what order.**

| omitted term | order in $\Sigma$ |
|---|---:|
| mixed: $\rho\,W_C\Pi_{0T}\,t\,J^T$ and $J^T\,t\,\Pi_{T0}W_C\,\rho$ | $c^{-2}$ |
| mixed feedback: $\rho\,W_C\Pi_{0T}\,t\,\Pi_{T0}W_C\,\rho$ | $c^{-2}$ |
| screened transverse: $J^T\,t\big[\Pi_{TT}+\Pi_{T0}W_C\Pi_{0T}\big]t\,J^T$ | $c^{-4}$ |
| retardation $t(\omega)$, vertex corrections, negative-energy states | not in any route |

The mixed terms are formally the same order as $\Sigma^B$. They are what
separates `bare_transverse` from a complete $c^{-2}$ RPA interaction. Their
size depends on the material:

* **With time reversal** the static mixed block vanishes: a static scalar
  potential cannot induce a current in a time-reversal-invariant state, so
  $\Pi_{T0}(\mathbf q,\omega=0)=0$. The mixed terms are then purely
  dynamical. On MoS₂ 3×3 the whole static difference between screened and
  bare current blocks is $1.2\times10^{-8}$ eV over 270 states
  ([heads §1.1](four-current-head-corrections.md#frozen-current-blocks)).
* **In a strongly spin-polarized metal or half-metal** a scalar potential
  induces spin density whenever the two spin channels respond differently,
  and that spin density carries a magnetization current. In a homogeneous
  static model with density-response polarization
  $p_\Pi=(\Pi_\uparrow-\Pi_\downarrow)/(\Pi_\uparrow+\Pi_\downarrow)$,
  strong screening gives $W_C\Pi_{s^zn}\to-p_\Pi$, with no extra small
  factor. The induced magnetization current per charge source is then
  $p_\Pi$ times an electron's own spin-magnetization current. The mixed
  terms can be comparable to the spin-magnetic part of $\Sigma^B$ there.
* **In orbital Hall or Chern states** an electrostatic potential drives a
  transverse Hall current, $\delta\mathbf j=-i\sigma_H(\hat{\mathbf z}\times\mathbf q)\phi$.
  That is a CT response at $c^{-1}$ in the source.

For these magnetic and Hall cases use `full_shared_pole` (§6).

**Where the counting fails.** The expansion holds at fixed
$(\mathbf q,z)$ when
$\big\|t\,[\Pi_{TT}+\Pi_{T0}W_C\Pi_{0T}]\big\|\ll1$. Near a collective pole,
or at long wavelength where the propagator can compensate the small
vertices, the full transverse denominator
$t^{-1}-\Pi_{TT}-\Pi_{T0}W_C\Pi_{0T}$ may be needed. The counting also says
nothing about the transverse electromagnetic response itself, which needs
$\Pi_{TT}$.

## 6. Routes: which terms each keeps {#routes}

`bispinor_gw ∈ {bare_transverse, full_static_cohsex, full_shared_pole}`
(`gw.gw_config.BispinorGWMode`), orthogonal to `compute_mode`. All routes use
the same raw kinetic-balance carrier. Admission rules and route predicates
are in [Four-current wiring](../architecture/four_current_wiring.md#routes-and-predicates).

| term (§3) | order | `bare_transverse` | `full_static_cohsex` | `full_shared_pole` |
|---|---:|---|---|---|
| $\rho W_C\rho$, four-component $\rho$ | $c^0$, $c^{-2}$ | yes, in the run's frequency model ($W_C=v$ under `x_only`) | `cohsex`: packed $W_{00}$ at $\omega=0$; `gn_ppm`/`hl_ppm`: scalar $W_C(\omega)$ | yes, full frequency |
| $\Sigma^B=J^T t J^T$ | $c^{-2}$ | yes | yes | yes |
| direct fields $V_H$, $\boldsymbol\alpha\cdot\mathbf A$ | $c^0$, $c^{-2}$ | yes | yes | yes |
| mixed $\rho W_C\Pi_{0T}tJ^T$ | $c^{-2}$ | no | at $\omega=0$ | yes |
| mixed feedback in $W_{00}$ | $c^{-2}$ | no | `cohsex` only, at $\omega=0$ | yes |
| screened transverse | $c^{-4}$ | no | at $\omega=0$ | yes |

`full_static_cohsex` builds all sixteen $\chi_0^{IJ}$ blocks at $\omega=0$
(no-pair, TT Ward-subtracted as $\Pi(q)-\Pi(0)$) and solves one packed
Dyson equation. Under GN/HL it keeps the scalar dynamic $W_C(\omega)$ for
CC and takes the fifteen current blocks from the static solve, so the mixed
feedback into $W_{00}$ is not in that arm. `full_shared_pole` builds the
ordered CC/CT/TC/TT sector bank, solves the four-current Dyson equation at
each bank frequency and fits each sector's poles
([shared-pole model](../architecture/shared_pole_model.md)). It keeps
$\Pi_{TT}$ too, which is more than $c^{-2}$ needs.

**`bare_transverse` has four implementations with the same terms.** They
differ in where the sum is contracted and in the Γ-cell head:

| implementation | taken when | charge $W_C$ | $\Sigma^B$ |
|---|---|---|---|
| exchange only | `compute_mode = x_only` | $v$ (no screening) | `sigma_x_bispinor` |
| incumbent | outside the packed envelope and not shared pole | scalar owner, any compute mode | `sigma_x_bispinor` |
| packed bare | `sys_dim = 2`, one-shot, `cohsex`/`gn_ppm`/`hl_ppm` inside the envelope | scalar owner, packed as $\mathrm{diag}(W_C,t)$ | `photon_sigma` |
| shared-pole hybrid | `compute_mode = mpa`, `sigma_w_model = shared_pole` | full-frequency shared-pole bank on the four-component charge | `sigma_x_bispinor` |

The packed and incumbent contractions are two orders of one sum; with the
head off they agree byte for byte (CLAIMS 581).

### Equation to code

| equation | implemented by |
|---|---|
| kinetic-balance lift (§1) | `common.bispinor_init.lift_to_4spinor` |
| vertices $\tilde\gamma^I$ | `common.gamma_matrices` |
| ISDF of $\rho$ and $J^i$ (§7) | `gw.isdf_fitting.fit_zeta_to_h5`, one fit per channel |
| $v$, $t$ tiles; the TT sign | `gw.v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5`; `vcoul.COULOMB_GAUGE_TT_SIGN` |
| $\Pi_{00}$, $W_C$ (bare_transverse) | `gw.screening.compute_screening_model` → `gw.w_isdf.compute_chi0`, `solve_w` |
| $\Pi_{IJ}$ at $\omega=0$, packed $W$ | `gw.w_isdf.compute_static_photon_response` (`compute_no_pair_dirac_current_block`) |
| full-frequency CC/CT/TC/TT bank | `gw.response_bank.compute_photon_bank`; poles `gw.shared_pole_sectors.construct_sector_poles` |
| $\Sigma^B$ | `gw.sigma_x_bispinor.compute_sigma_x_bispinor` |
| sixteen-block static $\Sigma$ | `gw.photon_sigma.compute_static_photon_sigma` (`contract_lorentz_blocks`) |
| dynamic sector $\Sigma$ | `gw.mpa.sector_sigma.compute_sector_sigma` |
| $V_H$, $\boldsymbol\alpha\cdot\mathbf A$ | `gw.hartree.direct_field_matrices` |
| route selection | `gw.gw_config`: `packed_bare_transverse_route`, `uses_bare_transverse_shared_pole`, `uses_full_bispinor_shared_pole`, `packed_photon_screens_current` |

The run record names the route on its `Photon route` line
([reading `gwjax.out`](../architecture/four_current_wiring.md#stage-5-outputs)).
`sigma_diag.dat` splits $\Sigma_{xc}$ into `sigCC`, `sigCT` (CT+TC) and
`sigTT`; on `bare_transverse`, `sigCT` is zero and `sigTT` is $\Sigma^B$.

## 7. ISDF on two centroid sets {#isdf}

Each channel density is fitted separately:

$$
\rho^{I}_{mn}(\mathbf r)\approx\sum_\lambda\zeta^{I}_{q,\lambda}(\mathbf r)\,
\rho^{I}_{mn}(\mathbf r_\lambda),\qquad I\in\{0,1,2,3\}.
$$

Charge and current use different centroid sets, because they weight space
differently. A bispinor deck names two files:

| file | channels | built by |
|---|---|---|
| `centroids_file` (`centroids_frac_<N>.txt`) | $I=0$ | `kmeans_cli` (default mode) |
| `centroids_file_current` (`centroids_frac_<M>_current.txt`) | $I=1,2,3$ | `kmeans_cli --density-mode current` |

A run without the current file refuses. For a band window $B$, with
$D_{B,k}(\mathbf r)=\sum_{n\in B}|\Psi_{nk}(\mathbf r)\rangle\langle\Psi_{nk}(\mathbf r)|$,
the current k-means weight is

$$
m_J(\mathbf r)=\sqrt{\sum_{k,i}w_k\,\mathrm{Tr}\big[D_{L,k}(\mathbf r)\,\alpha^i\,D_{R,k}(\mathbf r)\,\alpha^i\big]/\alpha_{\rm FS}^2},
$$

and the charge weight replaces $\alpha^i$ by the identity
(`centroid.sampling_metric`). The charge fit matrix is positive
semidefinite; each current one is Hermitian but indefinite and uses a ridged
pivoted LU. The three current channels share one fit loop and keep separate
matrices, factors and files
([ζ fit by μ-batches](../architecture/zeta_fit_mubatch.md)).

## 8. Which route to use {#which-route}

| situation | route | why |
|---|---|---|
| nonmagnetic material (time reversal), relativistic QP corrections | `bare_transverse` | complete for the static $c^{-2}$ interaction; mixed terms are dynamical only (§5) |
| strongly spin-polarized metal, half-metal, Hall or Chern state | `full_shared_pole` (`compute_mode = mpa`, `sigma_w_model = shared_pole`, `head_correction = no_local_fields`) | keeps the $c^{-2}$ mixed terms at full frequency |
| static cross-check of the current screening on a slab | `full_static_cohsex` (`sys_dim = 2` under `head_correction = full`, `linalg = distributed`) | all sixteen blocks at $\omega=0$ |
| exchange-only (Dirac–Hartree–Fock plus Breit-like) check | `bare_transverse` with `compute_mode = x_only` | $v$ and $t$ bare |
| two-component reference | `bispinor = false` | the $\alpha_{\rm FS}\to0$ limit |

The production calculation and its keys are
[Production QSGW](../how-to/production-qsgw.md).

## 9. Open questions

1. No external four-component (Dirac–Coulomb–Breit) reference certifies the
   absolute $\Sigma^B$ yet; internal carrier and mesh parity do not.
2. The size of the dynamical mixed terms on a time-reversal-invariant
   material has not been measured against `full_shared_pole`.
3. A positive-semidefinite band-pair Gram for the current ζ fit would cost
   more than the Schur CCT used today; no accuracy study justifies it yet.

## 10. q = Γ measurements on the bi4 deck (2026-08-01) {#q-gamma-measurements}

The argument for how each $(I,J)$ tile behaves at $\mathbf q\to0$ and the
correction the code applies are
[Four-current heads and frequency](four-current-head-corrections.md), §2.
This section keeps only the measurements that page and `vcoul.minibz` cite.
Deck: MoS₂ 4×4, 402 charge and 143 current centroids, P = 4, `sys_dim = 2`,
job 7885325 (Frontera; artifacts were machine-local and are not shipped).
They measure the former TT-slot overlay, not a current route; the packed
slab routes obtain $\langle D_{TT}\rangle$ from the coupled Γ completion.

| quantity | value |
|---|---|
| `vc0 = ⟨v⟩_mBZ` (bare, 4×4) | 2443.3 a.u. |
| `⟨v t^{11}⟩ / vc0`, `⟨v t^{22}⟩ / vc0`, `⟨v t^{33}⟩ / vc0` | 0.4993, 0.5007, 1.0000 |
| `⟨v t^{12}⟩ / vc0`; `t^{i3}` | 4e-4; exactly 0 (in-plane cell) |
| `‖ζ_T^i(Γ, μ, G=0)‖` | 1.17–1.21e3 |
| Frobenius ratio, missing rank-1 head / stored q=Γ TT slab (11/22/33) | 0.97 / 1.04 / 6.0 |
| whole q=Γ TT term (Γ-zeroed leg): Σ_X diag max / mean; eqp max | 0.347 / 0.059 meV; 0.347 meV |
| whole q=Γ TT term: share of the −1.553 eV Σ^B trace | −0.122 eV (7.8 %) |
| missing G=0 head (injected leg): Σ_X diag max / mean; eqp max | 0.209 / 0.037 meV; 0.209 meV |
| missing G=0 head: change of the Σ^B trace | −0.076 eV (4.9 %) |

The tabulated `⟨v t^{ij}⟩` are positive moments of the geometric projector;
the stored Coulomb-gauge slot is their negative. The restart legs shared
every bit except the TT q=Γ slabs (restart reproducibility exact 0).
