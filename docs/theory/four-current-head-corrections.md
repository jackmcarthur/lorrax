# Four-current GW: q→0 heads and the frequency treatment

This page owns the long-wavelength (Γ-cell) treatment of every Lorentz
channel of the four-current (bispinor) GW self-energy, and the frequency
dependence each channel carries. How the layer is wired into the code (each
producer, shape, sharding and refusal) is
[Four-current wiring](../architecture/four_current_wiring.md). Deck keys are
in the [input reference](../input_reference.md). The $S$ convention is
[its own page](s-tensor-convention.md).

Rydberg units throughout. The Lorentz index $I,J\in\{0,1,2,3\}$ labels the
charge channel C ($0$) and the Cartesian current channels T ($1..3$). The
stored vertices are $\tilde\gamma^0=1$ and $\tilde\gamma^i=\alpha^i$
(`common.gamma_matrices`), so every channel density is
$\rho^I=\psi^\dagger\tilde\gamma^I\psi$. The carrier is the raw
kinetic-balance lift $\psi=(\psi_L,\ (\alpha_{FS}/2)\,\boldsymbol\sigma\cdot\mathbf p\,\psi_L)$.
Because $\alpha^i$ couples the large and small components, every current
vertex carries one factor of $\alpha_{FS}/2$.

## 1. Routes, and what each channel carries {#four-current-phase-status}

`bispinor_gw` selects which Lorentz blocks are screened and which Σ owner
contracts them. All three values ride the same four-spinor carrier.

| route | selected by | screened blocks and frequency | Γ-cell head | Σ owner |
|---|---|---|---|---|
| packed, screened, static | `full_static_cohsex`, `compute_mode = cohsex` | all sixteen $\chi^{IJ}_0$ at $\omega=0$, one packed Dyson solve | coupled 4×4 completion (§4) | sixteen-block X/SX/COH |
| packed, screened, dynamic | `full_static_cohsex`, `compute_mode ∈ {gn_ppm, hl_ppm}` | CC: scalar $W_{00}(\omega)$ with the plasmon-pole model; the fifteen current-index blocks of the packed $\omega=0$ solve | CC: scalar dynamic head (§3.3) and $\langle v\rangle$ for $\Sigma_X$; current blocks: §4 completion | scalar $\Sigma_x+\Sigma_c(\omega)$ plus current blocks at $\omega=0$ |
| packed, bare | `bare_transverse` inside the packed envelope | $\chi_{TT}=\chi_{CT}=0$, so $W=\mathrm{diag}(W_{00},D_{TT})$; CC dynamic under GN/HL as above | §4 completion with a charge-only response: $\mathrm{diag}(W^{00}_h,\langle D_{TT}\rangle)$ | as the two rows above |
| incumbent, bare | `bare_transverse` outside the envelope | CC: scalar $W$ in any compute mode; TT: bare | charge: §3; TT: the bare overlay (§2.1) for GN/HL and `x_only` under `head_correction = full`, none otherwise | scalar Σ plus $\Sigma^B=X(D_{TT})$ |
| shared-pole hybrid | `bare_transverse`, `compute_mode = mpa`, `sigma_w_model = shared_pole` | CC: full-frequency shared-pole $W$ on the four-spinor charge; TT: bare | charge: `full`, or `no_local_fields` (direct $S(\omega)$, required on an ordered store); TT: bare overlay unless `off` | shared-pole $\Sigma_c$ plus $\Sigma^B$ |
| full shared-pole | `full_shared_pole`, `compute_mode = mpa`, `sigma_w_model = shared_pole` | ordered CC/CT/TC/TT sectors, each with its own poles ([shared-pole model](../architecture/shared_pole_model.md)) | first-order direct bulk head under `no_local_fields` (§5); `full` refuses | [sector Σ consumer](../dev/sector_sigma_consumer.md) |

**The packed envelope** is `compute_mode ∈ {cohsex, gn_ppm, hl_ppm}`,
`qp_solver = one_shot_dft`, `screening_diagrams = w_rpa`,
`head_correction ∈ {full, off}` and `linalg = distributed`. The bare route
also needs `sys_dim = 2`. The screened mode additionally refuses a named
scalar-head override, and it refuses `sys_dim ≠ 2` under
`head_correction = full`. Outside the envelope, `full_static_cohsex`
refuses; `bare_transverse` takes the incumbent route, and the run record's
`Photon route` line names the first unmet condition. The default
`linalg = local` keeps `bare_transverse` incumbent. On CUDA the distributed
Dyson plan needs a true 2-D mesh ($p_x,p_y\ge2$).

**Heads are always on** ([decisions, 2026-09-01](../architecture/decisions.md)).
`head_correction = off` is a DEBUG skip with a loud banner on every route.
`no_local_fields` refuses on every bispinor route except the two shared-pole
ones, where it selects the direct head. No other key moves the head or the
route.

Every bispinor deck also carries the transverse direct field
$\langle m|\alpha\cdot A|n\rangle$ ([Direct Hartree field](hartree.md)).

### 1.1 Why the current blocks may be frozen at ω = 0 {#frozen-current-blocks}

$\chi_{TT}$ enters $W$ at $\alpha_{FS}^2$. $\chi_{CT}$ enters at
$\alpha_{FS}$ times a transverse–charge overlap that vanishes in Coulomb
gauge as $q\to0$. The neglected $W_{AB}(\omega)-W_{AB}(0)$, $AB\neq CC$, is
bounded by the static current screening $W_{AB}(0)-D_{AB}$. The
Ward-subtracted no-pair current response is a positive-weight spectral
integral on the imaginary axis, so $|\chi_{TT}(i\omega)|\le|\chi_{TT}(0)|$.
The static term is exactly the difference between the packed bare and
packed screened modes on one deck. On MoS₂ 3×3 it is $1.2\times10^{-8}$ eV
over 270 quasiparticle states (CLAIMS 581), far below the sub-meV
transverse Γ-cell head (§2.1). Outside `full_shared_pole`, no current block
depends on frequency.

## 2. The bare propagator and its Γ-cell average

Eliminating the photon field in Coulomb gauge leaves an instantaneous,
block-diagonal interaction:

$$
D^{00}(\mathbf K)=v(\mathbf K),\qquad
D^{0i}=D^{i0}=0,\qquad
D^{ij}(\mathbf K)=-\,v(\mathbf K)\,P^T_{ij}(\hat{\mathbf K}),\qquad
P^T_{ij}=\delta_{ij}-\hat K_i\hat K_j ,
$$

with $\mathbf K=\mathbf q+\mathbf G$ and $v$ the (possibly slab-truncated)
kernel. The minus is the spatial metric. It is applied once, in the
propagator builder (`vcoul.COULOMB_GAUGE_TT_SIGN`), never in a vertex.

A finite k-grid never samples $\mathbf K=0$, and each block needs a
different replacement:

* $D^{00}$: $v$ diverges isotropically, and
  $M_{mn}(\mathbf q\to0,\mathbf G=0)\to\delta_{mn}$. The missing slot couples
  only the band diagonal, and the scalar cell average
  $\langle v\rangle_{\rm mBZ}$ replaces it completely.
* $D^{ij}$: $v$ diverges and $P^T$ has no limit (it depends on the direction
  of approach). The current matrix element
  $\langle m\mathbf k|\alpha^i|n\mathbf k\rangle$ is finite and not diagonal in
  band index. The replacement is the tensor
  $T_{ab}=\langle v(\mathbf q)P^T_{ab}(\hat{\mathbf q})\rangle_{\rm mBZ}$, and
  it must land in the $(\mu,\nu)$ centroid tile, not in a band-diagonal
  shift.
* $D^{0i}$: zero at every $\mathbf K$, so no bare CT head exists.

`bare_transverse` is the packed static mode with $\chi_{TT}=\chi_{CT}=0$.
The packed Dyson equation is then block diagonal:
$W=\mathrm{diag}(W_{00},D_{TT})$ and $W_{CT}=0$. The sixteen-block Σ is the
CC screened Σ plus $\Sigma^B=X(D_{TT})$, because $SX(D_{TT})=X(D_{TT})$ and
$COH(D_{TT}-D_{TT})=0$. The packed and incumbent contractions are two
orders of one sum. With the head off they agree byte for byte
(CLAIMS 581).

### 2.1 The bare TT overlay {#bare-tt-head}

Routes without the packed completion replace the $\mathbf q=\Gamma$,
$\mathbf G=0$ slot of the nine TT tiles of $V$ by $-T_{ij}/\Omega$
(`gw.v_q_bispinor._tt_head_tensor`, one $3\times3$ tensor per run). The
corrected tile flows through the ordinary $(\mu,\nu)$ convolution, so the
head reaches $\Sigma^B$ as a full band matrix. Exact identities serve as
gates:

$$
\operatorname{tr}T = 2\langle v\rangle_{\rm mBZ}\ \ (\operatorname{tr}P^T=2),\qquad
T=\langle v\rangle\,\mathrm{diag}(\tfrac12,\tfrac12,1)\ \ \text{(in-plane isotropic slab)},\qquad
T=\tfrac23\langle v\rangle\,\mathbb 1\ \ \text{(isotropic 3D)} .
$$

Bulk adds the Baldereschi–Tosatti analytic sphere for the $1/q^2$ part. The
slab average is a scrambled-Sobol Voronoi draw, not the exact polygon rule
of §3.1, so it keeps the 0.1–0.2 % cusp sampling error that rule removes
(`vcoul.Slab2D.q0_average_transverse_tensor`). Box truncation (`sys_dim = 0`)
never zeros the slot and is refused.

The head is frequency independent, because $\Sigma^B$ is. Its weight in Σ
is $\langle v\rangle/(\Omega N_k)$. In 2D, $\langle v\rangle\propto
q_c^{-1}\propto N_k^{1/2}$, so the head falls as $N_k^{-1/2}$. The overlay
is on for incumbent GN/HL and for `x_only` under `head_correction = full`
with `sys_dim ∈ {2,3}`, and for the shared-pole hybrid unless the head is
`off`. On `full_shared_pole` it stays in $V$ for exchange and is subtracted
from the screening root. Restart refuses on the first two, because a
restart $V$ does not stamp the choice
(`GATE bare_tt_gamma_restart_unstamped`). Inside the packed envelope the
completion owns this term, and the overlay would double count it.

## 3. The charge head {#charge-head}

### 3.1 Objects {#charge-head-objects}

The macroscopic charge response is a Cartesian quadratic form built from
velocity matrix elements over every energy-ordered pair
(`gw.qsgw_head.head_s_tensor_sharded`; `common.chi_from_dipole.compute_S_omega`
for the `dipole.h5` head):

$$
\chi_{00}(\mathbf q\to0,\omega)=q_a S_{ab}(\omega) q_b ,\qquad
S_{ab}(\omega)=\frac{4}{\Omega N_k\,n_{\rm spin}n_{\rm spinor}}
\sum_{\mathbf k}\sum_{\epsilon_i>\epsilon_j}
\frac{(f_j-f_i)\;\overline{v^a_{ij}}\,v^b_{ij}}
{\Delta_{ij}\left[(\omega+i\eta)^2-\Delta_{ij}^2\right]},
\qquad \Delta_{ij}=\epsilon_i-\epsilon_j .
$$

The occupation difference is signed, and there is no occupied-band
boundary. When Fermi-surface weights are supplied, the kernel takes the
removable $\Delta\to0$ limit through $-f'$ and adds the Drude term. Only the $(a,b)$-symmetric part is observable in
$q\cdot S\cdot q$.

Local fields enter through the two Γ wings $Y^a_\mu$, $Z^b_\nu$ (one
velocity leg, one centroid leg; `gw.qsgw_head.head_wings_sharded`) and the
headless body $W_{\mu\nu}(\Gamma,\omega)$. The bordered-Dyson (Schur)
reduction of head against body is

$$
S^{\rm eff}_{ab}(\omega)=S_{ab}(\omega)+\frac1\Omega\,Y^a(\omega)\,W_{\rm body}(\Gamma,\omega)\,Z^b(\omega)
\qquad(\texttt{gw.head\_correction.fold\_cartesian\_head\_wings\_sharded}).
$$

On a two-component store the centroid leg is the spin-traced pair density
and $n_{\rm spinor}=2$, so the fold is one contraction on both stores.
`head_correction = full` applies it once, `no_local_fields` skips it (the
direct head), and `off` removes the Γ contribution.

The head samples are cell averages of the bare and screened kernels
(`vcoul.<kernel>.q0_average`):

$$
v_h=\langle v(\mathbf q)\rangle_{\rm mBZ},\qquad
W_h(\omega)=\left\langle\frac{v(\mathbf q)}{1-v(\mathbf q)\,q_aS^{\rm eff}_{ab}(\omega)q_b}\right\rangle_{\rm mBZ}.
$$

On a slab both brackets use the exact Wigner–Seitz polygon cubature of the
mini-lattice: a Γ-to-edge Duffy triangulation on a fixed 16/24/32
Gauss–Legendre ladder, issued by `vcoul.slab_minibz_photon_cubature`. The
packed completion of §4 reads the same receipt, so both routes evaluate one
integral on one set of nodes. The call refuses if the 24→32 pair misses its
mixed tolerance. Bulk uses Sobol sampling plus the Baldereschi–Tosatti
analytic sphere; the polygon construction is two-dimensional.

Cost per frequency: $O(N_k n_b^2)$ for $S$ on a two-axis band-pair tile;
one pass over the resident body for the fold, with no gather; and
$n_{\rm edge}(16^2+24^2+32^2)\approx10^4$ scalar kernel evaluations for
the slab cell average.

The GW scalar route inserts the head as band-diagonal shifts (§3.2). The Γ
wings of $W$ (the $(\mathbf G=0,\mu)$ blocks) are not re-attached there.
Their leading term is odd in $\mathbf q$ and vanishes under the cell
average, so the omission is $O(1/N_k)$, not singular. Consumers that need
the head inside $W(q=0)$ (BSE, the densifiers) add it as the rank-1 update
$(W_h/\Omega)\,\overline{g_0}\otimes g_0$ with
$g_0(\mu)=\zeta_\Gamma(\mu,\mathbf G=0)$
(`gw.head_correction.apply_q0_head_rank1`).

### 3.2 Static COHSEX

These band-diagonal shifts are exact:

$$
\Sigma^X_n=-\frac{v_h f_n}{\Omega N_k},\qquad
\Sigma^{SX}_n=-\frac{W_h(0) f_n}{\Omega N_k},\qquad
\Sigma^{COH}_n=+\frac{W_h(0)-v_h}{2\,\Omega N_k}.
$$

### 3.3 GN-PPM and HL-PPM {#ppm-head}

One pole is fitted from the correlation part $W^c_h=W_h-v_h$ at two
frequencies (`gw.head_correction.fit_head_ppm`):

$$
\Omega_h^2=-z^2\,\frac{W^c_h(z)}{W^c_h(0)-W^c_h(z)},\qquad
B_h=-W^c_h(0)\,\Omega_h^2,\qquad R_h=\frac{B_h}{2\Omega_h},
$$

with $z=i\omega_p$ (Godby–Needs) or a real $z$ above all transitions
(Hybertsen–Louie). HL may instead take
$\Omega_h^2=\omega_p^2/(1-W_h(0)/v_h)$ from the f-sum rule, or a
deck-supplied $\Omega_h$. The head enters $\Sigma_c$ on the band diagonal:

$$
\Sigma^{c,\rm head}_n(\omega)=\frac{R_h}{\Omega N_k}\left[
\frac{f_n}{\omega-\epsilon_n+\Omega_h-i\eta}+
\frac{1-f_n}{\omega-\epsilon_n-\Omega_h+i\eta}\right].
$$

On shell this reduces to $\Sigma^{SX-X}+\Sigma^{COH}$ of §3.2. $S$, the
wings and the body are evaluated at the same two frequencies, $0$ and $z$.

### 3.4 MPA

The head is sampled on the body's stamped complex grid, the fold of §3.1 is
applied per sample with independent left and right wings, and the complex
poles feed `gw.head_correction.compute_complex_pole_head_sigma_diag`. The
metallic static limit (Thomas–Fermi) and its Schur fold are owned by
[Metallic MPA screening §4](metallic-mpa-screening.md).

### 3.5 Time-reversal breaking {#trs-breaking}

The derivation and branch assignment are owned by
[`DERIVATION_gnppm_nonhermitian.md`](../dev/notes/DERIVATION_gnppm_nonhermitian.md).

* Each response producer takes its ordered form from the measured
  `SymMaps.trs_allowed`: `gw.w_isdf.compute_chi0_imag_ordered` for the GN
  probe, `compute_chi0_contour_ordered` for MPA contour samples. Both
  weight the two particle–hole orientations independently and keep the
  anti-Hermitian, magnetization-odd part of $\chi_0$. At $\omega=0$, and on
  every time-reversal-symmetric deck, the even completion is exact.
* The GN fit splits $W^c(i\omega_p)$ into its Hermitian and
  anti-Hermitian halves and builds Hermitian $B$ and $D$ from them. It
  gives $R_+=B+D$ to the empty-state Σ branches and $R_-=B-D$ to the
  occupied ones, so each branch receives a Hermitian residue.
* The scalar charge head has no odd part. For one unordered pair,
  $(f_j-f_i)/\Delta^2\,[\bar T_{ab}/(z-\Delta)-T_{ab}/(z+\Delta)]
  =2(f_j-f_i)[\operatorname{Re}T_{ab}-iz\operatorname{Im}T_{ab}/\Delta]/(\Delta(z^2-\Delta^2))$
  with $T_{ab}=\bar v^a_{ij}v^b_{ij}$. The odd part is the antisymmetric
  $\operatorname{Im}T_{ab}$, which $q\cdot S\cdot q$ annihilates, so §3.1
  holds without time reversal. The k sum runs over the full zone, unfolded
  by the measured magnetic group. HL's real probe cannot resolve an odd
  residue and keeps the single-residue fit.
* A time-reversal-broken shared-pole store carries only the direct head
  (`no_local_fields`); `full` refuses (`GATE shared_pole_head_ordered`).

## 4. The packed static photon head

### 4.1 Body

The packed modes build one $C\oplus T_1\oplus T_2\oplus T_3$ operator on the
q-IBZ (`gw.photon_layout`). It holds the sixteen bare blocks
$D^{IJ}_q(\mu,\nu)$ from `v_q_bispinor.h5`. The screened mode adds the
sixteen no-pair blocks $\chi^{IJ}_0$ of the kinetic-balance current, with
the TT blocks Ward-subtracted as $\Pi(q)-\Pi(0)$, and solves
$W=(1-D\chi_0)^{-1}D$ in one distributed Dyson solve. `gw.photon_sigma`
contracts the $W^{IJ}$ blocks with the Lorentz vertices applied to the
wavefunction faces. The body lacks the $\mathbf K=0$ slot in every block.

### 4.2 The coupled Γ-cell solve

The completion fills the Γ slot of both $V$ and $W$. It solves the 4×4
Lorentz Dyson equation at every node of the exact Wigner–Seitz cubature of
§3.1:

$$
R(\mathbf q)=q_a H^a+q_aq_b S^{\rm eff,ab},\qquad
H^a_{0i}=-i\,\epsilon_{bai}\,\sigma_H^b,\ \ H^a_{i0}=\overline{H^a_{0i}},\qquad
W_h(\mathbf q)=\left[1-D(\mathbf q)R(\mathbf q)\right]^{-1}D(\mathbf q).
$$

$S^{\rm eff}$ is the charge head folded through the headless packed body
exactly as in §3.1. The response has charge support only (§4.3). The Hall
pseudovector $\sigma_H$ is never fitted. It is a separately produced input
(`static_gauge_hall.h5`, `gw.qsgw_head.static_gauge_hall_transaction`) and
the only admitted $q$-linear CT/TC structure. The persisted $\sigma_H$ is the
occupied-bra Berry sum, while the live Adler–Wiser response is energy-ordered
($P=-\Delta D$); that is the minus above. An unnamed
`static_gauge_hall_file` means $\sigma_H=0$, announced; by §4.4 that is
exact for the insulators this mode admits. A named but mismatched artifact
refuses in the loader. On the bare route only an exact-zero artifact is
accepted, because with the currents unscreened $W_{CT}=0$ at every finite
$q$, and a Γ-only CT/TC block would be the limit of nothing
(`GATE packed_bare_transverse_hall_unavailable`).

Only nine monomial moments survive the cubature:

$$
M_{uv}=\left\langle b_u(\mathbf q)\,W_h(\mathbf q)\,b_v(\mathbf q)\right\rangle_{\rm mBZ},
\qquad b=(1,q_x,q_y).
$$

The packed operators receive one bare and nine screened rank-4 outer
products (`gw.photon_layout.add_photon_q0_low_rank`):

$$
V_\Gamma\mathrel{+}=\frac{1}{\Omega}\,\overline{g_0}\otimes\langle D\rangle g_0,\qquad
W_\Gamma\mathrel{+}=\frac1\Omega\sum_{uv}L_u\otimes M_{uv}R_v,\quad
L=(\overline{g_0},\,(WZ)^x,\,(WZ)^y),\ \ R=(g_0,\,(YW)^x,\,(YW)^y).
$$

$M_{00}$ is the averaged head, $M_{0a}$ and $M_{a0}$ are the single-wing
moments, and $M_{ab}$ the double-wing moments. Each factor pair is
transported through every row of the Γ little group before the products
are averaged, so the update is covariant. Screening comes before
averaging, because
$\langle[1-DR]^{-1}D\rangle\ne[1-\langle D\rangle\langle R\rangle]^{-1}\langle D\rangle$.
The odd Hall term averages to zero in $M_{00}$ but survives in the crossed
first moments $\langle q_xW^{0y}\rangle$ and $\langle q_yW^{0x}\rangle$. The
single-wing moments carry the interband weight of the $P=-\Delta D$ head
convention (`gw.qsgw_head._head_wing_interband_weight`, one owner for both
wing layouts). Its sign cancels in the scalar fold $YWZ$ but not in
$M_{0a}$ and $M_{a0}$.

Certificates: folded Ward residual $\le10^{-8}$, Hermiticity
$\le10^{-10}$, Dyson forward-error bound $\le10^{-9}$, and the
convergence of the 16/24/32 ladder. The completion is slab-only. A bulk
analytic sphere cannot be added after the nonlinear coupled solve, and no
bulk integrator is derived for this route. Cost: $O(n_{\rm nodes})$
4×4 solves on replicated data, plus ten rank-4 local outer products per
little-group row into the packed body; no sample-by-centroid array exists.

### 4.3 What the response contains, by declaration {#response-content}

The complete list is the module docstring of `gw.static_gauge_response`.
Present: the charge $q^2$ head $S^{00}$, the charge wings $Y^0$/$Z^0$, and
the Hall CT/TC $q^1$ term. Omitted by model: the current $q^2$ response
(TT, CT/TC), the current wings, the diamagnetic/contact terms, and the
negative-energy (complement-space) closure. The uniform static current
response $\chi_{TT}(q=0)$ is zero by gauge invariance for an insulator. No
omitted term is stored as an accidental zero of a larger schema:
`S_direct` has charge support only.

### 4.4 The Hall coefficient is a topological invariant

For a gapped system the static long-wavelength charge–current response is
the Chern–Simons term, and its coefficient is quantized (TKNN):

$$
\sigma_{xy}(\mathbf q\to0,\omega=0)=C\,\frac{e^2}{h},\qquad
C=\frac{1}{2\pi}\sum_{n\in{\rm occ}}\int_{\rm BZ}\Omega_n^z\,d^2k\in\mathbb Z .
$$

The producer computes exactly this occupied Berry-curvature sum
(`gw.qsgw_head.raw_hall_pseudovector_sharded`,
$\sigma_H^b=-(\alpha_{FS}C_s/2\Omega)\,\operatorname{Im}c_B^b$ with state
capacity $C_s$). It refuses metals, and it refuses degenerate
differently-occupied states. Consequently:

* For a Chern-trivial insulator, $\sigma_H=0$ in the complete-basis,
  converged-k limit. The packed head then reduces to the charge head of §3
  with its wings carried, and the absent-artifact default is exact.
* For a Chern insulator, $\sigma_H$ is an integer multiple of
  $\alpha_{FS}C_s/(8\pi L_z)$ ($L_z$ the periodic cell height), known before
  the calculation.
* A static Hall response carries new information only in a metal, and the
  producer refuses metals.

The CT sector of a Chern-trivial insulator is bounded by symmetry and by
the current vertices. The surviving CT moment is
$\langle W^{0i}q_a\rangle$ with $W^{0i}\approx D_{00}R^{0i}D_{ii}$ and
$R^{0i}=i\epsilon\,\sigma_Hq+q_aq_bS^{0i,ab}+O(q^3)$. The quadratic
coefficient is the static linear magnetoelectric response, which needs
both inversion and time reversal broken. In a centrosymmetric crystal CT
starts at $O(q^3)$, so $W^{0i}=O(q)$ and the moment is $O(1/N_k)$: body
discretization order. Without inversion, $W^{0i}=O(1)$ survives averaging
like the CC screening correction, reduced by $(Z\alpha_{FS})^2$ for two
current vertices and by $\alpha_{ME}$, whose axion-strength ceiling is
$\alpha_{FS}/2$.

Two time-reversal-odd channels lie outside the static head. The
finite-frequency Hall/Kerr response $\sigma_{xy}(\omega)$ and the
antisymmetric TT response live at $\omega\ne0$ and $O(q^2)$. In a
ferromagnet the largest transverse screening channel is the
Goldstone-enhanced transverse spin susceptibility. It is a ladder (vertex)
effect outside RPA, and the ladder screening (`w_bse`) is charge-only.

### 4.5 Scalar versus packed insertion of the charge head

Both routes evaluate the same cell integral (§3.1). They insert it
differently. The scalar route uses the band-diagonal shift of §3.2, which
is exact for the bare CC head:
$\Sigma^X_n=-\langle v\rangle f_n/(\Omega N_k)$ is band diagonal and state
independent in a plane-wave basis. The packed route inserts the same head
as a rank-4 $(\mu,\nu)$ update through $g_0(\mu)=\zeta_\Gamma(\mu,\mathbf G=0)$.
It therefore carries the ISDF error of the $\mathbf G=0$ pair density:
$\sum_\mu\zeta_\Gamma(\mu,\mathbf G=0)|\psi_n(r_\mu)|^2\ne1$. The error is
state dependent and centred near zero; on MoS₂ 3×3 with 640 charge
centroids it reaches 11.5 meV in the bare-X head of individual occupied
states (CLAIMS 586). The wings and the current blocks have no band-diagonal
form, so they stay in the $(\mu,\nu)$ representation on every route.

## 5. The direct first-order bulk head (`full_shared_pole`) {#direct-bulk-head}

`full_shared_pole` with `head_correction = no_local_fields` completes the
ordered sector bank with a direct Γ head
(`gw.photon_direct_head.build_direct_photon_head`). It requires
`sys_dim = 3` and the current map's Fermi–Dirac occupations. `full`
refuses, because no wing/body fold exists for this route.

At each Γ-cell sample $\mathbf q$ and bank frequency $z$ the response is
first order in the long-wavelength vertex:

* **Interband.** There are six vertices: three charge jets
  $q\cdot v_{nm}/(E_m-E_n)$ and three uniform currents
  $(\alpha_{FS}/2)v_{nm}$ (the dipole-velocity approximation of the raw
  Breit current). Their $6\times6$ tensors are built on two-axis-sharded
  band pairs, with the same normalization as the CC block of $S$.
* **Intraband (FD).** At $z\ne0$ the Drude tensor $D_{ab}$ gives CC
  $=qDq/z^2$ and CT $=(\alpha_{FS}/2)\,qD/z$. At $z=0$ the finite-$q$ limit
  is Thomas–Fermi, CC $=-N(E_F)$, with TT $=-(\alpha_{FS}/2)^2D$.
* **Contact.** The bank's Fermi–Dirac contact $C$ is projected on the four
  uniform vertices.

The head solves
$W_h(\mathbf q,z)=[1-D(\mathbf q)(\Pi(\mathbf q,z)-C)]^{-1}D(\mathbf q)$
and stores cell averages: $W_h-W_\infty$ with $W_\infty=(1+DC)^{-1}D$, its
$z^2$ derivative, the ordered mirror (the response conjugated at
$-\mathbf q$, with CT/TC parity), the instantaneous constant
$\langle W_\infty-D\rangle$, and the large-$z$ expansion coefficients.
These enter the bank through the four literal-Γ vectors as rank-4 updates.
The bare TT overlay (§2.1) stays in $V$ for exchange and is subtracted
from the screening root.

The cubature uses $4\times2^{17}$ scrambled-Sobol samples of the mini-BZ
exterior. Inside the excised sphere an $8\times12\times24$
radial/angular rule runs through the same coupled 4×4 solve, with weights
calibrated to the analytic $8\pi/q^2$ sphere integral. The run prints the
replicate spread and the maximum Dyson residual. Only $6\times6$ and
$4\times4$ objects are replicated; band pairs stay sharded over both mesh
axes.

The charge row is the first-order expansion in $q\cdot v/\Delta$. It
degrades when a fractional-occupation pair lies closer than $|q\cdot v|$
on the Γ cell. Near-degenerate directed pairs ($|\Delta|\le10^{-8}$ Ry)
with unequal occupations refuse, because their charge jet is undefined
(`GATE photon_direct_degenerate_occupation`). The model folds no wings and
no microscopic local fields.

## 6. Code owners

| object | owner |
|---|---|
| bare $D^{IJ}$ tiles, bare TT overlay | `gw.v_q_bispinor` |
| cell averages, photon cubature, TT sign | `vcoul` (`minibz`, `slab_2d`, `bulk_3d`) |
| $S(\omega)$, wings, Hall pseudovector, per-map head samples | `gw.qsgw_head` |
| head resolution, Schur folds, static/PPM/MPA head Σ, packed Γ completion | `gw.head_correction` |
| bounded packed-head response and its content list | `gw.static_gauge_response` |
| packed layout and rank-4 updates | `gw.photon_layout` |
| packed body response and Dyson solve | `gw.w_isdf.compute_static_photon_response` |
| sixteen-block Σ and its current-only selection | `gw.photon_sigma` |
| incumbent $\Sigma^B$ | `gw.sigma_x_bispinor` |
| direct first-order bulk head | `gw.photon_direct_head` |
| route predicates and refusals | `gw.gw_config` |
| carrier resolution | `common.four_current_model` |
