# TD-aGW with an instantaneous dynamical self-energy

This page owns the design of a time-dependent GW self-energy that is rebuilt
from the instantaneous state during a TD-aGW propagation: the equations, the
split of the Green function into a frozen and an active part, the static limit
that is the present model, what the scheme conserves and what it loses, which
GW components it reuses, the cost per refresh measured from the GW stage
walls, and the refactors it needs. The static-screening member is implemented
(`tdagw.dynsig`). The GN-PPM member's two GW stages are callable per refresh
from arrays (`gw.ppm_sigma.screening_update`, `window_sigma_c`; §9 items 4–5)
and reproduce a GN-PPM run at t = 0 (§6.1). The collision wiring and earlier
population-based shared-pole refreshes are described below; the corrected
column-sampled MGKBA split is specified in §1.5. The full-density snapshot
modules implement general lesser/greater bubbles, a reduced rational Keldysh
interaction, and a converged material/coherent-self-energy update. The
`retarded_reference` choice instead builds only the full-density retarded
response and uses an explicitly equilibrium-like pole-branch self-energy
[reference](#retarded-reference). The
explicit `lossless_bath` alternative uses secular screening statistics with
full-density electronic lines, as specified in the
[transition-bath prescription](#lossless-transition-bath). Their
finite rank, support, regulator and refresh approximations are explicit. The
equation of motion, integrators and layout are owned by
[TD-aGW propagation](td_agw_propagation.md); the frozen-W kernel by
[TD-aGW interaction](td_agw.md). Equations in Ry unless stated.

The current full-microscopic snapshot contract is the
[retarded reference](#retarded-reference): full instantaneous density and
energies, column-sampled complex self-energy, and the material closure
$H^s=h_{\rm HF}[\rho]+M_H[\rho,H^s]$. It does not subtract a reference
collision source. Earlier carrier-only dressing and reference-subtracted
formulas below describe the earlier members, not additional operations to
apply to this closure. A DFT-prepared initial density is a preparation
quench, not an asserted stationary full-frequency GW ground state.
Implementation and small-matrix checks are not evidence that a material
pole bank has been admitted or a physical trajectory completed; the
[current numerical evidence](#full-microscopic-evidence) distinguishes them.

## 1. The scheme {#scheme}

This opening reference-increment scheme and §1.1's carrier-only reference
are retained to explain the earlier static/GN-PPM members. Their diagonal
reference energies are those actually supplied by the run: DFT bundle
energies when no `eqp_file` is set, QP energies otherwise. The label
$\varepsilon^{\rm QP}$ below does not imply that the historical CrI3 516
decks supplied QP energies. For the current full-microscopic member use
§1.5's explicit material Hamiltonian and counterterm choice instead.

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
H^G_{A,k}(t)=\mathrm{diag}\big(\varepsilon^G_{A,k}\big)+\lambda\,\big(K\,\delta\rho^{\rm car}\big)_k
=U_k\,\mathrm{diag}(e_k)\,U_k^\dagger,\qquad
\delta\rho^{\rm car}_k=U_k\,\mathrm{diag}\big(f_k-f^{\rm eq}_k\big)\,U_k^\dagger,\qquad
f_{ik}=\mathrm{Re}\,\big(U_k^\dagger\rho_kU_k\big)_{ii}, \tag{2}
$$

and the instantaneous orbitals are $\tilde\psi_{ik}=\sum_m\psi_{mk}U_{k,mi}$;
$f^{\rm eq}_k$ are the equilibrium occupations of the window bands,
$\rho^{\rm eq}_k=\mathrm{diag}\,f^{\rm eq}_k$.
$\varepsilon^G$ are the energies the GW run's $G$ used (DFT for
`one_shot_dft`), so that $\Sigma[\rho^{\rm eq}]$ is the run's own $\Sigma$
(§6.1). $\lambda=0$ (`dress = "none"`) keeps $U=1$, $e=\varepsilon^G$.
$\lambda=1$ (`dress = "instantaneous"`) dresses the basis with the frozen-W
update $K$ of the **carriers alone**, the population changes $f-f^{\rm eq}$
carried by the instantaneous orbitals, so Eq. (2) is a fixed point in
$(U,e,f)$, solved in every call.

* **Why the carriers alone.** In the instantaneous basis
  $\delta\rho=\delta\rho^{\rm car}+\delta\rho^{\rm sea}+\delta\rho^{\rm coh}$:
  the carriers of Eq. (2), the rotated equilibrium occupations
  $\delta\rho^{\rm sea}=U\mathrm{diag}(f^{\rm eq})U^\dagger-\rho^{\rm eq}$, and
  the coherences $\delta\rho^{\rm coh}=U\,(U^\dagger\rho U)_{i\neq j}\,U^\dagger$.
  Neither the field $e\mathbf E\cdot\mathbf r$ nor $\delta\rho^{\rm coh}$ nor
  $\delta\rho^{\rm sea}$ enters $H^G$. The coherent part of $K\delta\rho$ is the
  induced field (depolarization through $K^H$, the electron–hole attraction
  through $K^W$); it oscillates at $O(E)$ with the drive, like
  $e\mathbf E\cdot\mathbf r$, while $\delta\rho^{\rm car}$ is $O(E^2)$ and moves
  on the envelope. The basis follows the carriers, not the optical cycle, and
  linear response is the BSE (§2). $\delta\rho^{\rm sea}$ is the Fermi sea
  carried along by the rotation it causes: through $K$ it feeds the window's
  static response back into $U$. With the DFT $\varepsilon^G$ of CrI3 516 2v2c
  that response is near-singular: at the fixed point of a weak drive
  $\max|\delta\rho^{\rm car}+\delta\rho^{\rm sea}|$ is 15× the populations.
  Two red twins measure the excluded terms: `dress = "coherent"` dresses with
  the whole $\delta\rho$ (one diagonalization of
  $\mathrm{diag}\,\varepsilon^G+K\delta\rho$) and moves $e$ and $U$ at $O(E)$;
  `dress = "sea_rotation"` dresses with
  $\delta\rho^{\rm car}+\delta\rho^{\rm sea}=U\mathrm{diag}(f)U^\dagger-\rho^{\rm eq}$.
* **The fixed point** (`tdagw.dynsig._population_basis`): from $U=1$,
  $f=\mathrm{diag}\,\rho$, `DRESS_STEPS` = 32 plain evaluations of the map
  $\delta\rho^{\rm car}\mapsto U\,\mathrm{diag}(f-f^{\rm eq})\,U^\dagger$ in a
  static loop, without mixing. Without the sea the rotation feeds back only
  through the carriers it carries, so the contraction factor is of the order
  of the populations times $K$ over the window's splittings. On CrI3
  516 2v2c the plain iteration reaches round-off in 3, 5 and 23 steps at
  populations $5\times10^{-6}$, $5\times10^{-4}$ and $1.2\times10^{-2}$. Its
  contraction factor grows with the populations ($10^{-2}$ per step at
  $5\times10^{-4}$, 0.34 at $1.2\times10^{-2}$, about $p^{1.1}$), so on that
  window 32 steps hold the gate up to populations of about $2\times10^{-2}$
  and the plain map stops contracting near $3\times10^{-2}$ (both
  extrapolated). On the synthetic states of §8 it reaches an absolute residual
  of $10^{-12}$ in 3–7 steps at populations up to $4\times10^{-2}$, 17 on the
  driven state (populations $7\times10^{-2}$) and 36 on the hot-carrier state
  (0.39), which 32 steps leave at $1.1\times10^{-11}$. The sea-rotation map
  contracts only at the static response's rate (Jacobian eigenvalue $-0.88$
  on the synthetic driven state). On CrI3 516 its plain iteration stalls at a
  relative residual of 2.4–$2.7\times10^{-2}$ from step 12 to 40; Anderson
  mixing over 16 steps converges at populations $5\times10^{-6}$
  ($4\times10^{-11}$ relative in 32 steps) and does not at $5\times10^{-4}$.
  A residual $\max|U\mathrm{diag}(f-f^{\rm eq})U^\dagger-\delta\rho^{\rm car}|$
  above `DRESS_TOL` = $10^{-8}$ makes ΔΣ NaN
  (`GATE dynsig_dress_not_converged`, named by `check_state`).

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
  earlier correlation construction used the symmetrized GKBA,
  $g_{ij}(\tau)=\tfrac12(e^{-e_i\tau}+e^{-e_j\tau})\,\tilde\rho_{ij}$, a
  coherence entering at both of its energies. The corrected collision uses
  the ordered MGKBA line and column-energy sampling of §1.5 instead.

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

**Current collision admission.** The column-sampled MGKBA entry in
`tdagw.dynsig.window_collision` accepts `collision_screening = frozen`.
The runtime dispatches `joint` and `joint_frozen` to the full-density
`tdagw.snapshot_runtime.snapshot_collision` adapter described in
[the response/noise contract](#keldysh-screening).
Both `refreshed` and `shared_pole` refuse with
`GATE dynsig_collision_keldysh_screening`: their older population-only
refits lack the full-density instantaneous-energy adapter and explicit
reference-branch prescription of the native joint route.
The refit machinery below is retained for development and its earlier
arithmetic-operator benchmarks; those population-refit modes remain refused.

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

**GN-PPM refreshed-screening machinery** (stages built, §9 items 4–5;
`collision_screening = refreshed` currently refused). Two supports $0$ and
$i\omega_p$, Eq. (6) at
each; the per-element pole fit of `ppm_sigma.fit_ppm` (ordered, since the
magnet and a helicity drive break time reversal: conduction branches take
$R_+=B+D$, valence branches $R_-=B-D$; modes the fit marks invalid enter as
the static $W^c(0)$ term); $\Sigma^c_{ij}(\omega)$ by
the Σ τ sweep with $G=G_{\rm eq}+\delta G$ (symmetrized GKBA for the
coherences, §1.1); the operator at the instantaneous energies, Eq. (8) of §1.5, kept complex
(no Hermitian projection; [on-shell Σ](sigma-on-shell.md) for the
evaluation), rotated back with $U$.

**Shared-pole refit machinery** (`collision_screening = shared_pole`, currently
refused by the collision entry). The retained population-based implementation
and earlier arithmetic-operator results use
$W^c(q,z)=\sum_j b_j(q)b_j(q)^\dagger/(z^2-\Omega_j^2)$
with poles shared per $q$ (the even store; the ordered store pairs $q$ with
$-q$). A refresh is a **full refit**: the GW run's own W stage
(`gw.shared_pole_screening.screen_shared_poles`: response bank, moments,
constructor) rerun at the instantaneous occupation table — the run's step
occupations outside the window, diag ρ inside (dress none, §1.1) — after the
recipe is re-resolved at that state's census (`bind_shared_pole_census` +
`resolve_shared_pole_recipe`). $\chi_0(f(t))$ is the bank's own: it takes
fractional occupations at every complex support natively (the metallic route,
item 3), same-side pairs exact at $z\neq0$; no support is at $z=0$, so no
principal value enters. Eq. (6) is not used: the bank's Dyson solve is the
run's, with $v$. The head is frozen (the refit's `head_correction = off`; the
window Σ adds the run's head). At $\rho^{\rm eq}$ the refit is bitwise the run's
model (§6.1). The static limit subtracted from $\Delta\Sigma^{\rm h}$ is the
equilibrium model's own $W^c(0)$ (`gw.mpa.sigma.shared_pole_static_wc`; it
differs from the Dyson $W_0$ of the TD model's $K\delta\rho$ by the model's
static error, $1.6\times10^{-5}$ max-relative on 516). Two support sets:
the production ladder re-resolved at every state (V1), or an explicit set
(`collision_spm_support_sites_ev`, "line | imaginary | height" in eV). On 516
the explicit set that works at carrier states is V2′: the ladder plus a low
line at 0.25–2.3 eV, the whole line at height 1.0 eV, and imaginary sites 0.25
and 0.5 eV. An imaginary site at 0.1 eV is refused at carrier states by the
bank's rule construction (the response exponential fit, `tol 1e-8`, fails once
the occupation interval crosses zero); at $\rho^{\rm eq}$ it is admitted. V2′ itself
refits at $\rho^{\rm eq}$ and at the 5% hot state but is refused by
`GATE shared_pole_gram_valid` at q = 0 at the driven 3×10⁹ V/m states (Gram
min −5.8×10⁻⁷ at 1.5 fs, −7.8×10⁻⁶ at the pulse peak, against the rounding
floor 10⁻⁷; no PSD repair), so a driven run with V2′ stops at its sixth refresh;
V1 refits at every state measured.

**χ₀ in four blocks** (`collision_spm_chi0_active = [a0, a1]`, refits only; unset = the
GW run's one rule on every pair, unchanged). The refit's $\chi_0$ is split by hole space
× electron space, as $\delta\chi_0$ is (§1.3): F, the bands outside the active range
A = [a0, a1), at their 0/1 step; A, with f = diag ρ on the window. Hole weights f
($G^<$), electron weights 1 − f ($G^>$), so
$\chi_0=\chi_0^{FF}+\chi_0^{FA}+\chi_0^{AF}+\chi_0^{AA}$ exactly
(`gw.response_bank.response_blocks`). Each block has its own transition interval,
from band energies only (F's occupied bands as holes, F's empty bands as electrons, every
A band on both sides): on 516 with A = [88, 144), FF 16.19–83.40 eV, FA 6.99–80.15 eV,
AF 0.81–11.64 eV and AA ±8.40 eV (the nearer gaps around the window, 0.01–0.05 eV, would leave
FA and AF barely positive). Only AA crosses zero, and it is built without the
occupation envelope (decay rate 0). The intervals do not depend on f(t), so every
block's rule is built at the first refit and held for the run: no per-refresh rule
construction. Where a sample's AA pencil fails (V2's 0.25 and 0.1 eV sites), the
closed-form imaginary-axis set is taken in Gauss–Legendre panels of at most 384 nodes
with the quadrature's own weights, whose contributions add (`minimax`
`ResponseGroupTree(panels=True)`: 472 nodes at 0.25 eV, 1180 at 0.1 eV). The blocks
share the one compiled direct stream (union band bounds; each block's masked weights
and gauge references as operands) and add into one carry. Dyson, the exact moments
and the constructor are unchanged. A must end at band gaps (FA, AF strictly
positive) and F must stay 0/1, else `GATE response_blocks_*` refuses. On CPU
(`tests/test_response_bank_blocks_cpu.py`) the four blocks match the exact
ordered-pair scan (`compute_chi0_direct_fractional`) to 8.7×10⁻¹⁰ down to
Im z = 0.1 eV with fractional populations.
Holding the equilibrium poles and refitting only one real residue scale per
pole (V3) leaves 18–28% of the hot state's change of $W^c$ at the fit sites
(V1 target; 48–69% for V2′) and needs negative scales: the poles must move,
so the refresh refits them.

**Head.** The prototype keeps the run's $W_h$ (band-diagonal SX/COH shifts,
[static head](four-current-head-corrections.md#static-head)) frozen: $\delta\chi_0$
carries no $q\to0$ head. The head change is $q\cdot\delta S\cdot q$ from the
window's dipoles and populations (the `qsgw_head` route), plus a Drude term
once carriers exist (§4).

### 1.5 The on-shell collision term {#damping}

The corrected frozen-Hamiltonian Markov GKBA (MGKBA) prescription samples a
retarded self-energy on the **column energy**. It replaces the arithmetic
average of two on-shell tables as the collision definition. At each refresh,
$H_s=U E U^\dagger$, $\rho'=U^\dagger\rho U$ and, for $\tau>0$,

$$
g^<(\tau)=i\Psi'\rho'e^{-iE\tau}\Psi'^\dagger,\qquad
g^>(\tau)=-i\Psi'(1-\rho')e^{-iE\tau}\Psi'^\dagger.
$$

The ordering follows by putting the free evolution of $\rho(t-\tau)$ into
GKBA; it retains the full matrix $\rho'$. With
$\bar\Sigma^\gtrless(z)=\int_0^\infty d\tau\,e^{iz\tau}\Sigma^\gtrless(\tau)$,
define $A_{{\rm in},ab}=-i\bar\Sigma^<_{ab}(E_b+i\eta)$ and
$A_{{\rm out},ab}=i\bar\Sigma^>_{ab}(E_b+i\eta)$. For the frozen equilibrium
$W$ branch executor, its time-ordered hole table must first become retarded:

$$
R_p=\mathrm{cond}[1],\qquad R_h=\mathrm{val}[0]^\dagger,\qquad
M=R_p+R_h+S_{\rm static},\qquad A_{\rm in}=iR_h,
$$

where $S_{\rm static}$ is the separately Hermitian static coherent term, when
present. Slot 1 is the column-energy particle table; adjointing slot 0 of the
hole table reverses its time ordering and supplies its column-energy table.
Do not adjoint slot 1 or average slots before this conversion. The resulting
split, rotated back with $U$, is

$$
\Gamma=i\big[(R_p+R_h)-(R_p+R_h)^\dagger\big],\qquad
D=A_{\rm in}+A_{\rm in}^\dagger,\qquad
C_{\rm raw}=-i[\operatorname{Herm}M,\rho]
-\tfrac12\{\Gamma,\rho\}+D. \tag{8}
$$

Without the separately coherent static term this equals
$A_{\rm in}(1-\rho)+(1-\rho)A_{\rm in}^\dagger
-A_{\rm out}\rho-\rho A_{\rm out}^\dagger$, since
$A_{\rm in}+A_{\rm out}=i(R_p+R_h)$. Moving
$\operatorname{Herm}M$ into the coherent Hamiltonian removes its commutator
from the collision. In the legacy frozen-screening path, the static
screened-exchange response already present in $K\delta\rho$ must also be
removed from the added coherent correction, using the same full density matrix
and equilibrium $W_0$. Joint screening instead uses bare HF increments and
requires no such static subtraction. A time-ordered total or
the Hermitian part of an arithmetic on-shell average is not this $M$.

**Optional counterterm.** `raw` uses $C_{\rm diss}[\rho]=
-\tfrac12\{\Gamma[\rho],\rho\}+D[\rho]$ without subtraction.
`subtracted` adds the counterterm $-C_{\rm diss}[\rho^{\rm eq}]$ to make the
reference state stationary for the chosen frozen coefficients. This is an
additional approximation, not a Keldysh identity or a positivity argument.
With finite $\eta$, sharp occupations need not obey exact detailed balance;
raw equilibrium residuals and their $\eta$ convergence must be measured.
Recomputing the counterterm on a nonequilibrium $W(t)$ can remove physical
excitation by its lesser component, so the existing `rebase` option is not a
derivation of a full nonequilibrium collision operator.

* **Particle number.** Matching active internal and external state spaces,
  reciprocal vertices, bosonic endpoint relations and common quadrature are
  required for the self-consistent trace cancellation. Window leakage and
  nonreciprocal PPM fits must be measured. Even if the freshly evaluated
  $\sum_k\operatorname{Tr}C[\rho_*]$ vanishes, holding $D[\rho_*]$ and
  $\Gamma[\rho_*]$ gives
  $\sum_k\operatorname{Tr}C_{\rm held}[\rho]=
  \sum_k\operatorname{Tr}(D[\rho_*]-\Gamma[\rho_*]\rho)$, which generally
  changes as $\rho$ evolves. Refresh residual and trajectory number drift are
  separate diagnostics; cadence convergence is required.
* **Positivity.** For fixed held coefficients write the absolute-state flow as
  $\dot\rho=-\tfrac12\{\Gamma,\rho\}+D_{\rm eff}$, where
  $D_{\rm eff}=D$ for `raw` and includes the explicit equilibrium counterterm
  for `subtracted`. The full Pauli interval is invariant for every initial
  state iff $D_{\rm eff}\succeq0$ and
  $\Gamma-D_{\rm eff}\succeq0$. At a state-dependent boundary only their
  restrictions to $\ker\rho$ and $\ker(1-\rho)$, respectively, are required.
  Nonsecular column sampling guarantees neither inequality, even for positive
  spectral residues. Refreshes therefore report the minimum eigenvalues of
  both boundary matrices and of $\rho$; no coefficient or density is clipped.
  A future guaranteed closure must construct positive channel Gramians and
  paired destination-blocked gain/loss terms before summing channels. That can
  retain off-diagonal channels and exact instantaneous number cancellation,
  but it is a different closure, not a repair of Eq. (8). Negative dissipative
  substeps in the cfm4 triple jump would still prevent a guarantee for the
  composed numerical flow; a guaranteed mode needs forward dissipative steps.
  This is not only a warning about arbitrary held matrices: an exact rational
  three-state GW/RPA snapshot with Hermitian reciprocal density vertices,
  positive bare interaction, positive lesser/greater spectra and zero trace
  collision can still point outward on the actual current-density boundary.
  For example, in units with an electronic splitting of one, take
  $H=\mathrm{diag}(0,1,0.37)$,
  $\rho=|+\rangle\langle+|\oplus\tfrac12$ and
  $V=0.1\,[\begin{smallmatrix}a&1\\1&-a\end{smallmatrix}]
  \oplus0.2\sqrt{0.3-a^2}$, $a=0.4\gamma^2$.
  At $\gamma=0.05$, $\eta_{\rm coll}=0.01$, the empty/filled eigenvalue
  derivatives are $-6.65\times10^{-10}$ and $+6.65\times10^{-10}$ respectively
  (no rational fit or equilibrium subtraction). Thus a universal nonlinear
  invariance theorem would also be false. The small size of this example's
  excursion does not bound excursions in a material calculation.
* **Linear response.** Off-diagonal rates and the variation of $D$ couple
  coherences and populations; a scalar sum of two quasiparticle widths is
  valid only after further diagonal restrictions. The static-$W$ limit has
  no dynamical width; its coherent contribution must obey the double-counting
  subtraction above.
* **GN-PPM.** With real pole energies the rates are sums of broadened
  $\delta$ functions at $e_n\pm\Omega$; on a 30×30 mesh they depend on the
  regularization width. The shared-pole member gives smoother rates.
* **Integrator.** $\Gamma$ and the source
  $S=-\tfrac12\{\Gamma,\rho^{\rm eq}\}+D-C_{\rm diss}[\rho^{\rm eq}]$
  are held between refreshes (§7.1), like $\Delta\Sigma^{\rm dyn}$. The
  last term is omitted for `raw`. The dissipative flow
  $\dot x=-\tfrac12\{\Gamma,x\}+S$ is then affine and is
  solved exactly in the eigenbasis of $\Gamma$,
  $x'_{ab}(\tau)=e^{-(\gamma_a+\gamma_b)\tau/2}x'_{ab}(0)+\tau\,\varphi_1\big((\gamma_a+\gamma_b)\tau/2\big)S'_{ab}$
  with $\varphi_1(z)=(1-e^{-z})/z$. It is composed with the commutator flow
  by the triple jump, as the $\Gamma$ remainder is now, and is $k$-local: two
  $n_A\times n_A$ products per $k$ and application.
* **Code seam.** `window_sigma_c` supplies the branch pair tables. The collision
  consumer must construct Eq. (8) from the specified slots; its total
  time-ordered `onshell` table is not the retarded coherent operator.

**Branch executor and earlier benchmarks.** The wiring path is
`window_sigma_c(branches=True)` → `tdagw.collision` (rates, $C[\rho]$, the
held flow, the Hermitian update) → `tdagw.propagate.collision_program` (one
jitted `shard_map`: the dissipator, $C[\rho^{\rm eq}]$, the trace split) →
`ops.collision`, refreshed by `run_propagation`'s chunk-end hook
(`tdagw.dynsig.window_collision`); deck keys `[lorrax] collision*`
([input reference](../input_reference.md#td-agw)). The historical measurements
below used arithmetic pair averaging. They establish executor and cost
baselines, not validation of the corrected column-sampled Eq. (8):

* **One sweep, branches kept.** `gw.mpa.sigma._integrate_sigma_batches(keep_branches=True)`
  folds each window into its branch's accumulator (cond, val) instead of one;
  the total is their sum after finalization, equal to the incumbent's to
  rounding, and with the option off the incumbent's accumulator and outputs are
  unchanged bitwise. The head splits by its own terms: $f/(\delta+z)$ to the hole
  branch, $(1-f)/(\delta-z)$ to the particle branch; the invalid-pole static term
  is a third, Hermitian table. `window_sigma_c` now returns `onshell`
  ($U\tilde\Sigma U^\dagger$, complex) in place of the Hermitian `static`, and
  slices the window blocks on the device.
* **Hole conversion.** The sweep's hole branch is time ordered
  ($E-\epsilon+\Omega-i\eta$). Its sign convention is pinned by
  `tests/test_tdagw_dynsig_sigma.py` G7, twin R5. For the column-sampled
  collision the required conversion is the adjoint of slot 0, as above;
  changing the sign of an averaged hole width alone is insufficient.
* **Positivity is measured, not assumed.** Each refresh must report the
  smallest eigenvalues of $\Gamma$, $D$, and the propagated density bounds.
  Exact solution of a held affine flow does not make it positive.
* **The held Hermitian part.** $K\delta\rho$ already carries the static screened
  exchange of the populations, $-W^c(0)\,\delta f$, which is the static-$W$
  limit of the GN-PPM population response. The default
  (`collision_hermitian = dynamic`) therefore holds
  $\operatorname{Herm}M[\rho]-\operatorname{Herm}M[\rho^{\rm eq}]$
  minus the same change of that static limit (`window_sigma_c(static_limit=True)`:
  the window block of $-\langle G_{\rm occ}W^c(0)\rangle$ on every mode plus the
  head's static limit), the correction that vanishes as the pole energies go to
  infinity (§2). `full` is the literal difference (it counts the static part
  twice); `none` drops it.
* **Explicit methods.** dp54, rk4 and etdrk4 carry the generator
  $-\tfrac12\{\Gamma,x\}+S$ in the right-hand side; magnus2 and cfm4 apply
  the exact flow in the remainder slot (Strang, triple jump for cfm4), on the
  Schrödinger-picture $\Delta\rho$. Fourth order between refreshes does not
  establish fourth-order convergence of the refreshed nonlinear problem.
* **Refresh.** At a chunk end, by cadence (`collision_refresh_fs`) and/or
  $\max|f-f_{\rm last}|$ (`collision_refresh_df`): one window sweep per trial at
  $f=\mathrm{diag}\,\rho$ (dress none: $U=1$, $e=\epsilon^G$, the coherences not in
  the rates), the rates, the dissipator and $\Delta\Sigma^{\rm h}$, all handed
  to the next chunk as jit operands of the same shapes, so no chunk program is
  rebuilt. The carry absorbs the change: the explicit methods' last derivative
  is re-evaluated, and the Magnus predictor history gains the held term's change
  in each entry's frame (exact for a term constant in $\rho$).
* **Source of the poles.** A GN-PPM run persists $W(0)$ but not its pole model,
  so with `collision` on the TD driver runs the GW driver in its own process on
  a copy of `bundle_input` (`outdir/collision_source`, no restart written) and
  reads the Σ stage's carrier, poles and head at its return
  (`gw.ppm_pipeline.observe_window_sigma_source`). A restart with the collision
  on refuses (`GATE td_collision_restart`): the held rates are not in the carry.
* **Stability gate.** The gate measures the coherent problem; the collision is
  attached after it (the flow is a contraction where $\tilde\Gamma\ge0$).
* **Measured (CrI3 516, 07 GN-PPM arm, 4 A100, 2026-10-18).** The branches of
  one sweep sum to the incumbent pair to $1.5\times10^{-16}$ (relative);
  $\tilde\Gamma^>$, $\tilde\Gamma^<$ match an explicit $(q,s,\mu,\nu)$ NumPy sum with
  the run's residues to $4\times10^{-5}$ and $2\times10^{-3}$ of their maxima
  (the sweep's τ quadrature, $\sim5\times10^{-5}$ of $|\Sigma|$). At the run's
  regularization, 0.25 eV, the band-edge rates of the 2v2c window are
  regularization tails, $\propto\eta$: $\tilde\Gamma^>_{cc}$ 113–132 meV and
  $\tilde\Gamma^<_{vv}$ 125–196 meV, five times smaller at 0.05 eV; so are those of
  the 10v14c window [120, 144) (up to 222 meV, 44 meV at 0.05 eV: no
  $\eta$-independent channel inside it). $C[\rho^{\rm eq}]$ is 28 meV on the
  diagonal and 13 meV off it; the subtracted form keeps $\rho^{\rm eq}$ exactly
  (deviation 0 over 2.45 fs), the raw form drifts by 0.14 in $\mathrm{Tr}\,\rho$ per
  $k$. A refresh costs one window sweep (3.24 s against the incumbent call's
  3.22 s; 6.29 s against 6.27 s on one A100); the flow is 0.08 ms per call
  against 1.5 ms for one ΔΣ apply; one chunk program per run, no retrace across
  15–31 refreshes; the chunk program's collectives are unchanged. Physical
  rates at band edges therefore need a smaller regularization width than the
  Σ quadrature uses, or the shared-pole member.
* **On origin/main (2026-10-18).** Merged onto main 6f7b2ca26 (on-shell-only Σ:
  the accumulator factory carries `keep_branches`). The GW driver on the 07 deck
  is bitwise main's (15/15 `sigma_mnk.h5` datasets); main's driver differs from
  the 07 arm's files by ≤ 2.1×10⁻¹¹ eV in Σ_c (sigma_diag.dat identical), so
  the 516 rates move by ≤ 1.2×10⁻¹² (relative) and the driven runs by ≤ 2.4×10⁻¹²
  in P(t). The incumbent sweep is unchanged: 206 τ in 8 windows (val:resonant
  156), 3.26 s at P4 (3.22 s before the merge).
* **Rule switching (a non-smoothness of the incumbent).** A state enters a
  branch when its weight passes `OCCUPATION_WEIGHT_FLOOR` (10⁻⁵); the one-shot
  plan of the full sweep then re-plans its windows (516 drive at 5×10⁸ V/m:
  206 → 275 τ at 2.5 fs, when $n_c$ passes 10⁻⁵). The new rules differ from
  those of $\tilde\Sigma[\rho^{\rm eq}]$ by the quadrature error, and the held
  $\Delta\Sigma^{\rm h}$ jumps from 1.5×10⁻³ meV to 0.38 meV and stays there,
  while its physical value (the split refresh below, one held session) is
  2.9×10⁻³ → 0.9×10⁻³ meV. The full refresh should hold its rules too.

**Split refresh (`collision_refresh_sweep = active`).** With $W$ frozen and
dress none each branch is linear in the intermediate weights ($1-f_n$, $f_n$)
and so is the invalid-pole term's screened exchange, while the frozen bands keep
their populations. The window Σ splits by intermediate band,
$\Sigma=\Sigma^F+\Sigma^A[f]$ (`gw.ppm_sigma.InternalBands`;
`window_sigma_c(internal="frozen"|"active")`): $\Sigma^F$ (the bands outside the
window, with the Coulomb hole, no head) is swept once at t0 and each refresh
sweeps $\Sigma^A$ (the window's own bands, with the head). Each part runs on its
own held quadrature session planned on the window's own on-shell energies
(`window_sigma_plan(window_only=True)`; the rows of the Σ prefix outside the
window take the window's first energy at their $k$), the active one with every
active state live in both branches, so every value after the planning calls
comes from held rules and a refresh at $\rho^{\rm eq}$ reproduces t0. The
band-bracket reduction is not used: it keeps the matrices of the first and last
cumulative counts only, and the frozen part needs the off-diagonal blocks of a
middle bracket. Measured (516, P4): 2v2c active 44 τ, 0.95 s per refresh (88 τ,
1.56 s once both branches are live) against the incumbent's 206 τ, 3.27 s
(275 τ, 4.23 s), frozen part 48 τ, 1.09 s once; 10v14c active 68 τ, 1.29 s (182 τ,
2.82 s hot) against 206 τ, 3.34 s. The split equals the full sweep to the rule
tolerance (rates 2.1×10⁻³ of their maximum, pair 5×10⁻⁵ relative); along the
5×10⁸ V/m drive every refresh was checked against a full sweep at the same ρ.

**Earlier refreshed-screening baseline (`collision_screening = refreshed`,
currently refused; §7.1).** In the retained machinery, each refresh
re-screens $W$ at both GN-PPM supports from the window populations,
$W(z_s,t)=[1-W_0(z_s)\delta\chi_0(z_s,t)]^{-1}W_0(z_s)$, $z_s\in\{0,i\omega_p\}$
(`gw.ppm_sigma.screening_update` on the run's own two W samples and fit plan,
taken at the GW driver's seam by `observe_window_sigma_source(screening=True)`),
and sweeps the window Σ on the new poles; the $q\to0$ head stays the run's.
$\delta\chi_0$ (`tdagw.dynsig.delta_chi0`, dress none) uses the run's static
Laplace rule and its ordered probe rule: at $i\omega_p$ every node weighs
$\alpha_l-i\beta_l$, the Pauli product is the same kernel with the opposite
sign, and a same-side pair takes $(f_a-f_b)/(E_a-E_b-i\omega_p)$, the
orientation of the run's ordered $\chi_0(i\omega_p)$. Its real part alone
(the time-reversal-even coefficient) misses the magnet's odd channel (twin R7
fails by 0.21 on the synthetic bundle). At $z=0$ a driven distribution leaves
symmetry-equivalent $k$ unequally populated, and the exact static term of such a
degenerate pair is a Drude pole (884 pairs on the 516 drive); the refresh takes
the principal value of width γ, $(f_a-f_b)(E_a-E_b)/((E_a-E_b)^2+\gamma^2)$
(`collision_screening_gamma_ev`, 1 meV), exact for $|E_a-E_b|\gg\gamma$. The
frozen part now changes with $W$ and is swept at every refresh on its held t0
session. $\Delta\Sigma^{\rm h}$ subtracts the static limit of the run's $W_0$,
which $K\delta\rho$ holds, so the static and dynamical parts of $\delta W$ stay:
$\Delta\Sigma^{\rm h}=\mathrm{Herm}\,\tilde S[\rho,W(t)]-\mathrm{Herm}\,\tilde
S[\rho^{\rm eq},W_0]-(\Sigma^{\rm SL}[\rho,W_0]-\Sigma^{\rm SL}[\rho^{\rm eq},W_0])$.
With every mode invalid (the static-$W$ limit) this is the static member's
Eq. (5) for the populations (synthetic: 8×10⁻¹¹; the static limit taken at
$W(t)$ instead fails by 2.6, twin R8). At $\rho^{\rm eq}$, $\delta\chi_0=0$ and
$W=W_0$ exactly and the poles are the run's bitwise (516 and synthetic); the
run's $\chi_0$ rebuilt from the split at both supports gives its $W$ to
1.1×10⁻¹⁴ (516).

Measured on the 516 2v2c drives (8 fs, cfm4 25 as, refresh by cadence and
$\max|\delta f|\ge10^{-3}$; $E_g$ = 3.37 eV, the 07 arm's QP gap):
* **Counterterm sensitivity in the earlier population model.** With $C[\rho^{\rm eq}]$ at the
  t0 rates the final carrier density rises by 43 % at 5×10⁸ V/m and 58 % at
  3×10⁹ V/m against frozen $W$; with $C[\rho^{\rm eq}]$ at the rates of
  $\rho^{\rm eq}$ on $W(t)$ (`rebase=True`, +2.1 s per refresh) by 0.2 % and 19 %.
  These runs attributed the difference to regularization tails changing with
  $W$. This does not justify rebasing a future physical nonequilibrium $W^<$;
  the excitation it induces at $\rho^{\rm eq}$ need not be a tail.
* **$\delta W$ moves Re Σ more than the band-edge rates.** At a 5 % carrier
  transfer (all $k$) Re Σ of the window moves by 594 meV (17.7 % of $E_g$) and the
  band-edge rates by 37–39 meV; along the 3×10⁹ V/m drive the held
  $\Delta\Sigma^{\rm h}$ reaches 119 meV (3.5 %) against 0.13 meV with frozen $W$,
  while the smallest rate eigenvalue moves by 5.5 meV.
* **Not smooth.** $\Delta\Sigma^{\rm h}$ jumps by up to 77 meV (2.3 % of $E_g$)
  between refreshes at 3×10⁹ V/m (0.78 meV at 5×10⁸), correlated with the
  invalid-mode flips (r = 0.94; up to 1.7×10⁵ of 3.4×10⁷ modes per refresh):
  a mode whose fitted $\Omega^2$ changes sign moves between the pole and the
  static term. Refreshing every 0.125 fs does not converge P(t) at 3×10⁹ V/m
  (3.3×10⁻² between cadences; 3×10⁻⁴ at 5×10⁸ V/m, 2×10⁻⁵ with frozen $W$). The
  same-side principal value is a second sensitivity: γ = 10 meV instead of 1 meV
  moves the final carrier density by −55 % at 3×10⁹ V/m.
* On 10v14c at 3×10⁹ V/m (0.5 carriers per cell) $W$ moves by 85 %, 19 % of the
  modes turn invalid and the smallest rate eigenvalue reaches −1.8 eV: outside
  the member's range.

**Full-ρ in-scattering and active rate intermediates (2026-10-19, opt-in).**
Two deck keys change the rates of the split refresh (`collision_refresh_sweep =
active`). The arithmetic-pair construction and measurements in this paragraph
are an earlier comparison prescription, not the column-sampled MGKBA
definition in Eq. (8).

* `collision_in_scattering = full`. The internal line of the active part is the
  window's whole density matrix in the instantaneous basis, $\rho'=U^\dagger\rho U$
  ($U=1$ with dress none): hole $\rho'_{mn}$, particle $(1-\rho')_{mn}$, each
  off-diagonal pair at the symmetric prescription
  $$
  G_{\rm int}=\Psi'\,\tfrac12\{g(\epsilon),M\}\,\Psi'^\dagger,\qquad
  M=\rho'\ \text{(hole)},\ 1-\rho'\ \text{(particle)}, \tag{8a}
  $$
  i.e. $\tfrac12[g(\epsilon_m)+g(\epsilon_n)]$ for $M_{mn}$. It is
  contracted in the Σ sweep as $G=L\,\mathrm{diag}(w)\,R^\dagger$ on the panels $\Psi'$ and
  $\Psi'\rho'$ (Eq. 8c below; the $2n_A$ pseudo-bands $\Psi'(1\pm R)$ of the first version
  gave the same tables to 9×10⁻¹⁶), and the occupied projector of the static terms (the
  static limit subtracted from ΔΣ^h) is $\Psi'\rho'\Psi'^\dagger$. At a diagonal $\rho'$ it is
  the populations' sweep to rounding. The natural-orbital route (rotating the carrier to
  the eigenvectors of $\rho'$) is not used: it has no energy for the internal line.
* `collision_rate_intermediates = active` takes the window's own intermediate
  states in $\Gamma$ and $D$; frozen intermediate bands enter the coherent
  part only. Frozen populations never update, so including their dissipative
  transitions can make the window a particle source or sink.
* **Earlier trace benchmark.** The symmetric arithmetic-pair reference
  cancels its in/out traces on reciprocal synthetic poles. This is not proof
  of the corrected column split or its held propagation. CPU measurements
  (`tests/test_tdagw_collision_fullrho.py`, GN-PPM synthetic magnet, 2v2c
  window, random Hermitian ρ with spectrum in [0, 1]): diagonal ρ == populations
  to 2.8×10⁻¹⁶; both branches == the dense full-ρ reference to 1.1×10⁻⁸;
  covariance $U^\dagger\tilde\Gamma[\rho]U=\tilde\Gamma'[U^\dagger\rho U]$ at a unitary
  mixing degenerate states to 9×10⁻¹⁵ (populations alone: 3×10⁻²). Trace residual
  relative to the out-flux, on the fixture's poles made reciprocal: 2.5×10⁻⁸ (code,
  quadrature ε 10⁻⁷) and 3×10⁻¹⁶ (dense); left-index 0.34; populations + active
  5.6×10⁻²; full + all 0.18; populations + all 0.22. The fixture's own GN fit
  violates reciprocity at O(1) (residues 0.76, poles 0.70 relative), which leaves
  3.0×10⁻² for code and dense alike.

**Instantaneous basis, non-diagonal G builder and vertex pairing (2026-10-19, opt-in).**
The Markov step behind Eq. (8) must rotate the lines with the Hamiltonian that propagates ρ.
The GKBA hole line is $G^<(t,t-\tau)=i\,e^{-iH\tau}\rho(t-\tau)$ (particle: $1-\rho$), and the
interaction-picture step $\rho(t-\tau)=e^{iH\tau}\rho(t)e^{-iH\tau}$ turns it into
$i\,\rho(t)\,e^{-iH\tau}$: populations do not rotate, coherences do. In the legacy
frozen-screening path, `collision_basis = instantaneous` takes the basis from
the full-density reference material Hamiltonian,
$$
H^s=\mathrm{diag}(e_{\rm ref})+K_{W_0}[\rho-\rho^{\rm eq}]+V_{\rm dyn}
=U\,\mathrm{diag}(E)\,U^\dagger,\qquad
V_{\rm dyn}=\Delta M_H-\Delta M_H^{\rm static}. \tag{8b}
$$
The full current $\rho$, including coherences, enters $K_{W_0}$; no population
projection is made. `tdagw.dynsig.collision_snapshot_state` fixes $\rho$ and
iterates the coherent correction and eigenpairs together using bounded linear
mixing or Anderson acceleration with at most five history entries. Acceleration
does not relax the physical residual or add hidden self-energy sweeps. The
maximum matrix Frobenius residual over trial and k must satisfy
`collision_snapshot_tol_ev`; exhaustion of `collision_snapshot_steps` refuses
before installation. The accepted Hamiltonian is the input to the last sweep,
so its $U,E$, electronic phases and collision tables agree exactly while the
coherent map residual is bounded by the declared tolerance. The explicit
drive is excluded from $H^s$. The current mean-field coherences remain included.

This legacy convention has Hartree plus statically screened exchange. The
joint snapshot path instead reconstructs bare HF from authenticated kinetic/
ionic, Hartree and Fock source matrices; no $V_{xc}$ operator is required.
On the 516 decks, $e_{\rm ref}=\epsilon^G$
because no eqp file is set. At equilibrium the coherent correction is zero.
The full $\rho'=U^\dagger\rho U$ enters the internal line, and the final
coefficients rotate back to the fixed working basis. With joint screening,
the same instantaneous energies also enter both polarization bubbles at
every outer iterate. Saved basis receipts record the accepted $U,E$.
Frozen bands keep ψ and their 0/1 occupation and sit at $\epsilon^G(k)+\Delta_{\rm edge}(k)$,
$\Delta_{\rm edge}=e-\epsilon^G$ of the window's edge band on their side (lowest window band
below the window, highest above; 0 when the propagator runs on $\epsilon^G$); deep-state QP
corrections of an eqp file are not read. Their external states rotate, so the frozen part is
swept at every refresh; its static limit is t0's rotated (frequency independent).

*GEMM form.* The Σ executor builds $G_k(\mu,\nu,\tau)=L_k\,\mathrm{diag}(w(\tau))\,R_k^\dagger$ in
one GEMM per τ node, $L,R$ the $(N_\mu\times n_{\rm int})$ orbital panels at the centroids
(`ParentGreenCarrier.green_left`, `green_right`; None is the faces) and $w$ the branch's τ
weight at the inner index's energy. The incumbent is $L=R=\Psi$ with the occupation in $w$
(unchanged, bitwise). A non-diagonal line $C$ ($\rho'$ hole, $1-\rho'$ particle) carries no
occupation in $w$:
$$
G_L=\Psi' e^{-iE\tau}C\,\Psi'^\dagger:\ L=\Psi',\ R=\Psi'C;\qquad
G_R=\Psi' C\,e^{-iE\tau}\Psi'^\dagger:\ L=\Psi'C,\ R=\Psi'. \tag{8c}
$$
A state is live in a branch when $\max_n|C_{mn}|$ clears the occupation-weight floor
($10^{-5}$). The Σ box needs live states in both branches of every sweep, so the particle line
is the identity block minus the hole panels, $G_L[1-\rho']=\Psi'w\Psi'^\dagger-\Psi'w\rho'\Psi'^\dagger$:
the extra columns ($n_A$ per block, inserted after the window inside the Σ window) carry
$\Psi'$ or $\Psi'\rho'$, and the occupied projector of the static terms is $\Psi'\rho'\Psi'^\dagger$.
$\Psi'\rho'$ is one $k$-local product per refresh on the μ-sharded faces. $L$ and $R$ are
separate panels, so they may later sit at different $k$ (the phonon line $d_{q\nu}G_{k+q,k}$).

*Column sampling and vertex pairing.* In $\Sigma^<_{ij}(t,t-\tau)\,G^A_{jj}$ the external line closes on $j$ with
$e^{iE_j\tau}$ and the internal line $\rho'_{mn}e^{-iE_n\tau}$ puts $n$ on the $j$ vertex
($M^*_{jn}$): the phase is $e^{i(E_j-E_n-\Omega)\tau}$. Each external index pairs with the
internal index on its own vertex: $T_1=\Sigma_{ij}(E_i)$ takes $G_L$ and
$T_2=\Sigma_{ij}(E_j)$ takes $G_R$ (`collision_internal_pairing = vertex`:
two sweeps, each window block with an $n_A$ identity block). Eq. (8) uses
$R_p=\mathrm{cond}[1]$ and $R_h=\mathrm{val}[0]^\dagger$, without averaging
these tables. The earlier `symmetric` comparison keeps
$\tfrac12\{g(E),\rho'\}$, i.e. $\tfrac12(G_L+G_R)$, in both tables (Eq. 8a; one sweep on
$3n_A$ internal columns), which adds the two cross-vertex combinations. The q → 0 head pairs
each index with its own ($m=i$, $n=j$): $C_{ij}h_i$ and $C_{ij}h_j$ in either pairing.

*Particle number.* For the corrected column split, cancellation must be
checked on $C_{\rm raw}$ itself with reversed bosonic endpoints and a common
active state space. The earlier symmetric factors of arithmetic pair
averaging are not its proof. In fact, after column sampling the synthetic
reciprocal-pole gate gives a $1.39\times10^{-8}$ residual/flux for the vertex
code ($1.38\times10^{-15}$ for its dense reference), while the symmetric
comparison leaves $4.74\times10^{-2}$; a second full-$\rho$ state gives 0.381.
Collision-enabled propagation therefore refuses `symmetric`, populations-only
internal lines, and all-band rate intermediates: it requires the closed active
space, full $\rho$, and vertex ordering. The fresh-state and held-flow checks are
distinct, as specified above. Each refresh receipt therefore reports the
fresh generator's weighted number residual, its ratio to the mean in/out
flux, and separately the accumulated trace drift per unit time over the
preceding held interval. The first checks the snapshot collision algebra;
the second measures the finite-refresh-cadence approximation.

Current column-sampled CPU gates (`tests/test_tdagw_collision_instbasis.py`, the same fixture;
random Hermitian ρ with spectrum in [0, 1]): diagonal ρ' == populations for both line
builders to 4.0×10⁻¹⁶; vertex branch tables equal the dense vertex reference to
9.5×10⁻⁹; covariance at a unitary mixing degenerate states to 1.6×10⁻¹⁵; the
number residuals are as stated above; Gamma and D are Hermitian exactly; the basis map's
fixed point is 4.1×10⁻¹⁴, with U = 1 and E = e at ρ_eq.
End to end (`window_collision` on hand-built propagator operators, two trials): at ΔΣ = 0
and e = ε^G the instantaneous basis is the band basis to 1.4×10⁻¹³, and a refresh at
ρ_eq reproduces t0 exactly. The equilibrium path (no ρ) is bitwise the previous tree's,
the GW run's Σ(ω) cube included. Fixture walls: populations 1.04 s, symmetric 1.14 s,
vertex 2.24 s per sweep.

#### Full-matrix nonequilibrium screening {#keldysh-screening}

`collision_screening = joint` constructs the full microscopic snapshot
interaction with `collision_statistics = retarded_reference`.
`joint_frozen` uses the identical microscopic equilibrium interaction
throughout the run, providing a matched screening control. The native
response and self-energy retain every ISDF direction; frequency compression
is not a fixed spatial Coulomb projection. General signed/complex pole
residues have independent left and right factors. The passive equilibrium
GW shared-pole constructor retains its original contracts, owned by
[shared-pole theory](shared-pole-w-model.md) and its
[implementation](../architecture/shared_pole_model.md). The earlier
population-only `refreshed` and `shared_pole` TD modes remain refused.
The `finite_lag` and `lossless_bath` projected-model implementations below
are archived research helpers, not native propagation selections. Their
rank-one trajectories do not test full microscopic screening.

The snapshot stationary closure uses the one-sided transform
$\bar X(z)=\int_0^\infty d\tau\,e^{iz\tau}X(\tau)$ and its inverse/full
Fourier normalization first. Under the anti-adjoint negative-lag extension,
$\widehat X_\eta(\omega)=\bar X(\omega+i\eta)
-\bar X(\omega+i\eta)^\dagger$. The required identities are

$$
P^R=\bar P^>-\bar P^<,\qquad
P^R-P^A=\widehat P^>-\widehat P^<,\qquad
W^R=(1-vP^R)^{-1}v,\qquad
\widehat W^<=W^R\widehat P^<W^A,
$$

$$
\widehat W^>-\widehat W^<=W^R-W^A
=W^R(P^R-P^A)W^A.
$$

Here the products use full spectra at the same real frequency, with
$W^A=W^{R\dagger}$. Recover the positive-time $\bar W^<$ by a causal
Cauchy/Hilbert projection of $\widehat W^<$; replacing the full spectrum in
the product by $\bar P^<$ is incorrect. For the convention
$X(\tau)=\int d\omega\,e^{-i\omega\tau}\widehat X(\omega)/(2\pi)$,
the projection is
$\bar X(z)=i\int d\omega\,\widehat X(\omega)/[2\pi(z-\omega)]$,
with $\operatorname{Im}z>0$. Apply each declared regulator once: damping a
spectrum and adding the same $i\eta$ again changes the linewidth.

When $[H_s,\rho]\ne0$, the exact anti-adjoint identity exchanges both
time endpoints and therefore the endpoint density. Extending the frozen
positive-lag MGKBA line into a stationary spectrum is an additional closure;
it is not exact two-time RPA and does not imply passive response or positive
noise. Arbitrary coherences and population inversion need not satisfy the
current passive constructor's assumptions.

The joint representation must retain noise poles absent from $W_c^R$.
For example, $\rho=fI$, $0<f<1$, gives $P^R=0$ because all occupation
differences vanish, whereas $P^<$ has transition noise proportional to
$f(1-f)$. Thus $W^R=v$, $W_c^R=0$, but $\widehat W^<=v\widehat P^<v\ne0$.
A bank inferred only from retarded poles loses this noise. The CPU gates
include this counterexample, direct-frequency causal-projection agreement,
repeated poles, full-matrix endpoint ordering, and bare-bubble equilibrium
Bose residue weights. The last test is not a validation of screened FDT;
the finite-lag screened model has the known failure described below.

`tdagw.keldysh_rational` is the dense host reference for the algebra after
$P^<$ and $W^R$ have been supplied: it composes plain state-space arrays,
forms the sharp transfer, and performs the causal split by ordered complex
Schur decomposition plus a Sylvester solve. Its CPU gates compare the result
with the direct real-frequency Cauchy integral, exercise a repeated pole, and
retain the $\rho=fI$ noise pole when $W_c^R=0$. It refuses a pole on the real
axis until the caller declares a positive finite lag decay.

The archived projected snapshot helpers use a fixed Coulomb factor
$v_q\simeq F_qF_q^\dagger$ of declared `collision_density_rank`. In its
coordinates, $\widetilde P=F_q^\dagger P F_q$ and the bare interaction is the
identity. Centroid wavefunctions and $v_q$ remain distributed; a projected
band-pair block is streamed per q. Frozen bands and all active coherences
remain in both bubbles. A bounded full-block Loewner pencil uses both $P^<$
and $P^>$ at finite complex supports. Its stable modal candidates define a
causal basis; all matrix entries of both targets are refitted on that basis.
This is a new constrained fit, not a post-hoc shift of unstable poles. The
declared pencil-memory guard and final order bound are checked explicitly.
Retarded feedback and the
exact rational causal projection then produce $W_c^R,W^<$, retaining general
signed/complex residues, noise-only directions and the lesser $1/z$ tail.
The self-energy factors one exact common union realization for response and
noise; greater amplitudes are the sum of their matching outputs on that bank.
Independent held-out checks certify both bubbles, the screened response and
the dressed full-frequency noise. Unstable or unresolved poles refuse.
This reduces the joint polarization before the screening algebra rather than
fitting the screened retarded function alone.

Both native and archived snapshot models use the bare Hartree/Fock increment $\Delta h_{\rm HF}$,
not the reference screened-exchange kernel. With `collision_hamiltonian = hf`
the coherent fixed point is the specified material-Hamiltonian closure,
$H^s=h_{\rm HF}^{\rm eq}+\Delta h_{\rm HF}+M_H[\rho,H^s]$.
The authenticated kinetic/ionic plus equilibrium Hartree and Fock matrices
provide $h_{\rm HF}^{\rm eq}$. A DFT-prepared density need not commute with
this Hamiltonian or be a stationary collision state.
The explicit `reference` alternative uses
$H^s=\mathrm{diag}(e_{\rm ref})+\Delta h_{\rm HF}+
M_H[\rho,H^s]-M_H^{\rm eq}$ for comparisons anchored to the old reference.
It is a different effective-one-body model, not a claim of absolute HF/GW
self-consistency of DFT. Neither joint convention subtracts a static reduced-W
increment; doing so against a different full screened-exchange kernel would
leave an unintended density-dependent term.

The separate controls are density rank, rational order/support count,
fit-support height, collision regularization, and the declared exponential
finite lag decay of the lag kernels (historical key `collision_adiabatic_ev`).
Native `joint` and `joint_frozen` propagation requires the explicit
`collision_statistics = retarded_reference` selection; the default
`unspecified` refuses before GW setup. In the archived research helpers,
`finite_lag` acknowledges the earlier statistical model, not a
validated physical adiabatic limit. Its screened equilibrium FDT is known
not to hold in general, even when this decay tends to zero. The separate
`lossless_bath` choice below declares collective statistics directly; it
does not implement physical adiabatic preparation of arbitrary coherent
states.

An exact scalar counterexample needs no rational fit. Take zero-temperature
bare transition functions
$P^<(z)=g/(z+\Delta+i\gamma)$ and
$P^>(z)=g/(z-\Delta+i\gamma)$, with $v=1$ and positive $g,\Delta$.
Then $P^R=P^>-P^<$ and
$W^R=[(z+i\gamma)^2-\Delta^2]/[(z+i\gamma)^2-\Omega^2]$,
where $\Omega^2=\Delta^2+2g\Delta$. The implemented statistical construction
$W^<=\mathcal C_-[W^R(P^<-P^{<\sharp})W^{R\sharp}]$ has a
positive-frequency lower-pole residue

$$
r_+(\gamma)=\frac{g(\Omega-\Delta)(\Omega-\Delta-2i\gamma)}
 {4\Omega(\Omega-i\gamma)}
\ \longrightarrow\ \frac{g(\Omega-\Delta)^2}{4\Omega^2}>0.
$$

Equilibrium $T=0$ FDT requires zero positive-frequency lesser weight.
For $\Delta=1,g=1/2$ the nonzero limit is $0.0107233047034$;
the exact state-space regression checks both $\gamma=10^{-3}$ and $10^{-5}$.
Off-resonant lag-broadened bare noise is amplified at the collective poles,
so sending $\gamma$ to zero is a nonuniform statistical limit. This is a
known model failure, not fit error or insufficient numerical convergence.
At finite decay the supplied noise defines the stationary model; in a
lossless physical treatment collective occupations additionally require
initial correlations/switch-on information. No equilibrium or nonequilibrium
statistical shortcut inferred solely from retarded $W$ is implemented.
Subtracting the equilibrium collision term does not restore screened FDT.

##### Statistical preparation: resolved references and remaining closure

The counterexample above is a failure of an **equilibrium interpretation**,
not a proof that its noise cannot describe any preparation. A scalar RPA
oscillator makes the missing choice explicit. In units $\hbar=k_B=1$, set

$$
H_{\rm RPA}=\tfrac12p^2+\tfrac12\Omega^2q^2,\qquad
\Omega^2=\Delta^2+2g\Delta,\qquad Q=\sqrt{2g\Delta}\,q.
$$

Its positive-frequency response residue is $R=g\Delta/\Omega$. Every
stationary Gaussian state with occupation $N\ge0$ has

$$
W_c^R(z)=\frac{R}{z-\Omega}-\frac{R}{z+\Omega},\qquad
\bar W^<(z)=\frac{NR}{z-\Omega}+\frac{(N+1)R}{z+\Omega}.
$$

These are lossless functions evaluated for $\operatorname{Im}z>0$; no
arbitrary decay selects $N$. They obey $\bar W^>-\bar W^<=W_c^R$ for
every $N$, so that component identity does not specify the statistics.
Three distinct preparations give distinct answers:

* A correlated Gibbs state has $N=n_B(\beta\Omega)$, including $N=0$
  for the interacting ground state. It satisfies screened equilibrium FDT.
* An isolated, gapped adiabatic switch from a bare thermal oscillator carries
  $N=n_B(\beta\Delta)$ to the final mode. It gives the ground state at
  zero temperature, but generally not the Gibbs state at the original finite
  temperature. This finite-temperature distinction also occurs in
  [GKBA initial-state preparation](https://arxiv.org/html/1806.05639v1#S5).
* A sudden switch from the bare ground state, followed by dephasing of its
  anomalous covariance, has
  $N_{\rm quench}=(\Omega-\Delta)^2/(4\Omega\Delta)$.
  Consequently $RN_{\rm quench}=r_+(0)$ above. The same occupation is
  selected by a weak local damping/noise model tied to the bare oscillator.
  This equality is checked for this scalar example; it is not an equivalence
  theorem for arbitrary matrix or coherent snapshots.

The corresponding stable multi-mode reference uses
$H=\tfrac12p^Tp+\tfrac12q^TVq$, with $V=O\Omega^2O^T\succ0$.
Its symmetrized Gibbs covariance is

$$
S_\beta=\begin{pmatrix}
O\,\operatorname{diag}[(n_B(\beta\Omega)+1/2)/\Omega]O^T&0\\
0&O\,\operatorname{diag}[(n_B(\beta\Omega)+1/2)\Omega]O^T
\end{pmatrix}.
$$

It satisfies both $AS_\beta+S_\beta A^T=0$ and
$S_\beta+iJ/2\succeq0$, where
$A=\left(\begin{smallmatrix}0&I\\-V&0\end{smallmatrix}\right)$
and $J=\left(\begin{smallmatrix}0&I\\-I&0\end{smallmatrix}\right)$.
The small CPU reference checks these conditions, the microscopic Dyson
response for noncommuting two-channel couplings, residue KMS, and a numerical
adiabatic ramp. It specifies a preparation, rather than solving a Lyapunov
equation with an unspecified or bare diffusion matrix and calling its result
thermal. It does not cover unstable, zero-frequency or inverted RPA modes.

**A finite-width thermal reference.** If a bath model is intended instead
of lossless preparation, its fluctuation spectrum must also be specified.
For a genuine thermal reference it obeys

$$
\widehat P^<(\omega)=n_B(\beta\omega)[P^R(\omega)-P^A(\omega)].
$$

The matrix Dyson identity then implies screened FDT at the same real
frequency, including full local-field structure. Broadening transition
residues with their **bare-frequency** Bose weights is not this equation.
For commuting, fractional electronic populations one possible explicit
nonequilibrium bath model assigns each transition $i\to j$, $\Delta>0$,
the inverse temperature
$\beta_{ij}=\log[f_i(1-f_j)/(f_j(1-f_i))]/\Delta$ and applies the
frequency-dependent Bose factor to that transition's response. At thermal
Fermi occupations all these temperatures coincide. The equal-occupation
limit must be taken on the noise product, not by dividing by
$f_i-f_j$: finite fluctuation noise survives when the response vanishes.
Inversions give negative pair temperatures and require an independent
screened-stability check. This is a restricted bath model, not a derivation
for the full coherent density matrix, and is not wired into propagation.
At finite width its Bose/zero-temperature frequency factors are not in
general finite rational functions; a production pole representation would
need separately certified response **and** noise approximation.

**Why an equilibrium anchor alone is insufficient.** Replacing the
equilibrium noise by KMS noise and adding the old
$\delta\widehat P^<=\widehat P^<[\rho]-\widehat P^<[\rho_{\rm eq}]$
does restore equilibrium FDT algebraically, but need not define a positive
fluctuation spectrum even for diagonal populations. For a two-level system
with $f_{\rm low}=1-p$, $f_{\rm high}=p$, the proposed effective lesser
noise at $T=0$ and $\omega>0$ is proportional to

$$
2g\gamma\left[
\frac{p^2}{(\omega-\Delta)^2+\gamma^2}
+\frac{-2p+p^2}{(\omega+\Delta)^2+\gamma^2}\right].
$$

It is negative for arbitrarily small positive $p$ at fixed off-resonant
$\omega$. Dressing it by $W^R$ and $W^A$ cannot repair this scalar sign.
The missing correlated statistical response appears already at first order;
this reference subtraction is therefore not adopted as a physical fix.

**Coherences require a separate approximation decision.** A stationary
quasifree state of the frozen Hermitian $H$ requires $[H,\rho]=0$;
coherences within exactly degenerate subspaces are allowed. The upper-endpoint
line remains a defined approximation when this fails, but its stationary
Fourier extension need not be a positive-type correlation kernel. For
$H=\operatorname{diag}(0,\Delta)$ and
$\rho=\left(\begin{smallmatrix}1/2&c\\c&1/2\end{smallmatrix}\right)$,
its midpoint-frequency spectral determinant is nonnegative only if
$|c|\le\gamma/\sqrt{4\gamma^2+\Delta^2}$. The physical pure state
$c=1/2$ violates this condition at every finite $\gamma$ when $\Delta>0$.
Changing only collective occupations or imposing equilibrium FDT cannot
make that arbitrary-coherence stationary reconstruction exact.

A coherent alternative retains the **two** frozen time endpoints,
$g^<(t,t')=i\,e^{-iH(t-T)}\rho(T)e^{iH(t'-T)}$, and constructs screening
on an internal finite contour. With a declared virtual preparation time,
initial fluctuation covariance and switching profile, this is a snapshot
calculation: it need not access physical history or propagate another
physical state. A Gaussian reset plus quench, however, is the quench
preparation above, not correlated equilibrium. An equilibrium/correlated
preparation must be supplied explicitly and the internal window/switching
dependence converged. Silently discarding nondegenerate coherences in the
screening bath would instead be a different, secular-screening approximation.
Neither change has been made to the current runtime.

**Constructive two-endpoint reference (pure coherent kicks only).** The CPU
prototype starts from a correlated quasiboson RPA reference covariance
$C_{\rm ref}$, not a Wick-noise source alone. For equal-rank projectors
$P_0$ and $P=\rho$, one explicit local preparation convention is the direct
polar rotation

$$
B=PP_0+(I-P)(I-P_0),\qquad U_K=B(B^\dagger B)^{-1/2},\qquad
C_P=\mathcal R(U_K)C_{\rm ref}\mathcal R(U_K)^\dagger.
$$

Here $\mathcal R$ is the induced action on the fluctuation operators.
Set the auxiliary anchor $t_0=T$, $\rho(t_0)=P$, and assign $C_P$ there;
internal times can be integrated backwards from that anchor. A different
preparation endpoint must carry the consistently transported density and
covariance, not the current $C_P$ relabelled at an unrelated time.
The map refuses a singular overlap (a principal angle of $90^\circ$).
It is covariant under a simultaneous change of working basis, but is a
**declared preparation**: other unitaries with the same $UP_0U^\dagger=P$
can rotate $C_{\rm ref}$ differently. The density does not determine which
physical kick occurred. This map does not prescribe mixed/population-changing
states.

The smallest reference uses $H=\mathbf h\cdot\boldsymbol\sigma$,
$\rho=(I+\mathbf r\cdot\boldsymbol\sigma)/2$, a density vertex
$Q=\mathbf v\cdot\boldsymbol\sigma$, and interaction strength $\kappa$.
For the prescribed frozen background $\dot{\mathbf r}=2\mathbf h\times\mathbf r$,
the linearized ring fluctuation evolves with

$$
\dot F(t,s)=L(t)F(t,s),\quad F(s,s)=I,\qquad
L(t)=2[\mathbf h]_\times+
2\kappa(\mathbf v\times\mathbf r(t))\mathbf v^T,
$$

$$
J(t)=-2i[\mathbf r(t)]_\times,\qquad
LJ+JL^T=\dot J,\qquad
C(t,s)=F(t,t_0)C_PF(s,t_0)^T.
$$

Consequently $F(t,t_0)J(t_0)F(t,t_0)^T=J(t)$, and a positive
$C_P$ gives a positive block kernel over **both** time endpoints. For
this real Hermitian fluctuation basis,
$C_{ij}(t,s)=\langle\delta s_i(t)\delta s_j(s)\rangle$;
the lesser density kernel contracts the reversed ordering
$-i\kappa^2\mathbf v^TC(s,t)\mathbf v$. For
$\mathbf r_0=-\hat z$, $H=\Delta\sigma_z/2$,
$Q=\sqrt g\,\sigma_x$, $\kappa=1$, the correlated vacuum has transverse
block $\left(\begin{smallmatrix}\Delta/\Omega&-i\\i&\Omega/\Delta\end{smallmatrix}\right)$
and zero longitudinal variance. Its lesser kernel is
$-i(g\Delta/\Omega)e^{+i\Omega(t-s)}$, with no spurious positive-frequency
lesser weight. A coherent kick makes the statistical kernel nonstationary,
even in a case where the contracted retarded response remains stationary.
Independent ODE and time-discrete Dyson/Volterra constructions converge to
the same retarded kernel; the CPU test checks their step refinement.

This is **quasiboson** covariance positivity. In the two-level example its
squeezed variance $C_{yy}=\Omega/\Delta>1$ is not the exact finite-fermion
Pauli covariance at the same pure Slater density ($\sigma_y^2=I$).
The RPA reference neglects that fermionic correlation/depletion consistency;
no exact many-body representability claim is made. In a finite-time Keldysh
formulation $C_P$ supplies an explicit correlated boundary/homogeneous term.
Keeping only a bare Wick lesser bubble would put back the quench preparation.
Nor may the resulting $C(t,s)$ be collapsed into the old stationary
frequency-only screening algebra. An endpoint collision can instead integrate
the anchored $\Sigma^{</>}(T,T-\tau)$ directly with its column phases; its
finite internal window and numerical representation still need validation.

These references do not certify a full-coherence, nonequilibrium rho-only
statistical closure. The explicitly secular screening approximation below
provides a narrower implementation choice. Bosonic covariance positivity
and the separate nonsecular electronic collision positivity problem must
both be assessed before interpreting damping in a propagated trajectory.

##### Retarded-only equilibrium reference {#retarded-reference}

`collision_statistics = "retarded_reference"` is an explicitly approximate
equilibrium-like GW member. At each instantaneous Hamiltonian snapshot it
uses the complete density matrix in the ordered retarded bubble
$P^R=\bar P^>-\bar P^<$ and screens
$W^R=(I-vP^R)^{-1}v$ in the full microscopic density space. Only this
response is fitted and independently checked. Screening does not pinch the density.
The input response has the declared finite lag decay
`collision_adiabatic_ev`; collision evaluation uses its separate $\eta$.

The full-density retarded bubble is evaluated through the native complex-time
Green products and momentum FFT response. The ordered endpoint is
$\Psi\rho e^{-iE\tau}\Psi^\dagger$, not
$\Psi e^{-iE\tau}\rho\Psi^\dagger$. The terms quadratic in $\rho$ cancel
between greater and lesser bubbles, but the two remaining density insertions
are independent; one cannot replace the reverse product by the stationary
conjugation shortcut. All supplied virtual bands remain, including frozen-active
transitions. Frozen states have identity rotation and a per-side edge scissor.

An exact finite-band identity clarifies causality without introducing
propagated auxiliary states. With transition index $t=(k,b,d)$, vertex map
$T_{\mu t}$ carrying the spin/k/metric factors, define

$$
\Delta_t=E_{kb}-E_{k-q,d},\qquad
K_\rho=\bigoplus_k\left(I\otimes\rho_{k-q}^{T}-\rho_k\otimes I\right).
$$

The prescribed upper-endpoint bubble is
$P^R(z)=T K_\rho(z-\Delta+i\gamma)^{-1}T^\dagger$.
Density therefore belongs to the output map in this realization; moving
$K_\rho$ across the resolvent would change the coherent ordering unless
$[K_\rho,\Delta]=0$. Bare transition poles stay at $\Delta_t-i\gamma$.
Woodbury gives

$$
W_c^R(z)=vT K_\rho(z-A_D)^{-1}T^\dagger v,\qquad
A_D=\Delta-i\gamma+T^\dagger vT K_\rho.
$$

This identity is not a proposal to form the large transition matrix in
production. Physical $0\preceq\rho\preceq I$ makes $K_\rho$ Hermitian with
norm at most one, but does not make the feedback $A_D$ Hermitian, passive
or stable. For example, a valid inverted two-level projector has
$P^R(z)=-2\Delta/[(z+i\gamma)^2-\Delta^2]$; positive scalar $v$ gives
$(z+i\gamma)^2=\Delta^2-2v\Delta$, which can have an upper-half-plane pole.
Consequently, neither positivity of $\rho$ nor a fitted growing pole is
evidence that the current microscopic Dyson response is stable or unstable.
Suspect poles need independent current-response checks and refinement.

**Alternative numerical realization: transition Galerkin (internal bridge proof only).**
The preceding identity suggests preserving the known bare generator instead
of continuing sampled $W$ into a global unconstrained rational bank. Write
$A=T K_\rho$, $B=T$ and $D=\Delta-i\gamma I$, with exactly the same spin,
k-weight and packed-ISDF metric factors as above. For an orthonormal
transition-space basis $Q$ (not a spatial projector), define

$$
A_r=AQ,\quad B_r=BQ,\quad
D_r=Q^\dagger\Delta Q-i\gamma I,\qquad
P_r(z)=A_r(z-D_r)^{-1}B_r^\dagger.
$$

Its bare poles retain imaginary part $-\gamma$ because
$Q^\dagger\Delta Q$ is Hermitian. No square root or positivity assumption
on $K_\rho$ is needed. The corresponding reduced Dyson feedback is

$$
M_r=D_r+B_r^\dagger vA_r,\qquad
W_{c,r}(z)=vA_r(z-M_r)^{-1}B_r^\dagger v.
$$

All microscopic output ports remain in $A_r,B_r$; density stays on the
output side of the right-index energy phase. For $Q=I$ these are exact
finite-band identities, including the complex column-shell self-energy.
For smaller $Q$, they are an additional numerical approximation. Exact
full moments are not automatic: for example, containing the full ranges
of $B^\dagger$ and $\Delta B^\dagger$ suffices for the first two bare
moments, whereas a narrow tangential seed need not contain those ranges.

Current full microscopic Dyson samples can supply interpolating transition
states without fitting a $W$ continuation. For right and left port directions
$u,w$, the exact states are

$$
x=(z-D)^{-1}T^\dagger W_{\rm full}(z)u,\qquad
y=(\bar z-D^\dagger)^{-1}K_\rho T^\dagger
W_{\rm full}(z)^\dagger w.
$$

If both belong to $\operatorname{range}Q$, the reduced full interaction
interpolates the corresponding all-port right/left actions and the shared
tangential derivative $w^\dagger W'(z)u=-y^\dagger x$. In the dual state,
$K_\rho$ acts **before** the adjoint diagonal resolvent, reading products
right to left; interchanging them changes the coherent endpoint ordering.
These are numerical basis seeds, not propagated bosonic history or an
additional fluctuation-statistics model.

This is **not** a passivity or collective-stability guarantee. An arbitrary
orthogonal $Q$ can produce spurious collective gain by failing to preserve
the relation between $K_\rho$ and $\Delta$. Even a stationary binary-density
control should project its positive/negative occupation-gap sectors
separately; general coherent $K_\rho$ does not commute with $\Delta$.
The subspaces must also transform covariantly under the exact transition
exchange/conjugation relating $q$ and $-q$, including closure at self-negative
q. Independent microscopic $P$, $W$, dielectric and complex-$\Sigma$ checks
remain required. Genuine inverted-state gain is retained, not projected
away; no realization can cure a true microscopic pole above the specified
collision evaluation height.

`gw.transition_galerkin` implements internal native $T/T^\dagger$, $K_\rho$
and $D$ actions, distributed CGS2/TSQR basis construction, crossing closure,
stationary binary-density sign sectors and one-parent full-port modal
factors. The microscopic transition interaction is not formed. The basis
is k-major and whole-k sharded; the combined native spin/k normalization
is retained, including both wavefunction spin components. The tiny tests
in `tests/test_native_transition_galerkin_bridge_cpu.py` compare coherent
complex-time response with the existing native FFT for scalar/spinor input,
and exercise actual native factors through complex column-shell $\Sigma$,
dual derivative interpolation, repeated seeds, signed gain, q-crossing and
TSQR/capacity guards. Their ports are untruncated **tiny fixture** ports;
they do not certify a 978-port material calculation. The separate
`tests/test_full_density_transition_galerkin_cpu.py` supplies literal algebra
oracles.

The held basis may retain an exactly zero occupation-gap sector at a
stationary binary projector. Only its **current transfer** is omitted before
modalization, and only with exact coordinate-sector support, zero projected
free-generator crossblocks, and zero output coupling. The resulting lower
triangular generator has exactly the live-sector transfer even at coincident
dark eigenvalues/Jordan blocks. No computed pole is inspected or discarded;
the complete held basis remains available, and a coherent changed density
uses the full current generator. Tiny tests check both that exact identity
and reactivation; this is not a gain filter or a production admission.

This is **not a production route**: no all-q bank, runtime/default switch,
material response admission, cost/time-to-accuracy result or dynamics is
established by those CPU proofs. It is not a cheap substitution: at 36 k points and 160 bands
there are $36\times160^2=921\,600$ transition coordinates **per q**. A single
complex128 basis with $K=1000$ costs 14.75 GB globally per parent (before
orthogonalization workspace and other fields). Exactly zero spectator
pairs may be omitted, but coherent pairs cannot be dropped from their
diagonal occupations. With the present four-band coherent window, exact
same-filled frozen-pair elimination leaves 303,552 live coordinates; a
32-pair tile pads the carrier to 304,128. At $K=1000$, retaining one basis
for each of the 20 canonical parents on a 6x6x1 grid (four self-negative q)
costs about **97.3 GB globally**, before
vertices, duplicate basis/update buffers, seeds, provider workspace and
source arrays. Holding all parent bases during a Hamiltonian fixed point
and admitting their aggregate resources remain unresolved. One-parent
capacity receipts explicitly do not admit that full retention or a material
runtime. Numerical $P/W$/dielectric and complete complex-$\Sigma$ refinement
remain mandatory before any trajectory interpretation.

Full microscopic Dyson samples/actions feed a two-sided tangential Loewner
realization, using the source GW shared-pole support/direction recipe. Its
frequency pole bank has the form
$W_c^R=L\,\mathrm{diag}[(z-p)^{-1}]R^\dagger$ with full-ISDF endpoint factors.
The general pencil uses SVD rank reduction and a general eigensolve, not a
positive Gram whitening or a forced real-pole model. The response may have a
$1/z$ asymptote; no even-in-frequency or $z^{-2}$ constraint is imposed.
Only the reduced pencil may be staged for bounded per-parent host LAPACK;
full microscopic matrices/factors stay on their native distributed owners.
The source recipe owns frequency supports and pole capacity, not the
archived `collision_density_rank` or 48-mode prototype bound.

Screening and the complete complex self-energy are rebuilt from the current
snapshot. The Hamiltonian solve must meet its unscaled physical residual;
no unconverged last iterate is installed. Refusals retain the scalar
residual history. A W-fit tolerance alone is not a self-energy convergence
certificate: support/direction refinement must also converge complex
$\Sigma_{ab}(E_b+i\eta)$, $M_H$, $\Gamma$ and $D$.

For the causal pole expansion $W_c^R(z)=\sum_\ell R_\ell/(z-p_\ell)$,
the self-energy uses the zero-temperature reference branches

$$
B^-_\ell=-\theta(-\operatorname{Re}p_\ell)R_\ell,\qquad
B^+_\ell=\theta(\operatorname{Re}p_\ell)R_\ell,\qquad
\theta(0)=\tfrac12.
$$

Thus the internal hole line carries $\rho_{mn}B^-_\ell$ and the particle
line $(I-\rho)_{mn}B^+_\ell$, each with denominator
$E_b-e_n-p_\ell+i\eta$ at external column $b$. The complete complex
matrices enter Eq. (8): the Hermitian part contributes to the instantaneous
Hamiltonian, and the anti-Hermitian/injection terms define the collision.
No diagonal-density, diagonal-self-energy or real-self-energy replacement
is made. For a usual equilibrium pair of poles $\pm\Omega$ with residues
$\pm R$, these are the standard empty-boson particle/hole GW branches.

This branch choice is a reference ansatz for computing $\Sigma$ from
$W^R$, not a reconstruction of exact nonequilibrium lesser fluctuations.
There is no independent noise fit, bath-temperature solve, FDT test or
noise-positivity admission gate. Signed and complex residues are retained.
Fluctuations absent from the retarded response are intentionally absent
from this approximation: for example, $P^R=0$ gives $W_c^R=0$ and no
correlation self-energy from this member, even when fractional occupations
would carry noise in a fuller nonequilibrium treatment. Hartree/Fock and
the electronic lines still use the full density.

The retarded fit must reproduce held-out response samples. Its bare
polarization retains the declared lower-half-plane transition poles.
Causality of the feedback response does not require BIBO stability: a
positive-lag kernel may grow as $e^{\operatorname{Im}p_\ell\tau}$.
For this reference member, the self-energy integral is admitted only when
$\eta>\max_\ell\operatorname{Im}p_\ell$, so its explicit
$e^{-\eta\tau}$ factor makes the integral converge. Receipts report the
growth bound and positive evaluation-height margin; poles are never moved.
Supports interpreted as Laplace samples must also lie above this growth
bound. This finite-memory prescription does not assert a stationary
Fourier spectrum for a growing response. Distinct
q/$-q$ responses are paired algebraically, and self-inverse-q polarization
fits obey their crossing relation before Dyson feedback. The symmetric
zero-frequency branch convention preserves this pairing. Frozen and
updated controls use the same reference branch rule. Number drift and
Pauli bounds remain measured properties of the electronic propagation;
this approximation makes no general complete-positivity claim. The
separate nonequilibrium statistical constructions below remain research
alternatives and are not prerequisites for this reference calculation.

##### Lossless transition-bath prescription {#lossless-transition-bath}

`collision_statistics = "lossless_bath"` declares a stationary, lossless
screening bath at each snapshot. This is an additional approximation: only
the **screening** density is pinched into the instantaneous Hamiltonian's
spectral subspaces,

$$
\rho^{\rm bath}=\sum_E\Pi_E\rho\Pi_E.
$$

Within an exactly degenerate subspace it may be diagonalized without
changing the Hamiltonian; off-diagonal entries between different energies
are excluded from both response and statistical screening. The propagated
density, Hartree/Fock update, internal electronic lines and column-sampled
MGKBA collision retain the full density matrix. This is therefore not a
repair that preserves the earlier arbitrary-coherence screening bubble.
The discarded norm must be reported so its size can be compared with the
frozen/updated screening difference. Screening occupations outside $[0,1]$
by at most $10^{-12}$ are bounded only to make their logarithms well defined;
the correction is recorded, and larger violations refuse. This numerical
domain correction does not modify the propagated density or establish
electronic Pauli bounds.

For each ordered transition $j=(k,b; k-q,d)$, write
$e_j=E_{bk}-E_{d,k-q}$, $d_j=f_{d,k-q}-f_{bk}$, and let $v_j$ denote its
projected Coulomb vertex, including the square root of the k/spin weight.
The screening bath resolves transition gaps only to $10^{-12}$ Ry:
$\bar e_j=0$ when $|e_j|\le10^{-12}$ Ry, and $\bar e_j=e_j$ otherwise.
This is a declared statistical approximation, so a tiny computed splitting
does not silently select an infinite positive or negative transition
temperature. Receipts record the count and maximum gap adjustment.
Hamiltonian and electronic self-energy energies are unchanged; held-out
response checks still compare the final bank with the original gaps.
For resolved nonzero gaps the lossless projected polarization and pair
temperature are

$$
P_f^R(z)=\sum_{j:\bar e_j\ne0}\frac{d_jv_jv_j^\dagger}{z-\bar e_j},\qquad
\beta_j=\frac1{\bar e_j}\log\frac{(1-f_{bk})f_{d,k-q}}
 {f_{bk}(1-f_{d,k-q})}.
$$

The occupations specify **transition reservoir** temperatures, not a single
temperature fitted to the density. For populations that are Fermi--Dirac
on the resolved screening spectrum, all finite-gap $\beta_j=\beta$ and
degenerate levels have equal populations. Arbitrary occupation differences
at original tiny gaps are not such a thermal state after gap resolution.
Inverted transitions retain negative $\beta_j$ and gain.
The latter is a finite-level transition prescription, not a thermal state
of an unbounded boson at negative temperature.

Let the screened correlation response have distinct nonzero real poles
$\lambda$ and Hermitian residues $R_\lambda$. The lesser and greater
residues are prescribed by

$$
L_\lambda=R_\lambda S_\lambda R_\lambda,\qquad
G_\lambda=L_\lambda+R_\lambda,\qquad
S_{\lambda,f}=\sum_{j:\bar e_j\ne0}\frac{d_j n_B(\beta_j\lambda)}
 {(\lambda-\bar e_j)^2}v_jv_j^\dagger. \tag{8f}
$$

For a bath with only resolved finite-gap transitions,
$S_\lambda=S_{\lambda,f}$; the degenerate-reservoir term is specified below.

This is the weak colored-transition-bath pole prescription. For an
isolated collective pole it follows by giving the input transitions a
common small width $\gamma$: the off-resonant source is
$2\gamma S_\lambda+o(\gamma)$, and the dressed squared pole yields
$2\gamma R_\lambda S_\lambda R_\lambda/
[(\omega-\lambda)^2+\gamma^2]$. The declaration takes this pole limit
before evaluating the electronic collision. It does not assert that the
finite-width input source is positive for arbitrary complex vertices or
that a finite-width rational bank obeys exact frequency-by-frequency FDT.

Thermal correctness is a matrix identity. The residue of Dyson feedback
satisfies

$$
R_\lambda[-\partial_zP^R(\lambda)]R_\lambda=R_\lambda.
$$

Consequently a common $\beta$ on the resolved screening model gives
$L_\lambda=n_B(\beta\lambda)R_\lambda$ and
$G_\lambda=[n_B(\beta\lambda)+1]R_\lambda$ for the **screened** poles.
At zero temperature the positive-frequency lesser residues vanish,
including the scalar example that gives the spurious $0.0107233047034$
weight with bare Lorentzian noise. This is a lossless residue KMS/FDT
statement. A subsequently applied collision evaluation width is a
separate approximation. `collision_adiabatic_ev` does not enter this
lossless bank and is recorded as unused; the collision evaluation uses its
own declared regularization.

The production reduction must preserve that same identity. On a signed
transition branch, take $s=\operatorname{sign}d_j$,
$C_j=\sqrt{|d_j|}v_j^\dagger$, an orthonormal Krylov basis $Q$ for the
diagonal transition energies, $D_Q=Q^\dagger DQ$ and $C_Q=Q^\dagger C$.
The reduced response is
$sC_Q^\dagger(z-D_Q)^{-1}C_Q$. Its bath source is projected on the
**same** subspace:

$$
S_{\lambda,Q}=sC_Q^\dagger(\lambda-D_Q)^{-1}
 Q^\dagger\operatorname{diag}[n_B(\beta_j\lambda)]Q
 (\lambda-D_Q)^{-1}C_Q.
$$

Using the unreduced transition derivative with approximate screened poles
would break thermal normalization. This signed Hermitian reduction keeps
bare poles real; screening stability is checked after feedback. Arbitrary
complex poles may not be projected onto the real axis.

An optional preliminary Galerkin projection groups transitions into energy
bins before the Krylov stage. For a rank-one branch with coupling $c_j$,
the normalized vector in a bin is $Q_j=c_j/\sqrt{\sum|c_j|^2}$, its energy
is $\sum|c_j|^2e_j/\sum|c_j|^2$, and its colored occupation at a collective
pole is $\sum|c_j|^2 n_B(\beta_j\lambda)/\sum|c_j|^2$.
Averaging $\beta_j$ before applying $n_B$ would change the prescription.
For a matrix coupling, the local orthonormal basis projects the energy,
coupling and the full matrix $Q^\dagger\operatorname{diag}(n_B)Q$.
Composing this projection with Krylov preserves the same-subspace thermal
identity and avoids an all-transition by final-order basis. Held-out
microscopic response checks certify the complete projection, including
the binning approximation.

Equal occupations are handled before division: as $f_b,f_d\to f$,
$d_j n_B(\beta_j\lambda)\to f(1-f)e_j/\lambda$. Transitions with
$d_j=0$ therefore require their own noise representation. They contribute
this continuous limit to $S_\lambda$ **and** carry bare-pole noise. For a
noise-only spectral residue $F_e=\sum_{j:e_j=e}f(1-f)v_jv_j^\dagger$,
the latter has screened lesser and greater residue
$W^R(e)F_eW^R(e)^\dagger$. These are separate peak families of the weak
limit, so keeping only one loses fluctuations. In particular, when
$\rho=fI$, $W_c^R=0$ while both noise channels survive. A collective pole
coincident with a coupled noise-only bare pole can make this limit
singular; it needs an explicit collision/coincidence gate rather than a
small denominator replacement.

Resolved zero gaps have two explicitly different statistical preparations.
Writing $a_j=f_{bk}(1-f_{d,k-q})$, aggregate them as

$$
D_0=\sum_{\bar e_j=0}d_jv_jv_j^\dagger,\qquad
N_{\rm res}=\sum_{\bar e_j=0,\ d_j\ne0}a_jv_jv_j^\dagger,\qquad
N_{\rm static}=\sum_{\bar e_j=0,\ d_j=0}f_j(1-f_j)v_jv_j^\dagger.
$$

The response is $P^R=P_f^R+D_0/z$. Unequal-population degenerate
transitions are declared weak shot-noise reservoirs: their lesser input
has $2\gamma a_jv_jv_j^\dagger/(\omega^2+\gamma^2)$, with greater
coefficient $a_j+d_j=(1-f_{bk})f_{d,k-q}$. Both coefficients are
nonnegative, and the commutator is preserved. Their nonzero collective
lesser contribution is $R_\lambda N_{\rm res}R_\lambda/\lambda^2$;
thus $S_\lambda=S_{\lambda,f}+N_{\rm res}/\lambda^2$.
This preparation does not infer a temperature by dividing by a zero gap.

Equal-population zero-gap transitions instead retain **exactly static**
Wick covariance, a literal zero-frequency noise pole. They have no
collective-tail contribution. Treating this static covariance as the
previous weak Lorentzian reservoir would add such tails and break finite-T
thermal residue KMS. The two preparations are distinguished before taking
limits; the zero-gap, equal-population and weak-width limits do not commute.
For a thermal state of the resolved model there are no unequal-population
zero-gap channels, so the common-$\beta$ screened-pole KMS result remains
valid, including its zero-temperature limit.

To retain zero-frequency noise after response cancellations, let the
columns of $N$ span $\ker D_0$. The zero-frequency screened limit is the
Schur-complement expression

$$
W_0=N[N^\dagger(I-P_f^R(0))N]^{-1}N^\dagger,\qquad
L_0=G_0=W_0(N_{\rm res}+N_{\rm static})W_0^\dagger.
$$

For full-rank $D_0$, $W_0=0$. For $D_0=0$, this retains ordinary screened
static noise even when individual response contributions cancel. Signed
eigenfactors of the aggregate $D_0$ supply its response; keeping separate
canceling zero-transition realizations would create spurious dark/Jordan
states. A singular restricted inverse or a genuine unresolved collective
zero mode refuses. The reported numerical rank tolerance only removes
certified roundoff of the aggregate response.

Each admitted pole bank must pass full-matrix lesser/greater residue,
component identity and q/$-q$ reciprocity checks, including retained
noise-only poles. Under the paired transition map
$(e,d,v)\mapsto(-e,-d,v^*)$, $\beta$ is unchanged and
$n_B(-x)=-1-n_B(x)$. Thus
$R_{-q,-\lambda}=-R_{q,\lambda}^*$ and
$S_{-q,-\lambda}=[S_{q,\lambda}-\partial_zP_q^R(\lambda)]^*$
give $L_{-q,-\lambda}=G_{q,\lambda}^*$. A reduced calculation needs
paired subspaces or an independent check of this identity; separate
accurate response fits alone do not establish noise reciprocity.
The degenerate reservoir has the same crossing because reversing a
transition sends $d\mapsto-d$ and $a\mapsto a+d$.
The runtime derives one member of a distinct q/$-q$ pair algebraically
from the representative and checks its response against the actual partner
transitions. At a self-inverse q, it explicitly averages the screened bank
with its crossing partner:

$$
W_{c,s}=\tfrac12(W_c-\mathcal M W_c),\qquad
\bar W_s^<=\tfrac12[\bar W^<+\mathcal M(W_c+\bar W^<)],\qquad
\mathcal M X(z)=-\operatorname{conj}X(-\operatorname{conj}z).
$$

This is a convex mixture of two paired screened bath approximations.
It preserves positive lesser/greater residue matrices and thermal residue
KMS by linearity, and enforces crossing exactly. Its final response need
not be the Dyson response of the original compressed polarization; the
actual microscopic held-out response check is therefore repeated on the
mixed bank. Derivative normalization refers to its constituent reduced
Dyson banks, not a claimed single-polarization identity for the mixture.
Nonthermal baths are not automatically positive after
the projection; $L_\lambda\succeq0$ and $G_\lambda\succeq0$ must be
checked, and unstable or unresolved poles refused. These bath checks do
not prove positivity of the nonsecular electronic collision operator.
The electronic number residual, both gain/loss boundary matrices and
the propagated density eigenvalues remain independent acceptance checks;
no clipping or Pauli-bound guarantee is implied.

The archived `snapshot_collision.q_sigma_operators` evaluates the same ISDF
product in projected vertices, with column energy $E_b$ and internal phase
$e_n$, and admits a general biorthogonal pole expansion only after checking
its transfer reconstruction. Frozen states enter the coherent self-energy;
real collision destinations remain the active window.
This small projected executor uses direct momentum sums on the host, not the
large-grid FFT self-energy executor. It streams q and bounds k/pole contraction
temporaries, but no large-grid scaling claim follows from this implementation.

### 1.6 The active space: band window or disentangled subspace (proposed) {#active-space}

The split refresh of §1.5 uses the window as the active space of the
intermediate-band sum too: with $W$ frozen only the window's own intermediate
states change, so a refresh sweeps them alone, and the frozen bands' part is
the t0 one. With refreshed screening the frozen part changes with $W$ for every
band; restricting its change to the bands next to the window
(`collision_frozen_bands`) keeps the rates (16 + 16 bands: 2.8×10⁻³ of their
maximum, the rule tolerance) but not $\mathrm{Re}\,\Sigma$ (81 meV, 2.4 % of the
QP gap, at a 5 % carrier transfer): the screened exchange of every occupied band
responds to $\delta W(0)$.

$A$ is a band-index range. Where a band of other character enters $A$'s
energy range in part of the zone, the window holds that character at those $k$
only, and one Bloch character moves between $G_{\rm frozen}$ and
$G_{\rm active}$ across the zone; $\delta\chi_0$ and the $\Delta\Sigma$ basis are
then not smooth in $k$. The alternative is a disentangled subspace
(Souza–Marzari–Vanderbilt) of an outer window $O\supset A$ with a frozen
inner window $I$:

$$
\tilde w_{\alpha k}=\sum_{n\in O}\psi_{nk}V_{k,n\alpha},\qquad
V_k^\dagger V_k=\mathbb 1_{N_w},\qquad
\{\psi_{nk}\}_{n\in I}\subset\mathrm{span}\{\tilde w_{\alpha k}\}, \tag{9}
$$

with $N_w=56$ for CrI3 (20 Cr-3d + 36 I-5p spin orbitals, the present window
size).

Unchanged: the products $\tilde w_\alpha\tilde w_\beta^*$ lie in the span of
the $O\times O$ band products, so a $\zeta$ fitted over $O$ represents them;
the centroid values rotate as $\psi(r_\mu)V$; $W$ in $\mu$ space is untouched;
every contraction of §1.2–1.4 takes any orthonormal basis.

Changed:

* $P_k=V_kV_k^\dagger$ commutes with $H^{\rm QP}$ only inside $I$. The active
  Hamiltonian $V^\dagger\mathrm{diag}(\varepsilon_O)V$ is not diagonal, and its
  eigenvalues differ from the band energies outside $I$; the field-free $P$–$Q$
  block is dropped, at the level at which the field coupling to frozen bands
  is dropped now.
* $G_{\rm frozen}=G_{\rm eq}-PG_{\rm eq}P$ keeps the cross terms
  $PG_{\rm eq}Q+QG_{\rm eq}P$, frozen at equilibrium; they vanish for the bands
  inside $I$. $G_{\rm eq}$ is still composed from every band (Eq. 5), and
  $\delta G=G_S[\rho(t)]-G_S[\rho^{\rm eq}_S]$ with
  $\rho^{\rm eq}_S=V^\dagger fV$ keeps $\Delta\Sigma[\rho^{\rm eq}]=0$ exactly.
* Inputs over $O$. $V_k$ on the TD grid in the gauge of the TD states
  (htransform's, not the DFT states of a fine NSCF such as run 517). By
  projection only: the leading $N_w$ eigenvectors of the Löwdin projector
  $\langle\psi_n|P_{pd}|\psi_m\rangle_{n,m\in O}$ with $I$ enforced, a band
  operator that `bandstructure.orbital.band_operators` forms on the parents
  (channel table) and `interpolate_band_operator` carries onto htransform
  states. The SMV minimization needs the overlaps $M_{mn}(k,b)$ as well. The
  centroid values carry no inner products, so $V$ cannot come from them.
  $O$ must lie inside htransform's fitted band range; $\zeta$ and the
  downfolded $W$ over $O$ (DFOLD's `w88_144` covers $A$ only); QP energies
  over $O$. Dipoles: $v$ over $O$ rotated, $r_{ij}$ re-derived in the
  eigenbasis of the active Hamiltonian with the near-degeneracy mask.
* Validation: BSE ≡ TD in linear response (§2) and the §6.1 identity, repeated
  in the $S$ basis.

**Measured for CrI3** (run DEV/835; DFT 30×30, 200 bands, Löwdin
projections of `projwfc.x`). $A=[88,144)$ is an isolated group at every $k$:
global gaps 7.08 eV below and 1.50 eV above (k-local 143|144 gap ≥ 1.52 eV,
smallest at Γ). Every state of $A$ has Cr-3d + I-5p weight ≥ 0.742 (bands
92–143 ≥ 0.911); every band outside has ≤ 0.221. No band of other character
enters $A$ anywhere in the zone, so a Wannier construction for this group
needs no disentanglement and spans $A$. The projection-only p–d subspace
differs from $A$ by $d^2=56-\sum_is_i^2=1.22$–$1.40$ (outer window
$[88,200)$). Of this, 0.91 lies in the four Cr-4s(+I-5s)–I-5p σ directions
(cosines 0.85–0.93): bonding in bands 88–91, antibonding in bands 144–149. The
band-edge states (124–133) change active weight by ≤ 2.0 %. The partial-zone
hazard is the QP order at the upper edge. At Γ, GW lifts band 143 by 1.93 eV
and band 144 by 0.46 eV (eqp1, $N_\mu$ 9000), which closes the 1.52 eV
separation to 55 meV; eqp0 swaps up to four states at 8 of 10 coarse $k$.
This was checked on the 8×8 eqp grid only. For CrI3 the band window stays;
the QP-order margins at both edges on the TD grid belong in the run's gate.

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
and the populations are $O(E^2)$, so linear response is exactly TD-aGW, the
full BSE of the kernel page, with either `dress`: `"none"` keeps $U=1$,
$f=\mathrm{diag}\rho$, and `"instantaneous"` dresses with
$K\delta\rho^{\rm car}=O(E^2)$ (§1.1), so $U-1$, $e-\varepsilon^G$ and $\delta W$
are $O(E^2)$. Measured (§8) for $\rho=V\rho^{\rm eq}V^\dagger$,
$V=e^{-i\epsilon A}$ with $A$ valence–conduction: the dynamic correction scales
as $\epsilon^{2.00}$ dressed, and is zero on the pure coherence $\epsilon X$,
$X=-i[A,\rho^{\rm eq}]$; the coherent twin, which shifts $e$ at $O(E)$ through
$K\delta\rho^{\rm coh}$, scales as $\epsilon^{1.02}$. On CrI3 516 ($A$ the
$x$-dipole valence–conduction block) the exponents are 2.000 and 1.002.

TDMODEL's outlook ([td_agw.md](td_agw.md), outlook) keeps $K\delta\rho^{\rm coh}$
and adds $\Sigma^{\rm dyn}[f]-\Sigma^{\rm dyn}[f^{\rm eq}]$ for the populations.
With `dress = "none"` the present scheme is that split plus one term: the
coherent part is $K\delta\rho$ exactly, the population part is
$\mathcal W[\delta W](-\rho^{\rm eq}_{\rm tot}-\delta\rho^{\rm car}+\tfrac12)$,
and $-\mathcal W[\delta W](\delta\rho^{\rm coh})$, the screened exchange of the
coherences with the changed $W$ (second order, $\delta f\,\delta\rho^{\rm coh}$),
is the term their split drops.

## 3. Conservation, and what is lost {#conservation}

### 3.1 Kept

* for the static and frozen members: $\mathrm{Tr}\,\rho_k$ per $k$, Hermiticity
  of $\Delta\Sigma$ (measured $\le1.2\times10^{-17}$), unitarity of the
  commutator flow (the spectrum of $\rho$ at $\Gamma=0$). The dynamical members
  keep the Hermiticity of $\rho$. The collision requires separate fresh-state
  trace and held-flow drift checks (§1.5); neither Pauli bounds nor exact
  trajectory particle conservation follows from the Hermitian split;
* $\Delta\Sigma[\rho^{\rm eq}]=0$ by reference subtraction. The raw collision
  must be tested for stationarity independently; a subtracted counterterm can
  impose it for the selected frozen coefficients;
* covariance under a rotation of the window basis: the populations are those
  of the instantaneous basis, not of the band gauge (the red twin that reads
  them in the fixed basis misses the reference by 5%);
* the coherent static member has linear response = TD-aGW = BSE (§2), dressed
  or not. The dynamical collision adds the full linearized kernel of Eq. (8).

### 3.2 Lost

* **Time locality.** $\Sigma(t,t')$ is replaced by a function of $\rho(t)$ alone;
  no memory, no non-Markovian dephasing.
* **Φ-derivability.** With $W$ updated, $\Delta\Sigma$ is not $\delta E/\delta\rho$
  of any functional of $\rho$ (the GW Φ functional is a functional of the
  two-time $G$, and Eq. (2) is not its stationarity condition), so the energy
  is not conserved exactly. The coherent frozen-W member is Φ-derivable from
  $\tfrac12\mathrm{Tr}\,\delta\rho K\delta\rho$ and conserves its energy.
  This statement does not extend to the Markov collision with frozen
  screening, finite broadening, projected active spaces and held rates.
* **Memory in the rates.** The collision term (§1.5) uses a frozen Hamiltonian
  and is held between refreshes; there is no non-Markovian dephasing. The
  corrected full-matrix electronic line does retain coherence-driven
  scattering, unlike the population approximation.
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
| GN-PPM fit | `gw.ppm_sigma.pole_fit_plan` (:956) + `fit_pole_model` (:993) = `fit_ppm` (:808), the driver's call; `minimax_screening.fit_gn_ppm_from_wc_pair` (:309) | $(q,\mu_X,\nu_Y)$ | `screening_update` (:1006): Eq. (6) at both supports, then the fit (§9 item 4) |
| shared-pole W | `gw.shared_pole_screening.screen_shared_poles` (:443), `shared_pole_constructor.construct_shared_poles` (:57) | bank and model stores on disk | §9 |
| Σ τ sweep | `gw.ppm_tau_kernel._sigma_subtile_kernel` (:283), `_get_sigma_kij_kernel` (:482); `ppm_pipeline.compute_ppm_sigma_pipeline` (:691); window entry `ppm_sigma.window_sigma_c` (:1611) | parent Greens $\times$ $W(\tau)$, window projection | GN-PPM / shared-pole Σ |
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

**GN-PPM** (`07_gw_gnppm_bundle`: the 05 deck with `compute_mode = gn_ppm`). A
GN-PPM run persists $W(0)$ but neither $W(i\omega_p)$ nor its pole model, so
the check (`tests/tdagw/_dynsig_gnppm_516_case.py`) replays the GW driver in
the same process on the 07 deck and reads both supports, the fit plan and the
poles at the fit seam. The replay (branch) matches the 07 arm (run on the
base tree) bitwise: all 15 `sigma_mnk.h5` datasets and `sigma_diag.dat`.
$\chi_0^{ff}+\chi_0^{fa}+\chi_0^{aa}$ at both supports (10 static and 24
probe nodes on [0.110, 6.130] Ry, $\omega_p=2$ Ry, probe max error
$8.7\times10^{-9}$) through `screening_update` gives the run's $W(0)$ to
$1.3\times10^{-14}$ and $W(i\omega_p)$ to $3.4\times10^{-15}$ (relative, max over
36 q; q = 0: $9.6\times10^{-15}$, $1.3\times10^{-15}$), and `W0_qmunu` to
$1.1\times10^{-14}$. The poles match the run's fit with identical valid masks
(1,984,518 of 34,574,400 modes invalid, 5.7%): $B$ to $3.5\times10^{-12}$,
$D$ to $5.3\times10^{-11}$, $\Omega$ to $3.0\times10^{-8}$ (relative to the largest live
value); on the live modes $\max|D|/\max|B|=2.0\times10^{-3}$. $\chi_0^{fa}$ is 33% (static)
and 17% (probe) of $\max|\chi_0|$. `window_sigma_c` on these poles and
`exchange_sigma` (heads $W_h$, $v_h$) give the window block of `sigma_mnk.h5`
(both on-shell tables, off-diagonals included, float64,
`common.units.RYD_TO_EV`): $\Sigma^c$ to $2.5\times10^{-14}$ eV ($|\Sigma^c|\le5.03$ eV),
$\Sigma^x$ to $1.7\times10^{-13}$ eV ($|\Sigma^x|\le24.9$ eV), their sum to
$\le1.9\times10^{-13}$ eV (the file's `sigma_total_kij_ev` is $\Sigma^x+\Sigma^c+V_H$
exactly). The diagonal matches `sigma_diag.dat` to $9.5\times10^{-7}$ eV (six
printed decimals). On the run's own poles the sweep gives $1.8\times10^{-15}$ eV.
The COHSEX check's $4.7\times10^{-6}$ eV floor is absent: this check uses the
writer's constant and reads float64. The twins: $\chi_0^{fa}$ dropped
0.34 eV ($W(0)$ off by $1.6\times10^{-2}$), the pole order swapped ($D\to-D$)
8.2 meV, the frozen bands omitted from $G$ 3.59 eV.

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
| 516 6×6, 2v2c (160), GN-PPM (`07` replay, warm) | 36, 978 | 0.92 s (the run's χ0 stage, both supports) | 0.15 s (`screening_update`: both solves + fit) | 3.1 s (`window_sigma_c`) | **~4.2 s** | 2.95 ms |
| 516 6×6, 2v2c (160), shared-pole (`04`), full refit (measured, below) | 36, 978 | bank 10.0 s + moments 0.7 s | in the bank | 5.1 s + rebase 4.1 s | **34.3 s** (constructor 13.8 s) | 2.95 ms |
| 471 8×8 (898), N_μ 3000 (`14_bseprod_c3000`) | 64, 3000 | 1.1 + 2.4 s | 2 × 0.2 × 6.4 = 2.6 s | 16.6 + 11.7 s | **~34 s** | — |
| 471 8×8 (898), N_μ 6000 (`15_bseprod_c6000`) | 64, 6000 | 3.7 + 8.7 s | 2 × 0.9 × 6.4 = 11.5 s | 86.4 + 12.4 s | **~123 s** | — |
| 30×30, 42v × 14c (`34_bseprodb_df3000r8_w88_144`) | via the 8×8 parent | as 8×8 | as 8×8 | window on 30×30 | **8×8 + densify** | 95.4 ms |
| 518 12×12 (898), 30v × 30c (`32_bseprodb_k12_c3000`) | 144, 3000 | 2.4 + 14.6 s | 2 × 0.3 × 7.6 = 4.5 s | 46.3 + 13.9 s | **≤ 82 s** | 309.3 ms |

Measured, the static member (`screening="static_rpa"`: Eq. (5) cross-gap,
Dyson on 36 q, Eq. (7) on the window, 160 bands) costs 0.41 s per warm call on
516 at 4 A100, 139× the frozen `apply`; the exact same-side enumeration adds
0.54 s. The GN-PPM row is measured (4 × A100, warm compile cache): the
07 replay's χ0 stage, and the 516 case's `screening_update` and
`window_sigma_c` walls (first call 7.2 s); the TD split
(`dynsig.chi0_split`) itself was not timed. "Window Σ" is the Σ τ sweep plus the other Σ stages except exchange and
Hartree, which $K\delta\rho$ already carries, measured on the runs' own
160-band Σ window (`nval 130`, `ncond 30`): for a 30v × 30c window it is an
upper bound. The Dyson column scales the runs' wedge solves (10 q at 8×8, 19
at 12×12) to the full zone. The frozen
`apply` walls are TDPROP's (516 2v2c) and TDMODEL's (30×30, 12×12) at b = 1;
at 30v × 14c on 6×6 k it is 9.0 ms.

**Earlier arithmetic-operator baseline, the collision term on origin/main** (516 6×6, 160
bands, $N_\mu$ 978, 4 × A100, warm; τ = distinct Σ τ nodes of the sweep):

| refresh (2v2c window [128, 132)) | τ | wall per refresh |
|---|---|---|
| frozen W, full sweep on the GW run's on-shell list (incumbent) | 206 (275 once both branches are live) | 3.26–3.29 s (4.23 s) |
| frozen W, split: active part (frozen part 48 τ, 1.09 s, once at t0) | 44 (88) | 0.95–1.00 s (1.56 s) |
| refreshed W, split: δχ0 at 0 and $i\omega_p$ | — | 1.85 s |
| refreshed W: both Dyson solves + the run's pole fit | — | 0.40 s |
| refreshed W: frozen part on W(t), window plan (GW list: 207 τ, 3.26 s) | 48 | 1.10 s |
| refreshed W: active part on W(t) | 44–111 | 0.98–2.06 s |
| refreshed W, total | | 4.3–5.4 s |

On the 10v14c window [120, 144): incumbent 206 τ, 3.34 s; split active 68 τ,
1.29 s (182 τ, 2.82 s hot; frozen part 76 τ, 1.49 s); refreshed δχ0 5.16 s and
Dyson + fit 1.58 s. The frozen part's τ count is set by the window's own
energies (48 or 76 τ against 181–207 τ on the GW list); the active part's by its
live branches. δχ0 on the window is the bubble's change only ($\chi_0^{ff}$ is in
$W_0$): its same-side sum grows with $n_A$ (2v2c 1.85 s, 10v14c 5.16 s).

**Earlier arithmetic-operator baseline, the shared-pole collision term** (branch
`feat/td-agw-dynsig-spm-2026-10-18`; 516 6×6, 160 bands, $N_\mu$ 978, K = 1761
poles at every q, 4 × A100, warm; medians over the refreshes of the driven runs):

| refresh (2v2c window [128, 132)) | wall per refresh |
|---|---|
| refit, V1 (5×10⁸ V/m, 0.25 fs, 31 refreshes) | **34.3 s** |
|   bank: direct 2.1, Dyson 3.4, lines 2.1, rule nodes and host 2.4 | 10.0 s |
|   moments | 0.7 s |
|   constructor: Gram 8.0, directions 2.0, writer 2.9, other 0.9 | 13.8 s |
|   window Σ: frozen part 2.1 (52 τ), active part 3.0, census 0.4 | 5.5 s |
|   rebase: $C[\rho^{\rm eq}]$ on $W(t)$ | 4.1 s |
| refit, V1, 3×10⁹ V/m (hot sweeps, 130 τ) | 35.6 s |
| refit, V2′ ($\rho^{\rm eq}$; bank 18.7 s with 156 rule nodes, constructor 18.6 s) | 46.9 s |
| frozen shared-pole W, sweep only | 2.9 s |

The refit is 72% of a refresh. Against the GN-PPM refresh (4.3–5.4 s) it is
7×, against the frozen-W split (1.46 s) 23×. The constructor wall varies
between runs (13.7–18.1 s median; the Gram reduction 8–12 s). At a 0.25 fs
cadence an 8 fs drive spends 31 × 34 s ≈ 18 min in refreshes on 4 A100.

The 30×30 W update runs on the 8×8 parent (the 30×30 NSCF has 200 bands, not
the 898 of the sums) and is densified as W already is
(`bse_densify`, srlr/c1); the window populations are folded to the parent grid
through the htransform. **Every-call refresh is unaffordable**: at 80 ΔΣ calls
per fs (cfm4, 50 as, two corrections) the 12×12 refresh is up to 1.8 h per fs
on 4 A100, up to 265× the frozen `apply`. Memory: with no point-group symmetry the
$G_{\rm eq}$ parent tiles at 12×12 are $144\,(2\cdot3000)^2\cdot16$ B = 83 GB
each (20.7 GB/rank at P4), so the refresh needs P16, k-chunked Greens
(`feat/bse-wterm-chunking`) or a C3-symmetric drive (σ± at normal incidence on
M ∥ z keeps C3: one third).

### 7.1 Refresh policy {#refresh}

The column-sampled collision refreshes its electronic lines and, for `joint`,
both screening bubbles inside the converged material snapshot. `joint_frozen`
holds the same reduced equilibrium screening. A fixed positive refresh
interval tracks pure coherences, which may evolve with unchanged band
populations. Fresh-state number residual and trace drift during the preceding
held interval are reported separately. `collision_screening = refreshed` and
`shared_pole` are refused at entry by
`GATE dynsig_collision_keldysh_screening` (§1.4). The $W$-refresh policies,
costs and trajectories described below are prior arithmetic-operator
baselines from retained development machinery, not currently runnable
collision configurations.

$\delta W$ follows the populations, which follow the pulse envelope and the
carrier relaxation, not the optical period. The production design refreshes
Eq. (6) and the pole model when $\max_k|\delta f|$ has moved by a set
tolerance since the last refresh, and in between applies
$\Delta\Sigma=K\delta\rho+\Delta\Sigma^{\rm dyn}_{\rm last}$ (its coherent part
exact, its population part held). The refresh is a host-side decision at a
chunk boundary, so the jitted step stays one program. The collision term
(§1.5) refreshes this way now: `[lorrax] collision_refresh_fs` (cadence) and
`collision_refresh_df` (population tolerance), one line per refresh in the log's
`Collision refreshes` section. At one refresh per fs,
a 30 fs 12×12 drive costs at most 30 × 82 s + 2400 × 0.31 s ≈ 53 min on 4 A100.
A refresh can happen only at a chunk end, and `chunk` counts samples: at
`chunk = 5` and `dt_measure = 50` as a chunk is 0.25 fs, the shortest cadence (a
0.125 fs cadence then refreshes every chunk, as 0.25 fs does); 0.125 fs needs
25 as samples.

**Prior refreshed-screening baseline** (`collision_screening = refreshed`,
currently refused; §1.5). The retained machinery refreshes
Eq. (6) and the pole model with the rates. Its refresh line adds the walls (δχ0
at both supports, Dyson + fit, frozen and active sweeps), the invalid-mode count
and its flips (since the last refresh and against $\rho^{\rm eq}$), the largest
$|\Delta\Omega|$ over the modes valid in both, $\max|\delta W|/|W_0|$ at both
supports and the degenerate pairs taken at their principal value.
The subtracted form keeps $C[\rho^{\rm eq}]$ at the t0 rates (W0); with $W(t)$
the regularization tails of $C[\rho^{\rm eq}]$ (28 meV on the diagonal at
516) change with $W$ and are no longer cancelled. `window_collision(rebase=True)`
takes $C[\rho^{\rm eq}]$ at the rates of $\rho^{\rm eq}$ on $W(t)$ (one more split
sweep at $f^{\rm eq}$, 2.1 s on 516 2v2c); on the 516 drives it removes the
population model's measured tail-induced carriers (§1.5). Rebasing is an
optional counterterm approximation, not a formal default for a future
nonequilibrium lesser interaction.

**Prior shared-pole screening baseline** (`collision_screening = shared_pole`,
currently refused; §1.4). The retained machinery refits
the whole W stage at each refresh and used rebased $C[\rho^{\rm eq}]$ on
$W(t)$ by default (`collision_rebase = auto`). `on` with the admitted frozen
screening remains refused. Its source is the GW driver on `collision_source`
(the SPM deck; it may differ from `bundle_input` only for shared_pole). On 516
the refreshed run is one program per run (no recompiles over 15–63 refreshes);
at $\rho^{\rm eq}$ the collision stays zero through the refreshes (max
deviation $5\times10^{-14}$, held flow $3\times10^{-11}$ meV). Cadence
($\max_t|\Delta n_c|/\max n_c$, 8 fs drives, against 0.125 fs with 25 as
samples): 0.25 fs is within 0.9% at 5×10⁸ and 3×10⁹ V/m, 0.5 fs within 2.4% and
0.9%; P(t) within 0.35%. The refreshed W moves P(t) by 4.6% against the frozen
shared-pole W at 3×10⁹ V/m, 0.07% at 5×10⁸ V/m; the held $\Delta\Sigma^{\rm h}$
jumps by at most 13.8 meV at a refresh (3×10⁹ V/m; GN-PPM refreshed screening:
77 meV), the pole count stays K = 1761 and no gate refuses (V1).

## 8. Validation {#validation}

### 8.1 Full-microscopic implementation evidence {#full-microscopic-evidence}

The 20 October 2026 CrI3 516 proof used all 978 ISDF ports, all 36 q points,
all 160 virtual bands, and the full coherent density in the four-band
propagation window. Its input was a saved strong-drive snapshot at 9.5 fs
from an earlier projected-model run, not a new full-microscopic trajectory
or a converged material-Hamiltonian fixed point. The new screening
evaluation itself used no spatial projection. At source tree
`e58a455620e21903e8925271a0652c7a5da3b369` (commit `9021da7`), the model
**refused before self-energy evaluation**: the q=0 rational fit contained
a pole with growth 12.078764 Ry, above both the 0.1 eV collision evaluation
height and the response supports. This is an unadmitted fitted pole, not
evidence of physical gain or an unstable exact Dyson response. That run
did not retain the failed pencil before refusal; it cannot support a
retrospective pole-residue diagnosis or a damping claim.

The measured single-node P4 phase walls were:

| phase | seconds |
|---|---:|
| source preparation | 77.77 |
| scalar response planning | 18.28 |
| full-microscopic response stream | 302.44 |
| Dyson solves | 3.41 |
| direction selection | 7.99 |
| pencil construction | 9.47 |
| serial canonical host reductions | 700.17 |

The stream evaluated 8162 Green pairs for 43 frequency targets. A separate
five-second process sample found the rank-zero main thread actively using
one CPU core while the other ranks waited; it did not find a collective
deadlock. It does not establish the BLAS thread configuration for every
phase. These measurements motivate exact response blocking and concurrent
per-parent reductions, not a reduced spatial model or a relaxed pole gate.
The allocation was released after refusal.

The source, logs, failure receipt, accounting and process sample are
preserved in
`/home/jackm/mgkba_campaign_20261019/full_microscopic_v3_20261020/`.
The earlier all-rank P4 hsuite pass belongs to source tree `507b2793`, not
this final material-proof tree. Focused CPU oracles check the full complex
branch algebra and tiny native builder; neither substitutes for admission
of the full material model, self-energy convergence, or matched frozen-W
and updated-W trajectories. No full-microscopic 10 fs damping result is
established by this evidence.

A follow-up q=0 diagnostic at tree
`5cfab98ede3d8a78888e91f908720f272607872a` (commit `ee2ec85`) retained the
1226-by-1316 frequency pencil before refusal. The saved density was physical
($\lambda(\rho)\in[0.002574,0.996949]$). Its 1224-pole model again had growth
12.080938 Ry. All 24 native fitted/held sites had relative errors below
1.28%, but the additional collision-height sites failed by up to 46.1%.
Independent current microscopic Dyson evaluations near the suspect pole
disagreed with the fit by relative errors 41.16 and 29.08, not percentages.
This demonstrates an unsupported rational continuation, not a physical-gain
diagnosis. Full-port $C_1,C_2$ relative defects were 0.8743 and 0.0438;
these are diagnostics, not independently enforced moment gates.

That bounded diagnostic stopped after one q model: host reduction took
33.94 s (31.42 s eigensolve), not a measured 20-fold speedup of a complete
map. Its response stream took 313.55 s with the new block quadrature but
the old extensive added-verification geometry. Step cost was 0.53222 GPUh;
no self-energy or physical-time propagation was performed. The pencil,
exact-probe receipts and accounting are preserved in
`/home/jackm/mgkba_campaign_20261019/full_microscopic_v4_20261020/`.
The next diagnostic, tree `20537b99d2d320a1ed20cd7b4f6c2542ff85ab89`
(commit `991ac38`), retained all native sites, skipped only structurally
zero Green products, and trained three additional collision-height sites
inside the active energy span, with two independent interlaced held sites.
Its 3042-node stream took 60.58 s, versus 313.55 s above. This is a measured
response-stage improvement with changed numerical sampling geometry, not a
complete-refresh speedup or a self-energy convergence result. The one-q
step used 0.25556 active GPUh and its owned allocation was released.

That 1242-mode model still refused: maximum growth was 0.221185 Ry and the
worst full-matrix sample error was 8.47%, above the unchanged 2% criterion.
Full-port moment defects were 0.09214 and 0.002901 (diagnostics only).
Nearby microscopic Dyson comparisons had relative errors 0.4246% and
0.3090%, but **these do not confirm a physical growing pole**: the suspect
rank-one pole contributes only 0.4324% and 0.3065% of the corresponding
exact full-W norm. A weak spurious pole on a large smooth background is
compatible with both observations. Independent dielectric invertibility
or pole/null-vector checks, not total-W agreement alone, are needed.

A bounded CPU rank study on this captured pencil at tolerances 1e-6 and
1e-7 moved the most-growing pole substantially; both still failed the
Laplace criterion. Its physical Gram norms were unresolved. First-eight
descriptor-coordinate residue norms in the offline report must not be
interpreted as physical residues. These results establish neither gain
nor an admitted model. Captures, source hashes, probes and accounting:
`/home/jackm/mgkba_campaign_20261019/full_microscopic_v5_20261020/`.

The subsequent balanced ordinary eigensolver was checked on the captured
V4 pencil against the existing generalized-eigenvalue result: eigen time
fell from 46.34 s to 3.14 s on the same two-thread CPU environment, with
the same growing modes and nearest-pole differences below 2.6e-10 Ry for
the twelve largest-growth modes. This is a reduced-pencil benchmark,
not a material-screening admission. No full native material self-energy
or matched 10 fs trajectory is established by these follow-ups.

The next P4 diagnostic at commit `a91b532` (tree
`e4a2719fe7d2221a36609473593ceca4318e4f1c`) evaluated the **native dielectric
at the suspect fitted pole itself**, without an additional collision eta.
For the same strong snapshot, its physical 978-port block had
$\|\epsilon\|_F=31.4239$, $\|\epsilon^{-1}\|_F=31.1933$, and measured
inverse residual $4.32\times10^{-14}$. The inverse-residual bound, including
its FP64 allowance, gave $\sigma_{\min}(\epsilon)\ge0.0320581$; both identity
padding defects were zero (980 packed ports). This fitted pole is therefore
not a pole of the **computed** microscopic response. It does not establish
the absence of other physical poles or certify the response quadrature.

A stationary DFT-reference projector control used the same source energies,
$U=I$ and active occupations $(1,1,0,0)$, with no frozen-edge shift. Its
1218-mode fit also refused (growth 0.410557 Ry, maximum sample defect 12.70%).
At that fitted pole the computed dielectric again had a verified positive
smallest-singular-value lower bound, 0.0319715, with inverse residual
$3.77\times10^{-14}$ and zero padding defects. Thus the specific spurious
growth is not contingent on the strongly driven coherent state. This is a
screening-implementation control, **not** a self-consistent GW ground-state,
equilibrium-FDT, or raw-collision-stationarity test; no material self-energy
was reached in either diagnostic.

Both steps together used 0.43778 active GPUh; the owned reservation used
0.56889 GPUh and was released immediately after capture. Their complete
small receipts and the capacity-bounded frequency pencils are preserved in
`/home/jackm/mgkba_campaign_20261019/full_microscopic_v6_20261020/`.

A conditioning-only CPU replay of the captured V5 pencil with frequency
equilibration still produced growing fitted modes (maximum 0.363658 Ry).
The old capture lacks authenticated finite/infinity unit masks, so this
all-one-input-unit replay is not an exact test of the new producer policy.
It provides no model admission or self-energy result. Further production
fit-tolerance changes are not justified by this replay alone.

Separately, `gw.shared_pole_reference` supplies a verification-only exact
conversion of the authenticated equilibrium positive pole bank to the signed
ordered store consumed by the new full-matrix self-energy executor. Negative
frequency factors use the transpose at minus q, including the left/right
factor swap. Three CPU checks cover unequal factors, odd logical counts,
native full-density FFT self-energy and its frozen head exactly once.
This **source-frozen** control is not the production `joint_frozen`
initialization, does not update screening, and does not establish a material
Hamiltonian fixed point or trajectory. Its saved-state harness is
`tests/tdagw/source_frozen_saved_state.py`; material results, if obtained,
must be reported separately from these small algebra checks.

That source-frozen material control subsequently completed at commit
`62d9804` (tree `a9a10a3952d3a9b58f7926dcc1266539035b2998`), with
the same saved 9.5 fs strong-drive input, all 978 ports/36 q/160 virtual
bands and four real destination bands. Source preparation took 79.24 s,
exact positive-to-signed conversion 7.07 s, and the **new ordered
full-density self-energy** 10.77 s. It used 12 windows/96 tau pairs at
relative rule target 1e-4, collision eta 0.1 eV, and the unchanged source
head. The raw fresh trace was -2.373e-9 Ry against flux 0.00491225 Ry,
or 4.830e-7 relative. This does not meet the campaign's stricter 1e-8
fresh-number guard; no guard was relaxed and rule refinement remains needed.

The rate operators were indefinite: minimum eigenvalues of D, Gamma-D
and Gamma were -24.86, -54.23 and -2.53 meV. The input density was physical
and strictly interior, with eigenvalues [0.002574,0.996948]. These operator
signs preclude a general held-affine positivity guarantee; they are not a
measured density excursion or a trajectory. This calculation updated
neither W nor the material Hamiltonian fixed point, and establishes no
screening-induced damping, new band gap or 10 fs evolution.

An earlier attempt at `05fc7cb` stopped before the new self-energy because
a root-only general-store validator entered an all-rank broadcast. The
fix separates the host cache lookup from its collective wrapper; regression
checks retain file-generation rehash and corruption refusal. The failed
and successful P4 steps used 0.26778 and 0.13222 active GPUh (0.40000 total),
and both allocations were released. Authenticated raw complex branch
tables, operators, source/snapshot hashes and accounting are preserved in
`/home/jackm/mgkba_campaign_20261019/source_frozen_v2_20261020/` and the
combined status in `source_frozen_20261020/COMBINED_STATUS.json`.

### 8.2 Acceptance checks and earlier-member evidence

The corrected column-sampled MGKBA collision requires an independent dense
positive-time or direct-resolvent reference with noncommuting $H_s$ and
$\rho$. Acceptance must separately establish the slot conversion
$R_p=\mathrm{cond}[1]$, $R_h=\mathrm{val}[0]^\dagger$; $D$ and $\Gamma$;
the identity between Eq. (8) and the raw in/out expression; Hermiticity;
diagonal-state reduction; and covariance under unitary rotations within
degenerate energy blocks. Deliberately wrong hole slots and arithmetic
averaging must fail the noncommuting reference. Static-limit and coherent
double-counting checks must use the same full $\rho$ and $W_0$.

For reciprocal synthetic poles with matching active spaces, the fresh-state
weighted trace residual must agree with dense cancellation to the declared
quadrature tolerance. Also measure density eigenvalue bounds, raw equilibrium
residual versus $\eta$, and trajectory number drift versus refresh cadence;
none of these follows from a fresh-state trace test. CPU checks do not
establish end-to-end GPU correctness. Full-matrix screening additionally
requires the response/noise and convergence gates in §1.5's
[screening contract](#keldysh-screening). The measurements below predate the
corrected column split and do not by themselves satisfy these acceptance
criteria.

`tests/test_tdagw_dynsig.py` (synthetic 3×2×1 k, 9 bands = 3 frozen occupied +
2v2c + 2 frozen empty, two spinor components, a NumPy reference sharing no code
with `tdagw.dynsig`; CPU, 2×2 mesh of host devices):
$\chi_0^{ff}+\chi_0^{fa}+\chi_0^{aa}$ = dense $1.4\times10^{-12}$ (twin without
$\chi_0^{fa}$ 0.47); equilibrium COHSEX $5.9\times10^{-16}$; $\delta\chi_0$ of
an excited state $4.7\times10^{-13}$ (twins: no Pauli blocking
$4.8\times10^{-3}$, no same-side pairs 0.81); total ΔΣ $1.5\times10^{-13}$;
dressed (Eq. 2 after 32 steps against a NumPy plain iteration to $10^{-15}$)
$4.0\times10^{-14}$ on the driven state and $1.9\times10^{-9}$ on a hot-carrier
state (twins: fixed-basis populations 0.052, the coherent dressing 0.19); the
dressing's residual $9.0\times10^{-16}$ (driven) and $1.1\times10^{-11}$ (hot),
the sea-rotation twin's $2.1\times10^{-4}$ (driven, flagged); the dynamic
correction of a weak unitary drive $\propto\epsilon^{2.00}$ dressed,
$\epsilon^{1.02}$ coherent, $\epsilon^{2.00}$ undressed; literal static limit
$1.4\times10^{-14}$ (twin without coherences 1.75); frozen mode on a pulse =
`isdf_model` exactly; `static_rpa` departure $\propto E_0^{2.2}$. The same
script on 4 A100 (one rank per GPU, the runtime's communicator stack) reproduces
every row to round-off.

`tests/tdagw/_dynsig_516_case.py` (CrI3 516 6×6 COHSEX, 2v2c, DFT $H_0$, 4
A100): the loader's window ψ equal the TD model's exactly and the stored $W_0$
equals its $W$ at q ≠ 0 to $7\times10^{-16}$ (at q = 0 the TD model carries its
own head); §6.1 as stated there; a 6 fs pulse (rk4, 20 as, 1200 ΔΣ calls)
with `screening="frozen"` gives P(t) and ρ(t_f) equal to `isdf_model`'s to the
last bit; the literal Eq. (1) in a rotated instantaneous basis at ρ(t_f) equals
$-K^W\delta\rho$ to $1.1\times10^{-11}$ (twin without coherences 12.2); the
static member as in §4 and §7. Dressed (§1.1, cross-gap member): the dynamic
correction of a weak $x$-drive scales as $\epsilon^{2.000}$ (coherent twin
$\epsilon^{1.002}$, undressed $\epsilon^{2.000}$), is zero on the pure coherence
and is 0.51× the undressed one; at populations $4.8\times10^{-4}$ and
$1.2\times10^{-2}$ the dressing converges (residual $\le6.6\times10^{-16}$)
where the sea-rotation twin is flagged (residuals $1.2\times10^{-5}$ and
$3.3\times10^{-4}$, ΔΣ NaN). On 5 fs pulses of $5\times10^{7}$ and $10^{8}$ V/m
(rk4, 50 as) P(t) dressed minus undressed scales as $E^{3.000}$ (relative
$3.7\times10^{-7}$, $1.5\times10^{-6}$); at $10^{9}$ V/m (populations
$8.1\times10^{-4}$ at $t_f$) the dressed run is finite, and every dressed call
converged (residual at $\rho(t_f)$ $\le8.9\times10^{-16}$) at 1.19× the
undressed wall.

`tests/tdagw/test_spm_refresh_cpu.py` (CPU, NumPy references): the pole
census against a direct NumPy sum over a synthetic store, the occupation
duck, recipe-site equality, the "| height" field of the support sites, the
deck keys and their refusals, the refit receipts and refresh lines, and one
capacity ledger per Σ evaluation. `tests/tdagw/_spm_refresh_516_case.py` (516,
4 A100): the refit at $\rho^{\rm eq}$ is bitwise the run's model (factor,
poles, K); $W^c(0)$ of the model against the exact $W^c$ $1.6\times10^{-5}$, and
$V + W^c(0)$ against the COHSEX bundle's $W_0$ $2.1\times10^{-6}$ max-relative;
with the shared-pole options off, the GW outputs of the `04` and `07` decks and
the GN-PPM collision case are bitwise origin/main's.

## 9. Refactors for a production member {#refactors}

Each item makes a GW stage callable per refresh from in-memory arrays, with no
file I/O and no re-planning (lines at `feat/td-agw-propagator-2026-10-16`
7402a200b; the new entry points of items 4–5 at
`feat/td-agw-dynsig-sigma-2026-10-17` e7d86cd6e):

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
   pole model as arrays; the stores stay the GW driver's. **Done for GN-PPM:**
   `gw.ppm_sigma.screening_update` (:1006) runs Eq. (6) at both supports by
   `solve_w` (`meta=None`, `pref=1`; δχ0 donated) and then `fit_pole_model`
   (:993) on the plan of `pole_fit_plan` (:956); it returns
   `(PPMBuildResult, (W(0), W(iω_p)))`. The GW driver calls the same
   `pole_fit_plan` + `fit_pole_model` (`ppm_pipeline` :813, :830), so the TD
   update and the run share one fit; the run's outputs are bitwise unchanged
   (§6.1). **Done for the shared-pole member** (branch
   `feat/td-agw-dynsig-spm-2026-10-18`): `tdagw.dynsig.spm_refit` reruns the
   source's W stage (`gw.shared_pole_screening`, the entry seam of
   `gw_window_source`) at the instantaneous occupation table and returns the
   model on the store the window Σ reads; the bank and constructor are the GW
   driver's, the recipe is re-resolved at the state's census.
5. **Σ τ sweep on a window.** `ppm_pipeline.compute_ppm_sigma_pipeline` (:690)
   takes `wfns`, `config`, `meta`, `head_resolver`, `band_slices`, `wfn`,
   `sym`. Change: an array-level entry (`psi_full` faces, $G$ weights at the Σ
   nodes, the pole arrays, the window projectors), planned once; the
   quadrature session (`fixed_quadrature_session`) is the held plan. **Done
   for GN-PPM:** `gw.ppm_sigma.window_sigma_c` (:1611) runs the run's Σ stage
   (`compute_sigma_c_ppm_omega_grid`, :1301) on its parent carrier with the
   window's columns rotated by $U$, the instantaneous $e$, and $f$, $1-f$
   branch weights (new `populations=` argument; `None` is the incumbent
   call). `window_sigma_plan` (:1560) holds the plan: the session, the run's
   head and the on-shell frequency list. It returns $\Sigma_{ij}(e_i)$,
   $\Sigma_{ij}(e_j)$, an arithmetic-pair static operator
   $U[\tfrac12(\cdot+\cdot)]^{\rm h}U^\dagger$ and, on request,
   $\Sigma(\omega)$. Under §1.5 it returns both branches' pair tables from one
   sweep (`branches=True`), the time-ordered arithmetic total (`onshell`)
   and the static-$W$ limit of the populations'
   exchange (`static_limit=True`), done on branch
   `feat/td-agw-dynsig-collision-2026-10-17`. Limits: the projection is the band prefix $[0,b_{\rm hi})$,
   not the window alone, because the invalid-mode static term sums the
   occupied bands (window decode is item 6); the head is the run's
   `HeadSigmaDiag` at the instantaneous $e$, $f$ (frozen $W_h$, §1.4); only
   full-zone parents are accepted (`GATE window_sigma_full_zone`). The
   corrected collision consumes the branch slots defined in Eq. (8), not
   the arithmetic total as a retarded coherent operator.
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
9. **Held rules for the full sweep.** The incumbent full refresh plans one-shot
   rules per call, so the held $\Delta\Sigma^{\rm h}$ carries the rule change
   when a state crosses the occupation-weight floor (§1.5, 0.38 meV on 516).
   Change: a held session for the full sweep, planned with every window state
   live in both branches (as the split's active part), or the split as the
   default.
10. **Split by bracket.** The split refresh restricts the intermediate bands by
    mask (`InternalBands`), two sweeps at t0 (frozen, active). The band-bracket
    reduction (`BandBracketPlan`) keeps the matrices of the first and last
    cumulative counts only and asserts contiguous brackets over $[0,n_b)$;
    one bracketed sweep would need the middle bracket's full block.
11. **Instantaneous dressing in the split** (not implemented). With
    `dress = instantaneous` the window's orbitals and energies move ($\psi_AU$,
    $e(t)$). The active part already takes $(U, e, f)$. The frozen part keeps its
    internal bands and populations but not its external side: its on-shell tables
    are read at $e_i(t)$, $e_j(t)$ in the rotated basis. Needed: hold
    $\Sigma^F_{nm}(\omega)$ in the band basis on a frequency grid that spans the
    envelope of $e(t)$ (`window_sigma_c(omega_ry=…)`, the ω-grid route, planned
    once on that envelope), then per refresh $\tilde\Sigma^F_{ij}=\tfrac12[U^\dagger
    \Sigma^F(e_i)U+U^\dagger\Sigma^F(e_j)U]_{ij}$ by interpolation, with the
    interpolation error measured against the on-shell sweep; and the active
    part's session planned on the same envelope, so that no rule is re-fitted
    when $e(t)$ moves.
12. **Rebased subtraction and smooth invalid modes for refreshed screening.**
    The deck key exists (`collision_rebase`, `auto` = on for shared_pole); the
    default for `refreshed` (GN-PPM) remains off pending its deck-route check. The
    invalid-mode treatment continuous in $\Omega^2$ is still open for GN-PPM; the
    shared-pole member has no invalid modes (every pole is passive), which is the
    route measured on 516.
13. **Pole census.** $\max|\Delta\Omega|$ is set from the first refresh
    (δW/W0 ~ 10⁻⁹ on 516: 25 eV) by modes whose fit is ill-conditioned
    ($W^c(0)\to\mathrm{Re}\,W^c(i\omega_p)$); a residue-weighted measure would
    report the modes that move Σ. **Done for the shared-pole member**
    (`tdagw.dynsig.pole_census`: residue $|b_j|^2/2\Omega_j$ and static
    $|b_j|^2/2\Omega_j^2$ weights below 0.3 and 1 eV, $\Omega_{\min}$, q-resolved
    histograms, one line per refresh).
