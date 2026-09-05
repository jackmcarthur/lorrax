# Four-current GW: q→0 heads and the frequency treatment

This page owns the long-wavelength (Γ-cell) treatment of every channel of
the four-current (bispinor) GW self-energy, and the frequency dependence
each channel actually carries. It states what is computed, at which
frequencies, what is omitted by construction, and what is known to be
zero or unsupported. Deck keys are in the
[input reference](../input_reference.md); the phase-1 physics record is
[`BISPINOR_DHFB_DESIGN.md`](../BISPINOR_DHFB_DESIGN.md); the S-tensor
convention is [its own page](s-tensor-convention.md). **How this layer is
wired into the rest of the code** — every producer, object, shape, sharding
and refusal, stage by stage — is
[Four-current wiring](../architecture/four_current_wiring.md).

Rydberg units throughout. Lorentz index `I,J ∈ {0,1,2,3}` with `0` the
charge channel `C` and `1..3` the Cartesian current channels `T`. The
stored vertices are `γ̃^0 = 1` and `γ̃^i = α^i` (charge density `ψ†ψ`,
current density `ψ†α^iψ`).

## 1. Status in one table {#four-current-phase-status}

> **Integration status (2026-09-02).** Phase 1 is complete
> inside the slab one-shot static-COHSEX envelope. Phase 2 is implemented;
> the magnetic GN-PPM response and two-residue fit are W-level gated on CrI3,
> but there is no Sigma-level CrI3 number. Phase 3 is minimally implemented:
> GN/HL-PPM and MPA keep charge with their scalar dynamic owner and freeze the
> bare current blocks at ω = 0. On that route the packed CC block is absent,
> not zero-filled. Bare dynamic self-consistency reuses this orbital-independent
> operator but re-contracts it with each map's orbitals; charge-consuming and
> screened-current SC classes refuse by name.

| self-energy channel | frequency dependence | Γ-cell head | deck route | status |
|---|---|---|---|---|
| charge exchange `Σ_X` (CC) | none | `⟨v⟩_mBZ`, band-diagonal (§3.1) | every mode | production |
| charge correlation `Σ_SX+Σ_COH` / `Σ_c(ω)` (CC) | static, GN-PPM (two samples), HL-PPM, MPA | scalar `W_h(ω)` from `S_eff(ω)` (§3) | `head_correction = full` | production |
| bare transverse exchange `Σ^B` (TT) | none: instantaneous (bare Breit) | **inside the packed envelope**: `⟨D_TT⟩` from the Γ-cell completion, always on. **Outside it**: none — the `−⟨v P^T⟩_mBZ` tensor slot overlay (§2.1) survives as library code with no deck route | inside: `bispinor_gw = bare_transverse` + the §2 envelope. Outside: no deck key since 2026-09-01 | completion implemented and always on; the overlay is unreachable from a deck and **refused** on either packed route |
| screened TT/CT/TC (packed 4×4) | **static only** (`compute_mode = cohsex`) | charge CC `q²` + the charge wings + **optional** Hall CT/TC `q¹` (§4), **always on**; `head_correction = off` is a DEBUG skip behind a loud banner | `bispinor_gw = full_static_cohsex` | experimental, insulating slab, one shot |
| *unscreened* TT via the same packed operator | current block evaluated once at `ω = 0`; static COHSEX uses packed `W = diag(W_00,D_TT)`, while dynamic GN/HL/MPA uses `W_CURRENT = V_CURRENT` with packed CC absent | static COHSEX uses the coupled completion; dynamic mode inserts bare `⟨D⟩` only into V/logical current W and skips charge S/wings; Hall **refused** (§4) | `bispinor_gw = bare_transverse` inside the packed envelope | experimental slab; dynamic mode supports one-shot and self-consistent QP maps |
| the packed operator on a **dynamic** Σ | **CC dynamic scalar owner, current blocks static**: GN/HL/MPA own `W_00(ω)` outside the packed carrier; twelve current blocks are evaluated at `ω = 0` | CC: the dynamic model's own head for `Σ_c` plus scalar bare-X head; current: bare-only packed Γ completion | `bispinor_gw = bare_transverse`, `compute_mode` in {`gn_ppm`, `hl_ppm`, `mpa`}; screened MPA refuses its missing static role | experimental slab; current-frequency dependence neglected as bounded in §2.2; bare SC re-contracts per-map orbitals |
| retarded / dynamic photon `D^{IJ}(ω)` — the current blocks' own frequency dependence | — | — | none | **does not exist**; bounded from above in §2.2 |

Binding rule ([decisions, 2026-09-01](../architecture/decisions.md)): COHSEX
with bispinors always carries the Γ-cell head; `head_correction = off` is a
debug setting. A mode whose envelope forbids the head is a defect, not a
calculation. Since 2026-09-02 the grammar enforces it: a bispinor deck has
exactly two head values (`full`, `off`), `off` is labelled `WARNING -- DEBUG`
on **both** routes, and no other key — not
`bispinor_tt_head_correction`, not `restart`, not `no_local_fields` — can
reach a different head.

The carrier of the current channels is a separate axis from the routes in
the table: `bispinor_current_balance` picks the four-spinor the spatial-current
channels ride (§7). `kinetic` is the shipped σ·p carrier and the default;
`velocity` is on branch `feat/bispinor-velocity-lift-2026-09-04`, not on
`main`.

Two facts follow from the table and are easy to miss:

* The photon sector's **current** blocks have no frequency dependence
  anywhere. `Σ^B` uses the instantaneous Coulomb-gauge propagator, and the
  packed screened current response is built once at `ω = 0`. The minimal
  dynamic route leaves the **charge** block out of the packed layout:
  GN/HL/MPA carry `W_00(ω)` in their existing scalar owner, while the sole
  packed consumer reads only the current sector with `W_CURRENT = V_CURRENT`.
  `Σ^B` is its TT block rather than a separately added exchange term.
* The only time-reversal-odd term in the screened photon head is the
  static Hall (Chern–Simons) response. For a gapped system that
  coefficient is topologically quantized (§4.4), so for a Chern-trivial
  insulator it is exactly zero, so the packed mode's head reduces to the
  charge-only head of §3 — its Hall term is zero both by default (an absent
  `static_gauge_hall_file` sets `σ_H = 0`, announced) and by physics.

## 2. The bare propagator and its Γ-cell average

The [status statement](#four-current-phase-status) owns implementation phase
and route coverage. This section explains why the packed and incumbent
contractions represent the same bare-transverse body.

### Two routes to the same `Σ^B`

`bare_transverse` **is** the packed static mode with `χ_TT = χ_CT = 0`. The
packed Dyson equation is then block diagonal, so `W_packed = diag(W_00,
D_TT)`, `W_CT = 0`, and the sixteen-block `Σ_X` reproduces exactly the CC
screened COHSEX plus `Σ^B = X(D_TT)`, with `COH(D_TT − D_TT) = 0`. The two
descriptions below are the same physics; which code runs depends only on the
envelope.

| | inside the envelope | outside it |
|---|---|---|
| envelope | `bispinor`, `compute_mode ∈ {cohsex, gn_ppm, hl_ppm, mpa}`, `screening_diagrams = w_rpa`, `sys_dim = 2`, `head_correction ∈ {full, off}`, `restart = false`; `qp_solver = one_shot_dft`, or `self_consistent` only for bare dynamic mode | current bulk/restart/x-only/resolvent/explicit-local classes; screened MPA, screened SC, and bare COHSEX SC refuse by their dedicated gates |
| who contracts `Σ^B` | `gw.photon_sigma`, the TT part of the sixteen-block `Σ_X` | `gw.sigma_x_bispinor`, the nine bare TT tiles |
| the TT Γ-cell head | `⟨D_TT⟩` out of the Γ-cell completion, always on | **none** — the §2.1 overlay lost its deck key on 2026-09-01, and the run record says so |
| status | experimental; body byte-identical to the incumbent (below) | **the only certified route**, and unchanged |

The route is a predicate with a reason attached,
`gw_config.packed_bare_transverse_route(config) -> (taken, reason)`, reading
the same nine-row `packed_static_envelope` table the screened mode's refusal
walks, and the driver writes that reason into the run record as the
`Photon route` line — the two routes differ in the q→0 head mechanism, so
which one ran is a fact a later reader needs. No head dial moves the route:
`bispinor_tt_head_correction` is gone as a deck key (a hand-built `True` on
either packed route is refused,
`GATE packed_bare_transverse_tt_head_double_count`), `no_local_fields` is
refused rather than routed, and `restart = true` is refused on this deck
class rather than routed.

**Measured** (`reports/bisp_c_bare_as_packed_2026-09-01`, MoS2 3×3, 270
states, claim 581): with the head off the two routes are **byte-identical** —
`max|dE_qp| = 0.000 µeV`, `eqp1.dat` and every `sigma_diag.dat` data row
identical, despite entirely different contraction order and operator packing.
Declaring the twelve current `χ` blocks zero costs **0.012 µeV** (1.2e-8 eV)
against the screened packed mode on the same deck — the current vertices each
carry `α_FS/2`, so `χ_TT` enters at `α_FS²`. The former head-on comparison
mixed two quadratures and two insertion rules. The routes now share one
quadrature; §3.6 isolates the remaining insertion difference.

In Coulomb gauge the bare electron–electron interaction, after eliminating
the photon field, is block diagonal in `(C, T)`:

$$
D^{00}(\mathbf K)=v(\mathbf K),\qquad
D^{0i}=D^{i0}=0,\qquad
D^{ij}(\mathbf K)=-\,v(\mathbf K)\,P^T_{ij}(\hat{\mathbf K}),\qquad
P^T_{ij}=\delta_{ij}-\hat K_i\hat K_j ,
$$

with `K = q + G` and `v` the (possibly slab-truncated) Coulomb kernel. The
sign of `D^{ij}` is the spatial metric; it is applied once, in the
propagator builder (`vcoul.COULOMB_GAUGE_TT_SIGN`), never in a vertex.

The point `K = 0` is undefined in every block, but for different reasons:

* `D^{00}`: `v` diverges, isotropically. The charge structure factor obeys
  `M_{mn}(q→0, G=0) → δ_{mn}`, so the missing slot couples only to the
  band diagonal and a scalar cell average `⟨v⟩_mBZ` is the complete
  replacement.
* `D^{ij}`: `v` diverges and `P^T` has no limit (it depends on the
  direction of approach). The current structure factor
  `j^i_{mn}(k) = ⟨mk|α^i|nk⟩` is finite and not diagonal in band index.
  The correct replacement is the tensor cell average
  `T_{ab} = ⟨v(\mathbf q)\,P^T_{ab}(\hat{\mathbf q})⟩_{mBZ}`, and it must
  land in the `(μ,ν)` centroid tile, not in a band-diagonal shift.
* `D^{0i}`: identically zero at every `K`. No bare CT/TC head exists.

### 2.1 The bare TT head

This **was** the incumbent route's TT head. Inside the packed envelope of §2
the same physics arrives as `⟨D_TT⟩` from the Γ-cell completion instead, and
asking for this overlay there is refused
(`GATE packed_bare_transverse_tt_head_double_count`) — the two would double
count.

> **Status at `34228021`.** `bispinor_tt_head_correction` is no longer a deck
> key: heads are always on, so a dial that turns one off is not a dial
> (owner ruling 2026-09-01). The code below survives as
> `gw.v_q_bispinor._tt_head_tensor`, wired to `False`, for a hand-built
> incumbent non-packed configuration; it is reachable only from a
> hand-built config. **Consequence: a bispinor deck outside the packed
> envelope — for example bulk, restart, self-consistent, MPA,
> `x_only`/resolvent, or explicit-local — carries NO transverse
> q→0 head at all**, and its run record now says so on the `Photon head`
> line. The section is kept because the identities below are the gates the
> packed completion's `⟨D_TT⟩` is checked against.

In a hand-built config, setting the field replaces the `q = Γ, G = 0` slot
of the nine TT `V` tiles by `−T_{ij}/Ω_{\rm cell}`, where `T` is sampled by the
Sobol `vcoul.minibz_transverse_head_avg` debug estimator. It is not the
Wigner–Seitz polygon provider that owns the deck-reachable scalar and packed
slab heads.
Exact identities used as gates:

$$
\operatorname{tr}T = 2\langle v\rangle_{mBZ}
\quad\text{(any cell; }\operatorname{tr}P^T=2\text{)},\qquad
T=\langle v\rangle\operatorname{diag}(\tfrac12,\tfrac12,1)
\quad\text{(in-plane isotropic slab)},\qquad
T=\tfrac23\langle v\rangle\,\mathbb 1\quad\text{(isotropic 3D)} .
$$

The corrected tile flows through the ordinary `(μ,ν)` convolution into
`Σ^B`, which is why the head arrives as a full band matrix. Measured on the
MoS2 4×4 bi4 deck (`BISPINOR_DHFB_DESIGN.md` §11): the missing head is
comparable in Frobenius norm to the whole stored `q = Γ` TT slab and is
about 5 % of `Σ^B`, but only ~0.2 meV on quasiparticle energies at that
grid, decaying as `1/√N_k` in 2D. Box truncation (`sys_dim = 0`) never zeros
the slot and is refused. Bulk (`sys_dim = 3`) uses the Baldereschi–Tosatti
analytic sphere for the `1/q²` part, as `⟨v⟩` does.

This term is frequency independent because `Σ^B` is. If invoked from a
hand-built configuration, it is the only transverse Γ-cell correction outside
the packed routes; no parsed deck reaches it. Its own size
was re-measured on MoS2 3×3: switching the overlay on and off in
the incumbent route moves quasiparticle energies by **0.115 meV MAE /
0.251 meV max**, consistent with `BISPINOR_DHFB_DESIGN.md` §11's ≈0.2 meV at
4×4.

### 2.2 Freezing the current blocks inside a dynamic run

The packed dynamic route evaluates the charge block at every
frequency and the twelve current blocks once, at `ω = 0`. That is an
approximation, and it is bounded rather than asserted.

Write the packed screened propagator as

$$
W_{AB}(\omega) = D_{AB} + \sum_{CD} D_{AC}\,\chi_{CD}(\omega)\,W_{DB}(\omega),
$$

with `A, B ∈ {C, T1, T2, T3}`. Every current vertex carries one factor of
`α_FS/2`, so `χ_TT` and `χ_CT` enter `W` at `α_FS²` and `α_FS` × (a
transverse-charge overlap that vanishes in the Coulomb gauge at `q → 0`)
respectively. The neglected quantity is therefore
`W_AB(ω) − W_AB(0)` for `AB ≠ CC`, which is second order in the fine
structure constant to begin with.

It is bounded above by the **static** current-screening contribution,
`W_AB(0) − D_AB`, which the code can evaluate directly: the Ward-subtracted
no-pair current response is a positive-weight spectral integral sampled on
the imaginary axis, so `|χ_TT(iω)| ≤ |χ_TT(0)|` for every `ω`, and the
screened-minus-bare current correction at any frequency is no larger in
magnitude than its value at `ω = 0`. That value is exactly the difference
between `bispinor_gw = bare_transverse` (`χ_TT = χ_CT = 0`) and
`bispinor_gw = full_static_cohsex` (the sixteen-block `ω = 0` Dyson solve)
on the same tip and the same deck.

**Measured on MoS2 3×3, both heads on, 270 quasiparticle states:**
`max|ΔE_qp| = 0.012 µeV = 1.2 × 10⁻⁸ eV` in the static COHSEX mode
(GATE D, claim 581), with the same pair re-measured on the dynamic route
(`reports/bisp_n_dynamic_packed_2026-09-01/report.md`). Current
screening is worth ~10⁻⁸ eV on this deck, six orders of magnitude below the
0.1 meV scale at which the transverse Γ-cell head matters, so freezing its
frequency dependence is not the leading error in any bispinor number this
code currently produces. SCOPE: one system, one grid; this is a measurement
on MoS2 3×3, not a general statement about current screening.

## 3. The charge head: `S(ω)`, local fields, and the three frequency models

At `34228021` the magnetic GN path and the shared slab quadrature are live.
The [status statement](#four-current-phase-status) owns phase claims; §3.5
states the magnetic frequency behavior and §3.6 separates the now-closed
quadrature question from the still-distinct insertion rules.

### 3.1 Objects

The macroscopic charge response is written as a Cartesian quadratic form,

$$
\chi_{00}(\mathbf q\to0,\omega)=q_a S_{ab}(\omega) q_b ,
$$

built from velocity matrix elements over every energy-ordered pair
(`gw.qsgw_head.head_s_tensor_sharded`):

$$
S_{ab}(\omega)=\frac{4}{\Omega N_k\,n_{\rm spin}n_{\rm spinor}}
\sum_{\mathbf k}\sum_{\epsilon_i>\epsilon_j}
\frac{(f_j-f_i)\;\overline{v^a_{ij}}\,v^b_{ij}}
{\Delta_{ij}\left[(\omega+i\eta)^2-\Delta_{ij}^2\right]},
\qquad \Delta_{ij}=\epsilon_i-\epsilon_j .
$$

Only the `(a,b)`-symmetric part is observable in `q·S·q`; the antisymmetric
part (which is nonzero on a magnet) is discarded by the contraction and
canonicalized away at the static-gauge boundary.

Local fields enter through the two Γ wings `Y^a_μ`, `Z^b_ν` (one velocity
leg, one centroid leg; `gw.qsgw_head.head_wings_sharded`) and the headless
body `W_{μν}(Γ,ω)`:

$$
S^{\rm eff}_{ab}(\omega)=S_{ab}(\omega)+\frac1\Omega\,Y^a(\omega)\,W_{\rm body}(\Gamma,\omega)\,Z^b(\omega)
\qquad(\texttt{gw.head\_correction.fold\_cartesian\_head\_wings\_sharded}).
$$

This is the bordered-Dyson (Schur) reduction of the head against the body;
`head_correction = full` applies it exactly once, `no_local_fields` skips it
(the direct `ε` head, diagnostic), `off` removes the Γ contribution.

The head sample at frequency `ω` is the mini-BZ average of the screened
kernel with the ISDF body already folded in:

$$
v_h=\langle v(\mathbf q)\rangle_{mBZ},\qquad
W_h(\omega)=\left\langle\frac{v(\mathbf q)}{1-v(\mathbf q)\,q_aS^{\rm eff}_{ab}(\omega)q_b}\right\rangle_{mBZ}
\qquad(\texttt{vcoul.<kernel>.q0\_average}).
$$

**The cell average has one owner per dimension, and on a slab it is the
same rule the packed completion of §4 uses.** `Slab2D.q0_average` evaluates
both brackets on the exact mini-lattice Wigner–Seitz polygon cubature
(Γ-to-edge Duffy triangulation, fixed 16/24/32 Gauss–Legendre ladder)
issued by `vcoul.slab_minibz_photon_cubature` — the same authenticated
receipt, the same nodes and the same weights `complete_static_slab_photon_q0`
consumes. The superseded scrambled-Sobol draw survives only as the named
`rule="sobol_debug"`. `Bulk3D` keeps its Sobol + Baldereschi–Tosatti sphere
rule: the polygon construction is two-dimensional. See §3.6 and
`docs/services/vcoul.md`.

The body `W_{μν}(q=0)` is solved with the `G = 0` channel removed (the
`K = 0` slot of `v` is zero), so the scalar head is re-attached to it as a
rank-1 update `(W_h/Ω)\,\overline{g_0}\otimes g_0` with
`g_0(μ) = ζ_Γ(μ, G=0)`. The Γ wings of `W` (the `(G=0, μ)` blocks) are not
re-attached in this route; after cell averaging their leading term is odd in
`q` and vanishes, so this is an `O(1/N_k)` omission, not a singular one. The
packed route of §4 does carry them.

### 3.2 Static COHSEX

Exact band-diagonal shifts, from `v_h` and `W_h(0)` only:

$$
\Sigma^X_n=-\frac{v_h f_n}{\Omega N_k},\qquad
\Sigma^{SX}_n=-\frac{W_h(0) f_n}{\Omega N_k},\qquad
\Sigma^{COH}_n=+\frac{W_h(0)-v_h}{2\,\Omega N_k}.
$$

### 3.3 GN-PPM and HL-PPM

The dynamic head is a single pole fitted from the correlation part
`W^c_h = W_h − v_h` at two frequencies (`gw.head_correction.fit_head_ppm`):

$$
\Omega_h^2=-z^2\,\frac{W^c_h(z)}{W^c_h(0)-W^c_h(z)},\qquad
B_h=-W^c_h(0)\,\Omega_h^2,\qquad R_h=\frac{B_h}{2\Omega_h},
$$

with `z = iω_p` (Godby–Needs) or a real `z = Ω` above all transitions
(Hybertsen–Louie). The HL variant may instead take `Ω_h² = ω_p²/(1 − W_h(0)/v_h)`
from the f-sum rule, or a deck-supplied `Ω_h`. The head enters `Σ_c` on the
band diagonal only:

$$
\Sigma^{c,\rm head}_n(\omega)=\frac{R_h}{\Omega N_k}\left[
\frac{f_n}{\omega-\epsilon_n+\Omega_h-i\eta}+
\frac{1-f_n}{\omega-\epsilon_n-\Omega_h+i\eta}\right],
$$

which reduces to `Σ^{SX−X}+Σ^{COH}` of §3.2 on shell. The two frequencies
`0` and `iω_p` are the only ones the driver ever asks for on this route, and
`S(ω)`, the wings, and the body are all evaluated at both.

### 3.4 MPA

The head is sampled on the same stamped complex `z` grid as the body, the
fold of §3.1 is applied per `z` with the left and right wings independent,
and the resulting complex poles feed
`gw.head_correction.compute_complex_pole_head_sigma_diag`. The metallic
static limit (Thomas–Fermi, `κ_TF²`) and its Schur fold are owned by
[Metallic MPA screening](metallic-mpa-screening.md) §4.

### 3.5 What time-reversal breaking does and does not change here

Facts at `34228021`, with the derivation and branch assignment owned by
[`DERIVATION_gnppm_nonhermitian.md`](../dev/notes/DERIVATION_gnppm_nonhermitian.md):

* At `ω = 0`, and on every time-reversal-symmetric deck, the historical even
  completion is exact. When the measured `SymMaps.trs_allowed` verdict is
  false, `gw.w_isdf.compute_chi0_imag_ordered` instead weights both ordered
  particle–hole orientations independently and preserves the anti-Hermitian,
  magnetisation-odd part of `χ_0(iω_p)`. The `W` Hermiticity gate is scoped to
  the time-reversal-symmetric case.
* The GN fit splits `W^c(iω_p)` into Hermitian and anti-Hermitian halves. It
  constructs Hermitian `B` and `D`, then supplies `R_+ = B + D` to the empty
  (conduction) Sigma branches and `R_- = B - D` to the occupied (valence)
  branches. The crossing-window adjoint closure is unchanged because each
  branch now receives a Hermitian residue.
* The scalar charge head loses no magnetic component: its antisymmetric
  Cartesian response is annihilated by `q_a S_ab q_b`, so its odd residue is
  exactly zero. HL-PPM's real probe cannot determine an odd residue and keeps
  the single-residue fit.
* MPA follows the same measured `SymMaps.trs_allowed` verdict. On a
  time-reversal-broken deck its ordered non-Hermitian fit retains the odd
  closure with `R+ = B + D`, `R- = B - D` and reports `sigC_odd`. The packed
  bare current addition is TR-even and static: it never touches `B_odd`, `D`,
  the ordered residue branch, or that verdict.

### 3.6 CLOSED (quadrature): the 5.4 meV was the Sobol estimator

There were two evaluations of the **same** Γ-cell charge integral and they
disagreed. Two mechanisms differed at once — the quadrature and the
insertion — and the quadrature half is now settled and fixed.

| | incumbent (before 2026-09-02) | today, both routes |
|---|---|---|
| quadrature | Sobol Voronoi average of `v/(1 − v\,q·S·q)` | exact Wigner–Seitz Duffy–Gauss polygon rule, `vcoul.slab_minibz_photon_cubature` |
| insertion | band-diagonal scalar shift (§3.2) | scalar route: band-diagonal (§3.2); packed route: rank-4 update through the real Σ kernels (§4.2) |
| convergence certificate | none printed | scalar: polygon edges/orders/nodes/`⟨v⟩`/24→32 error ratio, refuses above budget. packed: `ward`, `hermiticity`, `dyson_forward_bound`, `cubature_orders` |

**What settled it** (claim 0586,
`reports/bisp_j_architecture_review_2026-09-01/report.md` §4).
Both rules were evaluated on one fitted head function for the MoS2 3×3
deck. The Wigner–Seitz ladder converges to `⟨v⟩ = 1652.678662 a.u.` at
order 24 and stays there through order 96 (`|W(32) − W(96)| = 3.7e-10`);
the production Sobol draw (2^18 × 10) gives `1655.334970` (+0.16 %) and a
second draw (2^20 × 20) gives `1652.378` (−0.02 %). The two draws straddle
the exact value. The error is deterministic per seed, so it never
presented as noise, and a second draw is not a convergence study. In Σ the
quadrature alone is **+5.72 meV on every occupied state's SX+COH**
(X +5.719, SX +5.723, COH −0.002); the independently measured charge-half
median was 5.55 meV. No k-grid sequence was needed: the two routes evaluate the same
function, and the difference is measured to 1e-10.

**One owner at `34228021`.**
`Slab2D.q0_average` now reads its nodes, weights and kernel values off the
same provider receipt, so the two routes are two reductions of one set of
numbers rather than two estimators that ought to agree. Measured on the
MoS2 3×3 deck geometry: `⟨v⟩` agrees to `2.274e-13` and the screened
companion to `0.000e+00`
(`runs/MoS2/08_bisp_k_head_quadrature_20260901/head_owner_check.json`).
Read back from a live GW pair (head on minus head off, legs 00 − 03 of that
run) the scalar route's `v_h` is `1652.678657 a.u.`, `3.0e-9` relative from
the exact rule and no longer the Sobol value. The packed legs are
`0.000 µeV` from `runs/MoS2/07_integ_regate_20260901` on all 270 rows: the
packed route was already on this rule and did not move.
Against the matched BerkeleyGW static-COHSEX reference on that deck (48
rows, 4 k × bands 19–30) the scalar route's `sigTOT` MAE moves
9.23 → 5.41 meV and its signed mean −8.49 → −4.68 meV, most of the way
onto the packed route's figure (4.76 / −3.50, claim 582, re-measured
unchanged here) — an independent referee moving the same way as the exact
rule, not proof that either is converged.

**Still open: the insertion, and it is the owner's choice.** The scalar
route keeps the band-diagonal shift of §3.2, which for the bare CC head is
*exact* (`Σ^X_n = −⟨v⟩f_n/(ΩN_k)` is band diagonal and state independent
in a plane-wave basis). The packed route inserts the same head as a rank-4
`(μ,ν)` update through `g_0(μ) = ζ_Γ(μ, G=0)`, and carries the ISDF
normalisation error of the `G=0` pair density with it: ±3.9 meV std,
11.5 meV max on bands 7–8 at 640 charge centroids, because
`Σ_μ ζ_Γ(μ,G=0)|ψ_n(r_μ)|²` is not exactly 1. That residual is now the whole
of the remaining scalar-vs-packed difference; it is an ISDF question, not
a quadrature one, and it is registered with two options (insert the CC
`M_00` band-diagonal part exactly, or price the error by centroid count).
The wings and the current blocks must stay in the `(μ,ν)` form either way.

Measured with one quadrature owner on both sides (leg 01's packed per-state
head against leg 00's band-diagonal head, same run):

| column | scalar band diagonal | packed mean ± std | residual mean / MAE / max |
|---|---|---|---|
| `x_head_CC` (occ, 234) | −3557.994 meV | −3557.077 ± 3.934 | +0.917 / 3.167 / 11.469 |
| `sex_head_CC` (occ) | −862.634 | −863.925 ± 1.182 | −1.291 / 1.334 / 5.280 |
| `coh_head_CC` (all 270) | −1347.680 | −1345.985 ± 1.813 | +1.695 / 1.884 / 6.738 |
| occupied `SX+COH` | −2210.314 | −2209.918 | **+0.396** / 1.769 / 6.155 |

The systematic offset on an occupied state's `SX+COH` was **+6.1 meV**
before the shared-cubature change (claim 0586 §4.3) and is **+0.40 meV**
now; what is left is state-dependent and centred near zero, which is what an
ISDF fit error looks like and what a quadrature error does not.

**Not covered by this owner: the hand-built bare-TT overlay.**
`q0_average_transverse_tensor` (the library-only q=Γ, G=0 overlay of §2.1)
still uses the Sobol draw, but no parsed deck reaches it. On the MoS2 3×3
geometry that debug estimator breaks the cell's
in-plane isotropy by 0.31 % (`diag = 826.41, 828.92` where the exact rule
gives `826.34, 826.34`) and is 0.16 % high on the `zz` component. The
packed route takes its TT head from the receipt and is unaffected.
Registered in `KNOWN_LORRAX_ISSUES.md`; the fix is three lines and moves
the incumbent route's numbers, so it wants its own gate.

## 4. The packed static photon head (`full_static_cohsex`)

The [status statement](#four-current-phase-status) owns implementation
coverage. At `34228021`, `full_static_cohsex` is the sole packed
screened-current spelling and the coupled completion is on under the default
`head_correction = full`.

### 4.1 Body and layout

`full_static_cohsex` builds one packed
`C ⊕ T_1 ⊕ T_2 ⊕ T_3` operator (`gw.photon_layout`): the sixteen bare blocks
`D^{IJ}_q(μ,ν)` from `v_q_bispinor.h5`, the sixteen no-pair response blocks
`χ^{IJ}_0` from the four-spinor current (the spatial-current carrier of §7)
(`gw.w_isdf.compute_experimental_no_pair_photon_chi0`, TT blocks
Ward-subtracted as `Π(q) − Π(0)`), and one distributed Dyson solve
`W = (1 − Dχ_0)^{-1} D`. `gw.photon_sigma` then contracts all sixteen
`W^{IJ}` blocks into `Σ_X`, `Σ_SX`, `Σ_COH` with the Lorentz vertices folded
into the wavefunction bundles. This body is the static analogue of §3's
charge body and carries the same missing `K = 0` slot in every block.

### 4.2 The coupled Γ-cell solve

The mode completes the Γ slot of both `V` and `W` by solving
the 4×4 Lorentz Dyson equation at every point of an exact Wigner–Seitz
cubature of the mini-BZ polygon (fixed 16/24/32 Duffy–Gauss ladder,
`vcoul.slab_minibz_photon_cubature`; slab geometry only):

$$
R(\mathbf q)=q_a H^a+q_aq_b S^{\rm eff,ab},\qquad
H^a_{0i}=-i\,\epsilon_{bai}\,\sigma_H^b,\; H^a_{i0}=\overline{H^a_{0i}},\qquad
W_h(\mathbf q)=\left[1-D(\mathbf q)R(\mathbf q)\right]^{-1}D(\mathbf q).
$$

Each `S^{\rm eff,ab}` is a 4×4 Lorentz block, folded with the Γ wings
through the headless packed body exactly as in §3.1. The Hall pseudovector
`σ_H` is never fitted: it is a separately produced input
(`static_gauge_hall.h5`, `gw.qsgw_head.static_gauge_hall_transaction`) and the
only admitted `q`-linear CT/TC structure is generated from it. It is also
**optional**: an absent `static_gauge_hall_file` means `σ_H = 0`, announced
with its reason, which by §4.4 is the exact answer for the systems this mode
admits; a present but mismatched artifact still refuses in the loader, so a
stale file cannot degrade the run silently. On the **unscreened** packed
route of §2 an authenticated exact-zero artifact is equivalent to the
unnamed default and is accepted. `GATE packed_bare_transverse_hall_unavailable`
refuses any nonzero component: with the current channels unscreened,
`W_CT = 0` at every finite `q`, so a Γ-only CT/TC block would not be the
`q → 0` limit of anything the model computes. The sign is the live band
orientation's: the persisted `σ_H` is the occupied-bra Berry sum while the
live Adler–Wiser response is energy-ordered (`P = −ΔD`), which is the minus
above; the oracle is
`tests/test_qsgw_parallel_transport_head.py::test_raw_hall_matches_orbital_cB_owner_and_documented_sign`).
Only the nine monomial moments survive the cubature,

$$
M_{uv}=\left\langle b_u(\mathbf q)\,W_h(\mathbf q)\,b_v(\mathbf q)\right\rangle_{mBZ},
\qquad b=(1,q_x,q_y),
$$

and the packed operators are updated by one bare and nine screened rank-4
outer products (`gw.photon_layout.add_photon_q0_low_rank`):

$$
V_\Gamma\mathrel{+}=\frac{1}{\Omega}\,\overline{g_0}\otimes\langle D\rangle g_0,\qquad
W_\Gamma\mathrel{+}=\frac1\Omega\sum_{uv}L_u\otimes M_{uv}R_v,\quad
L=(\overline{g_0},\,(WZ)^x,\,(WZ)^y),\; R=(g_0,\,(YW)^x,\,(YW)^y).
$$

`M_{00}` is the averaged head, `M_{0a}, M_{a0}` the single-wing moments, and
`M_{ab}` the double-wing (body) moments. Screening is done before averaging
because `⟨[1−DR]^{-1}D⟩ ≠ [1−⟨D⟩⟨R⟩]^{-1}⟨D⟩`; in particular the odd Hall
term averages to zero in `M_{00}` but survives in the crossed first moments
`⟨q_x W^{0y}⟩`, `⟨q_y W^{0x}⟩`.

`gw.photon_sigma` also reports the exact contribution of the completed Γ
blocks to `Σ` per `(X, SX, COH) × (CC, CT+TC, TT)` sector and checks that the
sector sum closes on the sixteen-block total.

### 4.3 What the mode contains, by declaration

The **module docstring of `gw.static_gauge_response`** is the complete list
(it replaced the `CHARGE_HALL_CUBATURE_AVAILABILITY` availability grammar,
which is deleted). Present: charge `S^{00}` (`cc_q2`), Hall `ct_q1`/`tc_q1`, and
the charge wings `y_charge`/`z_charge`. Omitted by model: the current
quadratic response `tt_q2`, the mixed quadratic response `ct_q2`/`tc_q2`,
the current wings, the diamagnetic/contact terms `contact_q0`/`contact_q2`,
and the negative-energy (complement-space) closure. `tt_q0` (the uniform
static current response) is zero by gauge invariance for an insulator and is
omitted rather than computed. They are never stored as accidental zeros of a
larger schema: `S_direct` has charge support only. The full-static-gauge
response that would carry all of these (`StaticGaugeHeadResponse`, its v2
head schema and loader, and both of the refusals that guarded it) had no
producer and is absent at `34228021`.

### 4.4 The Hall coefficient is a topological invariant

For a gapped system the static, long-wavelength charge–current response is
the Chern–Simons term, and its coefficient is quantized (TKNN):

$$
\sigma_{xy}(\mathbf q\to0,\omega=0)=C\,\frac{e^2}{h},\qquad
C=\frac{1}{2\pi}\sum_{n\in{\rm occ}}\int_{\rm BZ}\Omega_n^z\,d^2k\in\mathbb Z .
$$

The producer computes exactly this occupied Berry-curvature sum
(`gw.qsgw_head.raw_hall_pseudovector_sharded`, `:2585-2616`,
`σ_H^b = −(α_{FS}C_s/2Ω)\,\operatorname{Im}c_B^b`), and it refuses metals
(exact `0/1` occupations are required). Consequently:

* For a Chern-trivial insulator, `σ_H = 0` exactly in the complete-basis,
  converged-`k` limit. Whatever the producer returns is a band-truncation and
  `k`-sampling residue, and the CT/TC Γ-cell physics of the packed head is
  empty: the head reduces to the charge-only head of §3 (with the Γ wings
  carried, §3.1). This is why an absent `static_gauge_hall_file` defaulting
  to `σ_H = 0` is a correct default and not a silent omission.
* For a Chern insulator, `σ_H` is an integer times
  `α_{FS}C_s/(8\pi L_z)` in the code's units (`L_z` the periodic cell height),
  known before the calculation.
* The case in which a static Hall response carries new information, a metal,
  is the case the producer refuses.

CrI3 monolayer (6×6×1, 250 bands, `L_z = 43.0` bohr): the authenticated
producer gives `σ_H^z = −4.22×10^{-8}` bohr⁻¹ against a single-Chern quantum
of `6.8×10^{-6}` bohr⁻¹, i.e. `|C_{eff}| ≈ 6×10^{-3}`, sign- and
magnitude-unstable with band count (sandbox
`reports/cri3_full_ct_hall_head_2026-08-26`). That is the expected zero.

For a Chern-trivial insulator the CT sector of the Gamma-cell head is bounded
by symmetry and by the current vertices, not assumed away. The surviving CT
moment is the crossed first moment ⟨W^{0i} q_a⟩ with W^{0i} = D_00 R^{0i} D_ii
to first order, and R^{0i}(q) = iε σ_H q + q_a q_b S^{0i,ab} + O(q³). The Hall
term is quantized (zero for C = 0). The quadratic coefficient is the static
linear magnetoelectric response (a charge density induced at O(q²) by a vector
potential is ∇·(α_ME B)), which requires both inversion and time reversal
broken; monolayer CrI3 is D3d and has inversion, so it vanishes and CT starts
at O(q³), giving W^{0i} ~ (1/q)·q³·(1/q) = O(q) and a moment O(q_cell²) ∝
1/N_k, which is body-discretization order. Without inversion, W^{0i} ~
(1/q)·q²·(1/q) = O(1) survives averaging exactly like the CC screening
correction (0.66 eV on the CrI3 gap); its size relative to that is (Zα)² for
two current vertices (≲ 0.15 for iodine 5p states) times α_ME, whose natural
ceiling is α_FS/2 (axion strength) and whose ordinary value is a hundred times
smaller. The ceiling is therefore about 0.66 eV × 0.15 × 0.0036 ≈ 0.4 meV for
a heavy-element, inversion-broken, axion-strength magnet, tens of μeV for
ordinary magnetoelectrics, and zero for anything centrosymmetric. The finite-q
body is different: all sixteen W^{IJ}(q≠0) blocks, CT included, enter Σ at
first order and are computed by the packed Dyson; their size on CrI3 was
measured once (June 2026 supermatrix, screened versus unscreened Breit under
10 μeV, with the −2F shortcut) and is (Zα)²-suppressed.

The physically largest transverse screening channel in a ferromagnet, the
Goldstone-enhanced transverse spin susceptibility, is a ladder (vertex) effect
outside RPA; the bubble gives only M/Δ_exchange, α²-suppressed in the TT
response, and the code's ladder screening (`w_bse`) is charge-only.

The time-reversal-odd photon physics that is *not* quantized, the
finite-frequency Hall/Kerr response `σ_{xy}(ω)` and the antisymmetric part of
the TT response, lives at `ω ≠ 0` and at `O(q²)`. Neither is present in the
static packed head, and no dynamic photon propagator exists to hold the
former.

### 4.5 Audited seams and remaining limitation

At `34228021` the two implementation defects below are fixed. The remaining
head limitation is the insertion difference in §3.6; the scalar and packed
routes already share one slab quadrature.

* **Interband Γ wing sign.** Both wing kernels built the
  mixed head/body interband weight as `+F\,\overline{P}\,b/(z²−Δ²)`, while the
  shared head convention `P = −ΔD` implies a minus: the wing replaces one
  density-jet leg by the energy-scaled head vertex, and that substitution
  carries a sign the kernels never applied. It cancels in the scalar Schur
  fold `Y W Z` (§3) but not in the single-wing moments `M_{0a}`, `M_{a0}`
  here — exactly the moments the packed head reads. Both layouts now call
  the single owner `_head_wing_interband_weight` (`qsgw_head.py:1124`, from
  `:1357` and `:1682`).
* **The transverse current Hartree is on both routes** — the earlier
  statement here, that it is computed only on the packed routes and omitted
  by `bare_transverse`, was wrong. Its gate is
  `include_transverse = bool(config.bispinor)` (`sigma_dispatch.py:311`),
  which is the master bispinor switch and not `bispinor_gw`, and the term is
  added unconditionally wherever it was built (`:1045-1066`). Every bispinor
  mode and every compute mode carries it unless `omit_v_h` (density
  self-consistency, which rebuilds both fields itself).

## 5. Lineage of the screened four-current solve

This is implementation history, not a second status register; current
coverage is linked from [§1](#four-current-phase-status). Manual chapter 8 and
appendix B are present under `manual/` but are not part of the rendered site.

The packed Lorentz Dyson solve `W^{IJ}_q(μ,ν) = [(1 − Dχ_0)^{-1}D]^{IJ}` over
the compound index `(I⊗μ, J⊗ν)` was built twice. Neither incarnation
evaluates the packed current response at nonzero frequency. GN/HL-PPM and
MPA keep their frequency-resolved scalar charge owner beside the packed
operator and consume the latter's current-index blocks at `ω = 0`. The
packed CC block is absent on these dynamic bare routes, so MPA needs no
second static-W builder.

The second incarnation supersedes the first rather than extending it: it
computes the same sixteen blocks, but with both particle–hole orientations
summed exactly (the June kernel used the `−2F` time-reversal shortcut,
`alpha_chi = -2.0 * ...`, which is invalid on a magnet), with the reverse
orientation's vertex phase conjugated, with the TT blocks Ward-subtracted,
and with a mesh-interleaved packed layout instead of a replicated
`jnp.concatenate` supermatrix. Nothing physical in the June branch is absent
from the current implementation except its `breit_comparison.dat` diagnostic and the `Σ_mnk`
component dumps.

| incarnation | where | what it solves | frequency | status |
|---|---|---|---|---|
| `src/gw/w_bispinor.py`: channel-blocked supermatrix `[C n_C \| T1 n_T \| T2 n_T \| T3 n_T]`, the scalar `solve_w` at size `n_C + 3n_T` | `origin/agent/bispinor-supermatrix-w` (2026-06-16/17), carried to `origin/agent/bispinor-ibz-lorentz-unfold` with an IBZ→full-BZ Lorentz unfold, Σ^B folded into the static QP Σ_xc, and a screened-versus-unscreened Breit comparison (`breit_comparison.dat`) | milestone A: charge χ⁰⁰ only (W^{ij} = bare); milestone B: six TT χ^{ij} plus the three charge–current cross χ^{0i} by folding `γ̃` into the conduction ket of the scalar χ⁰ kernel | ω = 0 only, inside the static-quadrature section; GN-PPM probe screening charge-only | never merged. Measured on FM CrI3 6×6 (640/200 centroids): deeper bands −17…−40 meV Breit, screened vs unscreened differ by < 10 μeV |
| `src/gw/photon_layout.py` + `w_isdf.compute_static_photon_response` + `src/gw/photon_sigma.py`: mesh-interleaved direct sum, distributed Dyson, sixteen-block Σ | current implementation | all sixteen no-pair `χ^{IJ}_0` blocks for `full_static_cohsex`; zero current-response blocks for packed `bare_transverse`; dynamic bare layout has absent CC and no W carrier | current blocks at ω = 0; all sixteen blocks consumed by COHSEX, current-index blocks overlaid on dynamic scalar charge Sigma for GN/HL/MPA | live; `charge_hall_cubature` is a refused retired spelling |

The older full-frequency carrier proposal is not present. The implemented
dynamic seam is deliberately smaller: it does not evaluate
`χ^{IJ}_0(iω_p)` for current-index blocks or feed a frequency-resolved packed
operator into `photon_sigma`. It keeps the scalar charge PPM pipeline intact
and adds the packed `ω = 0` current contribution at `sigma_dispatch`.

## 6. Where the code is

The stage-by-stage wiring — every object's shape, sharding, route membership
and refusal at `34228021` — is
[`architecture/four_current_wiring.md`](../architecture/four_current_wiring.md).

| deck class | charge/current ownership | status |
|---|---|---|
| `bare_transverse` + dynamic `compute_mode ∈ {gn_ppm, hl_ppm, mpa}`; `qp_solver ∈ {one_shot_dft, self_consistent}` | scalar dynamic owner carries CC; packed CC is absent; fixed bare current operator is contracted with the current map's orbitals; SC rebuilds scalar screening/head and scalar+transverse Hartree per map | **served** |
| charge-consuming extensions: screened MPA; screened-current SC; bare COHSEX SC | require, respectively, the MPA exact-static W seam; sixteen χ blocks + packed solve + coupled head per map; or scalar W plus charge S/wings + completion per map | **refused by name**: `packed_screened_mpa_static_role_unimplemented`; `packed_screened_self_consistency_unimplemented`; `packed_bare_cohsex_self_consistency_unimplemented` |

| object | owner |
|---|---|
| bare `D^{IJ}` tiles, TT head slot | `src/gw/v_q_bispinor.py` (`_make_per_q_v_builder_for_tile`, `_tt_head_tensor`) |
| mini-BZ estimators: `⟨v⟩`, `⟨v q q⟩`, `⟨v P^T⟩`, photon cubature | `services/vcoul/src/vcoul/minibz.py` |
| `Σ^B` — the TT part of the packed `Σ`, static AND dynamic routes alike | `src/gw/photon_sigma.py`, the one sixteen-block consumer (`blocks = "all"` for `compute_mode = cohsex`, `blocks = "current"` for the dynamic route, whose charge block is the scalar `Σ_x + Σ_c(ω)`) |
| `Σ^B`, incumbent route only — current bulk/restart/x-only/resolvent/explicit-local classes outside the packed envelope | `src/gw/sigma_x_bispinor.py` |
| which of the two routes a deck takes, and the reason printed in the run record | `gw_config.packed_bare_transverse_route` |
| which compute modes the packed operator serves | `gw_config.PACKED_PHOTON_COMPUTE_MODES` (`cohsex`, `gn_ppm`, `hl_ppm`, `mpa`; screened MPA separately refuses its missing static role) |
| whether the packed operator owns the CHARGE Σ too, or only the current blocks | `gw_config.packed_photon_replaces_charge_sigma` (static) / `gw_config.uses_dynamic_packed_photon_route` (dynamic) |
| which of the two packed modes screens the current blocks | `gw_config.packed_photon_screens_current` |
| head sources, `HeadResolver`, static terms, PPM/complex-pole head `Σ`, Schur fold, rank-1 injection, packed Γ completion | `src/gw/head_correction.py` |
| `S(ω)`, Γ wings, velocity, Hall pseudovector, per-iteration head samples | `src/gw/qsgw_head.py` |
| packed layout and rank-4 updates | `src/gw/photon_layout.py` |
| packed body response and Dyson, for **both** packed modes and **both** compute-mode families | `src/gw/w_isdf.py` (`compute_static_photon_response`, keyword `screen_current`) |
| sixteen-block `Σ`, and its twelve-block current-only selection | `src/gw/photon_sigma.py` |
| the dynamic route's block split (charge half ↔ current half) | `src/gw/sigma_dispatch.py`, `compute_sigma_xc`'s `uses_dynamic_packed_photon_route` branch |
| bare dynamic SC rotation/re-contraction and per-map certificate | `src/gw/sc_iteration.py`, `wfns_transverse_qp` plus the `SC packed current map` record line |
| bounded packed-head response record (`StaticPhotonHeadResponse`; Hall optional), and the by-declaration content list | `src/gw/static_gauge_response.py` (the module docstring) |
| Hall artifact schema | `src/file_io/static_gauge_head.py` |
| four-current carrier resolution (`bispinor_gw` models, `bispinor_current_balance`; which lift each Lorentz label takes) | `src/common/four_current_model.py` (`resolve_four_current_representation`, `FourCurrentRepresentation.current_lift_for`) |
| head-source frequency plan (GN/HL/MPA) | `src/gw/ppm_pipeline.py`, `src/gw/screening.py` |
| the σ·p charge lift and the per-channel velocity lifts of the current carrier, with their provenance strings (§7.1–7.2) | `src/common/bispinor_init.py` (`lift_to_4spinor`, `VELOCITY_KINETIC_BALANCE_LIFTS`) |
| the projector velocity ket per channel and the exact `V_SR + V_SO` split (§7.2) | `src/psp/vnl_ops.py` (`nonlocal_velocity_lift`, `spin_orbit_split_E`) |
| the endpoint rule — one carrier per Lorentz label, and the operands of a Σ block whose two endpoints ride different carriers (§7.2) | `src/gw/wavefunction_bundle.py` (`LorentzCarriers`, `endpoint_bundles`) |
| the Σ-velocity of the current carriers and its two routes (§7.3) | `src/gw/wavefunction_bundle.py` (`add_covariant_current_velocity`), `src/gw/qsgw_head.py` (`covariant_velocity_correction`, `interband_dipole_velocity_correction`), `src/gw/sc_iteration.py` (`resolve_current_velocity_update`) |
| the per-axis derivative rule and the collapsed-axis connection (§7.4) | `src/common/parallel_transport.py` (`link_stencil_orders`, `fourth_order_covariant_derivative`), `src/common/mtxel_sweep.py` (`collapsed_position_operator`) |

## 7. The spatial-current carrier: kinetic balance, the per-channel velocity lift, its Σ-velocity, and the collapsed-axis connection {#current-carrier}

> On branch `feat/bispinor-velocity-lift-2026-09-04` (2026-09-04/05), **not on
> `main`**. The default deck is unchanged: `bispinor_current_balance = kinetic`
> is byte-identical to the shipped lift. Wiring is
> [here](../architecture/four_current_wiring.md#current-carrier-wiring); the
> deck keys are `bispinor_current_balance`, `pseudo_dir`,
> `sc_current_velocity_update` and `parallel_transport_file` in the
> [input reference](../input_reference.md). Evidence: the sandbox report
> `reports/bispinor_velocity_lift_2026-09-04/report.md` and the run directories
> named below.

### 7.1 Two carriers

The four-spinor is `Ψ = (ψ_L, ψ_S)` with `ψ_L` the DFT two-spinor. The charge
channel `γ̃^0 = 1` rides the kinetic-balance lift `ψ_S = (α/2) σ·p ψ_L`,
`p = k+G` in bohr⁻¹ ([`BISPINOR_DHFB_DESIGN.md`](../BISPINOR_DHFB_DESIGN.md)
§3). Its small-component density is the `O(α²)` Dirac density, and nothing in
this section changes it.

The spatial-current channels `γ̃^a = α^a` couple `L` and `S`:
`j^a_mn = ψ_m† α^a ψ_n = ψ_L,m† σ^a ψ_S,n + h.c.` With the σ·p lift, at
`q = 0`,

$$
\tfrac{2}{\alpha}\langle m|\alpha^a|n\rangle = \langle m|2(k+G)_a|n\rangle
\quad\text{(Ry)},
$$

the bare momentum, plus the Gordon spin current at finite `q`. The velocity of
the pseudo-Hamiltonian is `v = p + i[V_NL, r] = p + ∂V_NL/∂k`; the σ·p
carrier misses `∂V_NL/∂k`, which is 0.23 of `|2(k+G)|` on Bi and is what
`dipole.h5` and the static-gauge head already carry.
`bispinor_current_balance` selects the carrier of the spatial-current channels
only: the transverse ζ fits, `Σ^B`, the packed current blocks and the finite-q
`α` vertex.

### 7.2 One carrier per Cartesian channel

`bispinor_current_balance = velocity` lifts channel `a` with

$$
\psi_S^{(a)} = \tfrac{\alpha}{4}\Big[\sigma\!\cdot\!\big(2(k+G) + \partial V_{\rm SR}/\partial k\big)
+ \sigma^a\,\partial_a V_{\rm SO}\Big]\psi_L
\qquad\text{(Ry velocities)},
$$

`V_NL = V_SR + V_SO` the j-averaged and spin-orbit parts of the projector
operator. Why the split: `[σ, V_SR] = 0`, so
`σ^a σ^b V_b + V_b σ^b σ^a = 2 V_a` and a spin-scalar velocity may sit inside
the σ sandwich. `V_SO` does not commute with σ; in the sandwich it adds
`(i/2) ε_abc ⟨m|[σ^c, ∂_b V_SO]|n⟩`, which is not part of `∂H/∂k`
(measured 0.20 of `|∂V_NL/∂k|` on MoS2 and 0.92 on Bi with j-resolved
projectors). Behind `σ^a` alone, `σ^a σ^a = 1` returns `⟨m|∂_a V_SO|n⟩`
exactly. No single four-spinor removes the term for all three channels at
once (the sandwich map `X → σ^a X + X† σ^a` has 8 real degrees of freedom; a
spin-matrix-valued velocity needs 12), so there is one carrier per channel.
Result, gated to `1e-11` relative on the MoS2 `cohsex_debug` fixture
(`tests/test_bispinor_velocity_lift.py`):

$$
\tfrac{2}{\alpha}\langle m|\alpha^a|n\rangle^{(a)}
= \langle m|2(k+G)_a + \partial_a V_{\rm NL}|n\rangle
\quad\text{(Ry), spin-orbit included, no remainder,}
$$

plus the Gordon spin current at finite `q`. The spinor width stays four; the
transverse centroid samples triple; no `G` block, vertex table or contraction
changes.

**Endpoint rule.** Every Green's-function endpoint and every Σ bra/ket for
Lorentz label `a` is drawn from carrier `a`:
`G^{AB}(μ_A, ν_B) = Σ_l Ψ^A_l(μ) Ψ^B_l(ν)†`, and the two-component effective
operator is `Σ_AB (L^A)† Σ^{AB} L^B` with `L^A` the lift of channel `A`.
Hermiticity `Σ^{AB†} = Σ^{BA}` follows, and it is the machine check: under
`LORRAX_DEBUG_PRINT=1` the packed `χ_0`, `W` and `Σ_x/SX/COH` print
`max|B − B†|/max|B|`; a carrier leaking across a label would leave
`(relative carrier difference) × |CT|/|CC|`. Measured `2e-16` (`χ_0`),
`9e-14` (`W`), `1e-16` (`Σ_x`) with three carriers on MoS2 3×3 at P4
(`runs/DEV/322_bispinor_velocity_lift_20260904`, leg 10).

**Two other current constructions in the tree.** The static-gauge Hall/head
producer builds the same Hamiltonian velocity differently: the σ·p carrier
plus the explicit projector jet `(α/2) ∂V_NL/∂k ψ_L` in the large sector, no
spin sandwich. It refuses a velocity carrier, because the jet would then count
`∂V_NL/∂k` twice. The exact Hartree and the SC density rebuild load one ψ for
ρ and `J` and keep the charge lift for both; on a time-reversal-symmetric cell
the Hartree current vanishes identically, and the `V_NL` dressing of the
mean-field current is `O(α² V_NL)`. Both are announced in the run record.

**Measured** (MoS2 3×3 `bispinor_debug`, `nval = ncond = 4`,
`runs/DEV/322_bispinor_velocity_lift_20260904`): velocity minus kinetic moves
`sigX` by 0.875 meV max on the bare route and 1.78 meV max on the packed
route, and `sigC` by exactly 0 (`W_00` and `Σ_c` ride the charge carrier).
Kinetic is byte-identical to `origin/main`. On Bi the term the single-carrier
form would have added is 0.19 of the total velocity, so the effect there is
expected larger and is unmeasured.

### 7.3 The Σ-velocity of the current carriers in QSGW

With `H_QP = H_DFT + ΔH` the velocity is `v_QP = v_DFT + v_Δ`,
`v_Δ = −i[r, ΔH] = D_k ΔH` in the Bloch basis (the covariant derivative
`∂_k ΔH − i[A, ΔH]`). Each map, channel `a`'s small component gains

$$
\psi_S^{(a)} \mathrel{+}= \tfrac{\alpha}{4}\,\sigma^a \sum_m \psi_{L,m}\,(v_\Delta)^a_{mn},
$$

a band-space contraction of the resident large components on the head
manifold, added in the DFT basis before the map's QP rotation; then every
distinct channel carrier is rotated. No G-space work and no ζ refit: the term
is a linear combination of pair densities the fit already spans, and ISDF
interpolation is linear. Map 0 has `ΔH = 0`, so the one-shot equivalence
holds. The charge carrier is untouched.

Two routes for `v_Δ` (`sc_current_velocity_update`):

* `covariant`: `D ΔH` by the finite-link covariant derivative the head already
  uses under `sc_head_update = parallel_transport`, built once per map and
  handed to both consumers (one operator, two consumers).
* `interband`: `−i[r, ΔH]` with the interband dipole
  `r_mn = −i v_mn/(e_m − e_n)` from the exact DFT velocity in `dipole.h5`.
  Exact for the interband part: on a k-diagonal `ΔH` it is the scissor
  renormalisation `v_mn (E_n − E_m)/(e_n − e_m)`, on a rotated `ΔH` the
  explicit interband commutator. It **omits** the k-derivative part
  `∂_k Δ_mn − i(A_nn − A_mm) Δ_mn`: the change of the intraband group velocity
  and the diagonal-connection term. Pairs within `sc_exact_degeneracy_tol_ev`
  carry no interband dipole. The head keeps its own velocity on this route.

`auto` takes covariant when the links are resident, else interband when
`dipole.h5` is beside the deck, else off; the run record names the route and
what it omits.

**Measured** (MoS2 3×3 QSGW, GN-PPM, `sc_tol_ev = 1e-5`,
`runs/MoS2/31_sigma_velocity_sc_20260905`, arms 11–13): the converged `E_QP`
moves by 0.24 meV max under the carrier and by a further 0.22 meV max under
the interband Σ-velocity (gap +0.12 meV). Both are below the 2 meV gate of
TASTE 77 and below what a 3×3 grid can claim as converged.

Not done: entering `i[Σ, r]` through the pair densities themselves (the
band-rotation route, with or without a `ζ^0` component on the transverse set;
derivation in the report, round 3).

### 7.4 The collapsed-axis connection

Write `k = Σ_a κ_a b_a`. The finite-link covariant derivative along reduced
axis `a` is `D_a O = ∂_{κ_a} O − i[A_a, O]`, `A_a` the reduced Berry
connection. On a **collapsed** axis — one mesh point; the cell is
vacuum-padded along it, a slab normal or a wire's two transverse directions —
`∂_{κ_a}` of anything vanishes, and `A_a` is the band matrix of the position
conjugate to `κ_a`,

$$
Z_a = \langle m|\,b_a\!\cdot\! r\,|n\rangle = 2\pi\,\langle m|f_a|n\rangle ,
$$

`f_a` the fractional coordinate — a bounded operator because ψ vanishes in the
vacuum. So `D_a O = −i[Z_a, O]` exactly; no k derivative is fabricated. The
position enters as the sawtooth `2π wrap(f_a − f_a^0)` with its branch cut
half a cell from the atoms' circular-mean coordinate `f_a^0`, in the vacuum.
It is applied in G space through the sawtooth's exact Fourier coefficients,
`K(g) = i (−1)^g e^{−2πi g f_a^0}/g`, a Toeplitz product along the box's `a`
axis on the unwrapped `G_a` difference; sampling the sawtooth on the FFT grid
aliases its `1/g` tail (the two agree to four digits on the fixture, and the
grid form is kept as the reference). `r` is time-even (`v = dr/dt` is
time-odd), so the unfold to the full BZ takes no antiunitary sign; applying
the velocity's sign to `Z` put the wrong sign on every time-reversed `k`
(found and fixed 2026-09-05).

Per reduced axis (`common.parallel_transport.link_stencil_orders`):

| points on the axis | rule |
|---|---|
| ≥ 5 | fourth-order ±2 link stencil (the default, bit-identical to before) |
| 3 or 4 | second-order ±1 link stencil |
| 1 | collapsed: `A_a = Z_a`, `D_a O = −i[Z_a, O]` |
| 2 | refused by name: the ±1 neighbours coincide, and it is not a vacuum direction |

The velocity identity is the gate: on a collapsed axis
`−i Z_mn (e_n − e_m)/|b_a| = v_a,mn` against the exact `p + ∂V_NL/∂k`
(`p − i[r, V_NL]` in the stored convention; the only velocity since 2026-09-05,
`common.mtxel_sweep.VNL_VELOCITY_SIGN`).
MoS2 3×3×1 at 16 Ry: 1.6 % relative L2 over 424 resolved pairs; the in-plane
twin fails by 142 %; truncating the sphere to 12/9/6 Ry raises it to
5.3/8.2/7.5 %; vacuum leakage at the cut ≤ 2.4e-5 per band. The residual is
the plane-wave basis's incompleteness in the commutator (`−i[z, PHP]` against
`P(−i[z, H])P`), the same footing the in-plane links stand on. A 2D deck can
therefore run `sc_head_update = parallel_transport` and the covariant
Σ-velocity; on a 3×3 in-plane grid the gate refuses on the second-order
in-plane stencil, as a three-point stencil must, and not on the slab axis
(`runs/DEV/324_pt_collapsed_axis_20260905`).
