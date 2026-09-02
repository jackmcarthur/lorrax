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

## 1. Status in one table

> **Implementation status (2026-09-01).** This table now describes
> `lane/bisp-b-one-packed-mode-2026-09-01@41e2b6b2` (pushed, not on
> `origin/main`), which collapsed the two packed modes into one and made the
> Γ-cell completion the default. On `origin/main@8b6e3cc7` there are still
> two packed modes and the headless one is the defect the ruling names.

| self-energy channel | frequency dependence | Γ-cell head | deck route | status |
|---|---|---|---|---|
| charge exchange `Σ_X` (CC) | none | `⟨v⟩_mBZ`, band-diagonal (§3.1) | every mode | production |
| charge correlation `Σ_SX+Σ_COH` / `Σ_c(ω)` (CC) | static, GN-PPM (two samples), HL-PPM, MPA | scalar `W_h(ω)` from `S_eff(ω)` (§3) | `head_correction = full` | production |
| bare transverse exchange `Σ^B` (TT) | none: instantaneous (bare Breit) | `−⟨v P^T⟩_mBZ` tensor slot (§2) | `bispinor_tt_head_correction = true` | implemented, default off |
| screened TT/CT/TC (packed 4×4) | **static only** (`compute_mode = cohsex`) | charge CC `q²` + the charge wings + **optional** Hall CT/TC `q¹` (§4), **always on**; `head_correction = off` is a DEBUG skip behind a loud banner | `bispinor_gw = full_static_cohsex` | experimental, insulating slab, one shot |
| retarded / dynamic photon `D^{IJ}(ω)` | — | — | none | **does not exist** |

Binding rule ([decisions, 2026-09-01](../architecture/decisions.md)): COHSEX
with bispinors always carries the Γ-cell head; `head_correction = off` is a
debug setting. A mode whose envelope forbids the head is a defect, not a
calculation.

Two facts follow from the table and are easy to miss:

* The photon sector has no frequency dependence anywhere. `Σ^B` uses the
  instantaneous Coulomb-gauge propagator; the packed screened response is
  built once at `ω = 0`. Every dynamic self-energy (`gn_ppm`, `hl_ppm`,
  `mpa`) screens the charge channel only and adds `Σ^B` as a bare
  exchange term.
* The only time-reversal-odd term in the screened photon head is the
  static Hall (Chern–Simons) response. For a gapped system that
  coefficient is topologically quantized (§4.4), so for a Chern-trivial
  insulator it is exactly zero, so the packed mode's head reduces to the
  charge-only head of §3 — its Hall term is zero both by default (an absent
  `static_gauge_hall_file` sets `σ_H = 0`, announced) and by physics.

## 2. The bare propagator and its Γ-cell average

> **Implementation status (2026-09-01, in flight).** No lane changes the bare
> propagator or its Γ-cell average. `lane/bisp-c-bare-as-packed-2026-09-01`
> would route `Σ^B` through the packed contraction with `W = D`, which must
> leave §2.1 bit-identical — that is its gate.

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

`bispinor_tt_head_correction = true` replaces the `q = Γ, G = 0` slot of the
nine TT `V` tiles by `−T_{ij}/Ω_{\rm cell}`, where `T` is sampled by the
same mini-BZ estimator that produces `⟨v⟩` for the charge head
(`vcoul.minibz_transverse_head_avg`, weighted by `P^T` instead of `1`).
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

This term is frequency independent because `Σ^B` is. It is the only Γ-cell
correction on the transverse channels outside the packed modes of §4.

## 3. The charge head: `S(ω)`, local fields, and the three frequency models

> **Implementation status (2026-09-01, in flight).** Unchanged by the current
> lanes except §3.5: the `W[q=0]` Hermiticity consumer is being re-scoped to
> time-reversal-symmetric decks on `lane/bisp-g-trs-gates-2026-09-01`.

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

Facts that hold without time reversal (verified analytically and on a
random-state model, `reports/four_current_head_frequency_audit_2026-09-01`
in the sandbox):

* `χ_0(\mathbf r,\mathbf r';i\omega)` is **real** for any system once both
  ordered particle–hole orientations are summed (the incumbent
  `F + F^†` completion in `gw.w_isdf`). Hence `W_{-q} = \overline{W_q}` in a
  conjugation-equivariant ISDF basis, and the head scalar `q·S(iω)·q` is real.
  The GN fit of §3.3, which uses `Re W_h`, therefore loses nothing at the
  head level on a magnet.
* `χ_0(i\omega)` is **not Hermitian** in `(\mathbf r,\mathbf r')` when time
  reversal is broken and `ω ≠ 0`: its real antisymmetric part is
  `2ω\sum \operatorname{Im}[\overline{\rho_{vc}(\mathbf r)}\rho_{vc}(\mathbf r')]/(ω^2+Δ^2)`,
  odd in the magnetization and vanishing at `ω = 0`. So `W_q(iω_p)` is not
  Hermitian on a magnet, and neither are the elementwise GN pole tensors
  `Ω_q`, `B_q` derived from it. Two consumers assume otherwise: the
  `W[q=0]` Hermiticity gate at the `iω_p` role (`gw.screening._gate_w`), which
  on a magnet reports a physical asymmetry as a defect, and the crossing-window
  completion `(Z − Z^†)/2i` in `gw.ppm_accumulators`, whose derivation needs
  `B_q^† = B_q`. The Laplace windows need only bilinearity and are unaffected.
* The elementwise GN model keeps `Re Ω_q²` per element. On a magnet the
  discarded imaginary part contains the time-reversal-odd dynamic screening
  (the finite-frequency, non-quantized Hall-like response). GN-PPM is
  therefore not the model to use when that physics is the target; MPA fits
  the two frequency halves with independent complex poles and does not assume
  a conjugation relation between them.

## 4. The packed static photon head (`full_static_cohsex`)

> **Implementation status (2026-09-01).** This section describes
> `lane/bisp-b-one-packed-mode-2026-09-01@41e2b6b2` (`cff884e7` made the §4.2
> completion the default under one mode and deleted the producerless
> `StaticGaugeHeadResponse` seam; `cacc4e07` fixed the Hall sign of §4.2).
> On `origin/main@8b6e3cc7` the mode is still named `charge_hall_cubature`
> and the completion is opt-in.

### 4.1 Body and layout

`full_static_cohsex` builds one packed
`C ⊕ T_1 ⊕ T_2 ⊕ T_3` operator (`gw.photon_layout`): the sixteen bare blocks
`D^{IJ}_q(μ,ν)` from `v_q_bispinor.h5`, the sixteen no-pair response blocks
`χ^{IJ}_0` from the kinetic-balance current
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
stale file cannot degrade the run silently. The sign is the live band
orientation's: the persisted `σ_H` is the occupied-bra Berry sum while the
live Adler–Wiser response is energy-ordered (`P = −ΔD`), which is the minus
above (`lane/bisp-b-one-packed-mode-2026-09-01@cacc4e07`, from lane A's
register row; the oracle is
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
producer and **is deleted**
(`lane/bisp-b-one-packed-mode-2026-09-01@cff884e7`).

### 4.4 The Hall coefficient is a topological invariant

For a gapped system the static, long-wavelength charge–current response is
the Chern–Simons term, and its coefficient is quantized (TKNN):

$$
\sigma_{xy}(\mathbf q\to0,\omega=0)=C\,\frac{e^2}{h},\qquad
C=\frac{1}{2\pi}\sum_{n\in{\rm occ}}\int_{\rm BZ}\Omega_n^z\,d^2k\in\mathbb Z .
$$

The producer computes exactly this occupied Berry-curvature sum
(`gw.qsgw_head.raw_hall_pseudovector_sharded`,
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

### 4.5 Open items on this route

> **Implementation status (2026-09-01).** Both items below are closed; the
> section is kept because neither fix is on `origin/main` yet.

* **Interband Γ wing sign — fixed, on a branch.** Both wing kernels built the
  mixed head/body interband weight as `+F\,\overline{P}\,b/(z²−Δ²)`, while the
  shared head convention `P = −ΔD` implies a minus: the wing replaces one
  density-jet leg by the energy-scaled head vertex, and that substitution
  carries a sign the kernels never applied. It cancels in the scalar Schur
  fold `Y W Z` (§3) but not in the single-wing moments `M_{0a}`, `M_{a0}`
  here — exactly the moments the packed head reads. Fixed on
  `lane/bisp-a-fix-deltas-2026-09-01@60b7bbb7`, with both layouts routed
  through one owner (`_head_wing_interband_weight`) so they cannot drift
  apart again. Still open on `origin/main@8b6e3cc7`
  (`qsgw_head.py:1525,1844`).
* **The transverse current Hartree is on both routes** — the earlier
  statement here, that it is computed only on the packed routes and omitted
  by `bare_transverse`, was wrong. Its gate is
  `include_transverse = bool(config.bispinor)` (`gw.sigma_dispatch:310`),
  which is the master bispinor switch and not `bispinor_gw`, and the term is
  added unconditionally wherever it was built (`:930,937`). Every bispinor
  mode and every compute mode carries it unless `omit_v_h` (density
  self-consistency, which rebuilds both fields itself).

## 5. Lineage: which branch holds which screened four-current solve

> **Implementation status (2026-09-01).** Historical; no lane changes it.
> Manual chapter 8 and appendix B named at the end of this section have since
> been revived in the repo at `manual/08_bispinor/` and `manual/appendices/`
> (not part of the rendered site).

The packed Lorentz Dyson solve `W^{IJ}_q(μ,ν) = [(1 − Dχ_0)^{-1}D]^{IJ}` over
the compound index `(I⊗μ, J⊗ν)` was built twice. Neither incarnation ever
evaluated it at a nonzero frequency; the plasmon-probe and MPA screening
requests pass the scalar charge `V_q` in every branch.

The second incarnation supersedes the first rather than extending it: it
computes the same sixteen blocks, but with both particle–hole orientations
summed exactly (the June kernel used the `−2F` time-reversal shortcut,
`alpha_chi = -2.0 * ...`, which is invalid on a magnet), with the reverse
orientation's vertex phase conjugated, with the TT blocks Ward-subtracted,
and with a mesh-interleaved packed layout instead of a replicated
`jnp.concatenate` supermatrix. Nothing physical in the June branch is absent
from main except its `breit_comparison.dat` diagnostic and the `Σ_mnk`
component dumps.

| incarnation | where | what it solves | frequency | status |
|---|---|---|---|---|
| `src/gw/w_bispinor.py`: channel-blocked supermatrix `[C n_C \| T1 n_T \| T2 n_T \| T3 n_T]`, the scalar `solve_w` at size `n_C + 3n_T` | `origin/agent/bispinor-supermatrix-w` (2026-06-16/17), carried to `origin/agent/bispinor-ibz-lorentz-unfold` with an IBZ→full-BZ Lorentz unfold, Σ^B folded into the static QP Σ_xc, and a screened-versus-unscreened Breit comparison (`breit_comparison.dat`) | milestone A: charge χ⁰⁰ only (W^{ij} = bare); milestone B: six TT χ^{ij} plus the three charge–current cross χ^{0i} by folding `γ̃` into the conduction ket of the scalar χ⁰ kernel | ω = 0 only, inside the static-quadrature section; GN-PPM probe screening charge-only | never merged. Measured on FM CrI3 6×6 (640/200 centroids): deeper bands −17…−40 meV Breit, screened vs unscreened differ by < 10 μeV |
| `src/gw/photon_layout.py` + `w_isdf.compute_static_photon_response` + `src/gw/photon_sigma.py`: mesh-interleaved direct sum, distributed Dyson, sixteen-block Σ | `origin/main` since the 2026-08-24/25 integration branches (`integ/full-screened-bispinor-2026-08-24`, `integ/full-bispinor-*`) | all sixteen no-pair `χ^{IJ}_0` blocks (`compute_no_pair_dirac_current_block`, TT Ward-subtracted) and the exact-slab Γ completion of §4 | ω = 0 only (`compute_mode = cohsex`) | on `origin/main@8b6e3cc7` as the two modes `full_static_cohsex` / `charge_hall_cubature`; collapsed to the single `full_static_cohsex` on `lane/bisp-b-one-packed-mode-2026-09-01@41e2b6b2` |

What is on `origin/main` (2026-08-26 and 2026-08-28 integrations), judged
by content rather than commit ancestry: the packed layout, the sixteen-block
no-pair response with exact orientations, the distributed Dyson, the
sixteen-block Σ, the exact-slab Γ completion with the Hall term, the Hall
producer and its artifact, the bare TT head, the three `bare_transverse`
carrier variants, the four-spinor exact G-space Hartree including the
transverse current in every bispinor mode, and the face-layout bispinor ζ
fits. What stayed on `origin/feat/static-photon-effective-response-completion-2026-08-28`
(about 3,200 diverged lines): a frequency-axis carrier for the four-current
head (`gw.four_current_head.FrequencyResolvedFourCurrentHead`, holding
`Q0_direct`, `H_linear`, `S_direct` per stored ω, with an immutable writer
that refuses dynamic rows "until Q0/H/S share one causal response kernel";
its only producer call passes `(0.0+0.0j,)`), the normalized retained
static-response producer with the mixed CT response, the separated q→0 wing
face endpoints, and the completion-receipt persistence. Two items that were
on that list have since been taken from it: the Hall CT sign reconciliation
(`lane/bisp-b-one-packed-mode-2026-09-01@cacc4e07`) and the Adler–Wiser
interband wing-sign fix (`lane/bisp-a-fix-deltas-2026-09-01@60b7bbb7`,
§4.5). Its mode envelope is still `cohsex`-only.

The two 2026-08-22 design audits that preceded the second incarnation
(sandbox `RUNS_INFLIGHT.md` rows "codex-full-screened-bispinor-gw-audit"
and "codex-bispinor-screened-wings-q0-audit", at `origin/main@c344d57c`)
laid out a static-and-dynamic mode ladder, including a hybrid
"charge dynamic, non-charge static" rung. Only the static rungs were built;
no branch contains the hybrid or a dynamic photon block. (The audit texts
survive in the sandbox archive
`pre_2026-08-25_summary_archive_2026-09-01/reports/`.) The ingredient for
one is present: `compute_no_pair_dirac_current_block` accepts any minimax
`quad`, so `χ^{IJ}_0(iω_p)` and hence `W^{IJ}(iω_p)` is one call away, but
nothing wires it into `photon_sigma` or the PPM Σ_c.

A drafted manual chapter 8 ("Bispinor GW", four sections) and appendix B
("The q→0 head of W") exist on `origin/agent/manual@84b8c2aa` (2026-07-13)
and were never merged; chapter 8.4 already records "Not yet built:
transverse screening".

## 6. Where the code is

> **Implementation status (2026-09-01, in flight).** Owners move with lanes A,
> B and C. The stage-by-stage wiring — every object's shape, sharding, route
> membership and refusal — is
> [`architecture/four_current_wiring.md`](../architecture/four_current_wiring.md).

| object | owner |
|---|---|
| bare `D^{IJ}` tiles, TT head slot | `src/gw/v_q_bispinor.py` (`_make_per_q_v_builder_for_tile`, `_tt_head_tensor`) |
| mini-BZ estimators: `⟨v⟩`, `⟨v q q⟩`, `⟨v P^T⟩`, photon cubature | `services/vcoul/src/vcoul/minibz.py` |
| `Σ^B` | `src/gw/sigma_x_bispinor.py` |
| head sources, `HeadResolver`, static terms, PPM/complex-pole head `Σ`, Schur fold, rank-1 injection, packed Γ completion | `src/gw/head_correction.py` |
| `S(ω)`, Γ wings, velocity, Hall pseudovector, per-iteration head samples | `src/gw/qsgw_head.py` |
| packed layout and rank-4 updates | `src/gw/photon_layout.py` |
| packed body response and Dyson | `src/gw/w_isdf.py` (`compute_static_photon_response`) |
| sixteen-block `Σ` | `src/gw/photon_sigma.py` |
| bounded packed-head response record (`StaticPhotonHeadResponse`; Hall optional), and the by-declaration content list | `src/gw/static_gauge_response.py` (the module docstring) |
| Hall artifact schema | `src/file_io/static_gauge_head.py` |
| four-current carrier resolution (`bispinor_gw` models) | `src/common/four_current_model.py` |
| head-source frequency plan (GN/HL/MPA) | `src/gw/ppm_pipeline.py`, `src/gw/screening.py` |
