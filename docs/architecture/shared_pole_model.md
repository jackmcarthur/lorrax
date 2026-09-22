# The shared-pole screened interaction

This page is the design of the shared-pole route for the correlation part of the screened interaction,
`sigma_w_model = shared_pole`. It states the object, how the response bank samples it, how the constructor reduces
the samples to one pole set per parent q, what the store holds, how Σ consumes it, and which gate or test certifies
each step. Deck keys are in the [input reference](../input_reference.md). Equations are numbered `(SP n)`.

Notation. All frequencies are in Ry. `q` is a raw parent (one irreducible q; `−q` is its own parent when time
reversal is broken). `n` is the packed centroid count (`meta.mu_basis.n_packed`). `z` is a complex frequency and
`s = z²`. `b` is the stored factor (dataset `factor`). `Ω_j > 0` are poles and `Λ = diag(Ω_j²)`. `R_±` are residues
and `M_k` the physical high-frequency moments.

## 0 Where the code is

| section | owner |
|---|---|
| §2 the response bank | `gw.response_bank`, driven by `gw.shared_pole_screening` |
| §3 directions | `gw.shared_pole_directions` |
| §4, §5 the two pencils | `gw.shared_pole_pencil` |
| §4, §6 reduction, paired basis and cut | `gw.shared_pole_reduction` |
| §6 dedupe, §9 measured gates | `gw.shared_pole_gates` |
| the chain that calls them, per round of parents | `gw.shared_pole_constructor` |
| rounds, partner exchange tables, the round program | `gw.shared_pole_local` |
| §10 the byte model and its ledger rows | `gw.shared_pole_capacity` |
| §7 the store | `file_io.shared_pole_store` |
| §8 the Σ consumer | `gw.mpa.sigma` |
| the recipe, the gate tables and the receipt | `gw.shared_pole_recipe` |

## 1 The object

Σ needs `W_c(q, z) = W(q, z) − v(q)` on the `n × n` centroid basis. With the RPA response `χ = χ⁰ + χ⁰ v χ`,

$$ W_c(q,z) = v\,\chi(q,z)\,v . \tag{SP 1} $$

The route represents it by one set of real poles per parent, shared by every matrix element:

$$ W_c(q,z) = \sum_{j} \frac{R_{+,j}(q)}{z-\Omega_j(q)} \;-\; \sum_{k} \frac{R_{+,k}(-q)^{\mathsf T}}{z+\Omega_k(-q)},
\qquad R_{+,j} = c_j c_j^{\dagger} \succeq 0 . \tag{SP 2} $$

The minus sign and the transpose are the particle–hole pairing `R₋(q) = R₊(−q)ᵀ`, which holds without time
reversal. Each parent stores only its positive poles; the hole side of `q` is read from parent `−q`.

With time reversal `R₊(−q) = R₊(q)ᵀ`, so (SP 2) is even in `z`. With `b_j = √(2Ω_j) c_j`,

$$ W_c(q,s) = b\,(s-\Lambda)^{-1}\,b^{\dagger}, \qquad b \in \mathbb{C}^{n\times K} . \tag{SP 3} $$

Units: `b` in Ry^(3/2), `Λ` in Ry². The time-domain weight `d_j(τ) = exp[−i(Ω_j − E_ref)τ]/(2Ω_j)` stays separate
from `b`. When time reversal is broken and `q ≡ −q`, the odd part of (SP 2) is the imaginary part of the same
residue, so the odd channel costs no storage:

$$ W_c(z) = \sum_j \mathrm{Re}(c_jc_j^\dagger)\frac{2\Omega_j}{z^2-\Omega_j^2}
 + i\,\mathrm{Im}(c_jc_j^\dagger)\frac{2z}{z^2-\Omega_j^2} . \tag{SP 4} $$

## 2 The response bank

The bank (`gw.response_bank`) samples `W_c` and `∂W_c/∂s` at the recipe's support points and records the
high-frequency moments. It never forms `W` at a real frequency.

Production uses 18 fitted supports (line plus imaginary), with held supports additional,
and a retained-pole budget `ceil(1.8 N_mu)`. Line sites follow the band-structure support
rule in report §IV.B: equal quantiles of the square root of the crossing density,
broadened at the consumer's η. Delivered states within 5 eV of μ carry offsets from
−5 to +5 eV in 0.25 eV steps; crossings include levels at every k. The implementation
is `support_rule_line_sites`; this is distinct from the Si-only sparse n14 greedy search.
The existing SC support session retains the line-site tuple while its interval stays
enclosed and rebuilds it on an interval or policy change. Samples and the pole model
are rebuilt from the current state on every map.

The imaginary support top is set by the active-charge plasma scale plus the
recipe margin; it does not grow with the full input-band energy span. The 18
supports therefore target the consumer's low-energy and collective-mode
range. They are not a uniform approximation guarantee for arbitrary band
energy ranges. The response-bank time rule, the finite rational reduction,
and the projected Sigma error have separate certificates.

**Frequency-specific stream.** The scalar service fits each primitive independently:

$$ F(d,z)=\frac1{d-z}\simeq\sum_j c^F_j e^{-(d-r)t^F_j},\qquad
B(d,z)=\frac1{d+z}\simeq\sum_j c^B_j e^{-(d-r)t^B_j}. \tag{SP 5} $$

Here r is the scalar reference: L for a uniform fit, zero for the
occupation-weighted fit described below.

Times are complex with nonnegative real part. For nonnegative endpoint offsets,
the forward product is `Gu(t) conj(Gf(conj(t)))`; its reverse is formed as
`conj(A(conj(t)))`, preserving the same damping and reversing the orbital product.
It is **not** `conj(A(t))`. Both primitives contribute with a minus sign.
`minimax.response_frequency_rule` fits values and squared-denominator derivative
targets on the same nodes; its reported errors are sampled errors, not continuum
certificates. A matched QP-energy comparison is the observable acceptance check.

**Orientation.** Σ's `G_{k−q} W_q` contraction is exact for

$$ \mathcal F_q[f](\mu,\nu) = \sum_R f(r_\mu,\, r_\nu + R)\, e^{i q\cdot R}, \qquad W_q = \mathcal F_q[W] . \tag{SP 6} $$

The incumbent trace builds `conj(G(w, t)) = G(w̄, t̄)ᵀ` and gathers the forward transform at `q`, which returns
`F_q[χᵀ]`. With time reversal the two orientations are equal and the TRS bank keeps that trace bit-for-bit. An
ordered bank (`ordered=True`, and `pair_mode = "laplace_ordered"`) builds `G(w̄, t̄)` and gathers rows at `−q`,
which is `F_q[χ]`. With the transposed orientation on a TR-broken deck each Green's-function branch would take the
other branch's residues, `Σ[W^even] − Σ^odd`. Tests: `tests/test_shared_pole_stream_orientation.py` (ordered
retarded and Laplace rows equal `F_q[χ]` to 1e-10 on a TR-broken lattice and miss `F_q[χᵀ]`).

**Domain and derivative.** The domain spans every nonzero occupied/empty-weight
pair after the existing sample-only 1e-14 activity floor, including negative
transition energies and signed metallic weights; exact moments are untruncated. Scalar plans receive 2 eV padding per
one-particle endpoint for SC reuse, without changing occupations or energies.

The crossing fit uses the current occupation envelope
`|f_n u_m| <= A_f A_u min(1,exp(beta*d))`, where `d=E_m-E_n` and
`A_f=max(1,max|f|)`, `A_u=max(1,max|u|)`. The bank obtains beta from the
minimum one-sided log slopes of `|f|/A_f` above mu and `|u|/A_u` below mu;
no Fermi–Dirac assumption or additional occupation cutoff is made. Sampled
value and derivative errors use that envelope, with tolerance divided by
`A_f A_u`. For beta>0 the scalar reference is zero and both Green references
are mu; `0<=Re(t)<=beta` bounds the occupation-weighted Green factors.
Zero beta retains the uniform fit. SC reuse requires a current beta at least
as large and a current amplitude no larger than those used for construction.
This is an occupation-weighted approximation on the signed interval, not an
explicit triangular projection onto positive/negative transition energies.

$$ \frac{\partial F}{\partial s}=\frac1{2z(d-z)^2},\qquad
\frac{\partial B}{\partial s}=-\frac1{2z(d+z)^2},\quad s=z^2. \tag{SP 7} $$

**Dyson and storage.** A donated `[2,q,mu_X,nu_Y]` accumulator holds the value
and derivative at one frequency. Both coefficient rows consume the same
Green/FFT products in one scan. One collective bank transaction stores both
fields for all irreducible-q parents, draining bounded slices before publishing
its masks. Dyson commits W first; bounded reads recover full W for
`dW/ds = W (dchi/ds) W`, without a second solve or adjoint.
Charge storage is W−V; photon storage is W−W_infinity with its separate constant.
The existing per-field write masks allow a value commit before its derivative.

**Moments.** With `W_c(z) = Σ_{k=0}^{3} 2 M_k z^{−(k+1)} + O(z^{−5})`,

$$ M_0 = \tfrac12 m_0,\quad M_1 = \tfrac12 C_2,\quad M_2 = \tfrac12 m_2,\quad M_3 = \tfrac12 C_4, \tag{SP 8} $$

where `m_0 = Σ_j (R₊ − R₋)` and `m_2 = Σ_j Ω_j² (R₊ − R₋)` are the odd coefficients and `C_2`, `C_4` the even
coefficients of the full Dyson series. `M_0` and `M_2` vanish with time reversal and are written only to an
ordered bank. All four are exact band sums of the correlations the stream uses; the Coulomb prefactor and the
orthonormal FFT together scale them by `(1/√N_k)(1/√N_k) = 1/N_k`. Test:
`tests/test_shared_pole_bank_moment_roundtrip.py` (planted `M_k = m_k/2` survive bank.h5 bitwise and reproduce
`m_0..m_3` through the ordered infinity block).

## 3 Directions

For each fitted support the constructor selects a narrow direction set `Q` (`n × r`) from the sample, never the
full matrix, through `distrib_la`:

- line supports: right singular vectors of `W_c(s)` above `direction_cutoff` (relative);
- imaginary supports: leading eigenvectors of `−Herm W_c(iy)`, width `imaginary_width`;
- infinity: leading eigenvectors of `M_1`, width `infinity_width`.

Each state is `(node, Q, O = W Q, D = W' Q)`. A line support also takes its conjugate partner: `W(s̄) = W(s)^†`,
so the partner's directions are `O` itself and no extra selection is made.

## 4 The even pencil (time reversal)

For `X_b = (s_b − T)^{−1} b^† Q_b` the resolvent identity gives, without forming `X`,

$$ G_{ab} = X_a^\dagger X_b = \frac{Q_a^\dagger O_b - O_a^\dagger Q_b}{s_b - \bar s_a},\qquad
H_{ab} = X_a^\dagger T X_b = s_b\,G_{ab} - Q_a^\dagger O_b, \tag{SP 9} $$

with the confluent limit `G_aa = −Q_a^† D_a`. The infinity rows use `M_1` and `M_3`:

$$ G_{\infty b} = Q_\infty^\dagger O_b,\quad H_{\infty b} = s_b G_{\infty b} - 2 (M_1 Q_\infty)^\dagger Q_b,\quad
G_{\infty\infty} = 2 Q_\infty^\dagger M_1 Q_\infty,\quad H_{\infty\infty} = 2 Q_\infty^\dagger M_3 Q_\infty . \tag{SP 10} $$

**Reduction.** Equilibrate `G` by its diagonal, keep eigenvalues `γ > keep · γ_max` (`normalized_gram_keep`),
correct the retained metric by coupled Newton–Schulz to `Z^† G Z = I`, and solve the Hermitian problem
`Z^† H Z = U Λ U^†`. The model is `b = O Z U`, `Λ`. Poles below `lambda_cutoff_ry2` are dropped only within the
factor-weight budget (`zero_ritz_policy`).

## 5 The ordered pencil (time reversal broken)

The linear particle–hole pencil `(z σ₃ − M)` has the same resolvent-identity columns in `z`:

$$ \mathcal G = X^\dagger \sigma_3 X,\qquad \mathcal H = X^\dagger M X, \tag{SP 11} $$

with nodes `{z, z̄, −z̄, −z}` and `∂W/∂z = 2z ∂W/∂s`. Every state `X(z)` on `Q` is followed, after all originals, by
its mirror `X(−z)` on the same `Q`; for an exactly covariant representation,
`W_q(−z̄) = conj W_{−q}(z)` gives that mirror from parent `−q`. An approximate
ISDF photon operator need not obey this identity closely enough for the
finite Gram. A photon bank therefore stores `Wc_mirror` and
`dWc_mirror_ds` at `−conj(z)` from the same physical state, with the original
parent's V and contact. The constructor takes the stored value directly for
the conjugate mirror state and its adjoint for the original mirror state;
both derivatives acquire the actual mirror node's `2z` factor. CC, TT, CT
and TC use those same authenticated fields. M0..M3 are computed from the
original q operator, V and contact and still feed the unchanged infinity
block; finite-sample mirror positivity alone does not certify that block.
Poles
`ℋ ≻ 0` on the retained span is sufficient for Hermitian whitening, real
finite poles `1/μ`, and residues with the sign of the pole. A stable RPA
(`M ≻ 0`) guarantees this condition. Reality alone is insufficient to infer
stability: `G = I`, `H = diag(1,-1)` has real generalized eigenvalues despite
indefinite H. The constructor admits the positive retained metric; it does
not certify stability of discarded or unprobed physical states.

### Signed photon interaction and contact

The photon bank uses the physical paramagnetic response
`χp(z) = C (z J − H0)^−1 C†`, with particle/hole signature J and positive
transition energies in H0. The columns of C include the square roots of
positive occupation differences and the physical response prefactor; the
ordered negative-frequency residues have the opposite sign. Cartesian
current vertices stay in C, with their physical complex phases.

In `response_bank.response_algebra`, `χ = χp − D`, where
`D = cell_volume * TT_contact`, and `W = (I − V χ)^−1 V`.
The bare photon V is Hermitian and signed. If the contact solve exists,
`U = W∞ = (I + V D)^−1 V` is Hermitian. The resolvent identity gives

$$ W(z)-U = U C [z J-(H_0+C^\dagger U C)]^{-1} C^\dagger U. $$

Thus positivity of `H0 + C† U C` is a sufficient stable-realization
condition even for indefinite U. With this condition and Hermitian U,
the ordered residue at Ω has sign(Ω) times a positive semidefinite
matrix; a diagonal TT block inherits this property. Neither an indefinite
U alone nor a real spectrum proves the condition. Positive and negative
frequency sides must retain their ordered partner convention.

The scalar positive-V bound on the V-whitened `−Herm Wc(iη)` in `[0,I]`
does not apply to signed V. Its absence does not remove the retained-H
stability, finite-factor, zero-weight, partner or moment checks. Cross-sector
Cauchy–Schwarz applies to a positive spectral metric, not an arbitrary
complex-frequency W block. The separate constant `U−V` contributes to
Sigma independently of the pole model of `W−U`.

Sector stores stamp their actual realization as `raw-sector-endpoint-v1`
in the authenticated model recipe and sector manifest. This names raw
physical charge/current endpoints with their own centroid bases, followed
by the stored symmetry endpoint action. It performs no scalar little-group
averaging. CC and TT have independent poles; CT_C and CT_T share one
retained mask and pole ordering, with each endpoint passing its own lost
weight check. The manifest binds all four stores and the current-map
`W∞−V` bank constant.
Positive retained `H` certifies each projected realization. Since CC, TT
and CT are fitted independently, this does not certify positive spectral
residues of the assembled photon matrix: a CT-only pole can have a nonzero
cross residue while its CC and TT diagonal residues vanish. Assess the
assembled model through sector and integrated-Sigma accuracy receipts.

**Infinity block.** `k₀ = σ₃ C^† Q_∞` and `k₁ = σ₃ M σ₃ C^† Q_∞` need all four moments of (SP 8):

$$ \mathcal G_{\infty\infty} = \begin{pmatrix} P_0 & P_1\\ P_1 & P_2 \end{pmatrix},\quad
\mathcal H_{\infty\infty} = \begin{pmatrix} P_1 & P_2\\ P_2 & P_3 \end{pmatrix},\quad P_k = 2 Q_\infty^\dagger M_k Q_\infty . \tag{SP 12} $$

A bank without `M_0`, `M_2` builds finite states only and records the block NOT_MEASURED.

## 6 Paired basis, cut and dedupe

**Paired basis.** The congruence

$$ w_b = \tfrac12\big(X(z_b) + X(-z_b)\big),\qquad v_b = \frac{X(z_b) - X(-z_b)}{2 z_b},\qquad
w_\infty = k_1,\quad v_\infty = k_0 \tag{SP 13} $$

gives `ℋ' = [[H_ww, H_wv], [H_vw, H_vv]]` and `𝒢' = [[G_ww, G_wv], [G_vw, G_vv]]`. With time reversal
`ℋ' = diag(H_s, G_s)` and `𝒢'` is off-diagonal, `(G_s, H_s)` being the even pencil of §4.

**Cut.** Equilibration, the keep cut and the metric correction act on `H'_vv` alone, as the even route acts on
`G_s`; the kept span `Z` is applied to both halves. The restricted pencil `H_r = [[A, B], [B^†, I]]` gets a second
relative keep cut, `Ψ = L^{−†}` with `H_r = L L^†`, `μ = eig(Ψ^† 𝒢_r Ψ)` and `c = O_r Ψ rot`. The stored model is

$$ b = \sqrt2\,c\,\mu^{-1} \ (\mu > 0),\qquad \Lambda = \mu^{-2}, \tag{SP 14} $$

which equals the even model with time reversal. `|μ| ≤ keep · max|μ|` are poles at infinity; their output weight is
reported (`infinite_weight_ok`).

CC/TT spans retained for CT use `normalized_gram_keep.sector_threshold`
from the ordered gate table in both paired Gram cuts. Separately normalizing
nearly dependent sector directions can amplify their cross-block errors;
these cuts remove weak Gram directions before that normalization. The
ordinary ordered reduction, joint CT cut and inverse-pole filter still use
`normalized_gram_keep.threshold`. The negative-Gram validity threshold is
unchanged. Rank truncation must also be assessed through held-response and
integrated-Sigma errors; passing the Gram gate alone certifies neither.

**Dedupe.** For an imaginary support, or a line support with `Re z = 0`, the conjugate partner brings no new tangent
with time reversal: `W Q ∈ span(Q)`. Only the component of `O = W Q` orthogonal to `Q` above `direction_cutoff`
survives, so a time-reversal-symmetric bank adds no partner columns and the ordered model equals the even one at
equal rank.

**Ordered-sector numerical treatment.** Bispinor sector models use the
explicit ceiling

$$
 \Omega_{\mathrm{treat}} = 2\left[
 \max E_{\mathrm{cond},\chi}-\min E_{\mathrm{val}}
 \right]. \tag{SP 14a}
$$

This is a numerical treatment policy, not a bound on collective modes. The
first shared-pole SC map freezes the ceiling. A later map reports twice its
current chi transition span as the ceiling a fresh map would choose, but that
diagnostic does not change or invalidate the frozen treatment. The constructor
applies the same frozen mask to every current model, and the ceiling never
widens after map 0. The initial fixed Sigma rules certify
the intersection of `[0, Omega_treat]` with each existing pole selector, in
addition to the normal state and endpoint padding, so a retained pole may
move anywhere within that declared range without re-keying the nodes. Modes with
`Omega > Omega_treat` are inactive in the published model: both factor
endpoints are zero and their pole slots use the normal inactive sentinel.
CC and TT use independent masks. CT_C and CT_T use one common mask and pole
census, preserving the two ordered endpoint products. Constructor held rows
continue to score the untreated signed fit and do not certify this treatment;
accuracy requires a projected Sigma comparison at fixed state, occupations,
vertices, eta, and q weights. A measured result applies only to the states and
consumer window in that comparison.

**Layout.** The constructor reduces one round of parents at a time, one parent per rank (batch layout
`P(('x','y'), ...)`): `gw.shared_pole_local.round_program` packs each parent's panels to the round extent
(`round_tables`; ordered originals and mirrors as two halves of one extent), assembles and reduces its pencil with
local dense kernels, and sorts its poles, all on that rank; synthetic slots are skipped. The model checks (passivity,
held `W` and `dW/ds`, moments) run the same way (`round_checks`), and the store receives one write of every parent in
canonical order (`canonical_factors`). A round refuses before it
runs if eight `[R, R]` blocks and the eigh workspace do not fit one device (`distrib_la.fits_local`). Receipts report
each parent at its own extent: the round padding's zeros are dropped from the Gram spectrum
(`own_extent_receipts`). Called on face stacks instead, the same blocks are assembled and symmetrized through
`distrib_la` face blocks (`hermitian_block`, `hermitian_part`, `join_columns`, `on_face`): eager concatenation and
`a + a^†` of face-sharded operands would come out replicated on every rank.

## 7 Store schema

`model.h5` (schema `lorrax.shared-real-pole.v1`) and `bank.h5` (schema `lorrax.shared-real-pole-bank.v1`) share one
authenticated header (`file_io.shared_pole_store`):

| field | meaning |
|---|---|
| `identity` | current-state hamiltonian, occupations, wavefunctions, recipe and gate hashes |
| `recipe`, `recipe_hash` | the resolved recipe; restart and SC maps refuse a stale one |
| `representation` | model: `scalar-trs-even-s` (SP 3) or `scalar-ordered-ph` (SP 2); ordered bank: `charge-ordered-z` |
| `n_q_irr`, `q_irr_full_idx`, `qirr`, `operations` | raw parents and the authorized symmetry rows |
| `n_mu_logical`, `nspinor`, `centroid_digest` | basis identity; the operator is the `μ × μ` charge response on scalar and two-component decks |
| model: `factor [q, μ, 1, K]`, `poles2_ry2 [q, K]`, `K [q]` | (SP 3) per parent; inactive columns have `b = 0`, `Λ = 1 Ry²` |
| bank: `Wc`, `dWc_ds [q, a, μ, μ]`, `M1`, `M3` (+ `M0`, `M2` when `odd_moments`) | samples and moments of §2 |

## 8 The Σ consumer

Σ contracts `G` with `W(τ)` synthesized from the factors in bounded parent and column panels
(`gw.mpa.sigma._shared_pole_w_synthesis`):

$$ W_{c,+}(q,\tau) = b\,\mathrm{diag}\big(d_j(\tau)\big)\,b^{\dagger}, \qquad
d_j(\tau) = \frac{e^{-i(\Omega_j - E_{\rm ref})\tau}}{2\Omega_j} . \tag{SP 15} $$

Charge and CC/CT/TT synthesis use the existing Green-function GEMM with the
wavefunction layout: `low_mem_bands=true` keeps pole columns face-sharded;
`false` retains axis factors with replicated columns. Admission prices those
configured factors before reading them; the synthesized W always uses both
mesh axes. A resident factor set is read and placed once per Sigma call.

**Hole routing.** Conduction windows take `W₊(q)`. An ordered store routes valence windows to the particle–hole
partner,

$$ W_-(q,\tau) = W_+(-q,\tau)^{\mathsf T}, \tag{SP 16} $$

gathered at `−q` on the replicated q axis and transposed on its faces (`shared_pole_hole_kernel`); time-reversal
stores never take this branch. Tests: `tests/test_shared_pole_ordered.py` (the synthesized orientations are the
positive- and negative-frequency Lehmann sums) and `tests/test_shared_pole_lattice_sigma.py`: on a TR-broken
lattice the production τ kernel reproduces real-space `Σ = iGW` to 1e-10 relative, and the swapped routing misses
by more than 1e-3.

**Two-component decks.** `W` is spin-scalar; `G` carries the spinor axes, and the τ kernel broadcasts `W_q` over
both (`ppm_tau_kernel` `prep_w`). The factor spin axis is 1 on every admitted deck. The full Gamma head
(`gw.shared_pole_head`) is the same charge head on both stores: its wings trace the spinor index inside each
vertex, its capacity is `2/(n_spin n_spinor)` states per band, and the fold runs on the `n_mu × n_mu` body; the
head's capacity identity carries the store's `nspinor` because the ledger unit is `16 Q (N_spinor N_mu)² / P`.

**Ordered stores.** `head_correction = full` refuses (`GATE shared_pole_head_ordered`): the Gamma body evaluator is
the even form `b (s − Λ)⁻¹ b†`, not the signed model (SP 2), and the signed wing/body fold is deferred (owner scope
2026-09-21). The head an ordered store carries is `head_correction = no_local_fields`, the direct tensor `S(ω)`
finalized with no Gamma body; it needs no time-reversal assumption
([four-current heads §3.5](../theory/four-current-head-corrections.md), `tests/test_head_direct_ordered.py`).

**Debug.** `LORRAX_DEBUG_SHARED_POLE_EVEN_PART` (debug only) feeds `[W₊(q) + W₊(−q)ᵀ]/2` to both branches of an
ordered store, so `Σ^odd = Σ[W] − Σ[W^even]` can be measured; see `docs/dev/env_vars.md`.

## 9 Gates and tests

Every construction receipt row carries version, value, threshold and PASS/FAIL/WARN/NOT_MEASURED. The TRS table is
`shared_real_pole_gates_v1_r3b`; `shared_real_pole_gates_ordered_v1` copies it and replaces the rows marked
"ordered".

| gate | certifies | refuses? |
|---|---|---|
| `representation` | TRS: scalar and TRS allowed; ordered: TRS broken and an ordered bank | yes |
| `normalized_gram_validity` | equilibrated Gram (or `H'_vv`) spectrum min/max above threshold | yes |
| `normalized_gram_keep` | the retained rank at the recipe cut | diagnostic |
| `retained_subspace_moments` | TRS: projected `M_1`, `M_3` identity; ordered: `m_0..m_3` on the infinity directions | TRS yes, ordered diagnostic |
| `zero_ritz_policy` | dropped factor weight within budget; ordered also `infinite_weight_ok` | yes |
| `sector_pole_treatment` | bispinor-only map-0 ceiling and complete CC/TT/common-CT masks of (SP 14a); reports the current fresh-map candidate, counts and extrema, not accuracy | invalid mask yes |
| `finite_factors_poles` | finite `b`, positive finite active `Λ`, exact inert sentinels | yes |
| `passivity` | V-whitened `−Herm W_c(iη)` in `[0, I]`; ordered: Hermitian part, the anti-Hermitian part is reported | yes |
| `model_reciprocity` | TRS only: transpose symmetry of symmetric held samples | yes (TRS) |
| `held_w` | held `W`, `∂W/∂s` relative errors | diagnostic |
| `full_m1_defect`, `full_m3_defect` | full-matrix moment defects | WARN only |
| `capacity`, `stream_peak`, `sigma_peak` | device admission within budget; inherited peaks recorded separately | capacity yes |
| `rule_validity`, `sc_rebuild` | bank and Σ certificates cover the current domains; SC rebuilds from current bands | yes |

Fast CPU tests (4 host devices where a mesh is needed):

| test | pins |
|---|---|
| `test_shared_pole_ordered.py` | planted ordered oracle, projected moments per order, ordered = even on TRS data at equal rank, dedupe keeps no partner, generic-q assembly, Σ orientations, two-component routing against the Lehmann sum |
| `test_shared_pole_stream_orientation.py` | the ordered stream stores `F_q[χ]` (SP 6) |
| `test_shared_pole_lattice_sigma.py` | ordered Σ = real-space `iGW` on a TR-broken lattice; swapped routing fails (SP 16) |
| `test_shared_pole_bank_moment_roundtrip.py` | bank → constructor moments, `M_k = m_k/2` (SP 8) |
| `test_shared_pole_pencil_faces.py` | pencil blocks come out `P(None,'x','y')` in both routes |
| `test_shared_pole_head_two_component.py` | a spin-doubled two-component store reproduces the scalar `S`, `Y`/`Z`, static wings and folded `S_eff`; a global SU(2) rotation leaves them invariant |
| `test_shared_pole_head_capacity.py` | the head door admits N_spinor 1 and 2, refuses ordered stores and the N_spinor = 4 lift by name, and refuses a store whose spin geometry differs from the map's |
| `services/distrib_la/tests/test_eigh_keeps_operand.py` | the planned eigh does not overwrite its operand |
| `test_slab_io_mode_required.py` | `SlabIO`/`open_file` refuse a missing `mode=` |

## 10 Byte model

Per rank on an `x × y` mesh with `P = Px Py` and pencil side `R`:

$$ \text{reduction} \approx 16\,\big(14 R^2 + 12 n R\big)\,b/P + 16\cdot 3 n r\,b/P + \text{native eigh workspace}, \tag{SP 17} $$

(`shared_pole_capacity.shared_pole_byte_terms`; `ConstructorCapacity` beside it turns those terms into the map
ledger's rows). A round has `b = P`: every rank holds one whole parent (§6). The native cuSOLVERMp eigh adds a
private operand tile of `n²/P` next to its workspace, which `distrib_la.workspace_bytes_per_rank` includes; a byte
model without that tile under-counts the measured CrI3 q=1 construction peak (2.140 against 2.907 GiB per rank).


### Photon bank residency audit (2026-09-17)

This is a source audit of the literal-mirror producer, not a complete HLO or
native-workspace proof. Let K be full k count, m the packed photon extent,
s the spin count, B the band carrier, a the support count, q the admitted
union of exact original and negative momentum rows, and P=Px Py. Complex
arrays use 16 bytes per element. Run477/32 has K=64, m=1032, s=4, B=36,
a=22, q=22 (13 original parents), P=4. Its recorded JAX high water is
18,203,126,272 B/rank; the original panel admission was 5,209,765,972 B/rank.
Those differently scoped numbers do not by subtraction measure a missing
allocation. Native allocations are excluded from the JAX high water.

| Object | Actual production call / layout | Shape | Full bytes | Per-rank bytes |
|---|---|---|---|---|
| Prepared endpoints, four carriers | `prepare_photon_carriers`; mun `[K,s,m_X,B_Y]`, nmu `[K,B_X,s,m_Y]` on face layout | four K s m B | 64 K s m B | 64 K s m B / P |
| Bare photon V | `photon_bare_operator`, `[q_parent,m_X,m_Y]` | q_parent m² | 16 q_parent m² | 16 q_parent m²/P |
| Contact/reference buffers | `photon_static_contact` / slab read, `[1,m_X,m_Y]` | m² each | 16 m² | 16 m²/P each |
| Two live full-spin Green functions | `response_stream` / contour kernel, `[K,s,m_X,s,m_Y]` | 2 K s² m² | 32 K s² m² | 32 K s² m²/P |
| FFT and Green contraction temporaries | inside the compiled stream; explicit face stream signatures, interior HLO still to audit | compiler dependent | not inferred | compiled temporary bytes; external FFT workspace separate |
| Response carry | `integrate_response_frequency`, `[2,q,m_X,m_Y]`; one frequency, value + ds | 2 q m² | 32 q m² | 32 q m²/P |
| Dyson arguments/results | `response_algebra`, `[a,m_X,m_Y]`; original and mirror run sequentially | bounded a m² panels | 16 a m² each | 16 a m²/P each, plus native LU work |
| Ordered bare moments | `exact_bare_moments`, four `[q,m_X,m_Y]` arrays | 4 q m² | 64 q m² | 64 q m²/P |
| Sector samples and moments | sector reader; four photon sample fields and four moment fields, `[q_XY,a,mu,mu]` | sector dependent | 16 times element count | batch divided over P, padded to whole-parent rounds |
| Directions, Ritz pencil and factors | constructor local parent algebra; `[q_XY,mu,R]`, `[q_XY,R,R]` | b mu R and b R² | 16 b mu R; 16 b R² | 16 ceil(b/P) mu R; 16 ceil(b/P) R² |

The response carry grows by 22/13 for this exact-mirror union. It does not
create another Green stream. The four prepared face-layout endpoints alone
occupy 152,174,592 B/rank on this geometry. Family unfolding, canonical
unpacking, gamma action and photon packing have additional preparation
intermediates; retained-endpoint pricing does not certify their peak.

Face arrays remain constant per rank under m² proportional to P. Constructor
whole-parent rounds have a strong-scaling ceiling: with b=P each rank retains
a whole parent pencil and factor. Face/batch conversion uses explicit
all-to-all exchanges, not a host/global matrix copy. Replicated poles and
scalar diagnostics are small metadata. High-memory (`axis`) wavefunctions
have only one mesh axis in each carrier; this audit establishes face-layout
residency only. Both existing band-storage layouts bind the same charge and
photon Green/FFT kernels through the carrier's layout; response outputs stay
sharded on both endpoint axes. `tests/multi_device/response_bank_layouts.py`
checks complex ordered charge-stream parity and the all-P output placement.
This does not authorize a new layout or replicated bulk array.

Axis-layout frequency streams use the Green builder's prepared active-band
GEMM for both charge and photon vertices. Bounds enclose every parent's exact
occupied/empty weight support, are shared across frequencies, and are rebuilt
from the current SC occupations. Zero tails are skipped without changing any
weight or time node; underflow inside the enclosing interval remains harmless.

The producer must release ordered `o0/o1`, the per-parent operand tuple,
result list and contact constant after their synchronous writes. Otherwise
they survive into the next moment panel. Photon stream compiler temporaries
must be admitted alongside the already priced carry and endpoint buffers;
scalar streams retain their existing matched-reference admission route.
Compiler outputs and aliases are recorded separately to avoid charging the
donated carry twice. This is a pre-execution admission, not an automatic
panel retry or a measured whole-process peak.

Native LU remains a distinct gap: `batched_solve_lu_ffi.cc` allocates both
Getrf and Getrs workspaces simultaneously plus batch pivots outside XLA.
The existing `distrib_la.workspace` query supports GEMM/eigh only. A minimal
extension is an LU query in that same service/provider, with the actual
batch, n, nrhs, mesh/block geometry and dtype, summing both device workspaces
and pivots and reporting host workspace separately. No alternate backend
is needed. Until then the receipt must retain its native-workspace unknown
status; FFT custom-call scratch likewise must not be inferred as zero.

Scoped acceptance of the producer memory block: P4 job58495709.1,
Run477/38, sourceae2cabe1, sandbox claim2457. Actual photon stream compiled
temporaries476,768B plus arguments73,344B admit550,112B/rank; budget550,111B
refuses before execution. Donated output/alias6,144B; relative output
difference0. Optimized HLO Green[8,4,8,4,8] tiles global[8,4,16,4,16] onP4.
This proves the bounded admission path and tile geometry, not the native
custom-call interiors, endpoint preparation peak or a material peak decrease.

The charge constructor releases its sample matrices before reduction. Its local
route admits selection first, then prices the actual selected pencil before
allocating it; an oversized actual pencil still refuses at the existing capacity
gate. Coupled photon sectors retain samples for CT and retain the conservative
pre-read reduction check. Neither route relaxes the device budget.

### Whole-mesh photon constructor

`construct_sector_poles` resolves execution before reading a bank. The
`distributed` setting uses the complete supplied XY mesh; `local` uses
parent-local algebra when its existing capacity preflight admits it.
There is no retry or change of execution within a constructor stage.
Distributed rounds batch physical q parents at a width admitted before
reads, with `[b,s,n_X,n_Y]` sample faces. All samples, moments, direction
actions, original-pencil coefficient spans, joint CT pencils and factors
keep both matrix axes distributed. Scalar spectra and role metadata
replicate. No new input option or submesh is introduced.

Both paths call the same parent equation owner in `shared_pole_local` and
CT equation owners in `shared_pole_sectors`. Execution adapters supply
`distrib_la` GEMM/eigh and explicit matrix sharding. Local tracing passes
no face constraint. Distributed plans use the explicit `distributed`
backend with service `auto` batching and no plan-level capacity fallback.
Known matrix outputs have explicit schemas; no dtype heuristic classifies
results. Cached builders retain mesh and static role/layout information;
frequency nodes and numerical state remain operands.

Matrix selection, factor sorting and unequal CT block assembly use
`common.staged_reshard`'s shared bounded reindexing operation. This is the
centroid pack/unpack algorithm: exchange to slabs split over all ranks,
select or concatenate locally, then exchange back to the face. Per-parent
column maps preserve each parent's retained-direction order. Merely
constraining the result of a global take or concatenate is insufficient:
GSPMD can otherwise gather a large operand onto just one mesh axis.

Admission precedes each stage. For batch b, a complex matrix costs
`16*b*n*n/P`. Selection includes the live photon sample/moment faces and
its dense selection envelope. Reduction prices the actual closed direction
width, actions and live retained panels. CT original-pencil residency and
retained joint eigensolve have distinct extents: native eigh workspace is
queried at the latter, without reducing the original matrix residency
charge. Every native workspace query carries the actual batch width and
service route. Eigenvector carriers tile both mesh axes. The unchanged map
ledger refuses an oversized stage.

The P16 integrated fixture compares configured local/distributed CC, TT
and CT observables and committed stores against an independent signed
plant, including unequal parent ranks and asymmetric endpoint-loss refusal.
It is an algebra/store check, not an 18-support material accuracy or
hundreds-GPU scaling measurement. Interior compiled layouts must also be
checked at material sizes; output face specs alone do not establish memory
scaling. Sandbox report `reports/bispinor_pole_distributed_20260921/report.md`
owns the job receipts and material replay status.

The shared-pole MPA/W_RPA bispinor route also accepts `no_local_fields`:
its charge head uses the same direct response and authenticated dipole owner,
with the raw kinetic-balance charge representation and source-WFN state
capacity. This does not add a transverse Gamma-cell head, a wing/body fold,
or the disabled metallic velocity-head update. Packed static photon modes
retain their separate coupled-head contract.

The sampler computes the frequency-independent Coulomb roots once for all
irreducible q using the existing local/distributed batched solver, retaining
`H[q,mu_X,nu_Y]` throughout the frequency loop. Photon V already has this
lifetime. No full-zone Green replication or additional response field results.
