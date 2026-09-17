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

**Real-time stream.** For occupied and unoccupied weights `f`, `u`, the selected-q retarded correlation is
accumulated on a certified time rule `{t_a, w_a}` (`minimax.response_bank_rule`):

$$ A(R,t) = \sum_{k} G_u(k+R,\,t)\,\overline{G_f(k,\,t)}, \qquad
\chi^0(q,z) \approx \sum_a h_a(z)\,\mathcal F_q\!\left[-i\big(A(t_a) - \overline{A(t_a)}\big)\right], \tag{SP 5} $$

with `h_a(z) = w_a e^{i z t_a}`. Conjugating in R space before the q transform gives `conj(F_{−q})`, the partner
orientation with its own weight, so (SP 5) needs no time-reversal assumption.

**Orientation.** Σ's `G_{k−q} W_q` contraction is exact for

$$ \mathcal F_q[f](\mu,\nu) = \sum_R f(r_\mu,\, r_\nu + R)\, e^{i q\cdot R}, \qquad W_q = \mathcal F_q[W] . \tag{SP 6} $$

The incumbent trace builds `conj(G(w, t)) = G(w̄, t̄)ᵀ` and gathers the forward transform at `q`, which returns
`F_q[χᵀ]`. With time reversal the two orientations are equal and the TRS bank keeps that trace bit-for-bit. An
ordered bank (`ordered=True`, and `pair_mode = "laplace_ordered"`) builds `G(w̄, t̄)` and gathers rows at `−q`,
which is `F_q[χ]`. With the transposed orientation on a TR-broken deck each Green's-function branch would take the
other branch's residues, `Σ[W^even] − Σ^odd`. Tests: `tests/test_shared_pole_stream_orientation.py` (ordered
retarded and Laplace rows equal `F_q[χ]` to 1e-10 on a TR-broken lattice and miss `F_q[χᵀ]`).

**Remote Laplace cells.** Transitions far from the support region enter through a positive rule for
`1/(d² + η²)^{n+1}` on the same `t` (`minimax.response_laplace_rule`). The even kernel `d/(d² − z²)` weights
`forward − reverse`; the odd kernel `z/(d² − z²)` weights `forward + reverse`:

$$ \chi^0_{\rm even} \leftarrow \sum_a \rho^{\rm even}_a(z)\,(F - B)(t_a),\qquad
\chi^0_{\rm odd} \leftarrow \sum_a \rho^{\rm odd}_a(z)\,(F + B)(t_a), \tag{SP 7} $$

where `F = f_lower u_upper` and `B = f_upper u_lower` are the two one-particle correlations. Odd rows are computed
only for an ordered bank.

**Dyson.** Each sample solves `W = (I − v χ⁰)⁻¹ v` through the bounded panel GEMM (`distrib_la.panel_matmul`) and
the resolved LU plan, and stores `W_c` and `∂W_c/∂s`.

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
its mirror `X(−z)` on the same `Q`; `W_q(−z̄) = conj W_{−q}(z)`, so the mirror is one sample of parent `−q`. Poles
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

**Dedupe.** For an imaginary support, or a line support with `Re z = 0`, the conjugate partner brings no new tangent
with time reversal: `W Q ∈ span(Q)`. Only the component of `O = W Q` orthogonal to `Q` above `direction_cutoff`
survives, so a time-reversal-symmetric bank adds no partner columns and the ordered model equals the even one at
equal rank.

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

**Hole routing.** Conduction windows take `W₊(q)`. An ordered store routes valence windows to the particle–hole
partner,

$$ W_-(q,\tau) = W_+(-q,\tau)^{\mathsf T}, \tag{SP 16} $$

gathered at `−q` on the replicated q axis and transposed on its faces (`shared_pole_hole_kernel`); time-reversal
stores never take this branch. Tests: `tests/test_shared_pole_ordered.py` (the synthesized orientations are the
positive- and negative-frequency Lehmann sums) and `tests/test_shared_pole_lattice_sigma.py`: on a TR-broken
lattice the production τ kernel reproduces real-space `Σ = iGW` to 1e-10 relative, and the swapped routing misses
by more than 1e-3.

**Two-component decks.** `W` is spin-scalar; `G` carries the spinor axes, and the τ kernel broadcasts `W_q` over
both (`ppm_tau_kernel` `prep_w`). The factor spin axis is 1 on every admitted deck.

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
| `services/distrib_la/tests/test_eigh_keeps_operand.py` | the planned eigh does not overwrite its operand |
| `test_slab_io_mode_required.py` | `SlabIO`/`open_file` refuse a missing `mode=` |

## 10 Byte model

Per rank on an `x × y` mesh with `P = Px Py` and pencil side `R`:

$$ \text{reduction} \approx 16\,\big(14 R^2 + 12 n R\big)\,b/P + 16\cdot 3 n r\,b/P + \text{native eigh workspace}, \tag{SP 17} $$

(`shared_pole_capacity.shared_pole_byte_terms`; `ConstructorCapacity` beside it turns those terms into the map
ledger's rows). A round has `b = P`: every rank holds one whole parent (§6). The native cuSOLVERMp eigh adds a
private operand tile of `n²/P` next to its workspace, which `distrib_la.workspace_bytes_per_rank` includes; a byte
model without that tile under-counts the measured CrI3 q=1 construction peak (2.140 against 2.907 GiB per rank).
