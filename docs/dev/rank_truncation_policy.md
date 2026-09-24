# Rank truncation: one criterion, one gate, two dials

A user never chooses a truncation. Conditioning worsens with system size, so
every rank cut either stays inside a certified regime or refuses, naming the
number, the site and the fix.

| module (L2) | owns |
|---|---|
| `common/rank_criterion` | how many directions survive, whether that was allowed, and the `RankReport` |
| `common/spectral_closure` | where that many may land: a cut may not stop inside a degenerate block |
| `common/band_degeneracy` | the same question on the band axis, where a window names which states exist and rounds outward |

## The four questions every truncation site answers, in the log, every run

1. **How many, under what cap?** `rank_criterion.select_rank(spectrum, rtol)`
   keeps `σ_i > σ_max·rtol`, capping the amplification at `κ_cap = 1/rtol`.
   Never a knee or plateau search: ISDF and Galerkin overlap spectra are smooth.
2. **Was that many possible?** Every operator has a structural ceiling
   (`min(n_rows, n_cols)` for a rectangular fit, the candidate count for a
   Gram). `select_rank(..., ceiling=…)` clamps, and `RankReport.violations()`
   refuses an unclamped overshoot: a count above the ceiling is round-off.
3. **Did the cut land in a gap?** `spectral_closure` drops a straddled
   degenerate block whole. Splitting one makes the retained span a
   round-off-chosen slice of an eigenspace that differs between q and Sq,
   which breaks the k-star identity downstream.
4. **Was the cut load-bearing, and is that regime certified?** The gate below.

## The gate

$$
\kappa_\text{eff} = \sigma_\max/\sigma_\min(\text{kept}) \;\le\; \kappa_\text{cap} = 1/\text{rtol}
$$

holds by construction and is not sufficient: it certifies that the code did
what it was told, not that the result survives. The gate is:

> **When the criterion binds (discards at least one direction), `κ_eff` must
> not exceed the site's certified ceiling.**

If nothing was discarded the spectrum ended on its own; refusing would refuse
the input, not the policy.

`rank_criterion.KAPPA_CERTIFIED_GRAM = 1e8` for positive-definite overlap
Grams, calibrated on two decks with no shared code downstream of the fit:

| deck | rtol | κ_eff | binds | outcome |
|---|---|---|---|---|
| MoS₂ 4×4, nb = 1024, μ ≈ 10k | 1e-8 | ≤ 1e8 | yes (rank 6700) | eqp0 3.135 eV, correct |
| same | 1e-10 | ≈ 1e10 | yes (8290) | eqp0 −206.8 eV |
| same | 1e-12 | ≈ 1e12 | yes (9461) | eqp0 −5049.6 eV |
| Si 4×4×4 SOC, 128 bands, 1776 centroids | 1e-10 | 9.7e9–1.0e10 | yes (1469/1776) | Σ_c MAE 54.4 eV, rc = 0 |
| Si 4×4×4, 600 centroids | 1e-10 | — | no | Σ_c MAE 0.90 eV |

Everything at κ_eff ≳ 1e10 is wrong by electron-volts; everything at or below
1e8 is right. The `zeta_rcond` default, 1e-8, sits at the low end of the
over-complete recovery plateau `[1e-8, 1e-4]`.

Default action is refuse, naming κ_eff, the ceiling, the site, the drop count
and the deck key that fixes it. `LORRAX_RANK_POLICY=warn|off` is the one
override, so continuing is deliberate and logged.

**The κ arm is inert whenever `κ_cap ≤ κ_certified`** (`κ_eff < κ_cap` always,
so the comparison cannot fire). That is the case at the `zeta_rcond` default:
the arm becomes live only when the dial is loosened past the certified
plateau, which is where every registered catastrophe sat. `certify_numbers`
takes the criterion's `κ_cap` and announces the inert case, and
`RankReport.describe()` says so on its ceiling line: silence from this arm is
arithmetic, not a measurement. `isdf/core._certify_the_cut` (the device-face
ζ gate) prints only when it fires, so it does not yet distinguish inert from
passed.

**The discarded-weight arm** is independent of `rtol` and live on every
default. `RankReport.discarded_weight = Σ_dropped |λ_i| / Σ_all |λ_i|`; for a
charge Gram `C = PPᴴ`, `Σλ_i = tr C = ‖P‖_F²`, so it is exactly the fraction of
pair-density weight thrown away, at O(n) after the eigh. It is reported on
every run and gated at `DISCARDED_WEIGHT_MAX = 1e-3`.

### Refuted as gates: do not re-propose

* **Drop fraction.** MoS₂ at the certified rtol discards 33 % and is correct;
  Si 1776 discards 17 % and is wrong by 54 eV; Si 960 at `zeta_rcond = 1e-6`
  discards 34 % and moves the Σ star spread by 0.005 meV. Any threshold firing
  on 17 % fires on 33 %. Retained rank is not basis quality, in either
  direction. The drop count is reported and gates nothing.
* **A plane-wave upper bound on N_μ.** The good Si 600-centroid arm already
  exceeds `ngkmax = 588`, so the naive bound would refuse a 0.90 eV run.

## No absolute floors

Every threshold is relative to a scale the operator supplies:

| shape | used for |
|---|---|
| `σ_i > σ_max·rtol` | spectra of an operator being inverted |
| `‖r_after‖ > rtol·max(‖r_before‖, scale)` | Gram–Schmidt and probe independence (`rank_criterion.probe_is_independent`) |
| `\|λ_i − λ_j\| ≤ rtol_deg·max(\|λ_i\|, \|λ_j\|)` | membership of one degenerate block |

An absolute floor on a probe coefficient norm is a system-size-dependent
refusal: the norm scales like `‖ψ‖` at one sample point, i.e. like `1/√N_μ`.

## Indefinite operators

A fixed positive shift does not regularise a Hermitian indefinite operator
(negative eigenvalues move toward zero); rank truncation on `|λ|` is the
general policy. The equal-current ζ fit has `C = sQ`, `Z = s·RHS` with `Q` a
positive Gram and one sign `s` on both (`s = −1` for Γ₂).
`isdf.core._transverse_lu_ridge` gives the ridge the trace's sign, so
`(sQ + sδI)⁻¹(s·RHS) = (Q + δI)⁻¹RHS`. The ridge LU carries a κ lower bound
from `|diag U|` and refuses at `KAPPA_INDEFINITE_MAX = 1e12`; exceeding the
bound proves bad conditioning, and not exceeding it certifies nothing.

## The band axis rounds the other way

A rank cut says how many directions are trustworthy and floors to a block
boundary. A band window says which physical states exist and must contain
whole multiplets or refuse (`band_degeneracy.DEFAULT_MODE = "strict"`). The ζ
band window in `gw_init.py` is checked with
`check_band_window(..., mode="strict")` against the full mean field: a window
cannot certify its own edge. On the Si anchor deck a multiplet-splitting edge
at `nband = 60` gives a 1.957 meV within-star Σ spread; the clean edges 40 and
36 give exactly zero.

## Site register

"Certified κ" is the largest achieved amplification a measurement supports at
that site; `—` means uncertified, and the site warns instead of refusing.

| site | operator | rtol | ceiling | certified κ | gate |
|---|---|---|---|---|---|
| `isdf/core._charge_factor_math` `rank_truncate` | charge Gram `C_q`, PSD | `zeta_rcond` | `n_log` | 1e8 | refuse |
| `isdf/core._charge_factor_math` `transverse_rank_truncate`, `_factor_c_q_distributed_rank_truncate(indefinite=True)` | transverse CCT, indefinite | `transverse_zeta_rcond` | `n_log` | — | warn (not deck-selectable: `linalg` resolves the transverse factor to ridge) |
| `isdf/core._transverse_lu_math` (ridge) | transverse CCT | no truncation | — | κ ≥ 1e12 refuses | refuse |
| `common/zeta_projection.least_squares_transfer` | small-basis Gram `G_S` | caller `rcond` | `μ_S` | 1e8 | refuse (κ arm only: the route reduces over q before host, so it has no per-q trace for the weight arm) |
| `centroid/pivoted_cholesky` select | candidate Gram, PSD | `√ε` relative | candidate count | — | reports; see below |
| `bse/bse_w_exact` TRIM probes | probe independence | relative | block size | — | refuse on deficiency, naming k, block and norm |
| `bse/bse_pseudopoles._orthonormalize` | filtered-vector overlap Gram | `s_cutoff`, floored at absolute `1e-30` | — | — | not wired |

Wiring `_orthonormalize` changes numbers in a shipped eigensolver (a straddled
block is dropped, so the retained rank falls), so it needs an A/B on the Si BSE
deck first (`tests/known_failures/2026-08-11-bse-rank-cuts-outside-spectral-closure.md`).
Its `1e-30` floor violates the no-absolute-floor rule and binds only below
`s_max ~ 1e-24`.

### Where rank deficiency must not refuse

The centroid pivoted-Cholesky pool is over-complete by design, and its
numerical rank deficiency is anti-correlated with accuracy: the Si
960-point anchor set (numerical rank 799) gives the best BerkeleyGW agreement
on record for that deck (Σ MAE 0.644 meV). The select therefore reports the
deficiency with its consequence (the ζ back-solve will truncate about that
many modes per q) and refuses only what is structurally unsafe: a non-PSD
Gram, an exhausted candidate pool, a pivot outside the candidate range.
Orbit-mode selection admits a complete orbit only when it fits the remaining
point budget and applies a pivot update for every member, so its rank and
delivered count are both in points. The selection window must still cover
the pair-density space being fitted; that is a physics contract, not a way to
pass a rank gate.

## The two dials

| name | values | default | governs |
|---|---|---|---|
| `LORRAX_RANK_POLICY` | `refuse` / `warn` / `off` | `refuse` | every κ and weight gate above |
| `LORRAX_SPECTRAL_CLOSURE` | `snap` / `strict` / `off` | `snap` | whether question 3 repairs, refuses or is skipped |

`rtol` is not a dial for passing a gate: it is a convergence axis with a
measured plateau. Loosening `zeta_rcond` by four decades on the MoS₂ deck
raised the rank 41 % and moved a 2.2 eV gap to −5049 eV.
`RankReport.overcomplete_margin` is the distance to that cliff; a large margin
means the basis is over-complete and `rtol` must not be loosened.

## Scope

* The transverse channel is uncertified (`—` above): no production deck
  measures its rank-truncating factor.
* htransform model order is not governed here: `isdf.galerkin`'s randomized
  QRCP uses `htransform_qr_eps` (default 1e-3) as its criterion and
  `htransform_rank_multiplier` (default 20) as its search ceiling, and reports
  its own receipts.
* The thresholds are read from the measurements above; no production deck has
  been re-run under the gate itself.
