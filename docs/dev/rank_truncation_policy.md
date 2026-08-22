# The rank-truncation policy — one criterion, one gate, one override

*Design note, 2026-08-22.  Owner directive: a user must never have to think
about truncation.  Conditioning worsens with system size and WILL eventually
break, so the policy has to degrade gracefully or refuse informatively —
never silently.*

Companion pages, and the division of labour between them:

| module | owns | one-line discriminator |
|---|---|---|
| `common/rank_criterion` (L2) | **how many** directions survive, **whether that was allowed**, and the report | a cap on how much a pseudo-inverse may amplify round-off |
| `common/spectral_closure` (L2) | **where** that many is allowed to land | a cut may not stop inside a degenerate block |
| `common/band_degeneracy` (L2) | the same question on the **band** axis | a window says WHICH STATES exist and rounds outward |

Those three were already here.  What was missing, and what this note adds, is
the **gate**: every one of them measured the right number and then let the run
continue.  `TASTE.md`, 2026-08-15: *"an instrument that measures a defect and
then proceeds is not a gate."*  Six of the entries in that table are rank cuts.

---

## 1. The four questions every truncation site must answer

A site is not compliant because it calls `select_rank`.  It is compliant when
it can answer all four, in the log, on every run:

1. **How many, and under what cap?**  `rank_criterion.select_rank(spectrum,
   rtol)` — retain σ > σ_max·rtol, i.e. cap the achieved amplification at
   κ_cap = 1/rtol.  Never a knee/elbow/plateau search: these are ISDF and
   Galerkin *overlap* spectra and they are smooth by construction.
2. **Was that many even possible?**  Every operator has a **structural rank
   ceiling** that no arithmetic can exceed — `min(n_rows, n_cols)` for a
   rectangular fit, the candidate count for a Gram.  A criterion that returns
   more than the ceiling has counted round-off, and the number it prints is
   not a rank.  `select_rank(..., ceiling=…)` clamps; `RankReport.violations()`
   refuses an unclamped overshoot.
3. **Did the cut land in a gap?**  `spectral_closure` — drop the straddled
   block whole (owner ruling 2026-08-10).  Splitting a degenerate group makes
   the retained span a round-off-chosen slice of an eigenspace, which differs
   between q and Sq, and the k-star identity is gone for everything
   downstream.  **A band-window edge that splits a multiplet is the same
   defect on a different axis** and is treated identically — refuse, and for
   the band axis the repair rounds the other way (§5).
4. **Was the truncation load-bearing, and is that regime certified?**  This is
   the new one.  §2.

---

## 2. The gate: achieved amplification against a CERTIFIED ceiling

### The invariant

    κ_eff = σ_max / σ_min(kept)          achieved amplification
    κ_cap = 1 / rtol                     what the criterion asked for
    κ_eff ≤ κ_cap                        by construction

`κ_eff ≤ κ_cap` was already checked.  It is necessary and **not sufficient**:
it certifies that the code did what it was told, not that what it was told is
survivable.  Both registered catastrophes satisfied it exactly.

### The number, and why it is 1e8

The gate is on the **absolute achieved amplification**, and its ceiling is
**κ_certified = 1e8**, measured on two systems that share no code path
downstream of the fit:

| deck | rtol | κ_eff | truncation bound? | outcome |
|---|---|---|---|---|
| MoS₂ 4×4, nb=1024, μ≈10k (R19, `docs/dev/notes/ladder_rung1_R19_zeta_rcond.md`) | 1e-8 | ≤1e8 | yes, rank 6700 | eqp0 3.1350 — **correct** |
| MoS₂ 4×4, same | 1e-10 | ≈1e10 | yes, rank 8290 | eqp0 −206.83 |
| MoS₂ 4×4, same | 1e-12 | ≈1e12 | yes, rank 9461 | eqp0 −5049.59 |
| Si 4×4×4 SYM/SOC, 128 band, 1776 centroids (register 2026-08-15) | 1e-10 | 9.7e9 … 1.0e10 | yes, 1469/1776 | Σ_c MAE **54.4 eV**, exit 0 |
| Si 4×4×4, 600 centroids | 1e-10 | — | **no** | Σ_c MAE 0.90 eV |
| MoS₂ 4×4 htransform ψ@centroids (job 7883150) | 1e-8 | 4.44e4 | no (full row rank) | on-grid 3.2e-5 meV |

Read the third column, not the fourth.  Everything at κ_eff ≈ 1e10 or above is
wrong by electron-volts; everything at or below 1e8 is right.  That is the
same boundary `gw_config.py` already documents from the other side — the
`zeta_rcond` default sits at *"the LOW end of the over-complete recovery
plateau"*, the plateau being 1e-8…1e-4, i.e. κ_cap ∈ [1e4, 1e8].

### What the gate actually is

> **When the criterion BINDS — when it discards at least one direction — the
> achieved amplification must not exceed the site's certified ceiling.**

The "binds" clause is load-bearing and is not a softening.  If nothing was
discarded, the criterion made no choice: the spectrum ended on its own, the
operator is what it is, and refusing would be refusing the *input* rather
than the *policy*.  The Si 960-point anchor set is exactly that case at
production settings (768 of 768 retained) and it carries the record's best
BerkeleyGW agreement; a gate that refused it would be refusing the best
measurement in the tree.

Default action: **refuse**, naming κ_eff, the ceiling, the site, the drop
count and the deck key that fixes it.  One named override,
`LORRAX_RANK_POLICY=warn|off`, so that anyone continuing does so deliberately
and leaves a trace in the log — the shape `TASTE.md` prescribes for a guard
whose hard refusal would break an existing deck.

### What was REFUTED as a gate, and must not be re-proposed

**Drop fraction.**  The obvious gate — "refuse when n_keep/n_total falls below
X" — is dead, and it is dead by measurement in this tree, in both directions:

* MoS₂ production at the certified rtol discards **33 %** of the spectrum and
  is correct.
* Si 4×4×4 at 1776 centroids discards **17 %** and is wrong by 54 eV.
* Si 960 at `zeta_rcond` 1e-6 discards **34 %** and moves the σ-star spread by
  0.005 meV — nothing.

Any threshold that fires on the 17 % case also fires on the 33 % case.  This
is `TASTE.md` rule 12 in numbers: *retained rank is not basis quality, in
either direction.*  The drop count is **reported** — always, with the R19
anchor beside it — and it gates nothing.

**A plane-wave upper bound on N_μ.**  Attractive (it would be a parse-time
refusal) and not implemented, because the ceiling formula is unverified: the
same Si deck's *good* 600-centroid arm also exceeds `ngkmax = 588`, so the
naive bound would refuse a run measured at 0.90 eV MAE.  Left as an open
question in the register rather than shipped as a gate calibrated from one
payload.

### The accuracy statement that IS reported

`RankReport.discarded_weight` = Σ_{dropped} |λ_i| / Σ_all |λ_i|.

For a charge Gram `C = P Pᴴ` this is exact: `Σ_i λ_i = tr C = ‖P‖_F²`, so the
discarded fraction of the trace is the fraction of pair-density weight the
truncation throws away.  It costs O(n) after an eigh that already happened, it
is relative and scale-free, and unlike a rank count it is an *accuracy*
statement.  It is reported on every run and gated only at a loose ceiling
(1e-3) that no healthy deck measured here comes within four decades of —
a gate that exists to catch the case where the cut is eating real weight,
not to police ordinary conditioning.

---

## 3. No absolute floors.  Ever.

Every threshold in this policy is **relative to a scale the operator itself
supplies**.  Three shapes, and each replaced a hard-coded absolute number that
had already failed on a valid input:

| shape | used for | replaced |
|---|---|---|
| `σ_i > σ_max · rtol` | spectra of the operator being inverted | — (already relative) |
| `‖r_after‖ > rtol · max(‖r_before‖, scale)` | Gram–Schmidt / probe independence | `bse_w_exact`'s absolute `1e-6` coefficient-norm floor, which discarded every fixed-order Kramers probe on a valid fully relativistic LiF WFN |
| `\|λ_i − λ_j\| ≤ rtol_deg · max(\|λ_i\|,\|λ_j\|)` | "same degenerate block" | — (already relative; see `spectral_closure`) |

The middle row is the one worth stating twice.  A probe's coefficient vector
scales like `‖ψ‖` at one sample point, which falls like `1/√N_μ`: an absolute
floor is therefore a **system-size-dependent** refusal wearing a numerical
constant's clothes.  It works on the fixture and fails on the production deck,
which is exactly the class of failure this policy exists to remove.
`rank_criterion.probe_is_independent` is the one implementation.

---

## 4. Indefinite operators — a positive ridge is not a regularizer

The transverse (bispinor) CCT is Hermitian **indefinite**: both signs of λ are
physical.  Adding `+ε·I` moves every eigenvalue the same way, so it pushes the
negative ones *toward* zero.  Above κ ≈ 1e12 the ridge therefore makes
conditioning worse, which is measured (register `bispinor`, job 7885987) and
refutes the mechanism the ridge path's own docstring claimed.

The policy for an indefinite operator is the same one as for a definite one,
stated on |λ|: **truncate, do not shift.**  `transverse_zeta_solve =
rank_truncate` is that route and it is plumbed end to end.

The ridge path stays reachable because flipping a production default is a
physics ruling with a measurement attached, and no transverse measurement of
the truncated route exists yet on a production deck.  What it may not stay is
**uninstrumented**: it now carries a κ lower bound from `|diag U|` of its own
LU — O(n) after a factorization that already happened — and refuses above
κ ≥ 1e12 under the same `LORRAX_RANK_POLICY` dial.  A lower bound is the right
direction for a gate that fires when the number is *large*: exceeding it
proves κ exceeds it.  Failing to exceed it proves nothing, and the log says so
rather than reporting a clean bill.

---

## 5. The band axis is the same defect, and rounds the other way

`spectral_closure`'s TWO-RULE FAMILY already states this; it is repeated here
only because the band-window instance is the one that shipped:

* **A rank cut** says *how many directions are trustworthy* → floors to a
  block boundary, keeps fewer.
* **A band window** says *which physical states exist* → includes whole
  multiplets, or refuses.  `band_degeneracy.DEFAULT_MODE = "strict"`.

The ζ band-window closure at `gw_init.py` is the seam that used to print
`edge 60 min gap 0 meV` and continue.  It now refuses, through
`check_band_window(..., mode="strict")` on the **full mean field** rather than
on the window (a window cannot certify its own edge — `boundary_min_gaps`
returns `nan` there by construction since `b27f98c3`).  The measurement that
made this a refusal rather than a warning: at `nband=60` the within-star Σ
spread on the Si anchor deck is 1.957 meV; at the degeneracy-clean 40 and 36
it is **exactly 0.0000 meV on every Σ column**.  The band-window edge was also
the root of the band-window-dependent ζ conditioning behind the Si V_q
reciprocity break.

---

## 6. Site register

Status as of 2026-08-22.  "Certified κ" is the largest achieved amplification
any measurement in this tree supports for that site; `None` means the ceiling
is **uncertified** and the site warns rather than refuses, and says which.

| site | operator | rtol source | ceiling | certified κ | gate |
|---|---|---|---|---|---|
| `isdf/core._charge_factor_math` `rank_truncate` | charge Gram `C_q`, PSD | `zeta_rcond` (deck) | `n_log` | 1e8 | refuse |
| `isdf/core._charge_factor_math` `transverse_rank_truncate` | transverse CCT, indefinite | `transverse_zeta_rcond` (deck) | `n_log` | — (uncertified) | warn |
| `isdf/core._factor_c_q_distributed_rank_truncate` | same, 2-D pzheevd | as above | `n_log` | as above | as above |
| `isdf/core._transverse_lu_math` (ridge) | transverse CCT | — (no truncation) | — | κ ≥ 1e12 refuses | refuse |
| `bandstructure/htransform.streaming_galerkin_solve` | ψ@centroids Gram-eigh σ | `rtol` (1e-8) | `min(nk·nb, nspinor·n_μ)` | 1e8 | refuse |
| `common/zeta_projection.least_squares_transfer` | small-basis Gram `G_S` | `rcond` (caller) | `μ_S` | 1e8 | refuse (κ only — the route reduces over q before host, so it carries no per-q trace and cannot make the weight finding) |
| `centroid/pivoted_cholesky` select | candidate Gram, PSD | `sqrt(eps)` relative | candidate count | — | see §7 |
| `bse/bse_w_exact` TRIM probes | probe independence | relative (§3) | block size | — | refuse on deficiency, with k/block/norm named |
| `bse/bse_pseudopoles._orthonormalize` | filtered-vector overlap Gram | `s_cutoff`, floored at an absolute `1e-30` | — | — | **UNWIRED** — see below |
| `solvers/davidson._whiten_rank_revealing` | CGS2 residual-block Gram | hard-coded `1e-10` | — | — | **UNWIRED** — see below |

The last two rows are the policy's own outstanding debt, and the reason is
recorded rather than papered over:
`tests/known_failures/2026-08-11-bse-rank-cuts-outside-spectral-closure.md`
established that wiring either **changes numbers** in a shipped eigensolver
(under the closure default a straddled block is dropped, so the retained rank
falls and Davidson admits fewer directions), so the work is an A/B measurement
on the Si BSE Davidson deck, not an edit.  `_orthonormalize`'s absolute `1e-30`
floor is a §3 violation on paper; it binds only below `s_max ~ 1e-24` and
"agrees in practice" is not "same contract", which is exactly why it is filed
here instead of being quietly rewritten.

---

## 7. Where a rank deficiency must NOT refuse

`centroid/pivoted_cholesky`'s select is the counter-example that keeps the
policy honest.  Its rank gate refuses the shipped Si **960**-point anchor set
(certified 799) — and that set scores `sigTOT` MAE **0.644 meV**, the best
BerkeleyGW agreement on record for this deck, while the orbit-mode arm the
same gate *passes* at 960 is 20–56× worse.

Two separate defects, and only one of them is the rank:

* **Granularity.**  In orbit mode the greedy deflates the Schur complement by
  one direction per *orbit* while removing all `n_sym` members from
  contention, so its `rank` counts orbits.  "18/18 directions certified —
  PASS" over a delivered set of 768 *points* certifies nothing about the file
  being written.  The certification statement is now made at **point**
  granularity in both modes, which removes the asymmetry that made the
  refusal reachable only on the more accurate path.
* **Authority.**  A numerically rank-deficient *pool* is not an error — it is
  the ordinary state of an over-complete interpolation set, and it is measured
  anti-correlated with accuracy on this deck.  It is reported, loudly, with
  its downstream consequence (the ζ back-solve will truncate about that many
  modes per q).  What still refuses is what is structurally unsafe: a
  non-PSD Gram, a pool that has literally run out of candidates, and any pivot
  outside the candidate range.

The refusal text also no longer advises widening the prune window on this
deck: measured here, widening at fixed orbit setting changes `sigTOT` by <2×
and never recovers the orbit-mode loss.

---

## 8. The dials, and there are only two

| name | values | default | what it is for |
|---|---|---|---|
| `LORRAX_RANK_POLICY` | `refuse` / `warn` / `off` | `refuse` | the authority of every gate in §2 and §4 |
| `LORRAX_SPECTRAL_CLOSURE` | `snap` / `strict` / `off` | `snap` | whether §1.3 repairs, refuses or is absent |

`rtol` itself is **not** a dial for making a gate pass.  It is a physical
convergence axis with a measured plateau, and the R19 table is what happens
when it is used as an escape hatch: loosening it by four decades inflated the
rank by 41 % and moved a 2.2 eV gap to −5049 eV.  `overcomplete_margin` on
every `RankReport` is the distance to that cliff; a large margin means the
basis is over-complete and `rtol` must **not** be loosened on that run.

---

## 9. What this policy does not do

Stated so nobody quotes it as coverage it does not have.

* It does not certify the transverse (bispinor) channel.  There is no
  production-deck measurement of `transverse_zeta_solve = rank_truncate`, so
  that site's ceiling is `None` and it warns.  Getting a number there is the
  next measurement, not a code change.
* It does not bound N_μ from the plane-wave count.  §2 says why.
* It does not make the htransform exact-span route N_k-independent.  The
  paper-faithful finite-accuracy route is `htransform_rank_multiplier`
  (landed `7b6f9dd3`, opt-in, model order `ceil(m·N_b)` and therefore
  N_k-independent by construction).  Its default is still 0 — the exact-span
  path — because the only A/B on record is 2.218 meV RMS / 6.820 meV max on a
  MoS₂ exciton spectrum and is inspection-grade, not a production
  certification.  What changed is that the exact-span route's refusal now
  *names* it as the supported repair instead of telling the operator to buy
  centroids, which is measured wrong advice on a dense metal k-grid.
* It does not reach the two BSE-owned cuts in §6's last two rows.  Both
  wirings change numbers in a shipped eigensolver, so each needs an A/B first.
* It does not turn `κ_eff ≤ κ_cap` into a sufficient condition.  It never was.
* Nothing in it has been measured on a production deck **as of 2026-08-22**.
  The thresholds are read off measurements already in the tree; what is new is
  that a number now stops a run.  The gates were exercised by unit cells and a
  compute-node focused suite, not by a re-run of the Si 1776-centroid deck that
  motivated them — so "this would have refused that run" is an inference from
  its logged `kappa/q ≈ 9.7e9` against a 1e8 ceiling, not a re-measurement.
