# TARGETINFO result — dimension is a feasibility diagnostic, not an error weight

**Measured answer:** keep the delivered-envelope error contract.  The refusing
`ω<E_F val:resonant` window is unservable by the present ROQ representation;
loosening it because its measure is concentrated would violate, not preserve,
the delivered metric.  Its best stable rank-200 fit costs **69.85× the complete
plan budget by itself**.  The other 14 windows together cost only **0.2985×**
budget, leaving a feasible valence residual of **6.6166e-5**; the measured
stable residual is **6.58847e-3, 99.57× too large**.

The real signed sodium census contains 15 product windows (14 served, one
refused).  Exact dumped measures give the following crossing-window comparison;
`N_eff=(Σw)^2/Σw²` uses `w_cell = mass_cell Σ_ω 1/|d|`.  “Modes” counts positive
eigenvalues in the production weighted Gram (the last mode is roundoff-sensitive;
the P=4 replay counted 284 while this offline replay counted 283).

| crossing window | `N_eff`; top-10% mass | modes / ceiling | `A/γ` | allowance | achieved refined residual |
|---|---:|---:|---:|---:|---:|
| `ω≥E_F cond:resonant` | 198.36; 47.79% | 512 / 512 | 181.52 | 6.0592e-4 | **5.8948e-5** (served) |
| `ω≥E_F val:resonant[p1/2]` | 60.90; 65.61% | 38 / 38 | 12.38 | 0.5 | **1.3767e-4** (served) |
| `ω<E_F cond:resonant[p2/2]` | 41.96; 77.98% | 36 / 50 | 16.38 | 0.5 | **4.5314e-5** (served) |
| `ω<E_F val:resonant` | **38.37; 91.33%** | 283–284 / 512 | 204.37 | **9.4320e-5** | **6.5885e-3** at stable rank 200; 8.5320e-3 at production rank 512 (refused) |

There is no monotone “low information ⇒ hard fit” relation: the two other
low-`N_eff` supports (41.96 and 60.90) are served.  Failure needs the conjunction
seen only in the refused row: wide crossing geometry and a weighted subspace
that enters its numerical nullspace (228–229 of 512 modes zero in the two
replays).  Effective dimension is therefore useful for stopping the rank ladder
before its conditioning cliff, but it does not measure delivered error.

The apparent “mass-proportional share” is already the most permissive local
ceiling consistent with the global bound: for every uncapped window the code
sets `envelope_i * allowance_i = B`, the **entire** safety-adjusted budget
`B=7.96318e9`, then the selector checks the sum of achieved costs.  Here the
refused envelope is `8.44272e13`, **84.82%** of the combined pointwise envelope
scale `9.95398e13`.  At stable rank 200 it contributes `5.56246e11`, giving a
valence-only global relative bound **5.58818e-3 = 55.88×** the deck's `1e-4`
target (and 69.85× the 0.8-safety budget).  Concentration cannot hide that
error: the fitting residual itself already weights exactly those concentrated
cells by delivered mass and normalizes pointwise in omega.

**Recommendation:** do not add effective-cell count to error apportionment.
Use usable rank only as a feasibility/conditioning diagnostic and report the
best stable refusal early.  A serving change must improve the representation
or use a separately costed bounded subdivision; feasibility-aware redistribution
has no node cost to report here because it cannot satisfy the contract at any
measured rank.  The closest stable attempt already costs **200 window–tau
pairs** and exceeds the remaining global allowance by 99.57×.

Evidence: exact P=4 sodium census and supports at
`/pscratch/sd/j/jackm/wt_kappa_2026-08-31/tmp/kappa_probe/` (JID 57789884,
step 30, four ranks/four A100s, BFC@0.85, source
`5f750a774985936f1e6fea0fb28c4348dd2dbca0`); this lane remeasured all 15
census costs and the four crossing effective counts/Gram ranks offline from
`crossing_supports.npz`.  No new GPU leg is owed because this branch adds only
this report.  Branch: `study/target-vs-information-2026-08-31`.
