# ONESHOT — merged-tree Na 0–5 eV regression guard

## Achieved numbers

**FAIL — the merged tree exceeds the owner's 0.5 meV bar.**  The cold P=4,
BFC@0.85 run completed with **6 windows / 77 pairs / 62.923 s planning**.
Against `control_panes_24b`, the full complex `sigma_c_kij_ev` cube differs by
**0.760283 meV max / 0.023515 meV RMS**.  This is above both the **0.5 meV**
acceptance bar and the brief's **~0.3 meV** loud-warning line: **18/21** omega
points exceed 0.5 meV and **21/21** exceed 0.3 meV.  The worst point is
**2.25 eV**, not the historical 5 eV endpoint.

The merged tree is on the **pointwise contract**.  Its maximum certified
pointwise envelope spend is **0.734494** of the safety-reduced budget, so the
planner contract passes while consumer-level Sigma accuracy fails.  The known
74-pair pointwise artifact reproduces its stated control error here at
**0.235789 / 0.009575 meV**; merged versus that artifact is
**0.775083 / 0.021626 meV**, establishing a merged-tree regression rather than
a different control.

| omega (eV) | max (meV) | RMS (meV) | omega (eV) | max (meV) | RMS (meV) |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.439877 | 0.019828 | 2.75 | 0.755676 | 0.027217 |
| 0.25 | 0.386920 | 0.015095 | 3.00 | 0.731553 | 0.027547 |
| 0.50 | 0.456737 | 0.013182 | 3.25 | 0.742774 | 0.027447 |
| 0.75 | 0.538032 | 0.014899 | 3.50 | 0.707565 | 0.027618 |
| 1.00 | 0.625789 | 0.018534 | 3.75 | 0.710268 | 0.027348 |
| 1.25 | 0.675526 | 0.020570 | 4.00 | 0.685694 | 0.024998 |
| 1.50 | 0.726687 | 0.023615 | 4.25 | 0.673358 | 0.024725 |
| 1.75 | 0.744440 | 0.024948 | 4.50 | 0.665516 | 0.020921 |
| 2.00 | 0.758919 | 0.026232 | 4.75 | 0.643409 | 0.022669 |
| 2.25 | **0.760283** | 0.027250 | 5.00 | 0.639674 | 0.023094 |
| 2.50 | 0.757909 | 0.027079 | all omega | **0.760283** | **0.023515** |

## Planning and execution

All six windows were fitted on demand (`candidate_family=measure_adapted_roq`,
`source=live_fit`).  Their node counts were **26, 9, 14, 10, 9, 9**.  Of the
**62.923 s** cold plan, **41.544 s** was consolidation trials, **15.796 s** the
critical-rank window fits, and **5.480 s** window geometry.  Planning consumed
**54.6%** of the **115.310 s** total wall; Sigma was **97.438 s**, with 110 tau
dispatches in 9 sweeps.  Thus planning is nowhere near the owner's <1/10
execution rule.

## Merge and proof

All four requested branch tips are ancestors of `b266b7b8`.  In the semantic
conflicts I kept the screened-pole `E_ref_B` correction and the on-demand-only
fitter.  I left out marginal-cost's conflicting shipped-table iterator and its
table-specific test because that would restore the catalog forbidden by the
owner ruling; the surviving selector therefore receives one on-demand rule per
window.  I rewrote the screened-reference regression to test factor growth
directly.  The prescribed CPU gate passed **137/137 in 86.73 s**.

The first strict-degeneracy P=4 preflight refused before planning because the
legacy 48-band WFN has no spare band above `number_bands_chi=48`.  The measured
arm therefore retained the copied baseline's inherited `snap` condition for an
ancestry-matched comparison; it did not change the 24-band Sigma sum.  Final
provenance is JID **57804947**, step **58**, four A100s/four ranks,
`RUN_GIT_HEAD=b266b7b8e6709f641a427f9b4aa3899e031c6b8f`.  The artifact checker
reports every dataset finite and **PASS**.

Evidence:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829/oneshot_merged_p4_20260901`.
Branch: `test/oneshot-2026-09-01`.
