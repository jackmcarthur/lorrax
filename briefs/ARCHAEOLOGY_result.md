# ARCHAEOLOGY result — delivered-Sigma regression history

Lane weight: **heavy, read-only**. Numbers below are for the Na `0..+5 eV`,
24-band deck against `control_panes_24b` unless noted. I re-read the retained
full complex `sigma_c_kij_ev` cubes (350,784 elements) and independently
recomputed max/RMS; `plan_seconds` comes from the retained pickle opcode or the
named report. Exact duplicate executions are collapsed.

## Chronological attributable measurements

| tree / route | windows / pairs | plan (s) | Sigma_c max / RMS (meV) | note |
|---|---:|---:|---:|---|
| `f468902c`, pre-owner direct era | 9 / 102 **+ 2,794 direct terms** | not emitted | **0.151737 / 0.009526** | 975.074 s total; not a 102-node product plan |
| `e0f49270` -> `d56dc2da`, owner/legacy six panes | 6 / 137 | **245.693** (239.285--239.862 in instrumented twins) | **0.278609 / 0.010070** | cold plan; warm receipt executions were identical |
| `0db9fff8`, Occam lookup | 6 / 154 | **2.928** | **0.195847 / 0.008836** | 1.481 census + 1.447 fitting |
| `08ce1d52`, covered fallback prototype | 6 / 167 | **7.566** | **0.195916 / 0.008806** | same deck/control |
| `0ad6d39d`, omega reroute | 6 / 159 | **7.227** | **0.195849 / 0.008835** | duplicate parent rerun: 0.195850 / 0.008835 |
| `c2627ba8`, covered-density preservation | 6 / 154 | **7.347** | **0.195849 / 0.008835** | same deck/control |
| `c6cd506a`, consolidated/rank-parallel | 4 / 98 | **44.078** | **0.645712 / 0.016876** | fewer pairs, failed 0.5-meV bar |
| `0761062d`, final ROQ wiring | 6 / 115 | not persisted | **0.195876 / 0.008834** | Sigma stage 59.32 s |
| `50725bd0`, dials removed | 6 / 115 | **22.417** | **0.195876 / 0.008834** | `cond:resonant` = certified HGL 66 nodes |
| `a9999161`, SC census/fix lineage | 6 / 115 | **22.111** | **0.195876 / 0.008834** | same 66/9/11/12/9/8 plan |
| `53a76a10`, pointwise + contour integration | 6 / 74 | **105.581** | **0.235789 / 0.009575** | 25/9/11/12/9/8; no direct terms |
| `0bb0a6ba`, consolidated pre-on-demand | 6 / 115 | **51.678** | **0.195910 / 0.008835** | cold bisection rerun |
| `dd5a1227`, plus null-family fix | 6 / 115 | **52.547** | **0.195911 / 0.008835** | no accuracy change |
| `6fdad2c3`, first all-on-demand merge | 6 / 77 | **61.568** | **0.760271 / 0.023515** | first tree above 0.3 meV |
| `da274386`, plus Hackbusch seed | 6 / 77 | **63.033** | **0.760277 / 0.023515** | change only 0.000008 meV |
| `b266b7b8`, final merged one-shot | 6 / 77 | **62.923** | **0.760283 / 0.023515** | current bad 26/9/14/10/9/9 plan |
| `a5330758`, crossing-rank floor | 6 / **112** | **33.498** | **0.195974 / 0.008813** | on-demand `cond:resonant` rank **63** |

The old 115-pair plan has no clean earlier Pareto dominator on both pair count
and accuracy. The apparent 102-pair/0.1517-meV exception paid 2,794 direct
terms and 975 s, so it is not comparable. Against the all-on-demand 77-pair
state, however, `53a76a10` is a strict pair/accuracy dominator: **74 < 77** and
**0.235789 < 0.760271 meV**, albeit with a 105.6-s planner.

## What was better, ranked by actionable evidence

1. **The missing crossing rank was recovered.** `a5330758` fitted the dominant
   crossing window on demand at rank 63 (residual `3.988e-5`, kappa 33.14) and
   delivered **112 pairs, 0.195974/0.008813 meV**. This directly answers the
   66-node suspicion: yes, an on-demand configuration was measured near 66,
   and it recovered the observable. Caveat: its four sign-definite windows
   still used shipped tables, so it is not yet the owner's all-on-demand end
   state.
2. **A low-pair on-demand-era tree was silently better than its successor.**
   `53a76a10` delivered **74 pairs at 0.235789 meV**, then the on-demand merge
   selected 77 pairs at 0.760271 meV. The contract did not warn: the 74- and
   77-pair envelope spends differ only 0.0382%, while max error differs 3.224x.
3. **The table-backed 66-node anchor was both accurate and materially cheaper
   to plan than later copies.** `50725bd0`/`a9999161` delivered 0.195876 meV in
   **22.1--22.4 s**. The same 115-pair observable later cost 51.7--52.5 s.
   Cause of that same-plan planning regression was not established. This is
   archaeology, not a proposal to restore tables.
4. **Lookup was the fastest measured planner.** `0db9fff8` achieved the same
   ~0.196-meV consumer error in **2.928 s**, and the patch variants took
   7.2--7.6 s. They cost 154--167 pairs and violate the settled on-demand-only
   ruling, so they are historical performance bounds, not recovery candidates.

## Lost/unmeasured work and retractions

- Pushed but never deck-measured: `fix/route-sign-definite-2026-09-01`
  (six-window fit-only 2.261 s parallel / 3.245 s sequential),
  `feat/rank-acceptance-margin-2026-09-01` (+1 sign-definite rank),
  `perf/fit-ladder-faststop-2026-09-01` (8.25% offline fit saving), and the
  54-node/4.682-s `feat/roq-production-rules-2026-08-31` plan.
- Two filesystem-fault-era worktrees still contain tracked, uncommitted source:
  `wt_fit_ladder_2026-08-31` (28-line diff: whole-branch retry uses the hard
  rank cap) and `wt_lane_c_2026-08-31` (193-line state-by-omega catalog split).
  Neither has a deck measurement. The latter is superseded by the no-table
  ruling and by measured split inflation, so it is not a lost win.
- The catalog-coverage explanation for the formerly null easy flank was
  correctly retracted: the rule passed residual/kappa; factor growth was
  measured about the wrong screened-pole origin. The separate “SC map 2 only
  refuses” story was also correctly retracted: map 1 had replayed a cached
  plan; both branches refused from a truly cold map 1 before tightening.

## Not reproducible on today's intended tree; reruns owed

- Every legacy numeric arm used `LORRAX_BAND_DEGENERACY=snap`; today's strict
  policy refuses the 48-of-48 chi edge and sliced 24-band Sigma multiplet.
- The 102-pair direct-era route was deleted; old receipts with direct rows now
  refuse. The 115/154/159/167 table plans cannot be regenerated after catalog
  removal/cache-epoch change. The 112-pair floor plan also cannot reproduce
  exactly after table removal because its sign-definite rules were shipped.
- Highest-value rerun: combine the measured rank-63 crossing floor with the
  pushed on-demand sign-definite route, then run one cold P4 0..5-eV deck and
  report pairs, `plan_seconds`, and full-cube max/RMS. This is the missing
  all-on-demand recovery measurement.
- Also owed on a real deck: the +1 sign-definite margin and the 54-node
  production-rule plan. Run the former only as an A/B against the rank-floor
  tree; the latter has never crossed the consumer boundary at all.

Evidence roots: sandbox run 50 and run 60 named in the lane brief;
`/pscratch/sd/j/jackm/wt_regress2_2026-08-31/evidence/regress2`;
`/pscratch/sd/j/jackm/wt_resfloor_2026-08-31/tmp/resfloor_na_p4`.
