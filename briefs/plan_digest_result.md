# PLAN-DIGEST result — heavy lane

## Achieved numbers

The selected-plan path now emits **one fixed-width row plus one JSON row per
window**, and **one fixed-width summary per plan**, rank 0 only.  The mandated
CPU gate is **134 passed in 92.05 s**; the focused output contract is **7 passed
in 6.20 s**.  No fitter, rank ladder, selector, or acceptance decision changed.

The historical Na receipts show why the digest is useful.  Reconstructing the
new selected-plan fields from the two P=4 logs gives:

| selected window | 0.196 meV one-shot: nodes, residual, kappa | 0.760 meV merged: nodes, residual, kappa |
|---|---:|---:|
| `cond:resonant` | 66, 2.14352e-5, 559.937 | 26, **8.16353e-4**, 9.79145 |
| `cond:state_tail` | 9, 4.33471e-6, 1.07727 | 9, 1.94246e-5, 1.10727 |
| `cond:pole_tail` | 11, 8.31831e-5, 1.08161 | 14, 3.23359e-5, 1.05862 |
| `val:bulk` | 12, 1.07204e-5, 1.02051 | 10, **4.11083e-5**, 1.06233 |
| `val:resonant` | 9, 2.73165e-5, 1.12735 | 9, 2.91597e-5, 1.12764 |
| `val:pole_tail` | 8, 2.43819e-6, 1.08634 | 9, **6.75348e-5**, 1.04577 |

The corresponding summary scalar moves from
`sum_envelope_residual=1.33844e+09` to **`2.16942e+09` (1.621x)** while the
merged planner still reports `max_omega_spend=0.734494`.  The regression is
therefore visible from these lines alone: the cheap 77-pair merged plan spends
far more achieved envelope error than the 115-pair one-shot plan, especially
in `cond:resonant`, `val:bulk`, and `val:pole_tail`.  Historical logs did not
retain the weighted snapshot spectra, so their `rank/usable` values cannot be
reconstructed honestly; new runs print them directly from the already-built
spectrum.

## Change and evidence

The JSON census now records `rank`, `usable_rank`, and `nodes` for the selected
rule rather than a pre-selection candidate.  Human rows print name, kind,
A/gamma, rank/usable, nodes, residual, kappa p99, mass%, target, and family.
The summary prints windows, pairs, plan wall, achieved envelope sum, budget,
scalar spend fraction, and maximum pointwise budget fraction.  The usable-rank
count uses the established runtime-noise floor on the snapshot spectrum; it
adds only an O(rank) count to an SVD already completed.  Pointwise spend uses
the per-frequency envelopes already constructed during window geometry.

Evidence: the two historical P=4 logs are
`.../dialremoval_p4_20260831/dialremoval_p4.log` and
`.../oneshot_merged_p4_20260901/oneshot_merged_p4.log` under
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829`.
No new P=4 leg is owed: this lane is formatting plus CPU-only spectrum receipt,
and the dispatch explicitly prescribed the CPU gate.  Branch:
`feat/plan-digest-2026-09-01`.
