# RANKPREDICT result — heavy lane

## Achieved numbers

On 20 frozen-Na sign-definite cases (four real measures at target multipliers
0.5, 1, 2, 4, and 8), the existing analytic `rank(R, target)` seed predicted
the first consumer-passing rank exactly in **14/20 cases (70%)**.  It
underpredicted **6/20**, with a worst miss of **four ranks**.  A linear law in
`log(R)` and `log(1/target)` did not generalize under leave-one-window-out
testing: it underpredicted the held-out conduction pole tail by up to **three
ranks** and overpredicted the valence bulk by up to **three ranks**.  No Si
frozen-measure archive was present beside the supplied Na artifact.

At the four production targets, replacing the remaining uniform-error search
by one fixed predicted-rank solve plus measured upward correction reduced the
six-window sequential fitting-stage wall from **4.045327 s to 3.008230 s
(1.34x)**.  It was first-pass right in **3/4** sign-definite windows.  This
candidate was not landed because it changed two accepted rules and made their
delivered residuals 3.08x and 5.60x worse; that violates the lane's no-accuracy-
trade constraint even though both still passed their apportioned targets.

| window | wall current->fixed (s) | attempts | rank current->fixed | residual current->fixed | kappa p99 current->fixed |
|---|---:|---:|---:|---:|---:|
| `cond:state_tail` | 0.272572 -> 0.120881 | 6 | 6 -> 6 | 1.169503e-4 -> 1.169503e-4 | 1.076578 -> 1.076578 |
| `cond:pole_tail` | 0.960691 -> 0.771348 | 9,10,11 | 11 -> 11 | 6.470184e-5 -> 6.470184e-5 | 1.081603 -> 1.081603 |
| `val:bulk` | 0.700624 -> 0.321256 | 11 | 12 -> 11 | 5.951931e-6 -> 1.831447e-5 | 1.015588 -> 1.015548 |
| `val:pole_tail` | 0.201441 -> 0.124100 | 4 | 5 -> 4 | 1.090873e-4 -> 6.105069e-4 | 1.087039 -> 1.084915 |

The four sign-definite fits alone were **2.135328 s current vs 1.337585 s
fixed (1.60x)**.  The six-window number is the complete frozen-deck fitting
stage, not the full production planner; no full-planner claim is made.

## Decision and evidence

The measured result does not support replacing verification by this fitted
two-variable law.  Padding by four would cover the observed misses, but adds
four tau nodes to every window and therefore shifts planning cost into the
owner-declared expensive execution sweep.  Production remains on the seeded,
measured-correction path from `fix/route-sign-definite-2026-09-01`; no table or
deck dial was added.

Frozen input:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/evidence/causal_hankel/na_reconstructed_problems_v1.npz`.
Required CPU gate: **134 passed in 90.22 s**.  GPU verification is not owed
under the Four-GPU Rule's CPU-only solver/unit exemption.  Branch:
`feat/sign-definite-rank-law-2026-09-01`.
