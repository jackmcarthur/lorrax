# rank_acceptance_margin result — heavy lane

## Achieved numbers

On the four frozen-Na sign-definite windows, accepting one additional rank
reduced the refined residual by **1.43x–5.17x** while kappa stayed in
**[1.016, 1.086]**.  The measured sign-definite fit sum was **2.047 s** at
first-passing, **3.292 s** at +1, and **4.609 s** at +2.  Thus +1 cost
**1.245 s total / 0.311 s per added rank** on this cold login-node replay;
+2 cost a further **1.318 s / 0.329 s per added rank**.

| window | first rank / s / residual / kappa | +1 rank / s / residual / kappa | +2 rank / s / residual / kappa |
|---|---|---|---|
| `cond:state_tail` | 6 / 0.241 / 1.170e-4 / 1.0766 | 7 / 0.402 / 2.514e-5 / 1.0771 | 8 / 0.570 / 5.686e-6 / 1.0773 |
| `cond:pole_tail` | 11 / 0.916 / 6.470e-5 / 1.0816 | 12 / 1.517 / 4.538e-5 / 1.0816 | 13 / 1.895 / 3.303e-5 / 1.0816 |
| `val:bulk` | 12 / 0.691 / 5.952e-6 / 1.0156 | 13 / 1.051 / 1.600e-6 / 1.0156 | 14 / 1.688 / 6.249e-7 / 1.0156 |
| `val:pole_tail` | 5 / 0.199 / 1.091e-4 / 1.0870 | 6 / 0.322 / 2.112e-5 / 1.0863 | 7 / 0.456 / 4.558e-6 / 1.0863 |

## Change and verdict

Production now uses a fixed internal **+1** sign-definite acceptance margin;
there is no deck dial.  The first passing rank remains recorded in the
candidate receipt, and crossing windows remain first-passing.  The benchmark
tool can replay margins 0/1/2 explicitly.  +1 is the measured recommendation:
it buys the large first residual reduction; +2 has a similar marginal wall
for a second increment that is not yet justified by delivered Sigma evidence.

The requested cold P=4 three-policy one-shot comparison was **not run**.  The
15-minute funded line was consumed by bringing the two prerequisite commits
onto the mandated base, three cold frozen replays, and the required gate; a
claimed per-omega Sigma verdict would therefore be dishonest.  Consequently
this lane does **not** claim that +1 restores the 0.5 meV delivered bar.  The
one-shot bisection remains owed and should compare the fixed +1 policy against
`control_panes_24b` before landing.

## Evidence

Frozen input: `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/evidence/causal_hankel/na_reconstructed_problems_v1.npz`.
Required CPU gate: **135 passed** (134 base cells plus the new margin gate) in
**91.71 s**.  `git diff --check` passed.  GPU verification remains owed because
the production planner policy changed.  Branch:
`feat/rank-acceptance-margin-2026-09-01`.
