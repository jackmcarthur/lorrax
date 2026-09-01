# MARGINAL — measured-table accuracy curves

## Achieved numbers

On the frozen real sodium measure at relative target `1e-4`, the four
sign-definite windows exposed **2, 3, 2, and 3 Pareto rungs** in **1.365,
1.071, 0.938, and 0.864 s** (**4.237 s summed fit wall**).  The most relevant
cheap trade was `cond:pole_tail`: **11 nodes / 8.318314e-5 residual**,
**12 / 5.737510e-5**, and **13 / 4.542867e-5**, all at benign
`kappa_p99=1.08158--1.08161`.  Thus two extra sign-definite nodes bought a
**45.38%** residual reduction on the measured support.

| frozen sodium window | measured Pareto curve `(nodes, residual)` |
|---|---|
| `cond:state_tail` | `(9, 4.334706e-6)`, `(10, 1.241785e-6)` |
| `cond:pole_tail` | `(11, 8.318314e-5)`, `(12, 5.737510e-5)`, `(13, 4.542867e-5)` |
| `val:bulk` | `(12, 7.700864e-6)`, `(14, 2.556962e-6)` |
| `val:pole_tail` | `(8, 2.438188e-6)`, `(10, 1.023783e-6)`, `(11, 1.169015e-6)` |

The last 11-node rung survives in the delivered-cost curve because its
slightly lower measured amplification lowers the runtime-noise floor even
though its residual alone is higher.  The selector consumes `required_target`,
not residual alone.

## Change and proof

`_candidate_rules` now returns every accepted sign-definite catalog rung,
Pareto-pruned by measured node count and required delivered error.  The
existing exact pointwise MILP therefore allocates accuracy by the actual
discrete node/error curve.  Crossing behavior remains one accepted candidate,
so this change does not trigger the expensive fitted fallback after a shipped
crossing table passes.  A regression proves a measured miss is skipped and
two nondominated passing rungs are both offered.  The prescribed CPU gate
passed **137/137 in 80.69 s**; its two focused marginal/selector tests passed
**2/2 in 1.70 s**.

Evidence measure:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/evidence/causal_hankel/na_reconstructed_problems_v1.npz`.
The measurement used its shipped validation lattices; factor growth was not
reconstructed because that NPZ intentionally contains reciprocal problems,
not executor state/pole metadata.

## Funded-line boundary

The real 12-window Si plan and P=4 sodium deck were **not run** inside this
sprint, so Si planning wall/pairs and consumer-level sodium `sigma_c` parity
are not claimed.  The inherited pointwise sodium evidence is 74 pairs and
0.2359 meV maximum versus `control_panes_24b`; this branch still owes that P=4
recheck.  No P=1 GPU evidence was used.

Branch: `feat/marginal-cost-allocation-2026-08-31`.
