# FIT_LADDER result — heavy lane

## Numbers first

- Frozen sodium p0 widened to **A/gamma = 105.2** (106 frequencies): planning
  **70.966 s**, rank **111**, refined residual **5.172e-5**, kappa_p99
  **30.750**, angle **0 deg**, **3** search evaluations; target and noise gates
  both passed at **1e-4**.
- The proposed final-fit warm start was rejected: the frozen six-window gate
  changed acceptance and reached the conduction fallback. No performance
  shortcut was retained.
- Required CPU gate: **134 passed** in **82.32 s**. New focused regression:
  **1 passed** in **0.73 s**.
- Frozen ROQ suite on the base behavior is red: before this fix **5 passed,
  3 errors** in **80.22 s**, all masked by `NameError: group`. With this fix it
  reaches the physical refusal: **6 passed, 3 errors** in **105.24 s**, all
  `branch 'cond': no product-window ROQ plan meets ... gates`.

## Landed result

When a decay-compatible partition has no accepted row, the whole-branch
challenge now receives the hard sanity bound; `_try_whole_below` still derives
and applies the merged support's own rank ceiling. Previously the fallback
referenced an undefined loop variable and concealed the real selector refusal.
A regression forces this path and proves the challenge receives 512 rather
than raising `NameError`.

The performance result is negative but actionable: reusing quick-fit weights
is not acceptance-preserving, so production must continue cold-refitting the
chosen rank. The 70.966 s breakdown confirms that even the known passing
105.2 support remains far above the owner's planning/execution ratio.

Evidence: frozen input
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/evidence/causal_hankel/na_reconstructed_problems_v1.npz`;
reproduction commands and achieved outputs are recorded above. CPU-only lane;
the four-GPU rule does not apply because no GPU path changed.

Branch: `perf/fit-ladder-2026-08-31`.
