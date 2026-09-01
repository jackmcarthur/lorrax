Lane weight: **light** — cold, cache-disabled on-demand noncrossing fitting on the four sign-definite windows in the frozen six-window Na measure; no production code changed.

Achieved: **8.623488 s**, six solves, and **5–12 nodes/window** with retry factors `(1, 0.5, 0.125)`; all four delivered gates passed and measured `kappa_p99` was **1.0156–1.0816**.

Negative result: one-shot fitting is unsafe — `cond:pole_tail` achieved **1.69914e-4** against **8.71000e-5** at `1x`; `0.5x` still missed at **9.87908e-5**, while `0.125x` passed at **6.47308e-5** with 11 nodes.

Evidence: base `5f750a774985936f1e6fea0fb28c4348dd2dbca0`, `LORRAX_DISABLE_MINIMAX_DISK_CACHE=1`, and `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/evidence/causal_hankel/na_reconstructed_problems_v1.npz`; this report records every achieved number.

Verification/branch: prescribed CPU gate **134 passed in 81.51 s**; the four-GPU rule is exempt because this is an offline CPU measurement with no runtime-code change; branch `sprint/ondemand-fit-cost-2026-09-01`, and production wiring owes a measured retry ladder rather than a one-shot solve.
