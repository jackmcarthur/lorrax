# Lane B — high-A/eta ROQ result

Heavy investigation lane.  On the prescribed frozen Na p0 measure widened at
the original `Delta omega = eta` spacing, the amended contour scan serves
`A/gamma = 125.42` with **107 nodes, residual 5.513e-5, kappa_p99 22.23**, and
83.97 s fit wall at +1.1 degrees (target 1.20e-4).  The old scan selected 0
degrees and used 133 nodes, residual 5.349e-5, kappa_p99 38.28, and 105.86 s.

| A/gamma | frequencies | old angle | rank | residual | kappa_p99 | fit s |
|---:|---:|---:|---:|---:|---:|---:|
| 99.61 | 100 | 0 | 106 | 5.589e-5 | 28.69 | 55.30 |
| 109.53 | 110 | 0 | 116 | 5.419e-5 | 32.36 | 83.23 |
| 120.45 | 121 | 0 | 127 | 5.399e-5 | 36.43 | 97.45 |
| 125.42 | 126 | 0 | 133 | 5.349e-5 | 38.28 | 105.86 |
| 130.38 | 131 | 0 | 138 | 5.313e-5 | 40.17 | 114.30 |
| 140.31 | 141 | 0 | 149 | 5.238e-5 | 43.89 | 123.19 |

This is a slope, not a cliff: every point passes.  At A/gamma=125.42 the
full-fit rank curve is also smooth: ranks 90/100/106/110/116/120/125 give
residuals 4.551e-4/2.668e-4/2.002e-4/1.629e-4/1.072e-4/7.504e-5/5.595e-5,
while kappa_p99 stays in 38.15--38.29.  Thus neither QDEIM nor IRLS becomes
degenerate on this measure, and a moderate rank already passes.

The diagnosed omission is contour coverage.  At A/gamma=125.42 the actual
decay interval is only about -0.4 to +1.2 degrees.  Every old negative angle
is rejected, leaving 0 degrees; rank-12 probing instead finds +1.1 degrees.
Production fits at +0.4/+0.8/+1.1/+1.2 degrees use rank 107 and achieve
residual 1.024e-4/5.646e-5/5.513e-5/5.521e-5 with kappa_p99
30.24/25.05/22.23/21.44.  The derived horizon is flat at 249.19 Ry^-1 from
A/gamma=120.45 through 140.31, so horizon scaling is not the wall here.

The code adds +1.2 and +1.1 degrees to the fixed production scan; the existing
`_angle_decays` check refuses them on supports where they grow.  The reported
real-deck 4.26e-3/kappa 22878 cliff is therefore not caused by A/gamma alone:
it is absent from the directed frozen reproducer, so isolating that separate
cliff requires preservation of the real deck's fitted and validation measure.

Evidence input:
`runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/evidence/causal_hankel/na_reconstructed_problems_v1.npz`
under `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14`.
Verification: prescribed CPU gate, **134 passed, 8 warnings in 121.21 s**.
No GPU leg is owed: this change is entirely in the scalar offline fitter and
the CPU-cell exemption applies.  Branch: `feat/roq-high-aoe-2026-08-31`.
