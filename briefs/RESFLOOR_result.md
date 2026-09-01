# RESFLOOR result — heavy lane

## Achieved numbers

Cold Na P=4/BFC@0.85 selected **112 pairs** and recovered the accuracy anchor:
`sigma_c_kij_ev` differs from `control_panes_24b` by **0.195974 / 0.008813
meV max/RMS**, below the 0.5 meV bar.  The achieved
`sum(envelope*residual)` is **1.355955302e9** (merged regression:
2.16942e9; old good tree: 1.33844e9).  JID 57804947 step 79 completed in
85.888 s wall; Sigma was 70.168 s.  The finite-H5 checker passed every dataset.

| window | nodes | residual | kappa p99 |
|---|---:|---:|---:|
| `cond:resonant` | **63** | **3.98807e-5** | **33.1357** |
| `cond:state_tail` | 9 | 4.33471e-6 | 1.07727 |
| `cond:pole_tail` | 11 | 8.31831e-5 | 1.08161 |
| `val:bulk` | 12 | 1.07204e-5 | 1.02051 |
| `val:resonant` | 9 | 2.88256e-5 | 1.12744 |
| `val:pole_tail` | 8 | 2.43819e-6 | 1.08634 |

The frozen-measure `cond:resonant` frontier (target 8.07875e-4) is:

| rank | residual | kappa p99 | noise gate | accuracy gate |
|---:|---:|---:|:---:|:---:|
| 20 | 2.00713e-3 | 9.65867 | PASS | FAIL |
| 26 | 7.35415e-4 | 9.79236 | PASS | PASS |
| 40 | 1.25122e-4 | 9.87392 | PASS | PASS |
| 55 | 4.71051e-5 | 9.89671 | PASS | PASS |
| **63 (usable)** | **3.98768e-5** | **33.2591** | **PASS** | **PASS** |

Here `A/gamma=45.6628`, the target/mass-weighted cost-law floor is 90, the
geometric ceiling is 139, but the measured snapshot spectrum has only **63
usable modes** above the runtime-noise floor.  Therefore a rule near
`E_bw/eta ~= 92` is not admissible on this measure: it is 29 modes past the
usable-rank boundary, so there is no honest rank-92 noise-gate pass to report.
The planner takes rank 63, not numerical-noise modes.  This is also a better
conditioning point than the historical 66-node rule's kappa 559.937 while
recovering its 0.196 meV observable accuracy.

The low-rank waiver is measure evidence, not a default rung.  At rank 12,
`cond:resonant` misses its target (residual 1.90277e-2) and activates the
floor, whereas `val:resonant` passes residual/noise at 8.95969e-6 and kappa
1.13961, so it remains free to select 9 nodes.

Per-omega `sigma_c` max/RMS difference from `control_panes_24b`, in meV:

| omega (eV) | max | RMS | omega (eV) | max | RMS |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.056082 | 0.007592 | 2.75 | 0.072942 | 0.007884 |
| 0.25 | 0.071067 | 0.007654 | 3.00 | 0.080001 | 0.008150 |
| 0.50 | 0.058005 | 0.007520 | 3.25 | 0.094741 | 0.008585 |
| 0.75 | 0.058766 | 0.007541 | 3.50 | 0.101923 | 0.008955 |
| 1.00 | 0.059340 | 0.007518 | 3.75 | 0.102312 | 0.009101 |
| 1.25 | 0.061863 | 0.007582 | 4.00 | 0.095121 | 0.009407 |
| 1.50 | 0.064365 | 0.007645 | 4.25 | 0.133595 | 0.010367 |
| 1.75 | 0.066873 | 0.007737 | 4.50 | 0.156249 | 0.011065 |
| 2.00 | 0.071455 | 0.007811 | 4.75 | 0.160712 | 0.011107 |
| 2.25 | 0.070972 | 0.007811 | 5.00 | 0.195974 | 0.013385 |
| 2.50 | 0.073830 | 0.007798 | | | |

## Change, gates, and evidence

Crossing groups now derive a floor from `2.02*A/gamma` after discarding at
most their own apportioned target of delivered `mass/|d|`; the already-paid
rank-12 fit may waive it only by passing both delivered gates.  The floor is
then capped by the measure's runtime-noise usable rank.  No deck dial or
execution change was added.  Mandated CPU gate: **134 passed in 84.85 s**;
focused ROQ gate: **9 passed in 55.30 s**.  The first P=4 staging attempt produced no
measurement (missing restart); the repaired cold step is the result above.

Evidence: `/pscratch/sd/j/jackm/wt_resfloor_2026-08-31/tmp/resfloor_na_p4`
(`resfloor_na_p4.log`, `job_receipt.txt`, `sigma_mnk.h5`).  Branch
`fix/crossing-rank-floor-2026-09-01`, pushed through `a5330758` before this
report commit.
