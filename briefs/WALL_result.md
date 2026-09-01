# WALL result — merged planner remains 6.10x above five seconds

## Achieved numbers

The cold-cache frozen-Na whole-deck plan took **30.513 s** (`plan_cache=miss`;
no cache supplied), a **25.513 s gap** to 5 s.  The built-in planner profile
was **30.177 s angle/rank fits**, **0.331 s selection/scoring**, and **0.0035 s
whole-branch challenge** (the top three costs).  Thus fitting is 98.90% of the
measured wall.  The replay excluded imports and NPZ loading and used one
adjacent single-core-BLAS CPU process on this login host.

| frozen Na window | kind | fit wall (s) | rank | residual | kappa p99 |
|---|---|---:|---:|---:|---:|
| `cond:resonant` | crossing | 1.709 | 27 | 7.46754e-4 | 9.8188 |
| `cond:state_tail` | sign-definite | 3.281 | 9 | 1.94246e-5 | 1.1073 |
| `cond:pole_tail` | sign-definite | 9.908 | 14 | 3.23359e-5 | 1.0586 |
| `val:bulk` | sign-definite | 27.470 | 11 | 2.21630e-5 | 1.0146 |
| `val:resonant` | crossing | 0.690 | 9 | 2.91597e-5 | 1.1276 |
| `val:pole_tail` | sign-definite | 8.323 | 9 | 6.75348e-5 | 1.0458 |

Individually, sign-definite fits summed to **48.981 s** and crossing fits to
**2.399 s**; the all-window planner reduced the resulting deck to 3 rules / 53
nodes by branch consolidation.  The obvious next lever is therefore the
sign-definite planning wiring: production `_measure_adapted_candidate` routes
both kinds through ROQ, so the merged Hackbusch-seeded on-demand solver is not
reached.  I did not optimize this lane.

The comparable symmetric widened-p0 replays (21 omega points, target `1e-4`)
were:

| A/gamma | wall (s) | gap to 5 s | rank | residual | kappa p99 |
|---:|---:|---:|---:|---:|---:|
| 65.5168 | **3.820** | -1.180 | 60 | 9.43501e-5 | 16.8453 |
| 85.3708 | **6.121** | +1.121 | 102 | 5.36145e-5 | 23.2921 |
| 105.2248 | **8.850** | +3.850 | 119 | 7.02186e-5 | 29.0300 |

These are respectively 1.99x, 1.91x, and 1.44x faster than the previously
reported 7.6106 / 11.6611 / 12.7838 s adjacent series.  Only the narrowest
support is below 5 s.

## Tree, gate, and merge resolution

Measured tree before this report: `c252a86b74067d8961ae8918db8744bfaf6b0d30`;
the imported minimax service resolved inside this worktree.  Frozen input:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/evidence/causal_hankel/na_reconstructed_problems_v1.npz`.

All four requested branches have merge commits.  For the null-family conflict
I kept both the consolidated frequency-resolved envelope and the zero-centered
factor references.  For on-demand-only I kept catalog deletion and retained
the executor-reference assertions.  The marginal branch's table-walk
implementation was irreconcilable with the settled no-precomputed-tables rule,
so I kept the on-demand path and left that table enumeration out.

The prescribed CPU gate passed **136/136 in 158.87 s** with one warning.  No
GPU leg is owed: this lane changed no runtime code and measured the explicitly
CPU-only NumPy/SciPy fitting path.  Branch: `test/wall-2026-09-01`.
