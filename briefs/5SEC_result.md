# 5SEC result — defer cancellation scoring until rank selection

## Achieved numbers

Back-to-back login-node measurements used frozen sodium `p0`, target `1e-4`,
and the same 21-point frequency count widened to the three requested supports.
The control is `3280d83b` (the plan-wall salvage); the changed tree is
`ef52e919`.  The login node was visibly contended, so only these adjacent runs
are compared.

| A/gamma | before (s) | after (s) | speedup | rank | residual, before = after | kappa_p99, before = after |
|---:|---:|---:|---:|---:|---:|---:|
| 65.5168 | 15.1791 | **7.6106** | **2.00x** | 61 | 8.43955550e-5 | 16.8461095 |
| 85.3708 | 16.2064 | **11.6611** | **1.39x** | 91 | 5.83039962e-5 | 22.5255282 |
| 105.2248 | 22.3145 | **12.7838** | **1.75x** | 117 | 8.07731025e-5 | 29.0295535 |

Total wall fell **53.7000 -> 32.0554 s (1.68x, 40.3%)**.  The requested
five-second wall was not reached; the smallest case is 7.61 s on this host.
All three selected exactly the same rank, residual, kappa and 0-degree contour
before and after.

## What the profile cut

The 65.5 control profile took 9.121 s in a less-contended sample.  Exact
scoring consumed 4.730 s, including 3.343 s in `rule_amplification`; subspace
construction consumed 2.089 s and all four rank probes together consumed
6.609 s.  Rank bracketing acts only on the exact refined-lattice residual, but
the control also computed full kappa for every probe.  The change defers that
calculation and performs it once on the selected rule before acceptance.  It
does not change the node basis, IRLS solve, weights, delivered residual, or
noise threshold.  A focused regression proves a quick probe cannot be
accepted without the deferred amplification score.

The inherited plan-wall change also warm-starts final refits and interpolates
rank in log(error); this lane deliberately did not duplicate the sibling
usable-rank-cap work.  The frozen three-support measurements above exercise
four, three and four search fits respectively.

## Verification and scope

The prescribed gate collected **134 tests and passed 134** in 118.46 s (two
known warnings).  The new focused tests passed **2/2**.  The scalar-budget
one-shot reference remains 6 windows / 115 pairs / 0.1959 meV on the inherited
base; this change does not alter any accepted rule, as shown directly by the
bit-identical rank/residual/kappa rows above, and does not touch the planner or
executor.  No P=4 leg is owed under the CPU-cell exemption: this is scalar
offline NumPy/SciPy fitting and the prescribed integration gate ran on CPU.

Branch: `perf/five-second-planner-2026-08-31`.
