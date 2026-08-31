# ROQ production-rule result

Heavy lane. Branch `feat/roq-production-rules-2026-08-31`, based on
`1426d9f4`; implementation commits `7c131c1f`, `e31f7d2e`, and `5e078749`.
All measurements below use the frozen Na 24-band fit/refined-validation
measures named in the task, `eta=0.01837465441237269 Ry`, one CPU process,
four planning workers, and single-threaded BLAS. No GPU verification is owed:
this is scalar NumPy/SciPy offline planning and the CPU-cell exception applies.

## Achieved node and validation result

| route | total nodes | achieved refined-lattice error (cond / val) | amplification (cond / val) |
|---|---:|---:|---:|
| production six panes | 137 | `1.35633e-4 / 2.43017e-5` | not present in frozen export |
| certified lookup planner | 154 | not remeasured here | shipped `kappa0=1` |
| study ROQ, incumbent aggregate accuracy | 50 | `9.60254e-5 / 1.05378e-5` | study component values only |
| study ROQ, both branches below `1e-5` | 69 | `9.53990e-6 / 9.36858e-6` | effective `1.77 / 1.1825` |
| **production generator, this branch** | **54** | **`6.9276089e-5 / 1.1741761e-5`** | **effective `1.58217 / 1.10104`** |

The delivered plan is four nodes above the 50-node study result and 83 nodes
below the pane incumbent. It beats both incumbent aggregate branch errors and
the task's 69-node ceiling.

| selected product group | nodes | derived angle | derived horizon (Ry^-1) | achieved group error | kappa p99 |
|---|---:|---:|---:|---:|---:|
| conduction resonant | 27 | `0 deg` | `249.1893` | `7.6292809e-4` | `9.83552` |
| conduction state + pole tails | 15 | `-75 deg` | `83.0118` | `2.0467776e-5` | `1.03168` |
| entire valence branch | 12 | `-65 deg` | `30.8682` | `1.1741761e-5` | `1.10104` |

Every rule and both aggregate branches pass
`kappa_p99 * 6e-8 <= 0.05 * target`. Acceptance uses only production
`delivered_error` on the refined lattice. The all-conduction rotated-rule
challenge refuses from support geometry because no scanned negative angle
decays on its resonant cells; there is no explicit `(n,p)` evaluator or other
non-product-window escape hatch.

## Planning wall

The study's 363-prefix linear search cost 10.7 s. An intentionally complete
first implementation here cost 61.33 s because it optimized every alternative;
batching reduced that to 23.74 s. The final lookup-free production generator
cost **4.681769 s on the first run** and **3.890177 s on the immediate repeat**.

First-run breakdown:

| phase | wall (s) |
|---|---:|
| partition by shared contour decay | `0.003728` |
| fixed angle scan + bracketed rank fits | `4.105556` |
| whole-branch challenge/refusal | `0.006241` |
| per-window fallback | `0.000000` |
| refined scoring and selection | `0.566243` |
| **total** | **`4.681769`** |

The reduction comes from one rank-independent snapshot basis per finalist,
bracketed rank search rather than prefix scanning, all independent angle/group
fits in one four-core pool, and lazy fallback work. The fixed angle scan is
derived policy, not a user dial; it retains `-55` and the independently measured
`-58` discriminator. Horizon is `5 / max(eta, q_0.01%)`, where the quantile is
weighted by delivered mass; this reproduces the study's three contour scales
without a horizon dial.

## Reproducibility and tests

Two complete planner calls returned bit-identical time nodes and weights with
rule digest
`672e34978f1aa2cc2b44c32873f045555787605c41721353a0d6556fb6f74ca9`.
The focused suite reports **8 passed in 12.39 s**, including the two frozen Na
runs, angle selection, full-valence consolidation, growth refusal, node/error/
noise acceptance, and bit determinism.

The full service suite collected 134 tests: **129 passed, 5 failed** on stale
base catalog pins unrelated to this lane. The base contains 34 tables and now
selects the shipped `crossing_hgl_A_24...` table; those five tests still assert
31 tables and the older `A=40` selection. No source outside
`services/minimax/` was changed; this result page is the only additional file.
