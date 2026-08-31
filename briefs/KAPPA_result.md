# KAPPA result — the rank search enters a measure-induced nullspace

**Measured mechanism:** the refusing `ω<E_F val:resonant` support has a
mass-weighted ROQ snapshot subspace that loses numerical rank before its
apportioned target is reachable.  At the shipped 0-degree contour, the best
pre-cliff full fit measured here was rank 200, residual **6.58847e-3** and
`kappa_p99` **123.25**, still **69.85x** above its **9.43201e-5** target.
By rank 212 the snapshot singular ratio is **3.21e-8**; rank 216 jumps to
residual **47.0162** and `kappa_p99` **1.97815e7**.  Of the requested 512
subspace modes, **228 are exactly zero** numerically.  The production ladder
nevertheless continues to rank 512, where the exact P=4 replay reproduced
residual **8.53198e-3** and `kappa_p99` **93626.1**.  Thus the huge kappa is
caused by QDEIM/IRLS fitting beyond the weighted subspace's usable rank, not by
the smooth 13% increase in `A/gamma`.

| exact support measure | served cond | refused val | ratio / difference |
|---|---:|---:|---:|
| `A/gamma`; scale span | 181.521; 187.722 | 204.367; 208.752 | +12.6%; +11.2% |
| fit / validation cells; frequencies | 653 / 2534; 21 | 610 / 2298; 29 | comparable counts |
| legal decay-angle interval | -2.709..+0.316 deg | **-0.554..+0.289 deg** | val is 3.59x narrower |
| delivered mass on minority real sign | 1.395% | 1.800% | sign mix is similar |
| top 10% cells' delivered mass | 47.79% | **91.33%** | val is highly concentrated |
| delivered effective cell count | 198.36 | **38.37** | 5.17x smaller |
| delivered mass within `10 gamma` of resonance | 6.03% | **10.79%** | 1.79x larger |
| delivered envelope / apportioned target | 1.314e13 / 6.059e-4 | 8.443e13 / **9.432e-5** | 6.42x more mass / 6.42x tighter |
| 512-mode singular zeros | 0 | **228** | qualitative break |

The angle is not the fix.  The exact atom-decay interval leaves 0 degrees as
the only current production-grid point for both supports.  A fine rank-12 scan
preferred -0.4 degrees for valence, but its full rank-512 fit was worse
(residual **0.69793**, `kappa_p99` **982.18**).  Full rank-200 fits at
-0.5/-0.4/-0.3/0/+0.1 degrees gave residuals
**0.1723/3.2149/0.01439/0.006588/0.007939**: 0 degrees is best.  The target is
also not merely a little too tight: even the conduction window's 6.42x looser
target would still be missed by **10.87x** at the best stable valence rank.

A rank-search fix can stop before the singular-ratio cliff and return the
measured best stable refusal, cutting cancellation and avoiding meaningless
high-rank fits; it **cannot serve this support**.  A serving fix needs a
different representation or a bounded subdivision.  Changing the angle scan
or merely loosening the rank ceiling is disproved by these measurements, so no
production change is proposed in this diagnosis lane.

Evidence: `/pscratch/sd/j/jackm/wt_kappa_2026-08-31/tmp/kappa_probe/` contains
`crossing_supports.npz`, `probe.log`, and `job_receipt.txt`.  Exact replay:
JID **57789884**, step **30**, 4 ranks / 4 A100s, BFC@0.85, source
`5f750a774985936f1e6fea0fb28c4348dd2dbca0`.  No GPU gate is owed because the
branch lands only this report; the temporary capture hook was removed before
commit.  Branch: `study/crossing-conditioning-2026-08-31`.
