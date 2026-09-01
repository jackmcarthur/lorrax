# FIT_FASTSTOP result — light lane, shorter ROQ rank probes

**Numbers first.**  On the frozen real-sodium `p0` measure widened to the
same three 21-point symmetric omega grids, reducing quick IRLS from
`16 iterations / 3 stalled` to `8 / 2` preserved every selected rank and
printed residual/kappa while reducing adjacent whole-fit wall by **8.25%**
summed (**25.017660 -> 22.954204 s**).

| A/gamma | before (s) | after (s) | change | rank | residual | kappa p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 65.516770 | 5.971197 | 5.282854 | -11.53% | 70 | 5.12139312537e-5 | 16.8732937 |
| 85.370773 | 7.913872 | 7.199468 | -9.03% | 91 | 5.71644858346e-5 | 22.5255061 |
| 105.224776 | 11.132591 | 10.471882 | -5.93% | 134 | 4.52476445660e-5 | 29.0347283 |

An isolated A/B of the three actual rank probes on the widest support,
reusing one prepared subspace, took **4.438861 s** at `16/3` and
**4.280105 s** at `8/2` (**3.58% less**).  Ranks 89/111 remained measured
misses at `3.62789413643e-4` / `1.08921212406e-4`; rank 134 remained a
measured pass at `7.86495049342e-5`, with all three residuals and kappas
unchanged to the printed precision.  This is a bounded planning-path saving,
not evidence that the five-second owner rule is met.

The code change is two policy values plus a regression locking quick versus
final IRLS budgets.  The focused cell passed **1/1 in 0.81 s** and the exact
prescribed CPU gate passed **134/134 in 93.99 s**.  No GPU leg is owed under
the CPU-cell exemption: only the offline NumPy/SciPy fitter changed; Sigma
execution and FFT code are untouched.

Evidence artifact: `briefs/FIT_FASTSTOP_result.md`; frozen input:
`runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/evidence/causal_hankel/na_reconstructed_problems_v1.npz` in the supplied sandbox corpus.
Branch: `perf/fit-ladder-faststop-2026-09-01`, code commit `f648aeb9`.
