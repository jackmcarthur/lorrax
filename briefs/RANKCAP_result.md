# RANKCAP result — usable-rank refusal is total and 40.7% faster

**Measured first:** on the frozen refusing `omega<E_F val:resonant` support
(`A/gamma=204.367`, 29 frequencies, 610 fit / 2298 validation cells), the
same single-core CPU replay fell from **94.499 s** uncapped to **55.993 s**
patched (**38.506 s / 40.7% faster**).  The returned rule moved from rank 512,
residual **1.08447**, `kappa_p99=19826.5`, to the best locally probed stable
rank **197**, residual **6.72278e-3**, `kappa_p99=130.108`.  It correctly still
refuses the **9.43201e-5** target.  The published P=4 baseline on this same
measure was rank 512, residual 8.53198e-3, `kappa_p99=93626.1`; CPU and P=4
absolute cancellation differ, while both expose the same unstable high-rank
failure.

The implementation caps the cost-law ceiling by the weighted snapshot's
usable singular spectrum.  A ladder miss now probes a bounded four-rank
neighborhood below the numerical cliff, fully fits its best local rank, and
uses one common initialized final path.  Thus a ceiling below the rank-12
angle probe still evaluates at least one rank and always returns a `RoqRule`;
the former empty-range/`None.times` failure is removed.  The integration test
also now accepts the live ROQ serving this low-rank fixture before its shipped
fallback while retaining the complete crossing-rectangle assertions.

Verification: the requested CPU gate collected and passed **134/134** in
**146.38 s**; after the final refusal-selection change, the formerly failing
focused test passed **1/1** in **9.52 s**.
Timing used `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` and the
brief's supplied venv.  Frozen input:
`/pscratch/sd/j/jackm/wt_kappa_2026-08-31/tmp/kappa_probe/crossing_supports.npz`.
No GPU kernels changed.  The six-window scalar-budget sodium replay and its
0.1959/0.00883 meV comparison were not rerun inside this sprint, so that P=4
receipt remains owed rather than inferred.

Branch: `feat/usable-rank-cap-v2-2026-08-31`; commits `38d0b91c` (inherited
WIP) and `452d8fa7` plus the final measured follow-up.
