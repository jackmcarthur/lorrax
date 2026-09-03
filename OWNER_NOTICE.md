
# Owner/coordinator addendum (2026-09-03 12:02) — cost discipline for the two-level loop
The owner expects the nesting to cost more Sigma(omega) evaluations than a converging single-level QSGW, since the frequency integral, not the W build, is the expensive stage. Contain it and MEASURE it:
1. Innermost cycle on the FIXED Sigma table between Sigma recomputations: converge rotation and energies on the stored Sigma_c(omega) cube alone (Sigma_p = U^dagger Sigma_DFT U, the update law, eigh; this is the existing `run_fixed_sigma_evsc`/eqp2 machinery — reuse it, single owner), then recompute Sigma with the rotated G at frozen W. Only recompute Sigma when the fixed-table cycle has converged or stalled.
2. Cache the W-side time factors W_w(t_k) across inner iterations (the store is frozen, so they are identical); recompute only the G-side factors; the fused convolution still runs.
3. No re-planning inside the loop (SCFIX supplies frozen windows/rules; rebase onto it before validation).
Report cost as: number of Sigma(omega) evaluations to convergence, wall per evaluation, and the W-refit count, against the existing single-level loop on the decks where it converges (MoS2 P=30 at +-10/+-15 eV) and against its non-converging arms (Si P=20, MoS2 P=28) as evaluations-to-abandon.
