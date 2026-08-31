# Lane INT result — wide Sigma integration

**FAIL, measured at `53a76a10` on P=4/BFC@0.85.** The live -15..+15 eV
deck did not plan: after an observed **~112 s planner interval** (10:34:03
post-Hartree matrix close to the 10:35:55 planner exception; the refusal path
does not emit its normal profile record), the first refusing census row was:

```text
omega>=E_F cond:resonant: kind=crossing, source=live_fit,
A/eta=124.7105582, A/gamma_min=123.4939328, cells=652,
mass_share=0.31833936, apportioned_target=1.1955963e-4,
candidate_family=null, best_residual=4.2590361e-3,
best_kappa_p99=22877.7687, status=refused
```

Therefore there is **no plan window count, pair count, per-window selected
family/rank, or map-0 max|dE|** to report. The full P=4 step took 307 s; its
self-consistent-driver stage took 281.1 s and ended at the planner refusal.
The positive-angle contour scan did not change the previously measured
refusal (old residual 4.2590366e-3, kappa 22877.7709). Five census rows were
refused in total; the driver correctly raised the first one rather than
routing around it.

The required one-shot check completed (`rc=0`) but **regressed**: **6 windows /
74 pairs**, rather than 6/115, and `sigma_c_kij_ev` versus
`control_panes_24b` was **0.235789104 meV max / 0.009574672 meV RMS**, rather
than 0.195876/0.008834 meV. The 74-pair plan's selected per-window counts were
25, 9, 11, 12, 9, 8. This is a semantic integration conflict between the two
clean textual merges, so I leave it measured rather than choosing a winner
without another budget design.

The exact CPU gate passed **135/135** in 94.77 s. A strict/default-band-policy
preflight refused the legacy 48-of-48 WFN because closure is unverifiable; the
reported science runs then used the copied owner's launcher condition
`LORRAX_BAND_DEGENERACY=snap`, as did the registered controls. Both P=4 logs
print `RUN_GIT_HEAD=53a76a10846656cbf8cf717180894ff12e9f7bcf` and were placed on
separate nodes of shared JID 57781731.

Evidence: `/pscratch/sd/j/jackm/int_wide_sigma_20260831/{live_sc,regression}`.
Branch: `integ/wide-sigma-2026-08-31`; merged inputs are
`c6ebcd1a` (positive contour scan) and `eed87297` (pointwise budget).
