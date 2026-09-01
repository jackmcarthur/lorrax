# Contract-predictivity preregistration

Scoring was fixed before reading any `sigma_c_kij_ev` differences.  The archive
universe is every arm under
`runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829` with both (1) a readable
`sigma_mnk*.h5` containing `sigma_c_kij_ev` and (2) an attributable final
`[delivered-planner-window]` census.  Byte-identical output copies and repeated
logs of one plan are deduplicated; every exclusion is counted and named by
reason.

The response is the complex-array difference from
`control_panes_24b/sigma_mnk.h5`: maximum `abs(delta)` and
`sqrt(mean(abs(delta)**2))`, both in meV, over the full common dataset.  The
primary predictor is the planner quantity
`sum(delivered_envelope * best_achieved_residual) / delivered_envelope_total`.
Secondary plan-time candidates, fixed before scoring, are maximum window
residual, maximum envelope-weighted window contribution, residual of each
named window, and the same quantities without mass/envelope weighting.

Pearson and Spearman correlations are descriptive only and will be reported
with sample count.  With fewer than four independent matched plans, the lane
will report the pairs but will not call a correlation predictive.  Worst-error
indices will be reported; a window-to-element claim will be made only where
the archived artifacts expose that attribution, otherwise it is an explicit
measurement gap rather than an inferred assignment.
