# FIX-1 result — pointwise selector

**Measured first:** the audit case changed from a false `_BudgetShortfall`
after retaining cost **[0.1, 0.9]** to a feasible **3-node** selection costing
**[0.9, 0.1]** under the pointwise budget **[1.0, 0.5]**. The regression fails
on the imported pointwise-budget parent and passes on this branch.

The scalar-pruned DP is replaced by an exact bounded binary selection: one
binary variable per offered rule, one equality per window, plus the existing
pair ceiling and every live-frequency budget. It cannot discard incomparable
cost vectors and retains no growing Pareto frontier. A warmed sodium-shaped
probe with **6 windows / 12 offered rules / 115 frequencies** formed **12
variables / 122 constraints**, selected a feasible **105-node** plan, and took
**7.623 ms median / 9.569 ms p95 / 21.301 ms max** over 100 calls. This prices
the selector only; it does not claim the rank-ladder fits now meet the owner's
whole-planner one-tenth rule.

The prescribed CPU gate passed **136 tests / 0 failures** in **93.64 s**
(94.93 s process wall, 628736 KiB maximum RSS): the base 134, Lane D's
pointwise regression, and the new incomparable-choice regression.

The copied Na one-shot P=4/BFC@0.85 arm executed this branch at `f46bfd51` on
four A100 ranks, but strict band validation refused after **11 s**, before
planning: `number_bands_chi=48` reaches the 48-band WFN extent and multiplet
closure is unverifiable. I did not reuse the historical arm's prohibited
`LORRAX_BAND_DEGENERACY=snap`; consequently no Sigma artifact exists and the
**6-window / 115-pair, 0.1959 / 0.00883 meV** sodium parity remains unverified
on this branch. Evidence:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829/pointwise_dp_p4_20260831`.

Branch: `fix/pointwise-dp-2026-08-31`.
