# Lane D — pointwise delivered-error budget

**Measured:** the disjoint two-window regression closes at **2 nodes** with a
worst pointwise spend of **0.08 / 0.10 (80%)**, while the retired sum of window
maxima is **0.16 / 0.10 (160%)**.  This is zero extra nodes for a case the scalar
contract refused.

The planner now derives every window allowance from its live envelope at each
served omega, searches the existing scalar DP at the maximum-combined-envelope
frequency, and validates the selected rule vector against the delivered
contract at every omega.  Cached rules are re-certified pointwise; cache schema
version 7 prevents scalar-budget receipts from crossing the change.

The prescribed CPU gate collected and passed **135 tests** (the base's 134 plus
the new regression), with **0 failures** in **103.74 s**.  Evidence is the pytest
output for this worktree; the regression is
`test_rule_selection_enforces_budget_pointwise_not_sum_of_window_maxima`.

The real signed `-5..+5 eV` deck and the one-shot `control_panes_24b` numerical
twin were not completed inside the 15-minute sprint, so the reported zero-node
result is synthetic, not a claim about those decks.  No GPU leg was run; the
unit/CPU-cell exemption applies to the evidence above, and P=4 deck verification
remains owed before landing.

Branch: `fix/pointwise-budget-2026-08-31`.
