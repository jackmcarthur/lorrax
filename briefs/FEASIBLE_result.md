# FEASIBLE result — cheap sign-definite precheck landed; real-Si replay absent

**Numbers first.** Commit `62a48f46` adds a pre-fit pointwise necessary-condition
check and the prescribed CPU gate passed **139/139 in 79.08 s** (the integrated
base's 137 plus two new cells). The refusal cell proves a **2.200x** floor/budget
overshoot is rejected before any fit and names both contributing windows. The
working cell passes at **0.200x**, explicitly marked necessary-not-sufficient;
its arbitrarily massive crossing window contributes a conservative zero floor.

For every sign-definite window, the check performs one certified minimax-table
lookup at the first covering geometry, evaluates that rule on the already-built
validation measure, and charges `max(residual, runtime-noise requirement)`
pointwise against the existing budget. It raises one plan-level message with the
floor cost, budget, overshoot ratio, minimum deck target, blocking frequency,
and top three contributors. It runs immediately after product-window geometry
and before `_run_parallel_planner_jobs`; it does not change the exact selector,
candidate fits, or executor.

**Unfinished part, stated plainly.** Crossing windows use a certified floor of
zero. Usable rank bounds how many modes a crossing fit may use, but rank alone
does not provide a positive residual lower bound without solving weights; doing
that here would itself be the forbidden fit. Thus this slice can cheaply prove
sign-definite budget infeasibility but does not implement the requested
usable-rank crossing floor. A precheck pass may still fail the exact selector.

The frozen gapped-Si P=4 replay did **not** produce a measurement on this commit.
The stated JID 57781731 was unavailable and `lx` attached JID 57804947; its
`lorrax_A` module advertises a CUDA venv path that no longer exists. Four P=4
launch attempts all stopped before scientific Python/JAX initialization
(`ModuleNotFoundError: jax`, CPU-only JAX, `/usr/bin/python3`, then missing
module venv). Therefore I do **not** claim that this implementation reproduces
the inherited **2.463x** Si refusal or re-verifies the real sodium one-shot
pass. The existing planner fixtures, including the two-branch working plan,
remain green. Evidence:
`tmp/planner_materials_si/feasibility_p4{,_retry,_final,_measure,_measured}.log`
and `tmp/cpu_gate.log`; no Sigma sweep or output artifact was created.

Branch: `feat/plan-feasibility-2026-08-31`, based on
`integ/planner-consolidated-2026-08-31` at `0bb0a6ba`.
