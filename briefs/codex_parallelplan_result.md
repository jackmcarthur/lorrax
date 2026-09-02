# Rank-parallel delivered planning — result

Heavy lane. Branch `perf/parallel-planning-2026-08-31`, based on the complete
`feat/roq-wiring-2026-08-31` tip `0404cf40`. Final measured source is
`c6cd506a31f9d3d70cd4f183826e8a6d4a94268f` on JID 57760788, P=4,
BFC@0.85. Evidence directory:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829/codex_parallelplan_p4_20260831`.

## Change and determinism contract

Window `i` is fitted only by process `i % process_count`. Small candidate
dictionaries are pickled without numerical conversion, gathered through
`common.collectives.all_gather_processes` (one length gather and one padded
byte gather), checked for exactly one deterministic owner, and reconstructed
in index order on every process. No mesh is created. Exceptions are gathered
as data before any process raises; all processes choose the lowest-index
refusal message. Consolidation trials use the same exchange. The global-budget
retry reuses gathered adapted rules and one gathered merged result/refusal per
branch, rather than fitting either twice. The O(N^4) pair path remains absent.

The CPU rank-emulation gate builds complete plans at P=1, 4, and 16 and compares
raw bytes for window names/routes/masks, nodes, and executor weights. A second
gate injects refusals on two ranks and checks that all observers raise the same
lowest-index message. A retry gate proves each branch's consolidation trial is
fitted once. Required CPU set: **124 passed, 0 failed, 0 skipped** in 62.22 s
(`test_delivered_windows`, `test_hybrid_wiring`, `test_delivered_executor`,
`test_layering`).

## P=4 result and accuracy

The cold-cache artifact checker is **PASS** for all eight required datasets
(`sigma_checker.txt`). Plan: **4 windows, 98 (window,tau) pairs, 98 distinct
tau, 121 dispatches, 0 direct terms**. Against
`control_panes_24b/sigma_mnk.h5`, `sigma_c_kij_ev` is **0.6457120361 meV max /
0.0168763116 meV RMS**, preserving the stated 0.6457/0.01688 meV result.
Sigma stage wall is **78.421 s**; total driver wall is **93.530 s**
(`parallelplan_p4.log`, step 57760788.64).

The exact frozen 24-band arm cannot pass the strict band-edge guard: chi=48
ends at the 48-band WFN extent and Sigma edge 24 splits a multiplet (legal
nearby edges 20/26/46). `strict_preflight.log` records the refusal. The final
comparison used the source arm runner's declared diagnostic
`LORRAX_BAND_DEGENERACY=snap`; this guard continued at exactly 24 bands and did
not widen the calculation.

## Achieved wall and residual cost

Cold builder planning is **44.0784066 s**, down 33–56% from the brief's
66–100 s range. An intermediate implementation that repeated consolidation
cost 74.881 s; caching the gathered trials removed 30.803 s. The final receipt
(`plan_receipt_summary.txt`) costs the critical path as:

- premeasured-census receipt check 0.000091 s; window geometry 5.423440 s;
- ordinary window fits 6.055297 s: adapted 4.487607 s, shipped fallbacks
  1.567657 s, overhead 0.000032 s;
- merged trials **32.387245 s**: two jobs assigned to ranks 0/1, one accepted
  valence merge and one deterministic conduction refusal; ranks 2/3 have no
  independent merged branch to serve;
- gather/wait 0.170522 s; selection 0.000063 s; cache/output assembly 0.041750 s.

The pole census itself is produced upstream of this builder from premeasured
compact fields and was not separately profiled in the final leg; therefore
44.078 s is already a lower bound on total planning. Subtracting it from the
78.421 s Sigma stage leaves 34.343 s for the served remainder, whose one-tenth
allowance is 3.434 s. Planning is **12.8x that ceiling** (and 11.8x the 3.74 s
ceiling from the stated 37.4 s P=4 warm execution). The owner rule is **NOT
ACHIEVED**. Parallel rank distribution removes redundant ordinary fits, but
the single 32.39 s merged-conduction refusal is now the dominant indivisible
cost; further closure requires making or cheaply pre-refusing that trial, not
more rank scheduling.
