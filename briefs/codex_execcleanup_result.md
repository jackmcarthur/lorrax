# Executor cleanup result

Branch `perf/exec-cleanup-2026-08-31`; source verified in the GPU leg at
`a7e4fd01e286ab41b56aafcea1590f1ddbaefb23`.  Lane weight: heavy.

## Result

- Deleted `LORRAX_DELIVERED_MAX_DIRECT_TERMS`, its resolver, its GN/MPA call
  sites, executor accounting, census text, and tests.  Product windows are now
  the only executor contract.  A cached plan with explicit pair rows or a
  recorded nonzero direct-work count refuses before execution and names the
  receipt path.
- The base already performed the collective `sigma_c_kij_ev`,
  `sigma_sx_kij_ev`, and `hartree_kij_ev` write/close before mean-field load,
  `solve_qp`, and `eigh`.  The new regression gate pins that order.  The P=4
  leg confirmed the raw file was readable at the first QP marker.
- Bring-up audit: no low-risk edit taken.  Runtime initialization precedes GW
  imports; restart I/O is outside the executor; the MPA path already shares one
  collective pole reader across census and execution.  Removing its remaining
  small certification-ledger reread would duplicate the store validator.

## Before/after timings

Both rows use the copied Na 24-band delivered arm at P=4.  The prior row is the
warm-worker `onekernel_warm2_step2` measurement; the new row is a cold `lx run`
at canonical BFC, memory fraction 0.85, and therefore is not claimed as a speed
comparison.

| Stage | Prior P=4 warm (s) | Cleanup P=4 cold (s) |
|---|---:|---:|
| Runtime bring-up | 5.53 | 7.14 |
| Restart load | 3.69 | 6.16 |
| Sigma | 37.39 | 40.03 |
| QP solve + diagonalize | not separately quoted | 0.37 |
| Total wall | 51.68 | 59.56 |

The delivered receipt was a `complete_hit`: 6 windows, 137 `(window,tau)`
pairs, 194 tau dispatches, and no pairwise-work census.  The six raw numerical
datasets are bit-identical to the copied arm's preceding artifact
(`max_abs_delta=0` for omega, total, correlation, exchange, Hartree, and
evaluation-energy arrays).

## Verification and evidence

- CPU: `tests/test_delivered_executor.py tests/test_hybrid_wiring.py` collected
  20 tests; **20 passed** in 14.86 s.
- GPU: shared allocation **JID 57754440**, successful **step 2** on nid001064,
  P=4, 63 s launcher wall.  The strict-policy preflight was step 1 and refused
  before restart because the frozen WFN cannot certify its 48-band end edge;
  step 2 used the copied arm's diagnostic `snap` setting.  No allocation was
  created or released by this lane.
- Artifact parser: all 8 datasets finite, `FINITE_SUMMARY ... status=PASS`.
- Write order: collective `sigma_mnk.h5` close completed at 15:10:31; QP solve
  began at 15:10:32.  At the QP marker the file already had size 18,485,384 B,
  mtime `1788127833000000000` ns, and readable correlation, exchange, and
  Hartree datasets.  The filesystem's final mtime remained in the same coarse
  one-second tick after the small EQP-receipt append.

Evidence directory:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829/codex_exec_cleanup_24b_20260831`

Key files: `exec_cleanup_p4.log`, `job_receipt.txt`,
`pre_qp_checkpoint.txt`, `check_sigma_h5.txt`, `compare_prior_sigma.txt`, and
`sigma_mnk.h5`.
