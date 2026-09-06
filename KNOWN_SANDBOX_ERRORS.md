# BISP-PROF-S local scaffolding register

- 2026-09-06: prescribed once-only allocation failed QOSMaxSubmitJobPerUserLimit; fallback57955075 is invalid. No authorized pool, no compute measurement. Evidence: runs/Si/100_bisp_parent_route_2026-09-05/prof_s/00_allocation/alloc.log and fallback.log.
- Supplied historical GN checkpoint arms initially lacked quadrature caches and rebuilt in memory. New prepared P/F arms copy the same completed P tmp; actual selected-node/digest equality remains to check. Evidence: reports/bisp_prof_s_2026-09-06/checkpoint_receipts.txt.
- Shared ledger/report paths are outside the explicit write scope; lane-local registers/report are used. No shared sandbox file modified.

- 2026-09-06: tools/parse_lorrax_sigma_run.py rejects completed COHSEX gwjax.out because _parse_production_report requires absent Sigma rule plan/tau sweep rows. Static evidence retained as the production table; no invented zero-time parser output. Run prof_s/05_P_full_static_baseline, step lx-Xg4-011831-1203588-3371.

- 2026-09-06: first common-rule preparation incorrectly required optional Si dipole.h5; baseline has no such file. Step lx-Xg0-015048-1445882-7419 exit1 before science. Preserve partial Si11 directory; new prepared profiles use Si13/14 and copy optional dipole only when present.

- 2026-09-06: ablation13 imported w_isdf before gw_jax initialized distributed runtime and failed at startup (driver.rank0.log). Preserve failed arm; ablation19 imports the canonical driver startup first. Apply the same ordering to unrun18 and common-rule profile preparation.

- 2026-09-06: the sandbox HLO analyzer, as consumed by PERF2 profile_collective_census.py, omits collective-permute-start. M20 module4848 is reported with an empty static census despite optimized HLO lines701/744 containing two starts, corroborated by two native SendRecv kernels. Preserve the canonical output and supplement it with an explicit async-start census; never infer communication-free from that empty result. Evidence: JID57966610, M20 driver.1.log and xla_dump_rank0/module_4848.jit__tau.sm_8.0_gpu_after_optimizations.txt. This is a parser defect, not evidence that the kernels lack communication.
