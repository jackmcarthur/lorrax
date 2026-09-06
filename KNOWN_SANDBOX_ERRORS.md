# BISP-PROF-S local scaffolding register

- 2026-09-06: prescribed once-only allocation failed QOSMaxSubmitJobPerUserLimit; fallback57955075 is invalid. No authorized pool, no compute measurement. Evidence: runs/Si/100_bisp_parent_route_2026-09-05/prof_s/00_allocation/alloc.log and fallback.log.
- Supplied historical GN checkpoint arms initially lacked quadrature caches and rebuilt in memory. New prepared P/F arms copy the same completed P tmp; actual selected-node/digest equality remains to check. Evidence: reports/bisp_prof_s_2026-09-06/checkpoint_receipts.txt.
- Shared ledger/report paths are outside the explicit write scope; lane-local registers/report are used. No shared sandbox file modified.
