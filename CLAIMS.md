# BISP-PROF-S worktree-local evidence ledger

Shared sandbox ledger is outside this lane’s explicit write scope. IDs below are local.

| ID | Date | Status | Claim | Evidence | Artifact |
|---|---|---|---|---|---|
| 1 | 09-06 | **Pinned MoS2 static P/F baseline.** On branch perf/bisp-prof-s-2026-09-06, unmerged, production unchanged. BFC@0.85 P4 sequential Sigma150.63/18.24s; rank0 whole-run compilations1712/597 and compiler work147.90/27.74s. Sigma-only attribution pending. | MEASURED | JID57966610 steps lx-Xg4-011831-1203588-3371 and lx-Xg4-012232-1236210-4898 exit0 | reports/bisp_prof_s_2026-09-06/report.md |
| 2 | 09-06 | **Static Sigma compilation and native unit attribution.** On branch perf/bisp-prof-s-2026-09-06, unmerged. P64 contraction modules versus F4; P300 restore/add modules execute only6.607031ms projected GPU total. Warm TT SX P body/head2.185950/2.043967ms versus F combined6.757437ms. All P block/restore HLO modules have zero explicit collectives; FFI broadcasts remain. Both capture EQP files and90 printed Sigma rows exactly match same-source baselines. | MEASURED | JID57966610 steps lx-Xg4-012347-1245192-7444 and lx-Xg4-013253-1309401-8544 exit0 | reports/bisp_prof_s_2026-09-06/static_profile_receipts.json |
