Heavy lane BISP-FIX; branch `fix/bisp-review-followups-2026-09-06`, base `0e16bdb6`.

Evidence root: `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/121_bisp_review_fixes_codex_2026-09-06/`; allocation `58013495`, P4 GPU legs use BFC@0.85.

| Finding | Fix commit | Gate | Verdict |
|---|---|---|---|
| 1 | this commit | `01_restart_readers/`: fresh SOC/scalar GW; both downfolds complete; direct ζ reader P4 PASS both; CPU 68 passed | Reader fix PASS. Full exciton gate BLOCKED before interpolation: SOC spin axis 2 vs 4; scalar QRCP search saturation. No spectrum comparison available. |
