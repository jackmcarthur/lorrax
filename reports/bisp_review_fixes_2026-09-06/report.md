Heavy lane BISP-FIX; branch `fix/bisp-review-followups-2026-09-06`, base `0e16bdb6`.

Evidence root: `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/121_bisp_review_fixes_codex_2026-09-06/`; allocation `58013495`, P4 GPU legs use BFC@0.85.

| Finding | Fix commit | Gate | Verdict |
|---|---|---|---|
| 1 | `3678f103` | `01_restart_readers/`: fresh SOC/scalar GW; both downfolds complete; direct ζ reader P4 PASS both; CPU 68 passed | Reader fix PASS. Full exciton gate BLOCKED before interpolation: SOC spin axis 2 vs 4; scalar QRCP search saturation. No spectrum comparison available. |
| 2 | `8d9218ed` | `02_hartree_weights/retry.log`: 20 CPU tests passed | PASS; Hartree fixture widened to four bands for CPU P4 sharding. |
| 3 | this commit | `03_parent_live_set/probe_{before,after}/driver.rank0.log`, `comparison.txt` | PASS: two fewer faces (10,752,000 B/rank); eqp0/1 printed data identical; earlier 7.93 GB HWM unchanged. |
