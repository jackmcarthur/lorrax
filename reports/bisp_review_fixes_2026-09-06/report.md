Heavy lane BISP-FIX; branch `fix/bisp-review-followups-2026-09-06`, base `0e16bdb6`.

Evidence root: `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/121_bisp_review_fixes_codex_2026-09-06/`; allocation `58013495`, P4 GPU legs use BFC@0.85.

| Finding | Fix commit | Gate | Verdict |
|---|---|---|---|
| 1 | `3678f103` | `01_restart_readers/`: fresh SOC/scalar GW; both downfolds complete; direct ζ reader P4 PASS both; CPU 68 passed | Reader fix PASS. Full exciton gate BLOCKED before interpolation: SOC spin axis 2 vs 4; scalar QRCP search saturation. No spectrum comparison available. |
| 2 | `8d9218ed` | `02_hartree_weights/retry.log`: 20 CPU tests passed | PASS; Hartree fixture widened to four bands for CPU P4 sharding. |
| 3 | `0308b9b4` | `03_parent_live_set/probe_{before,after}/driver.rank0.log`, `comparison.txt` | PASS: two fewer faces (10,752,000 B/rank); eqp0/1 printed data identical; earlier 7.93 GB HWM unchanged. |
| 4 | `4fe7a147` | `04_transverse_finite/gate.log`: 2 CPU tests passed | PASS: each NaN transverse face refuses with its dataset named under strict sanity. |
| 5 | `00f98a6f` | `05_parent_gate/cpu.log`: 25 passed; `mos2/driver.rank0.log`: parent startup ON | PASS; separate env, startup enforcement, announcements and rank fingerprints; MoS2 eqp identical. |
| 6 | this commit | `06_parent_admission/cpu.log`: 24 passed; `mos2/driver.rank0.log` | PASS: ns=4 two-stage plan above SMEM floor, parent-only target probe; nk=36 native resident; eqp0/1 identical. |
