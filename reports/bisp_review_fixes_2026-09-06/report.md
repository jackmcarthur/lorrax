Heavy lane BISP-FIX; branch `fix/bisp-review-followups-2026-09-06`, base `0e16bdb6`.

Evidence root: `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/121_bisp_review_fixes_codex_2026-09-06/`; allocation `58013495`, P4 GPU legs use BFC@0.85.

| Finding | Fix commit | Gate | Verdict |
|---|---|---|---|
| 1 | `3678f103` | `01_restart_readers/`: fresh SOC/scalar GW; both downfolds complete; direct ζ reader P4 PASS both; CPU 68 passed | Reader fix PASS. Full exciton gate BLOCKED before interpolation: SOC spin axis 2 vs 4; scalar QRCP search saturation. No spectrum comparison available. |
| 2 | `8d9218ed` | `02_hartree_weights/retry.log`: 20 CPU tests passed | PASS; Hartree fixture widened to four bands for CPU P4 sharding. |
| 3 | `0308b9b4` | `03_parent_live_set/probe_{before,after}/driver.rank0.log`, `comparison.txt` | PASS: two fewer faces (10,752,000 B/rank); eqp0/1 printed data identical; earlier 7.93 GB HWM unchanged. |
| 4 | `4fe7a147` | `04_transverse_finite/gate.log`: 2 CPU tests passed | PASS: each NaN transverse face refuses with its dataset named under strict sanity. |
| 5 | `00f98a6f` | `05_parent_gate/cpu.log`: 25 passed; `mos2/driver.rank0.log`: parent startup ON | PASS; separate env, startup enforcement, announcements and rank fingerprints; MoS2 eqp identical. |
| 6 | `d5e2150e` | `06_parent_admission/cpu.log`: 24 passed; `mos2/driver.rank0.log` | PASS: ns=4 two-stage plan above SMEM floor, parent-only target probe; nk=36 native resident; eqp0/1 identical. |
| 7 | `0cca2900` | `07_service_bootstrap/gate.log`: 38 tests passed | PASS: bare-src imports of all four consumers and distrib_la bootstrap-order census. |
| 8 | `e224a753` | `08_restart_q_contract/parser.log`: auto/ibz PASS; `retry.log`: 4 collected, 4 setup errors | Parser/doc fix PASS; full closure gate BLOCKED: required CPU `liblorrax_ffi_host.so` absent. CPU fixture preserves its platform rather than attempting a hidden GPU run. |
| 9 | `af0942bf` | `09_parent_log/mos2/driver.rank0.log`: MoS2 P4 completed | PASS: eqp0/1 identical; ineffective detach block and false release message removed. |
| 10 | this commit | `10_real_weights/cpu.fused.log`: 42 passed, 7 deselected; `before/` vs `after_fused/`, `comparison.txt` | PASS: Si SOC GN P4 eqp0/1 identical; compile events 622 → 622; real-weight trace has one GEMM. |
