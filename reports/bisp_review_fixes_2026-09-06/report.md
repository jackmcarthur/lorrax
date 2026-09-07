Heavy lane BISP-FIX; branch `fix/bisp-review-followups-2026-09-06`, base `0e16bdb6`.

Evidence root: `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/121_bisp_review_fixes_codex_2026-09-06/`; allocation `58013495`, P4 GPU legs use BFC@0.85.

Evidence logs retain their original commit IDs; autosquash preserved the complete tree byte-for-byte (`e081af45efb0987fb8ce9876eaf42c8dbd36eefb`). The table names the final one-per-finding code commits. No suites were run.

| Finding | Fix commit | Gate | Verdict |
|---|---|---|---|
| 1 | `d60b85ef` | `01_restart_readers/`: fresh SOC/scalar GW; both downfolds complete; direct ζ reader and host-cache contract P4 PASS both (`reader_host.log`); CPU 68 passed | Reader fix PASS. Full exciton gate BLOCKED before interpolation: SOC spin axis 2 vs 4; scalar QRCP search saturation. No new exciton spectrum to compare. |
| 2 | `9cb8808a` | `02_hartree_weights/retry.log`: 20 CPU tests passed | PASS; Hartree fixture widened to four bands for CPU P4 sharding. |
| 3 | `e395e937` | `03_parent_live_set/probe_{before,after}/driver.rank0.log`, `comparison.txt` | PASS: two fewer faces (10,752,000 B/rank); eqp0/1 printed data identical; earlier 7.93 GB HWM unchanged. |
| 4 | `b793fc5e` | `04_transverse_finite/gate.log`: 2 CPU tests passed | PASS: each NaN transverse face refuses with its dataset named under strict sanity. |
| 5 | `cc449160` | `05_parent_gate/strict_on.log`: 26 passed; `mos2/driver.rank0.log`: parent startup ON | PASS; separate env, strict startup refusal for CPU/missing CUDA target, announcements and rank fingerprints; MoS2 eqp identical. |
| 6 | `527a20e1` | `06_parent_admission/cpu.log`: 24 passed; `mos2/driver.rank0.log` | PASS: ns=4 two-stage plan above SMEM floor, parent-only target probe; nk=36 native resident; eqp0/1 identical. |
| 7 | `954fa428` | `07_service_bootstrap/gate.log`: 38 tests passed | PASS: bare-src imports of all four consumers and distrib_la bootstrap-order census. |
| 8 | `9f8305c4` | `08_restart_q_contract/parser.log`: auto/ibz PASS; `retry.log`: 4 collected, 4 setup errors | Parser/doc fix PASS; full closure gate BLOCKED: required CPU `liblorrax_ffi_host.so` absent. CPU fixture preserves its platform rather than attempting a hidden GPU run. |
| 9 | `7524f288` | `09_parent_log/mos2/driver.rank0.log`: MoS2 P4 completed | PASS: eqp0/1 identical; ineffective detach block and false release message removed. |
| 10 | `6017f544` | `10_real_weights/cpu.fused.log`: 42 passed, 7 deselected; `before/` vs `after_fused/`, `comparison.txt` | PASS: Si SOC GN P4 eqp0/1 identical; compile events 622 → 622; real-weight trace has one GEMM. |
| 11 | `b9eb6cc8` | `11_local_gemm_donation/gate.log`: 30 distrib_la CPU tests passed | PASS: beta-zero out buffer remains live, result correct, no donation warning. |
| 12 | `8f055783` | `12_gij_refusal/gate.log`: 21 CPU tests passed | PASS: refusal names the explicit Gij and actual face/axis layout; config parameter is used. |
| 13 | deferred | main `0e16bdb6`, `src/gw/w_isdf.py:1276` | Registered: eager nine-source photon restore live set. |
| 14 | deferred | main `0e16bdb6`, `src/gw/cohsex_sigma.py:329` | Registered: no-plan axis projection reshards. |
| 15 | deferred | main `0e16bdb6`, `src/ffi/cpp/cufft/conv_kpair_cuda_ffi.cc:194` | Registered: repeated parent decode/phase inside spin loops; needs device rebuild and oracle. |
