| Stage 0 ceiling / integrator priority | Clock | ms | % of fit wall | Evidence |
| --- | --- | --- | --- | --- |
| MoS₂ 6×6 P4 fixed-tip whole ζ fit | host, cold, compile included | 48500 | 100 | 09_fixed_fit_profile; charge23400 + nonoverlapping transverse25100 |
| Cherry-pick FIRST: aa0fdb6e | fresh-fit refusal fix | — | — | Already pushed/gated by integrator;10 CPU passes, MoS₂≤0.018µeV; claim1411 |
| MoS₂ CCT charge/current | host | 9000 | 18.56 | CCT receipts4600+4400; not pure device tail |
| MoS₂ CCT charge/current modules | device | 43.527 | 0.090 | 0127 / 0379; includes their projectors |
| MoS₂ charge ZCT module | device | 2608.027 | 5.377 | 0225;9 tiles, includes projectors |
| MoS₂ coupled-current ZCT module | device | 1854.87 | 3.824 | 0470;7 tiles, includes projectors |
| MoS₂ eigensolve module | device | 389.042 | 0.802 | 0182; rank-truncated factor |
| Si leg20 whole ζ fit (actual ns=2) | host, cold, compile included | 15500 | 100 | 10_fixed_si_profile |
| Si CCT | host | 4600 | 29.68 | Includes compilation |
| Si chunk loop | host | 5900 | 38.06 | Includes projectors, ZCT, solves and I/O |
| Si CCT / ZCT modules | device | 35.886 / 506.144 | 0.232 / 3.265 | 0077 / 0198;2 ZCT tiles |
| Ceiling interpretation | — | — | — | Device rows exclude host compilation; host and device rows overlap and must not be added. ZCT module rows are upper bounds on tail device work. |

| Scope | Value |
| --- | --- |
| Lane / branch | Heavy BISP-ZFFI; perf/bisp-zeta-ffi-2026-09-06, unmerged; no rebase |
| Kernel commit | c2e847ad; additive CufftConvKParentCudaFfi, ABI3, existing handlers unchanged |
| Worktree | /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_zffi_codex_20260906 |
| Evidence root | /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/118_bisp_zffi_codex_2026-09-06 |
| Admission | Automatic ns=4 native; automatic ns=1/2 retains the existing decomposed tail to preserve required printed digits. Explicit on covers all shapes in the kernel oracle. |
| Owner stage ruling | JAX feed stage skipped; native CUDA implemented directly, unconditionally. No new full-k operand carrier. |
| Measurement geometry | P4=-N1 -G4 -n4; P16=-N4 -G4 -n16; BFC@0.85; cold JIT; one sample per arm |
| Stage 0 source | ISDF core equals aa0fdb6e; additive private FFI prototype loaded, new parent target never called;09/source.diff and HLO |
| Capture | Nsight CUDA capture bounded to fit_zeta, with per-module XLA receipts; no full-run Nsight capture |
| Comparison limitation | P4 final profile and P16 leg ran concurrently on separate allocations; no isolated repeat or confidence interval claimed. |

| Stage 0 per-module receipt | Module | Calls | Device ms | % of fit wall |
| --- | --- | --- | --- | --- |
| 09_fixed_fit_profile | module_0032.jit__psum | 4 | 9.634 | 0.020 |
| 09_fixed_fit_profile | module_0123.jit__local | 1 | 88.331 | 0.182 |
| 09_fixed_fit_profile | module_0125.jit__local | 1 | 2.317 | 0.005 |
| 09_fixed_fit_profile | module_0127.jit__fused | 1 | 31.720 | 0.065 |
| 09_fixed_fit_profile | module_0182.jit__scanned | 1 | 389.037 | 0.802 |
| 09_fixed_fit_profile | module_0215.jit_fn | 1 | 4.072 | 0.008 |
| 09_fixed_fit_profile | module_0225.jit_fn | 9 | 2608.027 | 5.377 |
| 09_fixed_fit_profile | module_0228.jit__ordered_pair_normal_equations | 9 | 9.179 | 0.019 |
| 09_fixed_fit_profile | module_0236.jit_gather | 9 | 1.342 | 0.003 |
| 09_fixed_fit_profile | module_0262.jit__block | 9 | 24.712 | 0.051 |
| 09_fixed_fit_profile | module_0272.jit__block | 9 | 10.485 | 0.022 |
| 09_fixed_fit_profile | module_0274.jit__identity_fn | 9 | 9.898 | 0.020 |
| 09_fixed_fit_profile | module_0304.jit__kernel | 9 | 51.299 | 0.106 |
| 09_fixed_fit_profile | module_0324.jit__kernel | 10 | 3.631 | 0.007 |
| 09_fixed_fit_profile | module_0326.jit__insert_tile | 10 | 1.140 | 0.002 |
| 09_fixed_fit_profile | module_0367.jit__local | 3 | 1.871 | 0.004 |
| 09_fixed_fit_profile | module_0369.jit__local | 3 | 1.869 | 0.004 |
| 09_fixed_fit_profile | module_0379.jit__fused | 3 | 11.807 | 0.024 |
| 09_fixed_fit_profile | module_0446.jit__kernel | 15 | 2.066 | 0.004 |
| 09_fixed_fit_profile | module_0448.jit_fn | 1 | 4.090 | 0.008 |
| 09_fixed_fit_profile | module_0452.jit_fn | 1 | 4.085 | 0.008 |
| 09_fixed_fit_profile | module_0456.jit_fn | 1 | 4.091 | 0.008 |
| 09_fixed_fit_profile | module_0470.jit_fn | 7 | 1854.870 | 3.824 |
| 09_fixed_fit_profile | module_0473.jit_gather | 7 | 1.298 | 0.003 |
| 09_fixed_fit_profile | module_0485.jit__fn | 1 | 1.974 | 0.004 |
| 09_fixed_fit_profile | module_0498.jit__reshard_z | 7 | 22.188 | 0.046 |
| 09_fixed_fit_profile | module_0504.jit__solve_one_q_and_update | 147 | 57.953 | 0.119 |
| 09_fixed_fit_profile | module_0506.jit__identity_fn | 7 | 12.962 | 0.027 |
| 09_fixed_fit_profile | module_0508.jit__identity_fn | 7 | 16.782 | 0.035 |
| 09_fixed_fit_profile | module_0516.jit_gather | 21 | 1.415 | 0.003 |
| 09_fixed_fit_profile | module_0548.jit__kernel | 21 | 44.274 | 0.091 |
| 09_fixed_fit_profile | module_0562.jit__lambda | 3 | 1.056 | 0.002 |
| 10_fixed_si_profile | module_0073.jit__local | 1 | 1.904 | 0.012 |
| 10_fixed_si_profile | module_0075.jit__local | 1 | 1.841 | 0.012 |
| 10_fixed_si_profile | module_0077.jit__fused | 1 | 35.886 | 0.232 |
| 10_fixed_si_profile | module_0132.jit__fn | 1 | 73.560 | 0.475 |
| 10_fixed_si_profile | module_0198.jit_fn | 2 | 506.144 | 3.265 |
| 10_fixed_si_profile | module_0201.jit__ordered_pair_normal_equations | 2 | 6.690 | 0.043 |
| 10_fixed_si_profile | module_0221.jit__reshard_z | 2 | 7.926 | 0.051 |
| 10_fixed_si_profile | module_0223.jit__solve_all_at_once | 2 | 22.681 | 0.146 |
| 10_fixed_si_profile | module_0225.jit__identity_fn | 2 | 5.540 | 0.036 |
| 10_fixed_si_profile | module_0227.jit__identity_fn | 2 | 5.917 | 0.038 |
| 10_fixed_si_profile | module_0247.jit__kernel | 2 | 5.749 | 0.037 |
| 10_fixed_si_profile | module_0261.jit__lambda | 1 | 2.314 | 0.015 |

| Final per-call device comparison | Before unfold/spin ms | Before FFT-layout ms | Before FFT/scaling ms | Before product/channel-store ms | After native gather+FFT+product ms | After unfold write / ABI-copy ms |
| --- | --- | --- | --- | --- | --- | --- |
| CCT charge | 13.936 | 0.000 | 7.404 | 4.764 | 15.173 | 0 / 0 |
| CCT current (per channel) | 1.103 | 0.000 | 0.872 | 0.569 | 3.750 | 0 / 0 |
| ZCT charge tile | 132.451 | 21.919 | 63.674 | 29.678 | 147.832 | 0 / 0 |
| ZCT coupled-current tile | 110.544 | 19.252 | 56.567 | 29.359 | 199.852 | 0 / 0 |

| Native change | Evidence | CCT charge ms | CCT current ms/channel | ZCT charge ms/tile | ZCT coupled ms/tile |
| --- | --- | --- | --- | --- | --- |
| Initial row-major parents | 13_native_fit_profile | 14.460 | 2.026 | 145.453 | 189.600 |
| Intermediate endpoint-row-major CCT | 28_layout_mos2_profile | 13.880 | 1.967 | 146.651 | 190.962 |
| Final GEMM column-major CCT | 38_column_mos2_profile | 15.173 | 3.750 | 147.832 | 199.852 |

| Accounting / HLO check | Result | Evidence |
| --- | --- | --- |
| CCT parent feed | GEMM custom-call → bitcast → parent custom-call; zero intervening copy/transpose | 38_column_mos2_profile/parent_operand_census.txt; modules0127/0379 |
| ZCT parent feed | Parent scan tuple → parent custom-call; zero intervening copy/transpose | Same census; modules0225/0470 |
| Unfold carrier | No full-k open-spin operand write in either native tail | Native LOAD uses owner-local maps, phases, antiunitary mask and spin coefficients |
| CCT physical storage | (parent,nu,spin,mu,spin), FFI logical rank5; static centroid_major=1 | Final HLO layout{1,2,3,4,0}; ZCT default rank5 layout |
| Earlier CCT copies | Initial parent-sized transpose≈0.956ms/charge call; intermediate≈0.467ms; final0 | 13/28/38 native censuses; existing GEMM input transforms excluded |
| Uncoalesced gathers | Included in measured native kernel time; no separate read-only timing or hardware-byte counter claimed | No extra SMEM permutation stage added; measured ZCT total device module improves2608.027→1669.627ms and1854.870→1525.909ms |
| Kernel read vs FFT split | Not separately observable in this fused-kernel capture | Report combined native time; do not infer eliminated-read milliseconds |
| Bucket scope | Named, HLO-checked kernels; small loop/control/init work is outside selected before buckets | 45_final_checks/device_accounting.json retains all native kernel rows; buckets do not claim exhaustive wall attribution |

| Fit wall (cold, BFC@0.85) | Before s | After s | Measured reduction | Evidence |
| --- | --- | --- | --- | --- |
| MoS₂ P4 fit-only profile | 48.5 | 45.9 | 5.4% | 09→38 |
| MoS₂ P4 plain fresh fit | 28.1 | 23.9 | 14.9% | 04→39 |
| Si SOC GN P4 plain fresh fit | 21.2 | 17.3 | 18.4% | 11→40 |
| Legacy Si leg20 ns=2 | 9.9 | 10.1 | -2.0% | 12→41; same decomposed tail |
| True Si ns=1 | 6.8 | 6.9 | -1.5% | supplied00→42; same decomposed tail |
| MoS₂ P16 plain fresh fit | 25.2 | 22.0 | 12.7% | 43→44 |

| Compile-event gate | Before | After | Result |
| --- | --- | --- | --- |
| P4 fit-only MoS₂ rank0 | 250 | 250 | equal;09/38 |
| P16 whole arm, EVERY rank0–15 | 620 | 620 | equal;43/44 |
| Legacy Si ns=2 whole arm | 454 | 454 | equal;12/41 |
| True Si ns=1 whole arm | 405 | 405 | equal;supplied00/42 |
| ns=1/2 wall interpretation | incumbent tail | incumbent tail | +0.1/+0.2s rounded cold samples; no native scalar speedup or statistically resolved slowdown claim |

| Final same-deck identity | eqp0 max µeV | eqp1 max µeV | Tolerance µeV | Result |
| --- | --- | --- | --- | --- |
| 39_column_mos2 | 0.007 | 0.007 | 20 | PASS |
| 40_column_soc | 3.680 | 3.680 | 20 | PASS |
| 41_column_si | 0.000 | 0.000 | 0 | PASS |
| 42_column_ns1 | 0.000 | 0.000 | 0 | PASS |
| 44_p16_native | 0.009 | 0.009 | 20 | PASS |

| ζ dataset comparison | Normalized max error | Tolerance | Result |
| --- | --- | --- | --- |
| 38_column_mos2_profile_zeta_q.h5.log | 7.9887e-09 | 1e-6 | PASS |
| 38_column_mos2_profile_zeta_q_mu1.h5.log | 1.8546e-12 | 1e-6 | PASS |
| 38_column_mos2_profile_zeta_q_mu2.h5.log | 1.0277e-12 | 1e-6 | PASS |
| 38_column_mos2_profile_zeta_q_mu3.h5.log | 9.9378e-13 | 1e-6 | PASS |
| 41_column_si_zeta_q.h5.log | 0.0000e+00 | 1e-6 | PASS |

| Correctness gate | Result | Evidence |
| --- | --- | --- |
| CPU touched-module set | 431 passed,1 platform skip,1 strict expected failure | 46_committed_cpu: c2e847ad,431 passed,1 skip,1 xfail,194.62s |
| CPU collection | 433 cases; all requested paths collected | test_isdf_zq_parent_parity.py, test_centroid_k_unfold.py, test_isdf_parent_conv.py, bispinor_physics_oracles.py, services/symmetry_maps/tests |
| Platform skip | WSL-profile test on Perlmutter | test_symmetry_maps_skip_honesty.py::test_the_wsl_profile_is_the_one_this_dev_box_selects |
| Expected failure | Known emulated CPU partitioned max discards NaN | test_symmetry_maps_multiproc.py::test_a_nan_survives_the_sharded_reduction |
| Native P4 oracle | 16 cases; ns=1/2/4, both arms, both parent layouts, current vertex; max4.997391306057662e-16 | 37_column_oracle/gpu.rank0.log |
| Typed-table CPU oracle | One test loops random ns=2/4 with mixed spin and antiunitary rows; tolerance1e-12 | tests/test_isdf_parent_conv.py |

| Refused change / preserved failure | Finding | Disposition |
| --- | --- | --- |
| Native ns=1 automatic admission | Initial eqp1 delta105.735µeV; decomposed repeat matches64/64 printed rows | Not admitted automatically; final42 eqp0/1 exact |
| Native ns=2 automatic admission | Initial candidate failed printed identity; mixed rule-cache choices also contaminated earlier comparison | Not admitted automatically; exact fixed cache + decomposed final41 gives224/224 exact rows |
| Legacy leg20 labeling | Two-component WFN under bispinor=false, not ns=1 | True scalar gate supplied separately; recorded in KNOWN_SANDBOX_ERRORS.md |
| P16 first attempt | Missing staged dipole.h5; failed before science | 21 preserved; corrected final43/44 passed |
| No convention changes | F1 spin action, antiunitarity, phase signs and post-unfold vertices retained | No altered tolerance, rank cutoff or physics deck |

| Build / publication | Value |
| --- | --- |
| Private library | /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/118_bisp_zffi_codex_2026-09-06/36_native_column_build/ffi_build_phdf5/liblorrax_ffi.so |
| Library SHA256 | c4c825a9006d98ddf6a69638f10c70a099f4d6ffdd08b99080fa63861641ae1b |
| Build verification | All build gates passed; ABI3 additive entry; shared runtime unchanged |
| Integrator order1 | aa0fdb6e — square/TR fresh-fit fix; integrator already owns its gates |
| Integrator order2 | c2e847ad — native parent CUDA load, direct layouts, automatic identity guard; pushed |
| Claims | 1411 fresh-fit fix;1421 gated native kernel; report-publication receipt45_final_checks/claim_report.txt |
| P16 pool | 58006946 released after final leg;20_p16_pool/release.log |
| P4 pool | 58005266 released after committed CPU confirmation;45_final_checks/release_pool.log |
| Status | Native ns=4 implementation pushed and gated; ns=1/2 refused automatic native admission and retain exact fallback; no outstanding science gate |
