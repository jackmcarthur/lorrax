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

| All-ns admission ruling | Result / evidence |
| --- | --- |
| Runtime delta | Deleted the three-line ns=1/2 auto-off in make_fused_conv_kparent; existing conv_kpair_plan is the only shape/SMEM admission policy |
| Geometry / cache | One combined P4 step; separate cold processes, BFC@0.85, JIT cache off; paired initial rule-cache files byte-identical; reduction_steps=0 |
| Legacy deck correction | Leg20 actually specified30 passes; new variants50/51/52 use the requested0. Arm50 seeds valid0-pass rules before copying the same cache to both paired arms. Original deck retained. |
| Selected rules | ns=1 node digests identical; ns=2 selected digests differ,700→702 window/tau pairs despite identical initial caches. Its eqp delta includes that adaptive response; no forced certificate or rule-selection bypass. |
| Kernel coverage | Existing16 native oracle cases cover ns=1/2/4 and both layouts/arms at≤5e-16; no CUDA or ABI change in this admission delta |

| Deck / fresh paired arms | ζ normalized max | eqp0 max µeV | eqp1 max µeV | Cold fit before→native s | Gate |
| --- | --- | --- | --- | --- | --- |
| ns=1: 48_ns1_decomposed→49_ns1_native | 3.945964e-09 | 93.562 | 105.735 | 6.9→6.1 | PASS;≤2000µeV; native faster |
| ns=2: 51_ns2_decomposed→52_ns2_native | 1.436599e-08 | 0.611 | 1.131 | 9.9→8.3 | PASS;≤2000µeV; native faster |

| Fit conditioning check | κ(C) measured max | ε·κ(C), ε=2.220446e-16 | rcond·δζ, rcond=1e-8 | Empirical relative eqp/ζ amplification |
| --- | --- | --- | --- | --- |
| ns=1 | 4.864828e+07 | 1.080209e-08 | 3.945964e-17 | 1382.72 |
| ns=2 | 9.979828e+07 | 2.215967e-08 | 1.436599e-16 | 3.14 |

| Amplification interpretation | Conclusion |
| --- | --- |
| Retained solve bound | κ(C)≤1/rcond=1e8. First-order solve sensitivity scales as κ(C)·(relative δCCT + relative δZCT); the oracle’s≈5e-16 scale per term gives an≈1e-7 relative fit-error scale. |
| Observed fit floor | Both measured δζ are below ε·κ(C), and rcond·δζ is below fp64 ε. The conditioning scale explains the ζ movement without a kernel inconsistency. |
| Eqp bound / limit | The fit rcond alone does NOT give a rigorous bound on the subsequent nonlinear GW/MPA/eqp stages. Reported amplification is empirical: max(relative Δeqp0/1)/δζ. The measured shifts105.735/1.131µeV are compatible with amplified roundoff and satisfy the owner’s2meV rule. |
| Source of receipts | 53_all_ns_checks/summary.json; registered compare_zeta_h5.py and eqp_ab.py outputs; per-rank driver stage and condition-number receipts |

| Updated native branch reference | Output arm |
| --- | --- |
| ns=1 | /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/118_bisp_zffi_codex_2026-09-06/49_ns1_native |
| ns=2 | /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/118_bisp_zffi_codex_2026-09-06/52_ns2_native |
| Reference manifest | all_ns_branch_references.json; validated_commit.txt added to each native arm after commit; in-leg source_head/source.diff preserved |

| Scope | Value |
| --- | --- |
| Lane / branch | Heavy BISP-ZFFI; perf/bisp-zeta-ffi-2026-09-06, unmerged; no rebase |
| Kernel commit | c2e847ad; additive CufftConvKParentCudaFfi, ABI3, existing handlers unchanged |
| Worktree | /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_zffi_codex_20260906 |
| Evidence root | /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/118_bisp_zffi_codex_2026-09-06 |
| Admission | Native parent tail admitted for every ns when the existing shape/SMEM plan admits; no ns=1/2 auto-off. CPU, missing-target and refused-plan fallback unchanged. |
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
| Legacy Si leg20 ns=2 | 9.9 | 8.3 | 16.2% | 51_ns2_decomposed→52_ns2_native; matched steps=0 |
| True Si ns=1 | 6.9 | 6.1 | 11.6% | 48_ns1_decomposed→49_ns1_native; matched steps=0 |
| MoS₂ P16 plain fresh fit | 25.2 | 22.0 | 12.7% | 43→44 |

| Compile-event gate | Before | After | Result |
| --- | --- | --- | --- |
| P4 fit-only MoS₂ rank0 | 250 | 250 | equal;09/38 |
| P16 whole arm, EVERY rank0–15 | 620 | 620 | equal;43/44 |
| Legacy Si ns=2 whole arm | 454 | 454 | equal;51/52 |
| True Si ns=1 whole arm | 405 | 405 | equal;48/49 |
| ns=1/2 wall interpretation | matched cold controls | native auto | Both fits faster in one P4 sample each; no statistical confidence interval claimed |

| Final same-deck identity | eqp0 max µeV | eqp1 max µeV | Tolerance µeV | Result |
| --- | --- | --- | --- | --- |
| 39_column_mos2 | 0.007 | 0.007 | 20 | PASS |
| 40_column_soc | 3.680 | 3.680 | 20 | PASS |
| 52_ns2_native | 0.611 | 1.131 | 2000 | PASS |
| 49_ns1_native | 93.562 | 105.735 | 2000 | PASS |
| 44_p16_native | 0.009 | 0.009 | 20 | PASS |

| ζ dataset comparison | Normalized max error | Tolerance | Result |
| --- | --- | --- | --- |
| 38_column_mos2_profile_zeta_q.h5.log | 7.9887e-09 | 1e-6 | PASS |
| 38_column_mos2_profile_zeta_q_mu1.h5.log | 1.8546e-12 | 1e-6 | PASS |
| 38_column_mos2_profile_zeta_q_mu2.h5.log | 1.0277e-12 | 1e-6 | PASS |
| 38_column_mos2_profile_zeta_q_mu3.h5.log | 9.9378e-13 | 1e-6 | PASS |
| 49_ns1_native_zeta.log | 3.9460e-09 | 1e-6; also below ε·κ | PASS |
| 52_ns2_native_zeta.log | 1.4366e-08 | 1e-6; also below ε·κ | PASS |

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
| Superseded ns=1 printed-digit gate | Initial eqp1 delta105.735µeV | Owner replaces bit identity with2meV for the conditioning-sensitive fit; native automatic admission restored |
| Superseded ns=2 printed-digit gate | Legacy mixed-cache comparisons retained as history | Matched-cache steps=0 pair51/52 replaces those branch references; native automatic admission restored |
| Legacy leg20 labeling | Two-component WFN under bispinor=false, not ns=1 | True scalar gate supplied separately; recorded in KNOWN_SANDBOX_ERRORS.md |
| P16 first attempt | Missing staged dipole.h5; failed before science | 21 preserved; corrected final43/44 passed |
| No convention changes | F1 spin action, antiunitarity, phase signs and post-unfold vertices retained | No rank-cutoff or physics change; owner explicitly revises scalar eqp acceptance to2meV |

| Build / publication | Value |
| --- | --- |
| Private library | /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/118_bisp_zffi_codex_2026-09-06/36_native_column_build/ffi_build_phdf5/liblorrax_ffi.so |
| Library SHA256 | c4c825a9006d98ddf6a69638f10c70a099f4d6ffdd08b99080fa63861641ae1b |
| Build verification | All build gates passed; ABI3 additive entry; shared runtime unchanged |
| Integrator order1 | aa0fdb6e — square/TR fresh-fit fix; integrator already owns its gates |
| Integrator order2 | c2e847ad — native parent CUDA load, direct layouts, automatic identity guard; pushed |
| Integrator order3 | This separate admission commit above317d754b: delete ns-specific guard, update matched scalar references and conditioning receipts |
| Claims | 1411 fresh-fit fix;1421 native kernel;1422 accepted report; admission claim receipt53_all_ns_checks/claim.txt |
| P16 pool | 58006946 released after final leg;20_p16_pool/release.log |
| P4 pools | Earlier58005266 released; admission58008072 released after matched pairs,47_all_ns_pool/release.log; failed allocation58007950 recorded |
| Status | Native parent tail admitted for every ns; both scalar ζ/2meV gates and cold-fit performance checks pass; separate admission delta ready for integration |
