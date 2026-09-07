| Stage 0 ceiling / module | ms | % of ζ fit wall | Evidence / scope |
|---|---:|---:|---|
| MoS₂ 6×6 P fresh fit, untouched ebee1467 | unavailable | unavailable | CCT refused before completion; no valid ceiling or tail profile |
| Si supplied leg20: whole charge fit | 15900 | 100.0 | P4, BFC@0.85; rounded host stage receipt, includes compilation; actual ns=2 |
| Si CCT | 4600 | 28.9 | Host stage, not device tail alone |
| Si rank-truncated inverse | 2100 | 13.2 | zeta_fit.cholesky receipt |
| Si ψ(r) cache | 600 | 3.8 | zeta_fit.build_psi_r_cache receipt |
| Si chunk loop | 6100 | 38.4 | Includes projectors, ZCT, solve and I/O; not a tail ceiling |
| Si G-flat output | 1200 | 7.5 | zeta_fit.write_g_flat receipt |
| Si remaining / rounded-stage residual | 1300 | 8.2 | Difference from rounded whole-fit wall; no device attribution |

| Lane | Value |
|---|---|
| Weight | Heavy |
| Branch | perf/bisp-zeta-ffi-2026-09-06, unmerged; no rebase |
| Baseline source | ebee146701b835de1a1e6aa6cc64d2aa694cf7a0; both in-leg source_head.txt match, source.diff empty |
| Worktree | /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_zffi_codex_20260906 |
| Evidence directory | /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/118_bisp_zffi_codex_2026-09-06/ |
| Pool | Dedicated 58004733, lx-alloc-jackm-BISP-zffi; premium QOS, P4 geometry -N 1 -G 4 -n 4 |
| Runner provenance | Copied supplied MoS₂ profiler; removed placement_patch/zeros_patch, warm repeats and unrelated stage wrappers; retained compile receipts, CUDA capture and fit-only stop |
| Library | Shared runtime library read-only; no build or ABI modification |

| Leg | Step / result | Artifacts relative to evidence directory |
|---|---|---|
| MoS₂ 6×6 P before | 58004733.0 / lx-Xg4-191104-211750-2358; exception, launcher exit143 after peer failure | 01_mos2_baseline/driver.rank1.log; 00_pool/baseline.log |
| Supplied Si leg20 before | 58004733.1 / lx-Xg4-191225-217159-1465; exit0, four fit_complete receipts | 02_si_baseline/driver.rank0.log; nsys_rank0.nsys-rep; stats_nvtx_gpu_proj_sum.csv; xla_dump_rank0/ |
| Si compile receipt | rank0 xla_compiles=100, 13.37 s; not a P16 count | 02_si_baseline/driver.rank0.log:592 |
| Actual ns=1 | Not run: supplied Si deck is ns=2, cache shape (5,8,16,2,13824) | 02_si_baseline/driver.rank0.log:465 |
| Si SOC GN / eqp identity | Not run; stopped at baseline blocker | No eqp deltas claimed |
| P16 MoS₂ before/after and compile count | Not run; no second pool requested | No P16 claim |

| Blocker / source finding | Evidence | Required resolution |
|---|---|---|
| Untouched CCT passes no transpose partner | src/isdf/core.py:1333 calls plan.unfold_operator(_projector(w_l)); src/gw/centroid_k_unfold.py:185 forwards operator_transpose=None | Source-owner correction or a designated valid baseline; outside this lane's frozen-tip performance comparison |
| Explicit right tables trigger rectangular admission despite square extents | services/symmetry_maps/src/symmetry_maps/maps.py:612 defines same_basis_tables by omitted right tables; :1451 explicitly supplies right tables with pair_transpose; :828 refuses on active TR rows with no partner | Resolve existing typed transport contract; do not remove TR rows or bypass refusal |
| Exact observed exception | ValueError: unfold_isdf_operator: rectangular pair_transpose requires trs_pair_q_ibz with reversed endpoint axes | Registered in sandbox KNOWN_LORRAX_ISSUES.md |
| Supplied CCT einsum description differs from frozen source | CCT uses gemm + plan.unfold_operator; kamn,knbr->kambr belongs to ZCT projectors | Future feed must target actual CCT unfold output |
| ns=1 gate input missing | Supplied non-bispinor Si leg is two-component | Provide a real one-component WFN/deck; registered in KNOWN_SANDBOX_ERRORS.md |

| Requested performance / landing item | Disposition |
|---|---|
| Per-change unfold/copy/FFT/fused-conv ms per tile | No changes; MoS₂ baseline failed before usable tail capture |
| HLO no-copy census between unfold fusion and custom call | Not applicable: no new custom-call feed implemented |
| ζ normalized error and eqp0/eqp1 deltas | Not measured; no changed arm |
| Stage 1 | Not implemented or landed; no runtime code changed |
| Stage 2 ≥15% decision | Undetermined; unavailable MoS₂ ceiling is not evidence below threshold |
| Final oracle suites | Not run; no source changes, prerequisite baseline failed |
| Integrator commit list | Documentation-only blocker report; publication receipt in 00_pool/publication.txt |
| Claims | One blocker-evidence row for the documentation batch, explicitly on branch perf/bisp-zeta-ffi-2026-09-06, unmerged |
| Final status | BLOCKED: untouched ebee1467 MoS₂ CCT transport refuses before the Stage 0 baseline; actual ns=1 gate deck also required. |
