# BISP-PROF-S local scaffolding register

- 2026-09-06: prescribed once-only allocation failed QOSMaxSubmitJobPerUserLimit; fallback57955075 is invalid. No authorized pool, no compute measurement. Evidence: runs/Si/100_bisp_parent_route_2026-09-05/prof_s/00_allocation/alloc.log and fallback.log.
- Supplied historical GN checkpoint arms initially lacked quadrature caches and rebuilt in memory. New prepared P/F arms copy the same completed P tmp; actual selected-node/digest equality remains to check. Evidence: reports/bisp_prof_s_2026-09-06/checkpoint_receipts.txt.
- Shared ledger/report paths are outside the explicit write scope; lane-local registers/report are used. No shared sandbox file modified.

- 2026-09-06: tools/parse_lorrax_sigma_run.py rejects completed COHSEX gwjax.out because _parse_production_report requires absent Sigma rule plan/tau sweep rows. Static evidence retained as the production table; no invented zero-time parser output. Run prof_s/05_P_full_static_baseline, step lx-Xg4-011831-1203588-3371.

- 2026-09-06: first common-rule preparation incorrectly required optional Si dipole.h5; baseline has no such file. Step lx-Xg0-015048-1445882-7419 exit1 before science. Preserve partial Si11 directory; new prepared profiles use Si13/14 and copy optional dipole only when present.

- 2026-09-06: ablation13 imported w_isdf before gw_jax initialized distributed runtime and failed at startup (driver.rank0.log). Preserve failed arm; ablation19 imports the canonical driver startup first. Apply the same ordering to unrun18 and common-rule profile preparation.

- 2026-09-06: the sandbox HLO analyzer, as consumed by PERF2 profile_collective_census.py, omits collective-permute-start. M20 module4848 is reported with an empty static census despite optimized HLO lines701/744 containing two starts, corroborated by two native SendRecv kernels. Preserve the canonical output and supplement it with an explicit async-start census; never infer communication-free from that empty result. Evidence: JID57966610, M20 driver.1.log and xla_dump_rank0/module_4848.jit__tau.sm_8.0_gpu_after_optimizations.txt. This is a parser defect, not evidence that the kernels lack communication.

- 2026-09-06: lx test -G0 -n1 --dry-run overrides the requested CPU geometry to G4/n1 and JAX_PLATFORMS=cuda,cpu. It launches nothing, but conflicts with the lane geometry. Saved receipt: prof_s/40_cpu_gates/lx_test_dry_run.log. Use the mandated lx run CPU recipe; do not disable evidence checks.
- 2026-09-06: CPU gate40 preparation raced script creation: mkdir refused an already-created directory, so rankwrap.sh was never copied. Step lx-Xg0-031725-2232478-2895 exited127 before pytest. Preserve40; prepare42 completely before launch. This is a lane scaffold failure, not a source-test failure.

BISP-PROF-S final sequence initially omitted the copied runner required JID positional argument; it refused before launching any step. Corrected sequence passes authorized57966610 explicitly; no completed run changed. Receipt: prof_s/00_tools/final_sequence.log.

BISP-PROF-S final gate0 did not pass: CPU step lx-Xg0-035328-91325-4986 (JID57966610) exit1. Artifact prof_s/41_ast_gate/gate0.log reports source rule/allowlist findings and malformed shared CLAIMS rows [254,341,349,384,514,519,523,625,672,694,720,823]. Its test-evidence check accepted job57909062 despite this lane using explicit lx run gates. Findings are being separated into pre-existing versus changed-file scope; no shared allowlist/ledger or evidence guard is modified.

BISP-PROF-S candidate manifests inherit the donor source.commit and sometimes its instrument label; candidate_source points to authoritative source_head.txt/source.diff, but the inherited field is misleading when read alone. Completed science is not rewritten. reports/bisp_prof_s_2026-09-06/final_run_audit.json records the actual runtime source snapshot and launcher for each leg; driver.sh and source.diff are authoritative. This affects the lane-owned copy helper, not the production code.

### 2026-09-06 BISP-PROF-S shape scan: allocation discovery timed out

The authorized named fallback `LX_ALLOC_MATCH=lx-alloc-jackm-BISP-PROF-S LX_ALLOC_NAME=lx-alloc-jackm-BISP-PROF-S lx alloc -N 1 --time 04:00:00` exited3 before allocation: Slurm `squeue --me ...` timed out three times at15s. This is not a QOS refusal. Evidence: `runs/MoS2/42_bisp_scale_2026-09-06/prof_s/allocation_retry_01.log`; the existing P4 scan leg remains queued on authorized57982945. Retrying the named allocation only; another lane's pool is not used.

### 2026-09-06 BISP-PROF-S: native gate wrapper permission (corrected in a new variant)

`prof_s/10_P_scan_oracles` copied a non-executable `driver.sh` from the older gate but invoked it directly through rankwrap. JID57982945 step `lx-Xg4-120420-417822-9746` exited126 before Python ran. Evidence: `runs/MoS2/42_bisp_scale_2026-09-06/prof_s/10_P_scan_oracles/driver.1.log`. New variant12 invokes the payload explicitly with bash; failed10 is preserved.

2026-09-06 phase3: GATES.md, INVARIANTS.md and TASTE.md are sandbox-owned; reading them at the LORRAX worktree root fails. Resolved from the known sandbox root. Common-basis arms104–107 likewise live under the sandbox runs root, not this worktree. No shared files changed.

2026-09-06 phase3 F37 diagnostic on JID57988457, step lx-Xg4-140021-988279-2173: ranks1/2 failed in fixed-main screening.compute_static_w:400 with RESOURCE_EXHAUSTED allocating2.42GiB; peer SIGKILL produced137. This is an incomplete diagnostic, not a timing verdict or proven source regression. Same BFC0.85 as common-basis107; preparing a fresh copy without HLO dumping to isolate diagnostic overhead. Evidence runs/MoS2/42_bisp_scale_2026-09-06/prof_s/37_F3_dynamic_receipts/driver.rank1.log.

- 2026-09-06 BISP-PROF-S: attempted ZW backlog path DEV/114_bisp_prof_zw_codex_2026-09-06 did not exist; bounded DEV listing resolves the owner to DEV/112_bisp_prof_zw_codex_2026-09-06 (read-only).

- 2026-09-06 BISP-PROF-S: phase3 seam command used nonexistent sandbox suffix2026-09-06; command never started; corrected to assigned2026-08-14 root.

- 2026-09-06 BISP-PROF-S: parse_lorrax_sigma_run.py is dynamic-only for production tables; static50 has no rule/tau rows and refuses parsing. Use existing stage parser for static; no static parse verdict was recorded.

- 2026-09-06 BISP-PROF-S: final compile-class writer initially assumed absent module classes had a count field; comparison receipts represent absence as an empty object. Corrected to count0; compile_class_dispositions.json now exists and is committed. No numerical result changed.

- 2026-09-06 PROF-S: INVARIANTS.md is sandbox-owned, absent from source worktree; read sandbox root file. Rebase onto b1d8b8f1 requires restoring lane evidence omitted by orchestrator cherry-picks.

- PROF-S run60: instrumentation preparation used a worktree-relative path while cwd was the run directory and failed; detached gate launched without HLO/warm wrapper. Receipts and identity valid, isolated warm time unavailable.

- 2026-09-06 PROF-S: `lx status --jid 57988457` is rejected by lx_pool; --jid is a run flag, not a status selector. No allocation or step affected.

- 2026-09-06 PROF-S run63 P16 fixed-main: timing wrapper treated present-but-None green_parent as a carrier, causing AttributeError before the first static block. Scientific source unchanged; retry in fresh67 uses green_parent or the full-k bundle.

- PROF-S parent regression: bisect7c5466be has lx_run1.log and rankwrap.sh but no driver.sh; copy via lane runner. One read used an invalid workdir spelling and was rejected without launching.

- Parent-Green gate28: copied Si56 rule cache missed four exact containment edges shifted by at most7.714e-13 Ry after floating-point reassociation; regenerated29-node schedule. The0.518 ueV difference from Si56 also occurs on unchanged c2f69987 (control32); it was not caused by the new Green algorithm or cured by rule pinning. Gate retry freezes the copied certificates through a run-local lookup with an explicit1e-10 Ry edge bound; production cache unchanged.

- Parent-Green controls: archived c2 control31 omitted root pyproject.toml and source closure refused; MoS2 gate68 overwrote its supplied per-window replay hook and consequently found no ordinary tmp rule cache. Fresh controls preserve the original replay hook and root project metadata.
