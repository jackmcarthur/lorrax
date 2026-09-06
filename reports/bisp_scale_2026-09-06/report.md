**Timing correction:** Slurm accounting proves overlapping full-GPU launches. The affected tables below are historical diagnostics, not valid scaling walls. See `runs/DEV/116_bisp_scale_codex_2026-09-06/gpu_overlap_audit.json`. Final audited intervals are being collected.

Lane weight: heavy. Branch `perf/bisp-scale-2026-09-06`, unmerged. Measurement in progress; no 6x6 GW speedup or identity claim yet.

| Unit | Charge PERF2 P/F (A/C) | Charge Na P16 claim 842 | Bispinor 6x6 P4 | Bispinor 6x6 P16 | Mapping / accounting scope |
|---|---:|---:|---|---|---|
| ζ tile | 87.608/169.771 = 0.516 device | fit loop 11/13 = 0.846 | pending | pending | Parent projector GEMMs; full-k FFT/tail remains. Unequal tile point counts in PERF2. |
| χ0 build | stage 4.60/6.02 = 0.764 | screening 176/1211 = 0.145 (includes W) | pending | pending | Typed child faces, parent projection; 16 ordered Lorentz blocks. |
| W restore | 4 A2A in both | not isolated | pending | pending | Not a removed charge-route collective. |
| τ node | 24.182/69.807 = 0.346 device | 175/2440 = 0.0717 host | pending | pending | Green/projection reduced; full-k convolution retained. |
| static block | no like-for-like ratio | not isolated | pending | pending | Static Green remains full-k in both routes. |
| projection | (3.783+0.816)/(29.505+7.336) = 0.125 | not isolated | pending | pending | Parent versus full-k band projection. |
| band unfold | not isolated | not isolated | pending | pending | Include antiunitary transpose collective permutes. |
| seam conversions | 139 eager → 10 compiled modules; same 38 A2A | not isolated | pending | pending | Compilation saving is distinct from communication count. |
| charge restores | retired bridges enumerated in PERF2 | not isolated | pending | pending | No historical G restore removed twice. |
| Lorentz restores/mixing | n/a | n/a | pending | pending | Bispinor-only: 100 source contributions/term, 300/static run. |
| family faces/current channels | n/a | n/a | pending | pending | C/T endpoint classes; coupled three-current ζ. |
| packed Dyson | n/a | n/a | pending | pending | Bispinor-only coupled (M_C+3M_T) solve. |
| Γ completion | n/a | n/a | pending | pending | Integration tip changes incumbent physics; exact P/F gate must be checked. |
| head attribution | n/a | n/a | P disabled; F still active | same source behavior | Fixed photon_sigma.py:380-423 ignores the false diagnostic flag when q0 factors exist; price this extra work. |

Charge references are harvested evidence, not this lane's measurements: sandbox `reports/psi_irr_perf2_2026-09-05/report.md`, JID57941637, and CLAIMS row842 (its separate claims/0842.md is missing). PERF2 used cuda_async@0.85; this lane uses BFC@0.85. Charge cross-source comparisons were not printed-digit identical. These reference ratios cannot establish bispinor identity or absolute timing across allocators.

Source provenance: started at S tip4a691b67, cherry-picked ZW0f8fbc3e/f811f734 as cee4011d/3046b911, resolving only the absent ZW report by retaining it. Preserved that state as local branch perf/bisp-scale-initial-2026-09-06. Fetched ORCH and switched measurement source to af85d4745fd1bebdde7b2f34421409ccdc6d0000: b455f781 integrates Σ executable/plan ownership and states the ZW ports are present; the tip additionally includes covariant Γ completion and the paired-Gram ridge-sign fix. Fixed-main e1559a07 remains read-only. Its absence of these physics fixes is a potential identity-gate blocker, not permission to waive identity.

QE evidence: `runs/MoS2/42_bisp_scale_2026-09-06/00_qe_6x6/`, pool57982945. NSCF reports12 spatial operations and7 stored k points. Both SCF/NSCF use6x6x1 and82 bands. wfn2hdf step lx-Xg0-105033-2302113-3374 exits0. Claim1214. Preprocessing requests600 charge/200 current orbit centroids. The inherited GW window remains80 bands with82 QE bands to witness the upper multiplet boundary.

Experiments (measured below): (a) restore16 source blocks once per term, retain only if exact and faster, with memory refusal tied to existing memory_per_device_gb; (b) enumerate actual module compile excess and share only shape-identical programs, with exact source-change gates. For complex128, heterogeneous resident sources occupy16*K*(M_C+3*M_T)^2/P bytes/rank, bounded by256*K*max(M_C,M_T)^2/P; this is carrier storage, not an allocator peak. At K36, M_C600, M_T200 the nominal resident carrier is829,440,000 bytes/P (207,360,000 at P4;51,840,000 at P16). It scales linearly in K, quadratically in centroids, independently of nb/ns, and cannot be treated as a fixed-size optimization.

Outstanding: GW source arms/fresh ζ, identity, P4/P16 stage/unit/compile tables, native/HLO census, both experiments, final verdict. No unrun cell is treated as zero or inherited speedup.

## P4 experiments measured (claim 1231)

All at BFC@0.85, same parent source and copied ζ. Combined step57982945/lx-Xg4-110752-42766-3793 exits0. EQP0/1 and all210 sector rows are exact in both experiments (`runs/DEV/116_bisp_scale_codex_2026-09-06/{compile_zeros_identity,restore_repeat_identity}.json`).

| arm | Sigma s | total s | compile events |
|---|---:|---:|---:|
| baseline05 |20.11|66.15|674|
| resident14 |19.69|65.22|674|
| baseline17 |20.08|65.24|674|
| resident18 |19.84|65.71|674|
| direct sharded zeros19 |17.99|62.39|610|

Reject resident sources for landing: only0.24–0.42s Sigma benefit, while total-time order reverses on repeat. The retained full-q carrier is203,233,536bytes/rank atP4 (K36, packed C600/T196); atP16 the same geometry would nominally require50,808,384bytes/rank. It is O(K*(M_C+3M_T)^2/P), independent ofnb/ns/n_parent once restored, in addition to the parent operator storage O(n_parent*(M_C+3M_T)^2/P). The prototype refuses above5% of the existing30GB/device deck budget. No production memory-lifetime change is retained.

Compile candidate: existing `services/distrib_la/src/distrib_la/matmul.py::_zeros` creates a new `jax.jit(lambda: zeros(...))` for every GEMM warmup. Profile12 attributes100 lambda modules to that owner in the fresh run; its source-preserving sharded-zero prototype eliminates64 compile events in the copied-ζ run. No new cache, decorator, public interface, env/deck key or retained arrays is needed. Its64-event saving does NOT establish the requested parity:610 remains32 above fixed-main578 on the copied static arm. Production adoption and broader gates remain pending; the run-local candidate is preserved at19/compile_experiment.py.

## Initial P4 source comparisons (failed exact P/F gate)

| scope / arms | P total s | F total s | P screening / Sigma s | F screening / Sigma s |
|---|---:|---:|---:|---:|
| fresh ζ 02/04 |99.35|88.71|25.20 /20.36|17.48 /20.81|
| identical copied ζ 05/06 |66.15|80.90|25.96 /20.11|18.82 /39.42|

The F Sigma wall changes substantially between fresh and copied arms, so one copied pair cannot establish a stable whole-run gain. Parent ζ stage15.68s plus transverse17.38s; F reports charge13.76s and includes its transverse fit inside23.84s ISDF setup/I/O, requiring timed-stage reconciliation rather than subtracting totals. Both P/F gates FAIL: fresh max4.470µeV; copied max4.467µeV, sectors4µeV (claim1220). Fixed-main remains an incumbent physics control, never a passing exact control.

Full parent profile12 completes267s under Nsight (not a baseline wall), JID57982945/lx-Xg4-105913-2356430-4328. Rank0 HLO/Nsight and PERF2 census are on disk. Fixed full profile13 fails at cuSOLVERMp external workspace cudaMallocAsync OOM; no timing claim derives from its missing completion. Scope27 static restart is refused by the incumbent config gate. Scope28 profiles copied zeta with restart=false and completes; exact against06. Initial dynamic copied cache selects different nodes (116 P versus85 F); the matched arms use PERF2's guarded single-rule lookup, leaving containment/error/noise guards unchanged.

## Common-rule dynamic P4 (claim1255) and P16 compatibility

New union-box certificates preserve all error/containment/noise guards; the eight schedules are identical (115 nodes). Runs25/26: total112.83/102.73s P/F, tau7.47/9.64s, screen28.50/23.96s. EQP0/1 max4.468/3.374µeV and sectors4µeV: exact FAIL. Profiles29/30 are exact against their own baseline. The overlap audit below invalidates their whole-wall/native timing comparison; trace-mode causation is NOT established.

P16 parent09 finishes94s step wall (lx-Xg4-112718-6609-6855),646 compile events/34.26s rank0; parent dynamic23 finishes133s (lx-Xg4-112855-174741-8396),892 events/52.91s. Fixed10/24 refuse logical transverse194 at P16 in core.py:3407. A common whole-orbit logical200 set is being built for a compatible P4/P16 matrix; old and new centroid measurements will remain separate.

Hourly fetch found integration71ae0bde (after af85d474), which removes both antiunitary collective permutes from each parent tau and defers band unfolding until after the frequency sum. Existing captures retain their af85d474 provenance; new-tip measurements will be separately labeled.

## Audited common-basis P4 matrix (claim1294)

Step57982945/lx-Xg4-123003-543237-1984 exits0 in655s. Slurm accounting shows no GPU overlap. Code71ae0bde, fixed e1559a07, common logicalC597/T200; 115 identical certified quadrature nodes. Artifacts: `runs/DEV/116_bisp_scale_codex_2026-09-06/final_P4.json`, `identity_104_105.json` through `identity_106_111.json`; run master103.

| Arm | Total s | Sigma other s | Tau sweep s | Compiles | Compiler s |
|---|---:|---:|---:|---:|---:|
|104 P static baseline|67.11|20.15|—|674|38.06|
|105 F static baseline|57.13|20.61|—|578|30.31|
|106 P dynamic baseline|96.51|31.34|7.29|894|57.49|
|107 F dynamic baseline|86.87|29.94|8.93|772|45.21|
|108 P static direct zeros|64.30|18.09|—|610|35.95|
|109 P dynamic direct zeros|92.84|28.56|7.15|798|54.28|
|110 P static plus host placement|62.35|17.98|—|589|35.64|
|111 P dynamic plus host placement|92.20|28.45|6.77|777|53.93|

All four prototype arms preserve both printed EQP files and every sector row exactly against their P baseline. P/F remains FAIL; no equivalent-science speedup is certified. Prototype compilation remains11 static/5 dynamic modules above F. Cold end-to-end P is slower even after these changes.

Dedicated four-node allocation57988457 succeeded on fallback attempt5 after four QOS refusals (`runs/DEV/116_bisp_scale_codex_2026-09-06/06_pool_fallback/allocation5.log`). Shared-pool queued master96 was cancelled before science. Dedicated master119 failed before science because its copied runner omitted preflight.py; corrected master120 preserves that attempt and is running. Future legs are pinned to our dedicated pool. Hourly fetch found ORCH443e95be; its additional head-factor placement changes are held separate until this71ae0bde matrix completes.
