| Unit | Charge PERF2 P/F; Na P16 claim842 | Bispinor 6×6 P4 P/F | Bispinor 6×6 P16 P/F | Arithmetic / logical bytes per rank | Collectives and compilation; mapping verdict |
|---|---|---|---|---|---|
| ζ charge tile | 0.516 native; 0.846 fit-loop wall | 1.616 observed time/point, unequal tiles | Native capture exists; timing excluded for overlap | Parent projectors leading 16 Q ns² M R nb/P real FLOPs; full-K transform tails remain; output16 K M R/P bytes | P4 P/F: five AG and five A2A/tile each; AR5/80. Not the same workload or tile width; no matched-tile speedup claim. |
| ζ current tile | No separate current unit | Coupled P versus three F channels: 0.967 observed time/point | Both routes coupled at P16; timing excluded | Same projector law per channel; P couples three channels, F P4 repeats the full-face tail three times | P4 coupled P5 AR versus F3×80 AR; per-tile counts differ from full-fit counts. Bispinor-specific sharing, not a scalar one-to-one row. |
| χ0 build | 0.764 stage; 0.145 screening including W | CC warm432.659/426.835 =1.014 | CC warm2243.668/2444.546 =0.918 | 22 distributed GEMMs/block at the 11-node static screening rule; full-K work, Θ(K nb ns² M_A M_B/P) per quadrature unit | P4 3168/P16 6336 broadcasts per block on BOTH routes. No Q/K reduction here; inherent full-child summation. |
| W restore | Four A2A on both charge routes; Na not isolated | Not separately isolated from restore/FFT | Same limitation | Full-K operator16 K M_A M_B/P bytes; transform Θ(K log K M_A M_B/P) | Do not count the charge route's four A2A as removed. Native restore modules below have no NCCL; transform/layout collectives belong to their surrounding modules. |
| τ node | 0.346 native; Na0.0717 host | 32.502/55.089 =0.590 native;33.454/56.034 host | 67.816/228.036 =0.297 native | P has TWO G GEMMs:16 Q d_A d_B nb/P; F8 K d_A d_B nb/P. Add projection and full-K convolution | P4 Broadcast112/432; P16 224/864. Integrated parent has four distributed GEMMs, F three. NOT the historic charge operation count. |
| Static block | No like-for-like charge ratio | CC0.267, CT0.249, TC0.258, TT0.257 native | CC0.263, CT0.209, TC0.310, TT0.282 | G8 K d_A d_B nb/P in BOTH; projection uses q=Q (parent), K (full) | P4 Broadcast200/720; P16 400/1440. F includes two extra head-diagnostic projections; ratios are not pure parent-route gains. |
| Band projection | 0.125 native; Na not isolated | (5.565+0.935)/(28.839+4.201)=0.197 | Q/K arithmetic class; constituent native thunks retained in trace, no isolated ratio asserted | 8 q (b d_A d_B+b² d_B)/P; q=Q or K; b30/P4,32/P16 | Two SUMMA calls:4 sqrt(P) q broadcasts; maps to charge Q/K reduction. |
| Antiunitary band unfold | Not isolated | Zero per-τ CP after integration71 | Zero per-τ CP | Final band tensor16 K b²/P; old per-node band exchange129600 bytes at P4 | Old P4: two CP/node, G161280000 bytes/exchanging rank plus band129600. New: second G GEMM and delayed final band unfold. This is a trade-off, not free deletion. |
| Seam conversions / operator spin transport | 139 eager→10 compiled modules; same38 A2A | Full-K convolution ratio0.997; parent spin/seam region14.360ms/node, not isolated rotations | Full-K tails persist | Two local spin rotations16 K ns³ M_A M_B/P FLOPs; full G16 K ns² M_A M_B/P bytes | Local rotations have no NCCL; ns=1 scalar cost differs. P4 CC rotations3.318 GFLOP/rank. Conversion compile events are priced in the module census. |
| Ordinary charge restores | Retired bridges enumerated by PERF2 | Parent typed restores remain; no second removal credited | Same | Θ(K M²/P) output traffic independent of nb | Native P4 small band restore ~0.05ms/call, no NCCL. Restore code is not the antiunitary G exchange. |
| Lorentz restores + mixing | Bispinor-only | 300 source restores/static run; TT ~0.012ms native/restore; host output-restoration aggregate median21.4ms | Same source multiplicity, no isolated timing asserted | Three terms ×100 contributions: output traffic3×16 K(M_C+9M_T)²/P bytes; mixing Θ(K M²/P) | CC/CT/TC/TT counts3/27/27/243, no NCCL inside restore kernels. Launch floor stays300; increasing K alone does not remove asymptotic cost. |
| Per-family faces / current channels | Bispinor-only | Four shape families; 16 ordered Lorentz blocks | P16 has greater carrier padding | Face storage Θ(Q nb ns M/sqrt(P)); current factor3, ordered vertex factor16; all typed actions from SymMaps | Shape families cause distinct modules; current coupling cuts repeat work. No new shape-identical kernel cache claimed beyond measured compile experiment. |
| Packed Dyson | Bispinor-only | Warm LU156.046/797.148=0.196; complete host167.376/820.453=0.204 | Warm LU513.159/2567.270=0.200 | L=M_C+3M_T; A+LU+2TRSM leading(56/3)qL³/P FLOPs; stored matrix16qL²/P bytes | P16 warm LU P/F: AR35/180, Broadcast259/1332, AG140/720, SendRecv144/738. Pivot traffic is data-dependent. Q/K reduction with padding. |
| Γ completion | Bispinor-only covariance correction | ~0.02ms/call,10 calls, zero NCCL | Not independently timed | Rank4 ×12 typed operations; Θ(n_sym rank L²/P), independent of nb, one-off | One P-only compiled module. Fixed-main lacks this physics fix; cannot erase it to force parity. |
| Head attribution | Diagnostic, not a charge gain | Requested false; P off, F erroneously active | Same source discrepancy | Two extra F band-projection GEMMs/static block | F photon_sigma.py:380–423 ignores false when q0 factors exist. Counted in actual static native traffic; registered source defect. |

| Acceptance class | Comparison | Measured residual | Verdict | Evidence |
|---|---|---:|---|---|
| Own ζ: µeV-class | Initial fresh6×6 P02/F04 | max4.470 µeV | PASS accepted µeV class | claim1220; D/static_fresh_identity.json; P02/F04 logs |
| Copied ζ: exact | 6×6 P104/F105 static | max0.895 µeV | FAIL exact; these arms copy ζ, despite the resume note calling them own-fit | claim1294; D/identity_104_105.json |
| Copied ζ: exact | 6×6 P106/F107 dynamic | EQP0 max0.894; EQP1 max0.675 µeV | FAIL exact | claim1294; D/identity_106_107.json |
| Copied ζ: exact | Parent code changes130/131/132/134/135 vs parent baselines | 0 printed-digit difference | PASS | claim1331 |

| Receipt scope | Value |
|---|---|
| R | runs/MoS2/42_bisp_scale_2026-09-06 |
| D | runs/DEV/116_bisp_scale_codex_2026-09-06 |
| Unit-source pins | P71ae0bde / F e1559a07; scalar/new9×9 P fe2a6937, main d3d4b03a |
| Native P4 | 57982945/lx-Xg4-130807-715734-2513 exit0; R72–75; clean interval |
| Native P16 | 57988457/lx-Xg4-132155-757259-3784 exit0; R66–69; clean interval |
| Geometry | K36,Q7,nb80,ns4; P4 P C600/T204 F C600/T200; P16 P C608/T224 F C608/T208 |
| Byte/FLOP scope | complex128; leading analytic real FLOPs and logical application bytes, not vendor wire-byte measurements; evidence/analytical_operand_counts.json |
| Charge reference | PERF2 report job57941637; Na claim842, separate claims/0842.md absent |

| Arm | ζ s | Screening support s | χ0 s | W s | Sigma other s | τ s | Total s | Compiles / seconds | Class | Step |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
|104_P_static_P4_baseline|reused|26.59|—|—|20.15|—|67.11|674 / 38.06|copied ζ: exact (P/F FAIL)|57982945/lx-Xg4-123003-543237-1984|
|105_F_static_P4_baseline|reused|17.18|—|—|20.61|—|57.13|578 / 30.31|copied ζ: exact (P/F FAIL)|57982945/lx-Xg4-123003-543237-1984|
|106_P_dynamic_P4_baseline|reused|19.3|4.22|5.71|31.34|7.29|96.51|894 / 57.49|copied ζ: exact (P/F FAIL)|57982945/lx-Xg4-123003-543237-1984|
|107_F_dynamic_P4_baseline|reused|14.4|2.97|5.67|29.94|8.93|86.87|772 / 45.21|copied ζ: exact (P/F FAIL)|57982945/lx-Xg4-123003-543237-1984|

| Arm | ζ s | Screening support s | χ0 s | W s | Sigma other s | τ s | Total s | Compiles / seconds | Class | Step |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
|90_P_static_P16_baseline|reused|48.18|—|—|22.57|—|93.61|646 / 33.77|copied ζ: exact (P/F FAIL)|57988457/lx-Xg4-124919-641781-9910|
|91_F_static_P16_baseline|reused|38.71|—|—|32.67|—|94.37|604 / 28.78|copied ζ: exact (P/F FAIL)|57988457/lx-Xg4-124919-641781-9910|
|92_P_dynamic_P16_baseline|reused|38.89|6.47|6.99|33.87|11.63|127.64|897 / 52.34|copied ζ: exact (P/F FAIL)|57988457/lx-Xg4-124919-641781-9910|
|93_F_dynamic_P16_baseline|reused|33.64|7.39|6.95|42.11|27.35|142.84|825 / 43.34|copied ζ: exact (P/F FAIL)|57988457/lx-Xg4-124919-641781-9910|

| Scalar non-regression | P source | F source | Class | State | Run |
|---|---|---|---|---|---|
| Si leg20 P4 | fe2a6937 | d3d4b03a | copied ζ: exact PASS; own ζ: exact PASS | measured copied pair | runs/Si/100_bisp_scale_scalar_2026-09-06/03_pair |
| Na8×8×8 P16 | fe2a6937 | d3d4b03a | own-fit exact comparison requested | P complete; F finishing | runs/Na/16_bisp_scale_scalar_2026-09-06/03_pair |

| 9×9 arm | P/F total s | Screening P/F s | Sigma P/F s | Compiles P/F | Class | Step / path |
|---|---|---|---|---|---|---|
| Prior static P4 |70.10 /69.03|29.14 /21.89|21.40 /23.60|641 /606|copied ζ: exact FAIL0.766µeV|57988457/lx-Xg4-131617-747058-1237; R124/125|
| Current static/dynamic P4 |initial matrix complete; matched dynamic queued|—|—|—|copied ζ: exact failures; unmatched dynamic rules|R204_speed_9x9_P4/05_matrix; matched R208_speed_9x9_P4_matched|
| Current static/dynamic P16 |queued|—|—|—|copied ζ: exact|R216_speed_9x9_P16/05_matrix|

| Unit compile events (captured families) | P4 P/F | P16 P/F | Module / scope | Mapping |
|---|---:|---:|---|---|
| χ ordered vertex families |4/4|4/4|jit_minimax_tau_integrate_chi_vertex|inherent full-child sum; equal compile families|
| Static contraction families |4/4|4/4|jit_contract_block, static capture|mapped family sharing|
| τ incl. projection/seams |1/1|1/1|jit__tau, dynamic capture|one executable each; extra parent G is inside it|
| Packed Dyson A assembly |1/2|1/2|jit__a_chunk, static capture|different q-tail shapes|
| Packed Dyson solve |1/1|1/1|jit__solve, static capture|mapped Q/K solve class|
| Typed restores |7/4|7/4|jit__do_unfold, static capture; dynamic8/5|inherent extra parent/operator families|
| Separate restore mixing |0/1|0/1|jit__do_mix|parent mixing not a separate executable|
| Γ completion |1/0|1/0|jit__photon_q0_factor_orbit|inherent additional covariance work|
| Head attribution |inside block|inside block|F extra GEMMs inside four contraction modules|defect in F diagnostic flag handling|
| ζ fit-only total |see per-module receipts|221/296|137/138 instrumented captures; timings excluded|tile/route-dependent, not whole-driver parity|

| CC τ SUMMA unit | P | Route | Calls/unit | A-panel bytes/call | B-panel bytes/call | Summed logical payload bytes/rank | Instances/τ |
|---|---:|---|---:|---:|---:|---:|---:|
|G (each)|4|P|28|768000|768000|21504000|2|
|projection1|4|P|28|23040000|288000|326592000|1|
|projection2|4|P|28|288000|288000|8064000|1|
|G (each)|4|F|144|768000|768000|110592000|1|
|projection1|4|F|144|23040000|288000|1679616000|1|
|projection2|4|F|144|288000|288000|41472000|1|
|G (each)|16|P|56|194560|194560|10895360|2|
|projection1|16|P|56|5914624|77824|167788544|1|
|projection2|16|P|56|77824|77824|4358144|1|
|G (each)|16|F|288|194560|194560|56033280|1|
|projection1|16|F|288|5914624|77824|862912512|1|
|projection2|16|F|288|77824|77824|22413312|1|

| Payload scope | Rule |
|---|---|
| SUMMA payload | Complex128 panels16(m/p)(k/p) and16(k/p)(n/p), p=sqrt(P), p broadcasts per panel per q. Actual NCCL wire/link replication is not measured. Shapes authenticated by native/HLO66–75; counts agree with native census. |
| Static body | Same formulas with G q=36 on both routes. F head-attribution adds two calls; their payload is excluded from this τ-only table. |

| Scalar completed arm | ζ s | χ0 / W s | Rule plan s | τ / other Σ s | Total s | Compile events / s | EQP0 / EQP1 vs historical reference µeV | Scope / step |
|---|---:|---|---:|---|---:|---|---|---|
| Si P4 fe2a6937 own ζ |12.18|4.37 /4.20|627.45|14.93 /22.36|707.74|503 /35.65|8.135 /15.091 vs leg20|own ζ; historical residual; current-main exact PASS below;57988457/lx-Xg4-143444-1183016-5007 exit0, contaminated startup|
| Si historical leg20 |11.48|3.45 /3.81|10.50|15.67 /22.14|88.22|not emitted in retained log|reference|runs/Si/99_psi_irr_zeta_2026-09-05/20_g0w0_packed_order|
| Na historical03 P16 |19.43|screening176.31|33.90|391.73 /29.65|675.15|reference receipt|reference|runs/Na/15_psi_irr_parents_only_2026-09-05/03_p16_packed_order|
| Na historical04 P16 |19.24|screening173.79|34.50|383.90 /28.49|672.65|461|reference|57955934/lx-Xg4-165800-844150-6998; runs/Na/15_psi_irr_parents_only_2026-09-05/04_p16_repro_coord_ade4fc66|
| Si planner attribution |—|—|cache12→18 files; rank0 compile agreement201.708s|—|—|gw.sigma_box_plan._rule_cache_lookup / plan_sigma_windows|both sources rebuild legacy unstamped rules; copied-input kernels compared below|runs/Si/100_bisp_scale_scalar_2026-09-06/01_bisp_tip_fresh/driver.rank0.log; identity_leg20_freshP.json|

| Si P4 copied ζ, exact | P fe2a6937 | F d3d4b03a | Verdict / scope |
|---|---:|---:|---|
| EQP0 / EQP1 printed digits |0 difference|reference|PASS; identity_copied.json|
| Compile events / compiler s |438 /28.19|438 /28.21|parity|
| χ0 / W s |4.41 /3.92|4.35 /3.89|combined8.33/8.24, +1.1%|
| τ / other Σ s |14.53 /20.90|14.27 /21.28|combined35.43/35.55, −0.3%|
| Rule planning s |609.99|0.61|cold rebuild versus compatible30-step cache; not a source timing comparison|
| Whole wall s |672.77|62.53|excluded from source-speed verdict because rule-cache states differ|
| Step / artifact |57988457/lx-Xg4-144241-1236696-5495 exit0|zero GPU overlaps|runs/Si/100_bisp_scale_scalar_2026-09-06/{04_bisp_tip_copied,05_main_copied}; claim below|
| Legacy rule-cache incompatibility |12 entries stamped−1|deck requests30|sigma_box_plan.py:271–277 correctly rejects both sources; KNOWN_SANDBOX_ERRORS.md; D/scalar_rule_cache_delta.json|

| Si P4 own ζ, current-source pair | P fe2a6937 | F d3d4b03a | Acceptance / scope |
|---|---:|---:|---|
| EQP0 / EQP1 |exact|reference|PASS; own ζ is identical here, stronger than µeV-class|
| ζ s |12.18|12.44|−2.1%|
| χ0+W s |8.57|8.17|+4.9%; copied-input comparison was+1.1%|
| τ+other Σ s |37.29|36.73|+1.5%|
| Rule plan s |627.45|592.86|both rebuilt incompatible historical cache; host/planning variation|
| Total s |707.74|671.46|+5.4%, primarily planner; no warm whole-driver claim|
| Compile count / s |503 /35.65|509 /34.84|six fewer modules; compiler seconds+2.3%|
| Step / evidence |57988457/lx-Xg4-143444-1183016-5007 exit0|CONTAMINATED startup: steps.93/.95 overlap74/89s|runs/Si/100_bisp_scale_scalar_2026-09-06/{identity.json,summary.json}; no repeated baseline needed|

| Audit correction | Effect | Evidence |
|---|---|---|
| Claim1360 clean-interval assertion withdrawn | Own-ζ EQP identity remains exact; own-ζ timing comparisons above are diagnostic, excluded from non-regression verdict. Copied-ζ pair remains clean. | D/gpu_overlap_audit.json:57988457.94 overlaps.93/.95 on nid001112; claim correction below |

| 9×9 P4 initial arm | Total s | Screening support s | Rule plan s | τ s | Other Σ s | Compiles / s | Acceptance / timing scope |
|---|---:|---:|---:|---:|---:|---|---|
| P static fe2a6937 |75.57|28.29|—|—|22.62|616 /40.64|copied ζ: exact FAIL0.766µeV; startup overlaps another GPU step, no clean wall claim|
| F static e1559a07 |72.68|22.94|—|—|24.52|606 /31.93|same pair; prior clean static124/125 retained|
| P dynamic fe2a6937 |123.83|22.22|10.57|12.50|36.56|854 /60.73|copied ζ: exact FAIL8.668µeV; independently selected rules|
| F dynamic e1559a07 |176.15|19.92|69.20|11.70|36.68|813 /48.89|different rule policy; not a matched-node speedup|
| Step / audit |57988457/lx-Xg4-145514-1252010-5546 exit0|.107 overlaps.106 for60s during first static startup|—|—|—|D/gpu_overlap_audit.json|matched rules: D12_nine_common_rules; P4 follow-up R208; P16 R216 uses same certificates|

| Na P16 completed arm | ζ s | Screening s | Rule plan s | τ / other Σ s | Total s | Compile events / s | Identity / scope |
|---|---:|---:|---:|---|---:|---|---|
| P fe2a6937 own ζ |22.68|196.70|0.50|474.32 /30.41|753.07|454 /39.83|EQP0/EQP1 exact vs historical04; current main pending; interval audit pending|
| Warm τ node |—|—|—|193.965ms median;204.334ms mean|2231 warm /2232 calls|one first-call compile|unit_summary_P.json; host synchronized observer on both sources|
| Step / artifact |57988457/lx-Xg4-150412-1355239-5472 running|—|—|—|—|—|runs/Na/16_bisp_scale_scalar_2026-09-06/{summary.json,identity_historical04_P.json,unit_summary_P.json}|

| Experiment | Before → after | Identity | Memory / disposition | Evidence |
|---|---|---|---|---|
| Restore each source once/term |300→48 restores/run; repeated total65.24→65.71s; Σ20.08→19.84s;674→674 compiles|EQP0/EQP1 and210 sector rows exact|REJECT: whole-wall win did not repeat. Retained sources203233536B/rank at P4 C600/T196; refusal above5% of30GB/device|57982945/lx-Xg4-110752-42766-3793 exit0; R17/18; D/restore_repeat_identity.json; claim1231|
| Resident scaling |16 K(M_C+3M_T)²/P bytes per retained term|all P ranks|Independent of nb/ns/Q; adds to parent O(QM²/P). Relative cost falls with nb/ns, not with K alone|run-local prototype only; no retained production residency change|
| Compile-event parity, commit902d01ca |P4 static674→584 vs F578; dynamic894→772=F772; P16 static646→556<F604; dynamic897→775<F825|Five final arms130–135 exact to parent baselines|RETAIN branch change; P4 static still six extra events. Exact fe2a6937 speed-phase snapshot excludes this local patch|57988457/lx-Xg4-141411-1064403-7247 exit0 and lx-Xg4-135842-973036-7653 science artifacts; D/final_source_gates.json; claim1331|
| Compile sites |64 static/96 dynamic zero-lambda events removed,21 typed-table placement events,5 surface-placement events|same source-change gates|distrib_la.matmul._zeros; host typed-map placement; absent surface/row table placement. No new cache or API|evidence/final_compile_module_delta.csv; source on perf/bisp-scale-2026-09-06, unmerged|

| 6×6 leading counts/rank | P | Family | Static G GFLOP | τ G GFLOP | Two projections GFLOP | Full G bytes | Operator bytes |
|---|---:|---|---:|---:|---:|---:|---:|
|P|4|CC|33.177600|12.902400|2.449440|829440000|51840000|
|P|4|CT|11.280384|4.386816|0.832810|282009600|17625600|
|P|4|TC|11.280384|4.386816|0.852768|282009600|17625600|
|P|4|TT|3.835331|1.491517|0.289941|95883264|5992704|
|F|4|CC|33.177600|33.177600|12.597120|829440000|51840000|
|F|4|CT|11.059200|11.059200|4.199040|276480000|17280000|
|F|4|TC|11.059200|11.059200|4.302720|276480000|17280000|
|F|4|TT|3.686400|3.686400|1.434240|92160000|5760000|
|P|16|CC|8.517059|3.312189|0.671154|212926464|13307904|
|P|16|CT|3.137864|1.220280|0.247267|78446592|4902912|
|P|16|TC|3.137864|1.220280|0.252772|78446592|4902912|
|P|16|TT|1.156055|0.449577|0.093127|28901376|1806336|
|F|16|CC|8.517059|8.517059|3.451650|212926464|13307904|
|F|16|CT|2.913731|2.913731|1.180828|72843264|4552704|
|F|16|TC|2.913731|2.913731|1.210319|72843264|4552704|
|F|16|TT|0.996803|0.996803|0.414056|24920064|1557504|

| Analytic scope | Limitation |
|---|---|
| evidence/analytical_operand_counts.json; same native/HLO steps66–75 cited above | Leading arithmetic and logical carriers; packing, beta, allocator peaks and vendor wire replication excluded. No new timing measurement. |

| Na P16 scalar pair | P fe2a6937 | F d3d4b03a | Verdict / scope |
|---|---:|---:|---|
| EQP0 / EQP1 |exact|reference|PASS; identity.json; parent also exact to historical Na04|
| Compile events / compiler s |454 /39.83|461 /39.05|seven fewer events; compiler time diagnostic|
| ζ / screening s |22.68 /196.70|20.32 /178.51|parent startup contaminated; not a regression verdict|
| τ / other Σ s |474.32 /30.41|474.17 /30.56|sum504.73/504.73: no scalar Σ regression; both sweeps after overlap ended|
| Warm τ median / mean ms |193.965 /204.334|192.650 /205.000|2231 warm calls each; median+0.68%, mean−0.33%|
| Whole wall s |753.07|731.07|excluded: startup contention in P|
| Step / audit |57988457/lx-Xg4-150412-1355239-5472 exit0|Slurm.118 overlaps.117 for110s|completed_receipts.json, identity.json, D/gpu_overlap_audit.json; Na04 parent-only repeat queued after9×9, main reference reused|
| Phase pin / refresh |measured fe2a6937|origin/main d3d4b03a|hourly fetch22:28UTC sees ORCH b1d8b8f1; immutable phase pin preserved per speed protocol|

| Matched9×9 follow-up | Result | Step / artifact | Disposition |
|---|---|---|---|
| Eight union-box certificates |eps1e-5, reduction_steps0; all original error/noise guards pass|57988457/lx-Xg0-152909-1494895-4356 exit0; D12_nine_common_rules/rules_receipt.json|same immutable certificates for P4/P16|
| R208 P4 replay selector |Both sources reject an already-contained box due to additional1e-3 proximity assertion; no EQP|57988457/lx-Xg4-153138-1497102-7816 exit1; R208/01 and02 driver.rank0.log|scaffolding failure; corrected containment selector in new R209 and unrun R216, original certificate guards unchanged|
| Quiescent measurement start |P16 matrix and Na parent repeat wait for other GPU steps to drain before imports; active-start receipt retained|D/quiescent_start.py; per-master quiescent_JOB_STEP.json|whole-step overlap and active-science overlap audited separately; no cancellation of shared jobs|
