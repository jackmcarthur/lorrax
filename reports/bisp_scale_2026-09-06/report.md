| Unit | Charge PERF2 P/F; Na P16 claim842 | Bispinor 6×6 P4 P/F | Bispinor 6×6 P16 P/F | Arithmetic / logical bytes per rank | Collectives and compilation; mapping verdict | Classification / scaling limit |
|---|---|---|---|---|---|---|
| ζ charge tile | 0.516 native; 0.846 fit-loop wall | 1.616 observed time/point, unequal tiles | unrun clean timing; capture excluded for overlap | Parent projectors leading 16 Q ns² M R nb/P real FLOPs; full-K transform tails remain; output16 K M R/P bytes | P4 P/F: five AG and five A2A/tile each; AR5/80. Not the same workload or tile width; no matched-tile speedup claim. | mapped projector reduction; full-K tails remain |
| ζ current tile | No separate current unit | Coupled P versus three F channels: 0.967 observed time/point | unrun clean timing; both routes coupled at P16 | Same projector law per channel; P couples three channels, F P4 repeats the full-face tail three times | P4 coupled P5 AR versus F3×80 AR; per-tile counts differ from full-fit counts. Bispinor-specific sharing, not a scalar one-to-one row. | bispinor-only; three channels, coupling depends on P |
| χ0 build | 0.764 stage; 0.145 screening including W | CC warm432.659/426.835 =1.014 | CC warm2243.668/2444.546 =0.918 | 22 distributed GEMMs/block at the 11-node static screening rule; full-K work, Θ(K nb ns² M_A M_B/P) per quadrature unit | P4 3168/P16 6336 broadcasts per block on BOTH routes. No Q/K reduction here; inherent full-child summation. | inherent; full-K child sum, no universal Q/K gain |
| W restore | Four A2A on both charge routes; Na unrun isolated timing | unrun isolated timing | unrun isolated timing | Full-K operator16 K M_A M_B/P bytes; transform Θ(K log K M_A M_B/P) | Do not count the charge route's four A2A as removed. Native restore modules below have no NCCL; transform/layout collectives belong to their surrounding modules. | inherent; full-K transform/traffic |
| τ node | 0.346 native; Na0.0717 host | 32.502/55.089 =0.590 native;33.454/56.034 host | 67.816/228.036 =0.297 native | P has TWO G GEMMs:16 Q d_A d_B nb/P; F8 K d_A d_B nb/P. Add projection and full-K convolution | P4 Broadcast112/432; P16 224/864. Integrated parent has four distributed GEMMs, F three. NOT the historic charge operation count. | mapped Q/K plus extra G GEMM; not operation parity |
| Static block | unrun charge counterpart | CC0.267, CT0.249, TC0.258, TT0.257 native | CC0.263, CT0.209, TC0.310, TT0.282 | G8 K d_A d_B nb/P in BOTH; projection uses q=Q (parent), K (full) | P4 Broadcast200/720; P16 400/1440. F includes two extra head-diagnostic projections; ratios are not pure parent-route gains. | inherent full-K G plus mapped projection; F diagnostic defect |
| Band projection | 0.125 native; Na unrun isolated timing | (5.565+0.935)/(28.839+4.201)=0.197 | unrun isolated ratio; Q/K arithmetic class, constituent thunks retained | 8 q (b d_A d_B+b² d_B)/P; q=Q or K; b30/P4,32/P16 | Two SUMMA calls:4 sqrt(P) q broadcasts; maps to charge Q/K reduction. | mapped Q/K |
| Antiunitary band unfold | unrun isolated timing | Zero per-τ CP after integration71 | Zero per-τ CP | Final band tensor16 K b²/P; old per-node band exchange129600 bytes at P4 | Old P4: two CP/node, G161280000 bytes/exchanging rank plus band129600. New: second G GEMM and delayed final band unfold. This is a trade-off, not free deletion. | mapped transport trade-off: extra G replaces exchanges |
| Seam conversions / operator spin transport | 139 eager→10 compiled modules; same38 A2A | Full-K convolution ratio0.997; parent spin/seam region14.360ms/node, not isolated rotations | Full-K tails persist | Two local spin rotations16 K ns³ M_A M_B/P FLOPs; full G16 K ns² M_A M_B/P bytes | Local rotations have no NCCL; ns=1 scalar cost differs. P4 CC rotations3.318 GFLOP/rank. Conversion compile events are priced in the module census. | bispinor-only spin transport; full-K seam inherent |
| Ordinary charge restores | Retired bridges enumerated by PERF2 | Parent typed restores remain; no second removal credited | Same | Θ(K M²/P) output traffic independent of nb | Native P4 small band restore ~0.05ms/call, no NCCL. Restore code is not the antiunitary G exchange. | inherent typed restores |
| Lorentz restores + mixing | Bispinor-only | 300 source restores/static run; TT ~0.012ms native/restore; host output-restoration aggregate median21.4ms | unrun isolated timing; same source multiplicity | Three terms ×100 contributions: output traffic3×16 K(M_C+9M_T)²/P bytes; mixing Θ(K M²/P) | CC/CT/TC/TT counts3/27/27/243, no NCCL inside restore kernels. Launch floor stays300; increasing K alone does not remove asymptotic cost. | bispinor-only; K M²/P, independent of nb; launch floor300 |
| Per-family faces / current channels | Bispinor-only | Four shape families; 16 ordered Lorentz blocks | P16 has greater carrier padding | Face storage Θ(Q nb ns M/sqrt(P)); current factor3, ordered vertex factor16; all typed actions from SymMaps | Shape families cause distinct modules; current coupling cuts repeat work. No new shape-identical kernel cache claimed beyond measured compile experiment. | bispinor-only; fixed16 vertices/four families, Q nb ns M/sqrt(P) faces |
| Packed Dyson | Bispinor-only | Warm LU156.046/797.148=0.196; complete host167.376/820.453=0.204 | Warm LU513.159/2567.270=0.200 | L=M_C+3M_T; A+LU+2TRSM leading(56/3)qL³/P FLOPs; stored matrix16qL²/P bytes | P16 warm LU P/F: AR35/180, Broadcast259/1332, AG140/720, SendRecv144/738. Pivot traffic is data-dependent. Q/K reduction with padding. | mapped Q/K solve on bispinor-only L=M_C+3M_T; cubic L |
| Γ completion | Bispinor-only covariance correction | ~0.02ms/call,10 calls, zero NCCL | unrun isolated timing | Rank4 ×12 typed operations; Θ(n_sym rank L²/P), independent of nb, one-off | One P-only compiled module. Fixed-main lacks this physics fix; cannot erase it to force parity. | bispinor-only; one-off symmetry completion, independent of nb |
| Head attribution | Diagnostic, not a charge gain | Requested false; P off, F erroneously active | Same source discrepancy | Two extra F band-projection GEMMs/static block | F photon_sigma.py:380–423 ignores false when q0 factors exist. Counted in actual static native traffic; registered source defect. | defect in F diagnostic flag; not an inherent physics cost |

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

| Scalar non-regression, P fe2a6937 / main d3d4b03a | Si leg20 P4 | Na8×8×8 P16 | Scope / evidence |
|---|---|---|---|
| EQP0/EQP1 printed digits |exact, own and copied ζ|exact, own ζ; also exact to historical04|PASS numerical gates; Si100/identity.json, identity_copied.json; Na16/identity.json|
| Compile events P/F |copied438/438; own503/509|454/461|PASS count non-regression; driver.rank0.log receipts|
| Σ τ + other P/F s |copied35.43/35.55 (−0.3%)|504.73/504.73 (equal)|PASS measured Σ scope; Na sweeps start after overlap ends|
| Screening P/F s |copied χ+W8.33/8.24 (+1.1%)|196.70/178.51, P startup contaminated|Si within small observed variation; Na clean screening comparison unrun|
| Fresh ζ P/F s |12.18/12.44, startup contaminated|22.68/20.32, startup contaminated|clean fresh-ζ non-regression unrun; no whole-stage PASS claimed|
| Whole wall P/F s |copied672.77/62.53; own707.74/671.46|753.07/731.07|excluded: Si cold/warm rules or overlap; Na parent startup overlap|
| Warm τ median P/F ms |unrun isolated receipt|193.965/192.650;2231 calls each|Na mean204.334/205.000ms; completed_receipts.json|
| Cache / interval audit |legacy12 rules unstamped−1 vs deck30; both sources rebuild. Copied pair clean; own pair startup overlaps74/89s|valid0-step rules both; step.118 overlaps.117 for110s before τ|D/scalar_rule_cache_delta.json; D/gpu_overlap_audit.json|
| Step IDs |57988457/lx-Xg4-144241-1236696-5495 copied exit0; lx-Xg4-143444-1183016-5007 own exit0|57988457/lx-Xg4-150412-1355239-5472 exit0|runs/Si/100_bisp_scale_scalar_2026-09-06; runs/Na/16_bisp_scale_scalar_2026-09-06|

| 9×9, K81/Q12, closed charge597/current200 | P/F total s | Screening support P/F s | Σ other / τ P/F s | Compiles P/F | Acceptance / evidence |
|---|---|---|---|---|---|
| Prior clean static P4 |70.10/69.03|29.14/21.89|21.40/23.60; τ n/a|641/606|copied ζ exact FAIL0.766µeV;57988457/lx-Xg4-131617-747058-1237 exit0; R124/125|
| Current static P4 |75.57/72.68|28.29/22.94|22.62/24.52; τ n/a|616/606|copied ζ exact FAIL0.766µeV; startup overlap; R204|
| Current dynamic P4 |123.83/176.15|22.22/19.92|36.56/36.68;12.50/11.70|854/813|copied ζ exact FAIL8.668µeV; independently selected rules, no matched speedup claim|
| Current P4 step |57988457/lx-Xg4-145514-1252010-5546 exit0|χ7.34/4.48; W6.85/6.99|rule plan10.57/69.20|compiler60.73/48.89s dynamic|R204/{summary.json,identity_static_sectors.json,identity_dynamic_sectors.json}; startup overlap60s|
| Matched dynamic P4 R208 |unrun completed result|unrun|unrun|unrun|both fail redundant replay proximity assertion before EQP;57988457/lx-Xg4-153138-1497102-7816 exit1|
| Static/dynamic P16 R216 |unrun|unrun|unrun|unrun|queued launcher stopped at owner closeout|
| Corrected matched P4 R209 |unrun|unrun|unrun|unrun|queued controller stopped at owner closeout|
| Union certificates |8 certified rules|eps1e-5 /reduction_steps0|original error/noise guards pass|unrun GW with corrected selector|57988457/lx-Xg0-152909-1494895-4356 exit0; D12_nine_common_rules/rules_receipt.json|

| Closeout | Disposition |
|---|---|
| Measurement pins |6×6 unit/matrix pins as above; speed phase P fe2a6937, scalar F d3d4b03a, bispinor F e1559a07. ORCH fetch22:28UTC saw b1d8b8f1; no claim that it was measured. |
| ZW_ROWS.md |Absent at closeout in this lane, ORCH handoff directory, ZW DEV directory and ZW report directory; no rows inferred. |
| Unrun work |9×9 P16, corrected matched9×9 P4, clean Na parent startup repeat; stopped on owner's instruction. Clean P16 ζ tile timing and exact vendor wire bytes remain unrun/unmeasured. |
| Verification |No pytest, lx test or gate0 in speed phase. Scalar numerical gates pass; complete fresh-ζ/screening timing non-regression is not established. |
| Branch |perf/bisp-scale-2026-09-06, unmerged; one-to-one table is an accounting verdict, not a claim of universal identity or compile parity. |

The bispinor parent route does not map one-to-one onto every charge-route operation: projection and packed Dyson retain the parent/full-k reduction, but full-k χ/G/convolution work, Lorentz/current multiplicity and spin transport remain, and the antiunitary implementation trades two exchanges for an extra G GEMM. On the clean6×6 P16 matrix, parent dynamic total is10.6% lower (127.64 vs142.84s), with Σ34.5% lower (45.50 vs69.46s) despite screening χ+W+support9.1% higher; static total is0.8% lower and Σ30.9% lower. These are diagnostic timing gains: copied-ζ P/F identity still misses the exact gate by sub-µeV amounts. P4 remains slower overall; clean9×9 P4 static is1.6% slower overall despite9.3% lower Σ, and no completed9×9 P16 result establishes a further crossover or a kernel-dominated cold driver. Scalar Si and Na EQP0/EQP1 are exact, compile counts do not increase, and measured Σ shows no regression; clean fresh-ζ and Na screening non-regression remain unestablished because startup intervals were contaminated. Restore residency was rejected, and static P4 compile parity remains six events short.
