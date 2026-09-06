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
| Si leg20 P4 | fe2a6937 | d3d4b03a | own-fit numeric residual reported; exact requested | detached queued | runs/Si/100_bisp_scale_scalar_2026-09-06/03_pair |
| Na8×8×8 P16 | fe2a6937 | d3d4b03a | own-fit numeric residual reported; exact requested | detached queued | runs/Na/16_bisp_scale_scalar_2026-09-06/03_pair |

| 9×9 arm | P/F total s | Screening P/F s | Sigma P/F s | Compiles P/F | Class | Step / path |
|---|---|---|---|---|---|---|
| Prior static P4 |70.10 /69.03|29.14 /21.89|21.40 /23.60|641 /606|copied ζ: exact FAIL0.766µeV|57988457/lx-Xg4-131617-747058-1237; R124/125|
| Current static/dynamic P4 |queued|—|—|—|copied ζ: exact|R204_speed_9x9_P4/05_matrix|
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
