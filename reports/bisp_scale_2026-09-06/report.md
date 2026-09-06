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
| head attribution | n/a | n/a | disabled | disabled | sigma_freq_debug_output=false; physical head stays full. |

Charge references are harvested evidence, not this lane's measurements: sandbox `reports/psi_irr_perf2_2026-09-05/report.md`, JID57941637, and CLAIMS row842 (its separate claims/0842.md is missing). PERF2 used cuda_async@0.85; this lane uses BFC@0.85. Charge cross-source comparisons were not printed-digit identical. These reference ratios cannot establish bispinor identity or absolute timing across allocators.

Source provenance: started at S tip4a691b67, cherry-picked ZW0f8fbc3e/f811f734 as cee4011d/3046b911, resolving only the absent ZW report by retaining it. Preserved that state as local branch perf/bisp-scale-initial-2026-09-06. Fetched ORCH and switched measurement source to af85d4745fd1bebdde7b2f34421409ccdc6d0000: b455f781 integrates Σ executable/plan ownership and states the ZW ports are present; the tip additionally includes covariant Γ completion and the paired-Gram ridge-sign fix. Fixed-main e1559a07 remains read-only. Its absence of these physics fixes is a potential identity-gate blocker, not permission to waive identity.

QE evidence: `runs/MoS2/42_bisp_scale_2026-09-06/00_qe_6x6/`, pool57982945. NSCF reports12 spatial operations and7 stored k points. Both SCF/NSCF use6x6x1 and82 bands. wfn2hdf step lx-Xg0-105033-2302113-3374 exits0. Claim1214. Preprocessing requests600 charge/200 current orbit centroids. The inherited GW window remains80 bands with82 QE bands to witness the upper multiplet boundary.

Experiments are preregistered, not implemented: (a) restore16 source blocks once per term, retain only if exact and faster, with memory refusal tied to existing memory_per_device_gb; (b) enumerate actual module compile excess and share only shape-identical programs, with exact source-change gates. For complex128, heterogeneous resident sources occupy16*K*(M_C+3*M_T)^2/P bytes/rank, bounded by256*K*max(M_C,M_T)^2/P; this is carrier storage, not an allocator peak. At K36, M_C600, M_T200 the nominal resident carrier is207,360,000 bytes/P (51,840,000 at P4;12,960,000 at P16). It scales linearly in K, quadratically in centroids, independently of nb/ns, and cannot be treated as a fixed-size optimization.

Outstanding: GW source arms/fresh ζ, identity, P4/P16 stage/unit/compile tables, native/HLO census, both experiments, final verdict. No unrun cell is treated as zero or inherited speedup.
