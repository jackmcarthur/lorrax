# BISP-PROF-S local issue register

Unmerged branch perf/bisp-prof-s-2026-09-06; source inspection only, quantitative impact unverified.

- `src/gw/w_isdf.py:2029`: `photon_blocks_full_q` defines a fresh jitted `add` inside each C,D/output/term iteration (300 creations for all16 blocks ×3 terms). Candidate compiler lifetime defect; implementation owned by screening lane ZW, consumption here charged to Sigma. See reports/bisp_prof_s_2026-09-06/report.md.
- `src/gw/photon_sigma.py:123`: static factory specializes by `(A,B)` in addition to family plans/shapes, unlike F's shape-class cache. Quantitative compilation cost unmeasured.
- `src/gw/photon_sigma.py:180`: separate q0 head kernel rebuilds G and unfolds both faces that the body just consumed. Candidate duplicated work; no change without profiling/parity gate.

- Measured, pending ablation: src/gw/w_isdf.py:2035 creates300 restore/add JIT bodies over X/SX/COH in the MoS2 full-static gate; native GPU projections total6.607031ms while enclosing restore compilation is95.008s in capture11.
- Measured factory design, pending ablation: src/gw/photon_sigma.py:127 constructs fresh projector and Green GEMM plans per vertex/head specialization, although endpoint shapes repeat; capture11 has64 contraction modules versus F4. Host-control14 explicitly measures factory work.
- Measured retrace, pending ablation: src/gw/photon_sigma.py:167 sends(k,band) occupied weights versus(band,) COH weights, causing a second executable per vertex/head kernel. Fix candidates remain unmerged on perf/bisp-prof-s-2026-09-06.
