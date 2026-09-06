# Claim 862 — independent bispinor controls (heavy lane)

Branch `feat/bisp-orch-controls-2026-09-05`, unmerged; source unchanged.

All ten prepared Si/MoS2 modes attempted in four combined P4 legs, pinned pool57955075, lorrax_A, BFC@0.85. Main actual tip de8dcfbc5505bb67ff31b86d4e78151ea2af4f9e; lane actual tip ade4fc66b7215650bdae552043ff1657ac7fde56.

| Pair | lx step | Slurm step | exit |
|---|---|---|---|
| Si main | lx-Xg4-174227-1081989-6309 | 57955075.89 | 1 |
| Si lane | lx-Xg4-174223-1082178-6037 | 57955075.87 | 143 |
| MoS2 main | lx-Xg4-174220-1104181-9749 | 57955075.88 | 1 |
| MoS2 lane | lx-Xg4-174225-1104416-2098 | 57955075.90 | 1 |

Nine drivers failed; Si main COHSEX driver completed, but its combined step exited1 due to GN failure. Therefore no certified control pair or QP parity verdict. Source errors and the secondary combined-run coordination limitation are registered in KNOWN_LORRAX_ISSUES.md and KNOWN_SANDBOX_ERRORS.md. Main packed calls reject charge_zeta_identity; lane screening rejects symmetry maps crossing identity-layout shards; main face GN passes int to padding strip_axis; lane Si GN rank1 fails HDF append metadata read. No source workaround or retry.

Disk results: all15 transverse ζ channel pairs have exact0 difference and no nonfinite values (tools/compare_zeta_h5.py, default1e-13); every available bare CC/TT V tile in all5 paired modes likewise exact0. No packed W was produced; lane W0 readiness false. Si main COHSEX eqp0/eqp1 each parse as8k/256 finite rows using tools/eqp_ab.py read_eqp; canonical sigma parser sector closure1.7763568394002505e-15eV. This single arm is not A/B evidence.

BFC@0.85 Si main COHSEX measured total55.69s, ζ7.74s, V5.21s, χ0 1.95s, W3.27s, Sigma12.64s. The four concurrent P4 jobs co-placed pairwise on nid001052/nid001092; timings are co-tenant observations, not isolated performance comparisons. Missing complete-stage times are not estimated. Main120-second quadrature reduction vs lane steps0 remains frozen and explicit; lane cache rejects clock-rule stamps, no cache workaround performed.

Evidence directory: `runs/DEV/111_bisp_orch_codex_2026-09-05/sub_02_controls/`. `control_inventory.json` has exact paths, tips, step receipts, readiness stamps, dataset shapes, errors and stage timing lines; `zeta_comparisons.json`, `v_comparisons.json`, `eqp_validation.json` carry disk measurements. All `tmp/` trees preserved. The staged MoS2 paths are03_full_static_cohsex,04_packed_bare_transverse,05_dynamic_packed_gn_ppm per final deck_inventory.json.
