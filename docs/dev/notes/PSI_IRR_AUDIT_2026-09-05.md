> Revision-scoped audit report; the sandbox evidence paths below are relative to `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14`. This report is evidence, not a new runtime contract.

# PSIIRR-AUDIT — Phase B result

Branch **`audit/psi-irr-packed-order-2026-09-05`**, unmerged. Audited and rebased onto **`7e1ae83d`**, the lane's `[packed-order] validated` commit, at the scheduled 19:44 UTC fetch. Numerical source tested: **`93166afc`**. The final source edits are comments/docstrings only, checked by normalized Python AST against that commit (`audit/16_final_comparisons/documentation_ast_check.json`). No main or peer-branch mutation.

**The requested Si gate passes.** The audit's fresh output equals the lane's validated packed-order leg 20 to every printed digit, and restart equals fresh. Against pre-restructure leg 14, max Δeqp1 is **5.343 µeV**, inside the **2 meV** gate. It is not printed-digit identical to leg 14. Final claim **832**, `claims/0832.md`.

Evidence root: `runs/Si/99_psi_irr_zeta_2026-09-05/audit/`. Historical ranked findings, design estimates and interim failures are preserved in `reports/psi_irr_audit_2026-09-05/phase_a_and_worklog.md`; the Phase A source snapshot is also in commit `458933bf`, `docs/dev/notes/PSI_IRR_AUDIT_2026-09-05.md`.

## Changes to cherry-pick

| Commit | Concern | Ownership and effect |
|---|---|---|
| `4d62f8ad` | Remove `ParentSigmaRoute` | `wavefunction_bundle.py:639` passes the existing `CentroidKUnfoldPlan` through COHSEX, static PPM and shared GN/MPA factories. Cache keys retain the plan itself. No copied row/symmetry/face-shape state or integer lifetime dependency. |
| `3a7f98c6` | One runtime extent across I/O | `common/centroid_basis.py:200,212` owns the solve receipt and conversions for either ordering of canonical/packed extents. `wfn_transforms.py:2631`, `v_q_g_flat.py:302`, `w_isdf.py:1535` consume the packed extent without reapplying the canonical extra-padding knob. The basis owns a read-only centroid table. |
| `66a696f9` | Preserve the transverse table | `gw_init.py:1495,4213` clears the inherited charge basis in fresh/restart transverse metadata; the current channel uses its own existing canonical loader path. |
| `93166afc` | Name parent-plan preparation | `gw_init.py:3222` owns eligibility, blocker messages and exact plan construction. The orchestration function shrinks by 48 lines to 1,031. Existing fit-memory planning and face loading remain distinct stages; no new state class. |

The final documentation commit corrects stale canonical-order comments and the padding owner page. Relative to validated `7e1ae83d`, production source is 218 lines added / 239 deleted (net -21, including comments). Tests and audit reports add evidence, not runtime API layers.

The preparation extraction is smaller than the initial 150–250-line estimate because the lane had already repaired reuse/pricing. Folding memory planning and all face loading into one helper would require a larger state-return contract. The secondary band-compaction consolidation was not taken. The lane's generic I/O permutation remains the single basis-conversion owner; restoring its deleted public symmetry reorder API solely for this seam would add another interface. No crystal symmetry/phase/TRS algebra was rebuilt.

## Verification

All compute work used pool **57941637**. The final GPU run was **one P4 step** with one rank per GPU, containing fresh and restart phases:

**`lx-Xg4-130842-1703447-2751`, exit 0 in 166 s**, source `93166afc`.

- Fresh: `audit/14_final_fresh/` — leg-14 deck with `write_restart_tensors = true`, leg-14 quadrature cache copied before launch.
- Restart: `audit/15_final_restart/` — `restart = true`, a **copy** of the fresh `tmp/`, made only after all fresh rank statuses succeeded.
- Parser results: `audit/16_final_comparisons/`. Uses `tools/eqp_ab.py`, `tools/sigma_diag_rows.py` (the canonical `tests.harness` parser), and `tools/compare_zeta_h5.py`; no second scientific text grammar.

| Comparison | eqp0 max | eqp1 max | Result |
|---|---:|---:|---|
| Fresh vs leg 14 | 1.410 µeV | 5.343 µeV | PASS at 2 meV |
| Fresh vs validated lane leg 20 | 0 | 0 | 224/224 rows identical in each file |
| Restart vs fresh | 0 | 0 | 224/224 rows identical in each file |

Named **real** Sigma columns vs leg 14: max ΔsigX = 0, ΔsigC = 2 µeV, ΔsigXC = 1 µeV; restart differences are zero. This parser does not compare the complete complex frequency cube. Zeta file normalized max difference vs leg 14 is **9.013994546e-8**, with zero nonfinite values and shape **(8, 836, 588)**. See `fresh_vs_leg14_sigma.json` and `fresh_vs_leg14_zeta.txt`.

Canonical restart datasets have μ=**836**, not the runtime packed width, in both files: V/W0 `(8,836,836)`, parent faces `(8,80,2,836)` and `(8,2,836,80)`, parent rows `(8,)`, no `psi_full_y`. See `restart_dataset_shapes.json`. All eight original rule-cache files remain byte-identical; two additional rule files were written, and restart's cache exactly matches fresh (`rule_cache_manifest.json`). The copied cache therefore does not justify claiming the original leg-14 schedule was wholly reused unchanged.

The final CPU leg was **`lx-Xg0-130138-1649043-7326`**, exit 0: **169 passed, 1 skipped**, 44.93 s pytest time. Artifact `audit/11_parent_preparation_cpu/pytest.log`; claim 830. It includes the eight requested suites, the new basis tests, and existing bispinor-reuse/centroid-loader planning checks. The skip is its own fractional-contour cell: host FFT FFI unavailable. Earlier per-concern CPU legs are claims 827–829.

New discriminators cover same-plan cache reuse and lifetime, distinct parent-row results, canonical widths 4/8/12 against fixed packed width 8, exact operator and flat-axis round trips, exact-zero pads, analytical Dyson values, and HLO without an input-carrier all-gather. Existing C/Z, spin/glide/TRS, projection and restart-geometry oracles remain.

Failed attempts are preserved: CPU legs 07/08 corrected mistakes in the new test oracle/HLO call; combined leg 12/13 completed fresh GW but failed before restart because the external copy helper assumed unavailable `mpi4py`. Step **`lx-Xg4-130429-1673793-6078`** is not counted as a successful gate. The replacement uses atomic status files and keeps stdout separate from driver-owned `gwjax.out`; `KNOWN_SANDBOX_ERRORS.md` records both scaffolding issues.

## Ranked findings still open

These are registered in `KNOWN_LORRAX_ISSUES.md`; none is a measured failure of the final Si deck.

1. **P2 — fallback can retain a parent-priced chunk plan.** `gw_init.py:1682` rejects HWM above the hard budget, while :3455–3466 falls back above the utilization target. For target < HWM <= budget with a full-k consumer blocker, chunks priced with `n_parent` ψ intermediates (`gflat_memory_model.py:539–547`) can survive selection of full-k contraction. Replan for the selected route; test a forced budget boundary.
2. **P2 — factor diagnostics count artificial mean-pad modes.** `isdf_fitting.py:846` adds the mean diagonal; `isdf/core.py:4580` counts the augmented spectrum. Do not read that count as physical ISDF rank. The trace-based Cholesky floor also includes pads (:4598); the mean-normalized opt-in ridge is preserved. The lane's mean-pad fix removes the demonstrated unit-pad cutoff error, but physical conditioning diagnostics still deserve a separate count.
3. **P3 — broad non-closure fallback.** `common/centroid_basis.py:121` catches both ValueError and RuntimeError from map construction. A typed non-closure refusal would distinguish a non-closed set from malformed metadata/programming errors.
4. **P3 — duplicate band compaction.** `isdf/core.py:2954` owns a helper while legacy/full-face Z builders retain parallel table construction. Consolidate only with remainder-bearing parity tests.

The Phase A stale-cache and unit-pad defects were measured (claims 824/825); they must not be reported as current uncorrected Si failures. Reuse admission, memory double-counting and torn-restart prechecks were fixed by the lane before this rebase. The residual budget-policy mismatch above is separate from double-counting.

Scope: the requested CPU/Si/restart gate, not a full census or a numerical bispinor/processor-count sweep. The bispinor change has source and structural-test evidence. No new speedup claim, no full complex Sigma-cube parity claim, and no blanket release certification.
