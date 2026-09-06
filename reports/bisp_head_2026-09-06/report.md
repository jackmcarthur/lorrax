Heavy lane — BISP-HEAD; branch `audit/bisp-head-2026-09-06`, unmerged; base `b8e036a8`, scoped velocity fix `ae36fa1b`.

# Verdict

The parent route reproduces the full-k Γ completion to floating-point accuracy, but **the inherited completion is not symmetry complete under the declared Lorentz-block action**. The actual TT Γ update fails its little-group covariance check by **0.556871776 relative to the largest TT entry**. Its bare update is exactly the fixed-main update, so equality to fixed-main cannot certify this property. Ward, Hermiticity, Dyson and polygon certificates all pass on the same failing object.

The scoped missing IBZ velocity consumer is fixed and pushed. It authenticates the existing dipole artifact, reads only file-wedge rows and the active chi band window, and restores the velocity through the existing `symmetry_maps.unfold_file_wedge_polar_matrix`. No new action, head builder, kernel, deck key or file schema was introduced. Both static and dynamic parent QP outputs retain exact printed-digit identity to their parent controls. The inherited TT covariance failure remains open; changing it conflicts with §3.8's unchanged-completion requirement and the fixed-main identity requirement. A scope clarification has been requested.

# Controls and method

The source worktree was verified at `b8e036a8f00fcbb9092cc711ab53438c129576fb`; fixed-main is read-only `wt_main_de8dcfbc_fixed` at `e1559a071e244b4f049c924781b668d9e1560739`. The stale branch ancestry embedded in the copied hard-rule text was not used. No orchestrator worktree was modified.

`run_manifest_receipts.json` records actual step receipts and harness hashes. Several copied initial manifests retain stale run-id/step fields; those fields are superseded by this separate receipt inventory without rewriting completed runs.

All production verification used P4, one rank per GPU, pool **57966610**, allocator **BFC@0.85**. The MoS2 deck has 12 spatial operations, 3 stored / 9 full k points, 80 chi bands, C192/T98 centroids. Both new static arms reuse the same canonical ζ files from Mo84 and rebuild V and screening. Diagnostic wrappers capture only O(N_mu) head factors, wings and small tensors on every rank; no dense centroid square is gathered. CPU comparisons contract one row of each low-rank update at a time. Canonicalization uses `PackedCentroidBasis.unpack_host` and the photon layout's shard interleaving; comparing the raw packed arrays was explicitly rejected as a coordinate mismatch.

Evidence paths below are relative to `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14`:

- `DEV` = `runs/DEV/115_bisp_head_codex_2026-09-06/`.
- `MO` = `runs/MoS2/41_bisp_parent_route_2026-09-05/`.
- `HEAD` = `MO/head/`.

# Completion input inventory

Source line references in this table describe **b8e036a8**, before the scoped fix.

| Input | Source and representation | Action / consumer | Proven or flagged |
|---|---|---|---|
| Four literal `g0` factors, `(1,n_mu)` each | V projection; `v_q_bispinor.py:555-599`, `v_q_g_flat.py:569-583`; q-IBZ ζ and canonical logical file factors | `unfold_isdf_one_leg`, scalar charge / polar time-odd current, explicit q-policy row; three source current components accumulated; `pack_photon_channel_vectors` produces `(4,Npacked)` | Canonical factors exact against fixed-main. At Γ, parent G is zero and translation/wrap phases vanish, but the centroid permutation and Cartesian component action remain. Selecting one Γ representative is not little-group averaging. The two-index TT update fails covariance. |
| `S_direct (2,2,4,4)` | `static_gauge_response.py:240-275` embeds only in-plane charge CC from `build_dft_head_response` | Shared scalar `head_s_tensor_sharded`; source-state spinor normalization, full-k energy/occupation sums | Parent/fixed-main max2.2862e-19. Current q² entries are omitted by the declared model, not missing symmetry images. |
| DFT velocity | `qsgw_head.py:3487` reads whole `dipole_cart` and unused `deltaE`, then slices bands | Before fix: full-BZ payload. Producer `get_dipole_mtxels.py:1564-1581` already computes analytic file-wedge velocities and uses the typed polar time-odd unfold | **Fixed** at existing consumer owner: only `kirr_fullids` rows read. Rebuild max2.2204e-16; gather-only red twin2.683864. Selected parent rows0,1,4 have identity operation. File remains full-BZ indexed and authenticated. |
| Charge `Y/Z` wings | `qsgw_head.py:2047-2080`; static surface counterpart `:2330-2348` | Already streams `iter_parent_children_faces`, one raw-parent star at a time; vertices act after child transport; velocity selects the corresponding full-k rows | No new wing builder needed. Canonical static parent/fixed-main errors Y2.7756e-17, Z2.7799e-17. Changed velocity input alone changes Y/Z by at most3.5e-18. Current wings are omitted by model. |
| Headless `W(Γ)` for Schur fold | q-IBZ packed Dyson in `w_isdf`; `head_correction.py:1397-1405` takes packed row0 | Same two-dimensionally sharded packed operator; four small folds and existing wing-half owner | Completion executes on q-IBZ successfully. These algebraic certificates do not measure covariance of the inserted Γ operator. |
| Optional Hall `sigma_H` | `static_gauge_hall_transaction`, persisted by `file_io/static_gauge_head.py:155-205`, authenticated on reload | Full-BZ uniform-current/Berry transaction, not the ordinary dipole reader; `static_hall_linear_response` makes CT/TC with the documented occupied-Berry sign | MoS2 has no Hall artifact and explicitly uses zero. Nonzero two-band toy has exact odd sign under typed antiunitary velocity transport. No CrI3 production claim. |
| `trs_allowed` | Authenticated `WfnLoader.symmetry()` / `SymMaps` | Controls allowed global-TR images; does not independently zero `sigma_H` inside completion | Broken global TR does **not** universally imply no antiunitary operations: authenticated magnetic operations can remain. The completion consumes the sealed full-BZ Hall result; it has no second TR override. |
| Parallel-transport links | `qsgw_head.load_parallel_transport_head` → `file_io.parallel_transport.load_full_bz_links` | Directed neighbor-edge reconstruction for SC covariant velocity | Not consumed by one-shot static/dynamic DFT head. These are not ordinary scalar row gathers and were not changed. |
| Cubature receipt / `q_cart` | `vcoul.slab_minibz_photon_cubature`, `services/vcoul/src/vcoul/minibz.py:732-888` | Every Γ-to-edge triangle, fixed16/24/32 Duffy–GL ladder, weighted small Dyson solves before averaging | No extra direction-star average. The polygon and edge rules supply symmetry when the mini-lattice has it. MoS2 point-group node residual3.6312e-12 and weight residual1.4095e-16; supplied Cartesian actions themselves have orthogonality defect2.6953e-11. Strict1e-12 node equality fails and is retained. |
| Completion certificates | `head_correction.py:1185-1339,1405-1448` | Fold Ward/Hermiticity; finite/conditioned Dyson backward/forward errors; mixed-scale polygon convergence | Passed on the q-IBZ completion: Ward0, Hermiticity2.191e-17, forward bound6.733e-15, final polygon error ratio0.03678. **None certifies little-group covariance.** |
| Restart factors | Four `photon_g0_vectors_A` datasets in V file before readiness stamp (`v_q_bispinor.py:605-612`) | Logical canonical serialization, fresh mesh packing, same completion rebuilt from velocities/wings/Hall/cubature | Historical Mo84→85 and Mo84→91→93 evidence retained. This lane's P16 request is separately tracked below; historical success is not presented as a new measurement. |

# Numerical results

| Check | Result | Evidence |
|---|---|---|
| Full-BZ velocity from 3 IBZ rows | maxabs2.220446049250313e-16; plain gather2.683864068895179 | Claim1006, `DEV/01_velocity_inventory/velocity.json`; CPU `lx-Xg0-014502-1406640-3994`, exit0 |
| Focused tests | 37 passed, no skips; includes poisoned nonparent rows, nonzero band-start slice, typed TR minus, provenance refusal, parent/face wings, sealed response, magnetic operation typing | Claim1022, `DEV/05_cpu_gates/pytest.xml`; CPU `lx-Xg0-015148-1454206-7406`, exit0 |
| Static isolated velocity A/B | All90 eqp0 and eqp1 rows and real printed sectors exact | Claim1022, `HEAD/01_parent_static` P4 `lx-Xg4-014817-1431804-9810`, exit0,210s; `HEAD/05_parent_full_velocity` P4 `lx-Xg4-015708-1502620-1294`, exit0,205s; BFC@0.85; `DEV/12_velocity_ab_dynamic/comparison.json` |
| Static parent vs fixed-main inputs | g0 exact; bare average exact; S2.2862e-19; Y/Z≤2.8e-17; screened moments5.6845e-14 | Claim1023, fixed-main `HEAD/02_fixed_main_static`, P4 `lx-Xg4-014817-1431800-8440`, exit0,57s BFC@0.85; canonical comparison `DEV/07_canonical_compare/comparison.json`, CPU `lx-Xg0-015355-1471730-3855` |
| Actual Γ low-rank updates | Bare V exact; screened W maxabs1.7463990836692122e-10 at scale417759.4944886803, relative4.1804e-16 | Same factors and canonical comparison; absolute1e-10 is not claimed |
| Static QP vs fixed-main | 82/90 rows exact; remaining8 differ≤1neV; all90 real printed sector rows numerically exact, including signed-zero equivalence | `DEV/12_velocity_ab_dynamic`; original Mo73 vs Mo11 has the same82/90 result in `DEV/09_eqp_little_group/historical_eqp1.txt` |
| Dynamic packed GN-PPM | All90 QP/real-sector rows exact against parent Mo80; fixed-main Mo79 eqp1 has81/90 exact and max1neV; real sectors exact | Claim1022, `HEAD/03_parent_dynamic`, P4 `lx-Xg4-015315-1465075-9083`, exit0,225s BFC@0.85; `DEV/12_velocity_ab_dynamic` |
| Full Γ TT two-index covariance | Bare worst0.5568717760215355 relative, maxabs72662.38535028041; screened same to rounding | Claim1023, `DEV/10_operator_covariance/comparison.json`; CPU `lx-Xg0-020008-1525909-9808`, exit0. Exit0 means measurement completed, **the symmetry result is FAIL**. |
| Scalar S at0 /0.5i Ry; rotated whole cell | S≤5.84e-19;0.37rad in-plane cell rotation gives node4.22e-16 / weight3.25e-18 | Claim1032, `DEV/13_rotated_cell_scalar_S/comparison.json`; CPU `lx-Xg0-020658-1587184-2094`, exit0 |
| Hall domain reversal | sigma_z−13.703599917759012 → +13.703599917759012; sigma and CT/TC odd residual0 | Claim1024, `DEV/07_canonical_compare/comparison.json`; toy only |

The CPU comparison for static/dynamic QP and real sectors is `lx-Xg0-020403-1563901-5283`, exit0. `eqp_ab.py` prints its tolerance to one decimal; the relaxed fixed-main comparison's actual argument is0.002µeV, although that field renders0.0. Strict0 comparisons are retained for the changed-vs-parent gates. `parse_eqp_rows` reports real named sectors; no claim about equality of every dynamic complex-spectrum element follows from this table.

# Why the existing certificates miss the TT failure

At Γ, every operation leaves q unchanged. The selected q table chooses one operation for that target; it does not apply every member of the little group. The completion packs channel-specific literal factors and inserts the bare product `conj(g_A) D_AB g_B / volume`. Positive/negative metric signs, Hermiticity and converged polygon averaging constrain this product but do not require its centroid/Lorentz representation to be covariant.

The accepted test takes the actual stored low-rank factors, converts their centroid axes through the canonical owner, and applies both Lorentz indices with `SymMaps.cartesian_action(axial=False,time_odd=True)` and both centroid pullbacks from `centroid_source_map_and_wrap`. Antiunitary conjugation is delegated to `apply_band_matrix_symmetry`. It contracts rows before/after the action, retaining O(N_mu) memory. Operation0 is exact; genuine mixing rotations give the0.557 residual. This is independent of the changed velocity reader because the **bare** update does not consume velocities, S or wings and is exactly the fixed-main bare update.

The earlier one-vector test in `DEV/09_eqp_little_group` is a diagnostic of the factors, **not** the two-index operator gate; using it alone would omit the second Lorentz index. The evidence above is probe10. The fit's separate current-channel normal equations (`isdf/core.py:3308-3315`) are a lead for the representation mismatch, not a proven root cause. No ISDF-rank explanation or symmetry projection has been asserted without its physical gate.

# Changes and open scope

`ae36fa1b` extracts the existing authentication into `qsgw_head.read_authenticated_dipole_velocity`, replaces the full payload read with parent hyperslabs, and routes through the already-public typed unfold. Shared parent wings and the completion are unchanged. Tests and the velocity register are included in the same commit. The producer continues writing full-BZ-indexed `dipole_cart`; accepting a differently indexed compact file would require a distinct authenticated format and is not silently inferred from shape.

The dynamic charge path calls this same builder (`gw_jax.py:735`), folds against its role-specific W, and installs the resulting samples into `HeadResolver` (`:936-942`). The resolver's separate direct diagnostic fallback can still read the legacy full payload; it is not the installed full-local-field head used by this deck. QSGW neighbor-link and uniform Hall transactions retain their existing owners.

The TT covariance defect is registered in sandbox `KNOWN_LORRAX_ISSUES.md` under **2026-09-06 — BISP-HEAD: Gamma TT completion is not little-group covariant**. Repairing it while retaining exact fixed-main bare factors is mathematically incompatible with the measured failure. No second head builder, unrequested projector, or current-fit physics change was introduced. The owner must resolve whether to extend the scope and change the full-k reference or retain this as an inherited defect beyond the scoped velocity fix.

# Restart and remaining evidence

The supplied report's initial P16 failure is superseded by its later Mo91/Mo93 chain and Mo95→97 actual-writer test; those are inherited evidence only. This lane copied `HEAD/01_parent_static/tmp` into `HEAD/04_restart_p16/tmp`, set restart=true only on that copy, and submitted P16 with `--jid57966610 --wait1800`. The authorized pool could not provide four free nodes. After1362s in prelaunch, only this lane’s waiting launcher PID1466581 was terminated; wrapper exit143, no compute step and no P16 result. No shared allocation or running step was canceled. Claim1040 records this absence; `HEAD/04_restart_p16/driver.1.log` retains the capacity refusal. The branch's P4 runs already rebuilt and captured all four factors, but a physical P16 result requires a completed step and artifact comparison.

No broad core-suite claim, CrI3 production claim, or complete current-q²/wing model claim is made. Failed exploratory launches are retained and registered in `KNOWN_SANDBOX_ERRORS.md`; none is counted as a passing numerical gate. Claims1006,1022,1023,1024,1032,1040 each have same-session `claims/NNNN.md` records. The full symmetry-completeness goal remains open because of the inherited TT failure.
