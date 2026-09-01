# SC — signed ±5 eV live self-consistency

## Achieved numbers

**Two SC maps completed; map 2 planned and entered its Sigma sweep.**  The
historical map-1 `_BudgetShortfall` is gone: all **36/36** emitted census rows
across maps 0--2 were `served`, with zero null candidate families and zero
budget shortfalls.

| map | result | `max|dE|` | windows | pairs | protected bands (1-based) |
|---:|---|---:|---:|---:|---|
| 0 | complete | 0.992367 eV | 12 | 138 | 9--20 |
| 1 | complete | 0.326653 eV | 12 | 183 | 9--16 |
| 2 | sweep started, then funded wall | not produced | 12 | 179 | 9--16 at map entry |

The protected mask was **not stable across all maps**: its exact 24-bit value
changed from `000000001111111111110000` at map 0 to
`000000001111111100000000` at map 1.  Map 2 entered with the latter eight-band
partition again, so the measured sequence is 12, 8, 8 protected bands rather
than the old immediate 12, 8, 12 alternation.  Two completed maps are not
enough to certify that a later period-2 cycle is absent.

The run did **not converge** and did **not reach `sc_max_iter=8`**.  The
800-second attempt cap sent SIGTERM to all four ranks during map 2; after its
grace period the launcher reported exit 137 at 924 s.  This timing and the
four simultaneous SIGTERM records identify the funded wall, despite the
launcher's generic “usually OOM” wording.

## Tree, merge, and proof

Source was `test/sc-2026-09-01` at
`0db36749a07936eae21dfe8468fce5a8ea5559fb`, on four A100s/four ranks,
BFC@0.85.  The prescribed CPU gate passed **136/136 in 158.42 s**; the count is
one below the consolidated 137 because the removed catalog's rung test is no
longer collected.  After adding the per-map mask print, the focused hybrid
gate passed **8/8 in 2.78 s**.

All four requested branches are merge ancestors.  Conflict resolution kept
both per-frequency pointwise envelopes and the screened-pole factor reference.
The marginal branch's table-enumeration implementation was incompatible with
the owner's on-demand-only ruling, so the merged production path keeps
on-demand fitting and does not resurrect `_sign_definite_table_candidates`.
Every census row in this run reports `candidate_family=measure_adapted_roq`.

The first strict-policy launch refused before SC because the 48-band chi edge
equals the WFN extent and cannot prove multiplet closure.  The measured arm
therefore restored the supplied launcher's explicit legacy diagnostic
`LORRAX_BAND_DEGENERACY=snap`; it also used a fresh planner-cache filename, and
all three plans report `plan_cache=miss`.  The inherited deck printed three
ignored legacy keys (`mpa_material_class`, `sigma_omega_layout`, and
`mpa_sigma_crossing_target_error`); startup independently inferred metal,
resolved the Sigma layout to sharded, and used the recognized sector target
of `1e-4`.

Evidence:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/60_sc_delivered_20260831/test_sc_codex_20260901`
(`sc_live_merged_snap.log`, `eqp0_iter000{0,1}.dat`, `job_receipt.txt`).

Branch: `test/sc-2026-09-01`.
