# The Σ evaluation stamp and the eqp receipt sit on different wedges (open ruling)

Moved here from the deleted branch report
`docs/reports/INTEG_CHECKLIST_LANDINGS_2026-08-27.md` §1a, which the error
message in `file_io/restart_bundle.read_eqp_assembly_receipt` used to cite.

`sigma_eval_rel_ev` is written by the Σ cube writer, so it is on the STAR
wedge. The eqp receipt is on the FILE wedge
([symmetry register §8](../../docs/architecture/symmetry_register.md)).
`read_eqp_assembly_receipt` pairs them by asserting equal shapes, so it
refuses a deck whose two wedges differ (measured: stamp `(5, 46)` against
receipt `(9, 46)`). The refusal names the mismatch.

Both fixes are physics rulings, so neither landed:

- Unfold the stamp along the star. That is the substitution `k_irr_rows_for`
  exists to refuse. These are DFT energies and the cube's
  `star_spread_diag_ev` read exactly `0.0` on that deck, which argues for
  calling them star-invariant but does not prove it.
- Stamp the evaluation energies on the file wedge. That is a schema change.

Consequence: `make_eqp_bgw` cannot consume a receipt on a deck whose two
wedges differ.
