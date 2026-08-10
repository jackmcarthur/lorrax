# AMENDMENT — `kirr_fullids`: THE WEDGE ROW MAP, FIXED, AND THE ONE FIXTURE IT LEAVES STALE (2026-08-08)

`fix/kirr-fullids-2026-08-08`, off `main` @ `bc37b4d3`.  Nothing on this list
turns red.  The amendment is here for the two things a green suite does not
say: which committed fixture now disagrees with the live code, and which live
code path stops answering and starts refusing.

**THE DEFECT.**  `SymMaps.kirr_fullids[i]` is supposed to be the full-BZ row
that IS the WFN file's irreducible k-point `i`, and every wedge-shaped output
in the tree gathers with it — `gw_output.write_results` builds every column of
`eqp0.dat` / `eqp1.dat` that way, and `sc_iteration.dump_qp_wfn_artifacts`
reads `U` at those rows.  It was built instead from the STAR LABELS: the first
full-BZ row carrying label `i`, taken out of `irr_idx_k`, with a silent
identity fallback `kirr_fullids[i] = i` for labels no row carries.  Labels do
go uncarried: `find_symmetry_ops_simple`'s op-selection policy has no `break`,
so a full-BZ row reachable from more than one stored IBZ point is labelled
with the HIGHEST of them and the lower ones are orphaned, which happens on any
deck whose stored wedge has two entries in one orbit.

**MEASURED at `bc37b4d3`, all four in-tree decks:**

| deck | shipped | correct | damage |
|---|---|---|---|
| `gnppm_debug` | `[0,1,1,3,4,5,3,5,4]` | `[0..8]` | 4 rows name a k 1/3 of a **b** away; 3 row pairs collide; IBZ k2, k6, k7, k8 never emitted |
| `bispinor_debug` | `[0,1,1,3,4,5,3,5,4]` | `[0..8]` | identical to the above (same mesh, same group) |
| `cohsex_debug` | `[0,1,1,4]` | `[0,1,2,4]` | row 2 duplicates row 1; IBZ k2 never emitted |
| `si_cohsex_debug` | `[0,1,2,5,6,7,10,27]` | same | correct **by luck** — 48 ops, eight disjoint stars, no orphaned label |
| `hbn_cohsex_debug` | `arange(18)` | same | `ntran = 1`, so the trivial branch, which is right by construction |

The fix matches on k itself — the row whose `unfolded_kpts` entry equals
`wfn.kpoints[i]` modulo a reciprocal lattice vector, to
`find_symmetry_ops_simple`'s own 1e-6 — and raises instead of falling back
when a stored k is not on the grid its file's `kgrid`/`shift` generate.  New
gate: `services/symmetry_maps/tests/test_symmetry_maps_kirr_fullids.py`,
14 cells, of which 10 are red on the pre-fix construction.

**FIXTURE ADJUDICATION — one file, registered, NOT re-frozen.**

`tests/regression/cohsex_debug/qp_wfn_rotations.h5` carries a
`kirr_to_kfull` dataset whose stored value is `[0, 1, 1, 4]` — the broken map,
frozen on the day the file was written.  The live class now produces
`[0, 1, 2, 4]`, so the blob and the code disagree.  **It is not re-frozen
here**: re-cutting a committed deck artifact is the re-cut wave's row, not a
fix branch's, and the manifest discipline is that a stale blob is registered
with its measured value rather than quietly replaced.  Whoever next re-cuts
`cohsex_debug` should regenerate the file and strike this paragraph.

Everything else in `tests/regression/` was checked and is CLEAN.  The reason
is worth stating because it is what kept the defect out of the gates: every
committed `eqp_*.dat` / `sigma_diag_*.dat` in the tree is the SIGMA-DIAGNOSTIC
format (`k-point N:` blocks) written on the FULL BZ un-subset — 9 blocks on
the 3×3×1 decks, 18 on hBN, 8 only on Si — and the sigma-diagnostic writer
never touches `kirr_to_kfull`.  No committed fixture is a BGW-format wedge
`eqp{0,1}.dat`.  `gnppm_debug/eqp_rotations_fixedpoint_ref.npy` is `(9, 46)`,
i.e. full-BZ, and `cohsex_debug/sigma_mnk.h5` carries no wedge map at all.

**ONE LIVE PATH CHANGES ANSWER TO REFUSAL, and it is the designed one.**
`eqp_bgw`'s post-hoc CLI pairs `kirr_to_kfull` with a wedge-stored
`sigma_mnk.h5` through `file_io.sigma_output.k_irr_rows_for`, which refuses
any requested row that is not itself a stored row.  On `cohsex_debug` the k_irr
store keeps rows `[0, 1, 4]`; the broken map asked for `[0, 1, 1, 4]`, which
all land on stored rows, so the CLI silently handed back **k1's matrix under
k2's label**.  The corrected map asks for `[0, 1, 2, 4]`, row 2 is not stored,
and the call now raises by name.  That is `k_irr_rows_for`'s whole purpose —
"the refusal is the whole point", `sigma_output.py:775` — reaching a real deck
for the first time.  It is a silent wrong answer becoming a loud one, not a
regression, and no collected test exercises that pairing on that deck.

**§7.7.1 OF `BGW_CD_COMPARISON_DESIGN.md` IS CORRECTED BY THIS BRANCH.**  That
section attributes its mislabelled k rows to `kirr_fullids` evaluating to
`[0..7]` on the Σ_x probe's tree.  It did not: that tree (`si_mpa_0808/wt` @
`59fa874b`) carries the same construction as `main`, its deck's stored wedge is
the same eight k in the same order as `si_cohsex_debug`, and the run's own
`eqp0.dat` lists all eight correct IBZ k.  What the probe read was
`eqp_r1128x.dat`, the **sigma-diagnostic dump, 64 `k-point` blocks on the full
BZ**, whose first eight rows are full-BZ rows 0-7 with IBZ parents
1, 2, 3, 2, 2, 4, 5, 6 — exactly the table §7.7.1 prints.  The probe's remap
and every measurement built on it stand; the attribution does not.  The defect
above is real and was found independently, on the other three decks.
