# The checklist-landings integration — branch report

`integ/checklist-landings-2026-08-27`, base `chore/local-dev-base-2026-08-27`
(`5119bdd0`). Written because five things on this branch lived only in commit
bodies on an unpushed branch: two ask for an owner ruling, one is an
API-shape change a three-way merge cannot see, one is a removed capability
that turned nothing red, and one is a test that could not run.

**Status.** Local only. `git ls-remote --heads origin
integ/checklist-landings-2026-08-27` returns no output, so nothing here is
pushed or landed. The branch name says "landings"; it is not landed.

---

## 1. The receipt gate refused the real deck

The driver died in `write_results` on the tracked `gnppm_debug` deck, after
the whole Σ chain and before `eqp0.dat`:

```
File "src/file_io/sigma_output.py", line 1618, in append_eqp_assembly_receipt_h5
ValueError: EQP assembly receipt row count disagrees with the raw cube's
canonical star map: completed assembly has 9 k rows, raw artifact resolves
to 5 canonical rows.
```

A CrI3 bispinor run produced the same message on 2026-08-26 (Stampede3).
All 105 cells of the receipt's own suites passed at the same time.

**Cause: two different k-sets share one name.**

| | rows | produced by | who is on it |
|---|---|---|---|
| file wedge | `sym.nk_red` = `wfn.kpoints` | `symmetry_maps.reduce_full_bz_to_file_wedge` | `eqp0.dat`, `eqp1.dat`, `sigma_diag.dat`, `kirr_to_kfull`, the completed `EqpAssembly` |
| star wedge | one row per symmetry orbit | `compact_star_tables` via `extract_and_stamp_k_irr` | every dataset inside `sigma_mnk.h5` |

They have the same length only when `nk_red == n_orbits`. From
`symmetry_maps`' own table: `gnppm_debug` 9 and 5, `bispinor_debug` 9 and 5,
`cohsex_debug` 4 and 3; equal on `si_cohsex_debug`, `si_bse_debug`,
`hbn_cohsex_debug`. Every fixture the schema was developed against is in the
second group.

The writer computed the star wedge, stamped it under the file wedge's
attribute name, and then required the assembly to have that many rows.
Confirmed on the artifact the failing run wrote: `sigma_c_kij_ev` is
`(41, 5, 46, 46)` with `k_storage=ibz`, `nk_full=9`,
`irr_idx_k = [0,1,1,2,3,4,2,4,3]`; the deck's `WFN.h5` has `nrk = 9` on a
`[3,3,1]` grid, so the WFN stores the whole zone.

The star map cannot supply the file wedge: it labels every full-BZ k with
its orbit and says nothing about which k the WFN stored. `cohsex_debug`
keeps four rows for three orbits, one of them the TRS partner of another.

**Fix.**

1. `append_eqp_assembly_receipt_h5` takes `file_wedge_full_bz_rows` as a
   required keyword. `write_results` passes `_wedge(np.arange(nk_full))` —
   the same reduction that produced the assembly's rows, applied to an index
   vector, as it already does for `kpts_irr`.
2. The count check becomes: one row per assembly row, distinct, in range for
   the cube's mesh, and every stored star reached by at least one row.

The old check compared counts only. On this deck it refused the correct
9-row assembly and would have accepted any 5-row one, including
`[0,1,2,3,6]`, which covers 3 of the 5 stars (`old_gate_probe.log`).

**Cells in `tests/test_sigma_eqp_assembly_receipt.py`:** the real 9-vs-5
topology accepted; a 5-row wedge missing two stars refused; wrong count,
out-of-range row and repeated row refused; rows `[3,2]` stamped verbatim
where the star map would have said `[0,2]`; and
`test_receipt_lands_on_the_real_gnppm_run`, which consumes `gnppm_session`
and reads the `sigma_mnk.h5` a driver run wrote, asserting that this deck's
two wedges do differ so the cell cannot pass by coinciding.

## 1a. Open ruling: the evaluation stamp is on the other wedge

`sigma_eval_rel_ev` is written by the cube writer, so it is on the star
wedge `(5, 46)` while the receipt is on the file wedge `(9, 46)`.
`read_eqp_assembly_receipt` pairs them by asserting equal shapes, and
therefore refuses the real deck. Measured on the accepted run's artifact:

```
ValueError: sigma_eval_rel_ev has shape (5, 46); expected the receipt's
file-wedge/window shape (9, 46).  The stamp has 5 k rows against the
receipt's 9: the stamp is on the cube's STAR wedge and the receipt on the
FILE wedge, and pairing them needs the star substitution this module
refuses (k_irr_rows_for).
```

Not resolved here, because both answers are physics rulings:

- unfolding the stamp along the star is the substitution `k_irr_rows_for`
  exists to refuse — though these are DFT energies and the cube's own
  `star_spread_diag_ev` on that dataset reads exactly `0.0` on this deck,
  which is an argument for calling them star-invariant, not a proof;
- stamping them on the file wedge instead is a schema change.

What landed is a named refusal that says which mismatch it is. Consequence:
`make_eqp_bgw` cannot consume a receipt on a deck whose two wedges differ.
It could not consume such a deck before either — the legacy reader routes
`kirr_to_kfull` through `k_irr_rows_for`, which refuses a full-BZ row that
is not itself a stored row — so no route is lost.

## 2. API-shape changes other lanes cannot see

| # | Change | Who it bites |
|---|--------|--------------|
| a | `file_io.append_eqp_assembly_receipt_h5` gains a required `file_wedge_full_bz_rows=`. | any caller; one in `src/`, four in `tests/`, all updated |
| b | `gw.gw_output.write_results`'s `qp_solver` loses its `None` default. | the one caller already passes it |
| c | `gw.sigma_dispatch.finalize_dynamic_sigma` returns `sigma_omega_h5_path=None` when `write_sigma_omega_h5=False`, where it returned a path for a file it had not written. | anything reading that field as "the path I would write" |
| d | `gw.gw_jax` raises at driver entry on `qp_solver=self_consistent` beside a dynamic `compute_mode` (imported; §3). | any deck with that pair |

(b) is the one the audit asked to be surfaced rather than fixed silently:
one cherry-picked commit shipped two policies for one parameter — required
on `write_qp_wfn_oneshot`, defaulted to `None` on `write_results`, where the
`None` arm wrote an unstamped `qp_wfn_rotations.h5`. Resolved toward the
sibling; the arm was unreachable from the one call site.

(c) closes a trap armed for the change that re-enables SC output:
`write_qsgw_sigma_cube` opens that path `"a"`, so h5py would create a file
with no ω axis and no raw operators, and the receipt append raises
`FileNotFoundError`.

## 3. A capability was removed on purpose upstream

`qp_solver = self_consistent` with a dynamic `compute_mode` now raises at
driver entry. Upstream `92130176` says so in its own message:

> Refuse dynamic self-consistent output before the expensive SC loop because
> finalized H/X are dft_band while its C(omega) remains qp_band, and retain
> the late post-Sigma invariant.

The defect it fail-closes is real and never measured: the SC finalize
rotates V_H / Σ_x / Σ_xc / Σ_SX / Σ_COH to `dft_band` and leaves
`sigma_c_omega_kij_ry` in `qp_band` by design (it is the QSGW ansatz's
operand, `sigma_dispatch.SIGMA_BASIS_FIELDS`), and the EQP sum holds only at
`U = identity`. See `tests/KNOWN_FAILURES.md`, "eqp0.dat / eqp1.dat mix two
bases on the self-consistent path", open since 2026-08-05.

Kept, with three things it was missing:

1. The refusal now names the alternative (`cohsex`, `one_shot_dft`,
   `fixed_point`) and carries got / want / why / fix / doc. It was a bare
   `ValueError` telling the reader to rotate the correlation operator, which
   a deck author cannot do.
2. The docs no longer advertise it unqualified: `docs/input_reference.md`,
   `docs/drivers.md`, `tools/gen_input_reference.py` (kept identical to the
   `.md`, since the generator itself cannot run).
3. It now turns something red. `tests/regression/gnppm_debug/gnppm_sc.in` is
   the tree's only self-consistent deck and its `gnppm_sc_session` fixture
   had no consumer, so the removal was invisible.
   `test_the_one_self_consistent_deck_is_the_pair_the_driver_refuses` reads
   that deck and AST-checks the guard. Falsified against the trees without
   it: 0 matches at `5119bdd0` and `18c5b718`, 1 from `44ee0dfa` on.

The refusal is still not discoverable at parse time: `gw_config` accepts the
pair and the driver refuses it. Moving the check next to the sibling
solver × mode gate at `gw_config.py:4053` is a further step not taken here.

## 4. Flags that were buried in commit bodies

**(a) The eqp2 receipt gap.** `append_eqp_assembly_receipt_h5` runs at
`gw_output.py:~1486` and the fixed-Σ `eqp2.dat` block runs after it at
`~1498`, so `eqp2.dat` carries no receipt provenance. Nothing breaks — the
paths are sequential — and no test covers the combination, because the two
branches were developed against different bases. Closing it means new schema
fields, so it is recorded rather than fixed here.

**(b) Direct-field branch divergence (resolved).** At this report's baseline,
`18c5b718` omitted upstream's `hartree_already_resolved=True`; this was a
numerical no-op because the folded-source matrix was zeroed, but it left a
repeatable merge conflict and changed the reported rule label. `62f1fa22`
deleted source selection: live G-space is now the only Hartree path.

**(c) The evaluation-stamp wedge ruling** — §1a.

**(d) A false discriminator claim.** `44ee0dfa`'s imported message claims
discriminators for "early/late mixed-basis refusal". No test in this tree
reached either raise: `grep -rn "basis-consistent" tests/*.py` returns 0, no
test imports `gw.gw_jax`, and the deck that would have (`gnppm_sc.in` via
`gnppm_sc_session`) had no consumer. The commit message is amended to say
so; §3.3's tombstone supplies the missing discriminator.

## 5. The permanently-red cell

`test_capped_lru_write_cost_is_explicitly_unmeasured` failed deterministically
at the branch tip — the only new local failure the branch introduced. The
cell sets a finite `compilation_cache_max_size`, which puts jax's `LRUCache`
on its eviction branch, and that branch raises `RuntimeError: Please install
the filelock package to set jax_compilation_cache_max_size`
(`jax/_src/lru_cache.py:73`). The introducing commit's body quoted
`ModuleNotFoundError: No module named 'filelock'`, which is not what the run
produces; that quote is corrected in the amended message.

The cell now skips with a reason saying a skip here is not a pass, and the
skip was watched firing with `filelock` uninstalled. With `filelock` 3.32.4
installed it passes (39 passed).

**Still open:** `filelock` is in neither `dependencies` nor any extra in
`pyproject.toml`, so on a fresh checkout the cell silently skips and the
eviction receipt goes unverified. Declaring it as a test extra is the
follow-up.

## 6. What was measured

SCOPE: WSL login-class box, one process, CPU, `JAX_PLATFORMS=cpu`,
`/home/jackm/projects/lorrax_cloud13/.venv/bin/python` (jax 0.9.1),
`PYTHONPATH=<worktree>/src` plus each `services/*/src`, both FFI legs built
out of tree from `config/cloud`. A snapshot of this worktree, not a property
of `main`. No compute-node launch, no multi-rank run, no GPU, no BerkeleyGW
comparison — nothing here is evidence about numbers.

| what | result |
|---|---|
| `gnppm_debug` deck end to end, 1 process / 1 CPU device, `memory_per_device_gb` 3 | **rc 0**; `eqp0.dat`, `eqp1.dat`, `eqp_g0w0.dat`, `qp_wfn_rotations.h5`, `WFN_qp.h5`, `sigma_mnk.h5` written |
| the receipt in that file | 9 k rows, `file_wedge_full_bz_rows = 0..8`, over a 5-row cube; every stored star covered |
| `read_eqp_assembly_receipt` on that file | refuses, with the §1a message |
| `tests/test_sigma_eqp_assembly_receipt.py` | 14 passed (+1 deck cell, run against the real run dir) |
| `tests/test_qp_solver_config.py` | 41 passed |
| `tests/test_sanity_gates_jax.py` with both FFI legs | 8 failed → 38 passed |
| `tests/test_compile_cache_jax_compat.py` | 1 failed → 39 passed (38 passed / 1 skipped without `filelock`) |
| the 16 touched suites, both FFI legs, at `6fafd126` | 1 failed / 301 passed / 8 skipped |
| the same 16 at the branch tip | 1 failed / 304 passed / 8 skipped |

The audit measured 9 failures on those 16 suites with no FFI library built:
8 missing-`liblorrax_ffi_host` names and the `filelock` cell. Building both
legs clears the 8 and installing `filelock` clears the ninth. The one
failure left is the same cell on both sides of the fix —
`test_w_head_densify.py::test_the_loader_does_not_defer_when_the_grids_are_equal`,
which passes when its file runs alone and fails after another suite has
changed the process's device set. Recorded in `tests/KNOWN_FAILURES.md`;
it is not a branch regression.

`test_receipt_lands_on_the_real_gnppm_run` is deselected from that subset
because its fixture runs the deck at the tracked `memory_per_device_gb =
28`. Its body was executed against the accepted run's directory instead,
and it also fails in the falsifying direction (a receipt stamped with the
star wedge).

Logs: `scratchpad/fix_integrations/` in this session (`gnppm_accept.log`,
`receipt_check.log`, `old_gate_probe.log`, `real_cell_probe.log`,
`sub_head.log`, `sub_prefix.log`). Session scratch is not durable, which is
why the numbers are excerpted above.

The deck asks for `memory_per_device_gb = 28`; this box has 7 GB, so the
acceptance run set it to 3. A 4-device single-process mesh is refused on
this branch by `ffi/io.py:121` (the emulated-mesh SlabIO tier is
`feat/emulated-multidevice-cpu-2026-08-27`, not this branch), so the run is
one device.
