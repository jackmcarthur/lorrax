# AMENDMENT — MERGE CHECKPOINT, `integration/merge-checkpoint-2026-08-08` @ `6a4f73da` (2026-08-08)

**This supersedes the hBN amendment below for the counts, and nothing else.**
The checkpoint merges `feat/batched-canonical-2026-08-08`,
`fix/ffi-odr-2026-08-08`, and `chore/post-wave-cleanup-2026-08-08` onto
`main` @ `a16a241c`, with both `.so` legs rebuilt from the merged tree
(the ODR fix changed symbol visibility and the phdf5 type tags while the
kchunk conversion, already in `main`, changed the read-dispatch signatures —
no pre-merge pair is valid for this head).

| | |
|---|---|
| machine | Perlmutter, 4-node lx pool (JID 56485516), 4×A100 per node, Shifter, `lx test` |
| module | `LX_BASE_MODULE=lorrax_J070`, jax 0.7.0 |
| trees | `/pscratch/sd/j/jackm/merge_ckpt_2026-08-08/lorrax` (`047f2929`; the one-cell layering fix `6a4f73da` re-verified on its own file, 78/78) and `/pscratch/sd/j/jackm/wt_main_pristine` (`a16a241c`) |
| `.so` pins | the MERGED pair, built from this tree: device md5 `c680c229…`, host md5 `91f330c3…` (`merge_ckpt_2026-08-08/build_{dev,host}/`); baseline leg ran the kchunk_conv pair `main` requires. Neither leg pinned `LORRAX_FFTW3_SO`, so the FFT-engine block is red on BOTH sides and invisible to the set-diff |
| artifacts | `bwrun/suite_base.xml` — **1935 testcases**; `bwrun/suite_merged.xml` — **1966 testcases**; set-diff by `bwrun/setdiff_mc.py` |
| runs | one `lx test` each: baseline 304 s, merged head 572 s |

## The census at this head

| leg | pass | fail | skip | error | total |
|---|---|---|---|---|---|
| `tests/` (lorrax monorepo) | 1178 | 36 | 61 | 0 | 1275 |
| `services/distrib_la` | 136 | 1 | 32 | 0 | 169 |
| `services/lxkit` | 120 | 0 | 0 | 0 | 120 |
| `services/symmetry_maps` | 150 | 1 | 14 | 0 | 165 |
| `services/vcoul` | 33 | 1 | 0 | 0 | 34 |
| `services/wfn_loader` | 77 | 0 | 15 | 0 | 92 |
| `services/zeta_loader` | 110 | 0 | 1 | 0 | 111 |
| **ALL** | **1804** | **39** | **123** | **0** | **1966** |

## SET-DIFF vs `main` @ `a16a241c`

| direction | result |
|---|---|
| newly RED | **1 at `047f2929`, 0 at this tip** — `tests.test_layering::test_only_the_substrate_constructs_a_mesh` flagged the new GATE 10 file building its own cpu Mesh inside a CUDA process; sanctioned as a mesh owner at `6a4f73da` (the one construction `single_device_mesh`/`resolve_mesh` cannot express), file re-run green 78/78 |
| newly GREEN | **9** — the `services.symmetry_maps` import-isolation cells, red at `a16a241c` in full-suite order only; the per-scope skip-honesty fix (`9455e1d8`, "services stop disarming each other") is the mechanism |
| newly SKIPPED / no longer skipped | **0 / 0** |
| collection delta | **+32 new cells, all green** (17 distrib_la batched-scan + shape-algebra, 3 so_acceptance ODR-surface, 2 wfn_loader per-scope skip-honesty, 3 compile-cache agreement, 2 env registry, 5 sigma_output columns); **1 renamed away** — `test_this_gate_did_not_disarm_distrib_las` (red at base) became the two per-scope cells |
| carried red | **38**, identical by name on both sides |

The mixed-process ODR proof was re-run at the merged head against the merged
pair: gate-off arm B 46 P / 1 skip in 12 s; the deployed-pair falsification
arm C died (hung to its 900 s timeout after the cells preceding the kchunk
read path — the arity mismatch BUILD_NOTES predicts), and the kchunk probe
passes on the merged pair while its deployed-pair twin fails.  GATE 10 ALL
PASS; acceptance tier 12/12 (`merge_ckpt_2026-08-08/_reports/`).

**A trap this census recorded on the way** (it cost one invalid leg): a
census suite launched with a stale `LORRAX_CHECKOUT` ran an unrelated
worktree and produced a plausible-looking false red on the BSE anchor —
the eigenvalues matched that worktree's expanded band window exactly.  The
`[lx] source tree:` line in the log names the tree that actually ran;
READ IT before believing any leg.
