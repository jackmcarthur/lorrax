# AFIXES — lane note, 2026-09-11

Branch `lane/sp-afixes-2026-09-11`, control `810c260b`, candidate **`58a1aad1`**
(was `ff02272a` when this session opened; `58a1aad1` adds the refusal-fidelity
repair below). Worktree `/pscratch/sd/j/jackm/wt_sp_afixes`.

Pool **58209192** (`lx-alloc-jackm-SP-INT2`, interactive, 4 nodes), shared with
ACARRY and ACONJ. The originally assigned SP-M2 pool `58190627` was **cancelled**
before it ever ran, which is why the prepared supervisor stopped; interactive
QOS is at its per-user submit limit, so this lane took no new pool.

Full write-up with every number and its job.step:
`reports/shared_pole_push_2026-09-07/afixes/report.md`.
Evidence root: `runs/frequency_integration_sandbox/364_afixes_20260911/`.

## Dispositions

| Item | Disposition | Evidence |
|---|---|---|
| 1a `tagged_arrays.py:218` | **Repaired.** Agree the serial-read error and the membership hash before the collective validator; stop double-wrapping `SharedPoleMemberRefused`. Plus a second defect this session identified: the agreed receipt truncates at 240 characters (`common/collectives.py:1422`) and lost the reason on this tree's long paths. `file_io/commit_state.py:agree_io_refusal` is now the one owner for both readers. | 58209192.8 red (reason truncated), 58209192.11 green on 4 ranks |
| 1b `shared_pole_store.py:543` | **Repaired; diagnosis narrowed.** Only `shard` could be unbound — `b` is bound at `:531` and `local = None` at `:533`. | `test_digest_empty_addressable_shards`, 58209192.11 |
| 2a `sigma_box_plan.py:342` | **Repaired.** Authenticate the stored `eps`/`relative`, then filter; an adjacent-ULP request now reuses the certificate instead of discarding it. | `test_cache_one_ulp_request_reuses_authenticated_stored_eps`, 58209192.4 (P1, 150 passed) |
| 2b `sigma_box_plan.py:297` | **Repaired for visibility and invalidation; pruning deliberately deferred** (removing files would mutate completed evidence, and a prune dial is forbidden). Live example: run 337's Na cache holds **twelve** readable `rule_<digest>.npz` files invisible to the current listing. | `test_old_cache_namespace_is_not_opened`, 58209192.4 |
| 3 acceptance cloud | **Measured, nothing changed.** 4.99x cloud costs exactly **one removal pass** and **one extra tau node** on two independent Na crossing boxes; the accepted rule is slightly more accurate; planning wall did not rise. Exactly reproducible on repeat. | 58209192.4 and 58209192.12 |
| 4 `shared_pole_constructor.py:104` | **Repaired.** One production call site, aliased panels, so `b` is identically `a^H`; the general two-GEMM path is kept for distinct panels. | `test_pencil_alias_and_distinct_panels`; Si band below |
| 5a `polar.py:206` | **Repaired on the production (batch_reshard) route; distributed route deliberately deferred.** The validation now costs **zero** collectives; the pre-check it replaced cost 3 all-gathers + 1 all-reduce and a 3.2x temporary. XLA lowers it as all-gather + all-reduce, **not** all-to-all. | HLO census on every rank, 58209192.11 |
| 5b `shared_pole_constructor.py:764` | **Repaired.** Workspace resolves through the matmul route, not the eigh plan; transpose staging priced separately and is **zero** on both reference decks. | `test_afixes_workspace.py`, 58209192.11 |
| 6a `shared_pole_store.py:1149` | **Repaired.** 1624 header validations and 1624 collective opens on Na become one validate and one append handle. | `test_shared_pole_outputs.py` asserts `['w','a']` and one validate; 58209192.11 |
| 6b `shared_pole_screening.py:150`/`:224` | **Repaired.** Public `wfn.path`, resolved and refused before any screening work; empty `wfn_file` rejected at deck parse. | two contracts, 58209192.11 |
| 6c `gw_config.py:3211` | **Repaired.** Membership in `_NULLABLE_INT`, not a literal key name. | `test_nullable_integer_none_is_uniform` |
| 6d `efermi.py:502` | **Repaired.** Messages name `solve_smearing_occupations`; the width names kBT for FD. | `test_smearing_width_diagnostic_names_family` |

## Registered, not fixed here

- `common/collectives.py:1422` — the agreed I/O receipt truncates the reason at
  240 characters. Not this lane's file; in `KNOWN_LORRAX_ISSUES.md`.
- `tests/test_sigma_box_plan.py` — P=1-only by construction and unmarked
  (`sigma_box_plan.py:601` shards windows over `process_count`). At P4 it
  reports 20 failures that are pure harness artifacts; 150-green at P1 on the
  same source. In `KNOWN_LORRAX_ISSUES.md`.

## Scope

`src/gw/mpa/sigma.py` untouched (ACARRY). `shared_pole_constructor.py:511-560`
untouched (ACONJ). `services/minimax/src/minimax/uniform_rule.py` untouched —
changing the acceptance density is the owner's call. Budget policy, currency,
defaults and wall-clock termination in `sigma_box_plan.py` untouched. No new
dial, env var, cache, fast path or magic constant.

**Bit-identical Sigma is not required (ruling 65d).** Never combine figures
across source assemblies or job.steps: control arms run `810c260b`, candidate
arms `58a1aad1`, both in the same session on the same pool and node.
